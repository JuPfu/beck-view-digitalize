# DigitizeVideo.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
DigitizeVideo - top-level capture & orchestration for beck-view-digitize.

Design notes (Option B):
- Ft232hConnector is treated as the owner of the FTDI device and the poller.
  DigitizeVideo will attempt to instantiate a Connector in the "Option B"
  constructor form:
      Ft232hConnector(signal_subject, max_count, gui)
  If that constructor isn't available (older connector), DigitizeVideo will
  create and open a pyftdi.Ftdi() instance and pass it into the older API:
      Ft232hConnector(ftdi, signal_subject, max_count, gui)

- DigitizeVideo subscribes to the signal_subject for incoming frame events,
  maintains shared memory buffers which are handed to worker processes, and
  writes frames via WriteImages (process pool).

- Capture is decoupled from the poller:
  The poller thread (Ft232hConnector) only publishes (count, timestamp) events.
  The subscription hops over to a dedicated single-thread scheduler
  (observe_on(EventLoopScheduler)), so camera reads (cap.read()) never block the
  poller. A slow read can therefore no longer make the poller overlook the next
  OK1 pulse (OK1 is only ~10 ms wide, the projector cycle is ~155 ms).
  Events are queued and processed in order; on_completed is queued behind the last
  event, so final_write_to_disk() always runs after the last queued capture.
  NOTE: decoupling prevents lost *events*; a capture that starts late still delivers a *late*
  image. A warning is logged when a capture starts later than the earliest moment its frame could
  be selected (trigger + d_min), i.e. when the capture thread is really behind. A capture that is
  merely waiting for its camera frame (d_min + up to one camera period) is not a backlog.

- Shutdown sequence:
    1) FT232H poller signals EOF by calling subject.on_completed()
    2) The completion is delivered on the capture thread after all queued captures
       and starts final_write_to_disk() to flush remaining frames to disk.
    3) After final write finishes, the Connector is closed, pools are closed,
       shared memory is unlinked, and the program exits gracefully.
"""

import cython
from cython.view cimport array

cimport numpy as np
import numpy as np

from datetime import datetime

from libc.string cimport memcpy

import platform
import logging
import multiprocessing
from collections import deque
from multiprocessing import shared_memory
import concurrent.futures
import threading
import time
import signal
import sys
from argparse import Namespace
from pathlib import Path
from typing import List


import cv2
from reactivex import operators as ops, Observer
from reactivex.scheduler import EventLoopScheduler
from reactivex.subject import Subject

from TypeDefinitions import ImgDescType, SubjectDescType
from WriteImages import write_images

from Ft232hConnector cimport Ft232hConnector
from Ft232hConnector import get_timing, get_timing_view

from TimingResult cimport TimingResult as CTimingResult

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Number of uint32 columns per frame in the descriptor shared memory handed to WriteImages:
#   0 = bytes per image, 1 = film frame number, 2 = bracket index, 3 = suspect flag (1 = suspect)
# write_images() gets this number as desc_cols, so both sides always agree on the layout.
# Suspect captures are written as frameNNNNNx_SUSPECT.png.
DESCRIPTOR_COLUMNS = 4

# d_min: earliest delay after the trigger at which a camera frame is taken as "the current film
# frame". A frame that arrives sooner was exposed while the rotating shutter still covered the lens
# (exposure + sensor read-out + USB transfer + driver all lie between exposure and arrival).
# The default is a deliberately cautious placeholder: one camera period at 30 fps (33 ms) plus the
# exposure (8 ms). It is NOT calibrated. Determine the real value with d_min_from_images.py and pass it
# as capture_min_delay_ms. It belongs to the camera setup (camera, resolution, exposure, fps, USB,
# computer), not to the film or the projector speed.
DEFAULT_CAPTURE_MIN_DELAY_MS = 40.0

# width of the shutter blade in degrees (the lens is covered by it during film transport)
DEFAULT_SHUTTER_DEGREES = 70.0

# A capture that starts later than this (s) after the earliest moment its frame could be selected
# (trigger + d_min) is logged: the capture thread is behind. About one camera period.
CAPTURE_START_LAG_WARN_S = 0.035


cdef class DigitizeVideo:
    """
    Cython extension class for digitizing Super8 frames.

    Responsibilities:
    - Initialize camera and shared buffers.
    - Subscribe to photo-cell signal subject and capture frames into shared memory.
    - Manage a process pool to write chunks of images to disk asynchronously.
    - Wait for completion (EOF) and perform a clean shutdown.
    """

    cdef:
        # configuration & runtime state
        int device_number
        object output_path          # pathlib.Path at runtime (store as object)
        bytes output_path_b         # path converted to bytes
        object exposures            # list of (exposure, suffix) tuples
        object timing               # TimingResult singleton (Python object)

        int width
        int height
        bint bracketing
        int frame_multiplier
        int chunk_size              # images (not film frames) per buffer / per worker task
        bint settings
        bint gui

        # RX subject, logger, and completion sync
        object signal_subject       # reactivex Subject (Python)
        object logger               # Python logger
        object _completion_event    # threading.Event set when the final write has finished

        # camera runtime parameters (after open)
        int img_width
        int img_height
        int img_bytes               # bytes per frame (w*h*3)

        # per-frame bookkeeping
        list img_desc               # list for lightweight logging descriptors
        object img_desc_lock        # threading.Lock

        # blank frame to copy on read errors
        np.ndarray blank_frame

        # thread/process pools and related objects
        object executor                 # ThreadPoolExecutor (python object)
        object signal_observer          # SignalObserver instance
        object photoCellSignalDisposable  # subscription disposable
        object completion_disposable     # subscription for on_completed -> set event
        object wait_subject             # completed when the shutdown is finished

        # continuous camera reader ("grabber") and frame selection
        object _grab_thread             # thread that reads the camera all the time
        object _grab_stop               # threading.Event: ask the grabber to end
        object _frame_ring              # deque of (seq, arrival_time, frame), oldest first
        object _frame_cond              # threading.Condition protecting the ring
        object _cap_lock                # serializes cap.grab()/retrieve() and cap.set()
        long _grab_seq                  # sequence number of the newest frame
        double capture_min_delay        # s: d_min, earliest arrival after the trigger that counts
        double capture_window           # s after d_min within which a frame must arrive
        double open_fraction            # share of a projector cycle with the lens open
        bint _min_delay_is_default      # True: d_min was not configured (uncalibrated default)
        object _trig_hist               # last trigger timestamps (cycle estimate)
        long _budget_warned_at          # film frame of the last budget warning
        double capture_timeout          # s: longest wait for a frame after a trigger
        int bracket_settle_frames       # frames to skip after an exposure change
        int ring_size
        long suspect_count
        object capture_info             # film frame -> (delay_ms of first bracket, suspect 0/1, work_ms)

        # decoupling of capture from the poller thread
        object capture_scheduler        # EventLoopScheduler: single "capture" thread

        # shared memory buffers for workers
        list shared_buffers            # list of shared_memory.SharedMemory objects
        list _shm_arrays               # numpy views (per shared buffer)
        list _desc_shms                # descriptor shared_memory objects
        list _desc_arrays              # numpy descriptor arrays (per buffer)

        int shared_buffers_index
        list buffers_in_use
        object buffer_lock             # threading.Lock, protects buffers_in_use and processes
        list processes                 # list of (AsyncResult, shm, desc_shm, buf_idx)
        int process_count
        int _frames_per_buffer

        # camera handle and settings
        object cap
        int compression_level
        object pool                    # multiprocessing.Pool (worker processes)

        # FTDI connector (set by connect())
        object ft232h                  # Ft232hConnector instance

        # stats
        long processed_frames
        double start_time
        double new_tick

        # administration
        bint _final_write_done
        bint _cleaned_up

        CTimingResult timing_view

    def __cinit__(self):
        # nothing heavy here — ensure numpy API will be initialised in __init__
        pass

    def __init__(self, args: Namespace, signal_subject: Subject, wait_subject: Subject) -> None:
        """
        Initialize the DigitizeVideo instance.

        This constructor creates:
         - camera capture object
         - shared memory buffers and worker pool
         - FT232H connector (Option B), starts poller, and subscribes to subject

        The method blocks very briefly while opening camera and setting properties
        but returns quickly; the FTDI poller runs in its own thread.
        """
        np.import_array()

        # command-line values
        self.device_number = args.device
        self.output_path = Path(args.output_path)
        self.output_path_b = str(self.output_path).encode('utf-8')
        self.width = args.width
        self.height = args.height
        self.bracketing = bool(args.bracketing)
        self.frame_multiplier = 3 if self.bracketing else 1
        self.chunk_size = args.chunk_size - (args.chunk_size % self.frame_multiplier)
        self.settings = bool(args.settings)
        self.gui = bool(args.gui)

        # frame selection (optional attributes of args; defaults are placeholders to be tuned
        # to the real rest time of the film frame in front of the camera)
        min_delay_ms = getattr(args, "capture_min_delay_ms", None)
        self._min_delay_is_default = min_delay_ms is None
        if min_delay_ms is None:
            min_delay_ms = DEFAULT_CAPTURE_MIN_DELAY_MS
        self.capture_min_delay = max(0.0, float(min_delay_ms)) / 1000.0
        # capture_window_ms is counted from d_min: a frame arriving later than d_min + window is LATE
        # (60 ms = one camera period plus jitter; it only trips when the camera stalls)
        self.capture_window = float(getattr(args, "capture_window_ms", 60.0)) / 1000.0
        shutter_deg = float(getattr(args, "shutter_degrees", DEFAULT_SHUTTER_DEGREES))
        self.open_fraction = min(0.95, max(0.05, (360.0 - shutter_deg) / 360.0))
        self._trig_hist = deque(maxlen=9)
        self._budget_warned_at = -1000
        self.capture_timeout = float(getattr(args, "capture_timeout_ms", 150.0)) / 1000.0
        self.bracket_settle_frames = int(getattr(args, "bracket_settle_frames", 2))
        self.ring_size = int(getattr(args, "frame_ring_size", 12))
        self.suspect_count = 0
        self.capture_info = {}

        # subject to receive signals from FT232H connector
        self.signal_subject = signal_subject

        # small event we set when the subject completes (EOF)
        self._completion_event = threading.Event()

        # logging
        self.initialize_logging()

        # capture settings & buffers
        self.initialize_bracketing()
        self.initialize_camera()
        self.log_camera_properties()
        self.initialize_grabber()
        self.initialize_threads()
        self.initialize_process_pool()

        # bookkeeping
        self.img_desc = []
        self.img_desc_lock = threading.Lock()
        self.blank_frame = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # administration
        self.wait_subject = wait_subject
        self._final_write_done = False
        self._cleaned_up = False

        # timing singleton from connector module (may be None until connector created)
        self.timing = None

        # stats
        self.processed_frames = 0
        self.start_time = time.perf_counter()
        self.new_tick = self.start_time

        self.register_signal_handlers()


    def initialize_logging(self) -> None:
        """
        Configure logging: one handler for this module, no propagation to the root logger
        (otherwise every line is printed twice when started from the GUI).
        """
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # stdout when started from beck-view-gui, stderr otherwise
            handler = logging.StreamHandler(sys.stdout if self.gui else sys.stderr)
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def register_signal_handlers(self) -> None:
        """Graceful shutdown on SIGINT/SIGTERM (and SIGBREAK, which the GUI sends on Windows)."""
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._on_signal)

    def initialize_bracketing(self) -> None:
        """Prepare exposure triplets (or single) depending on platform/flags."""
        self.exposures = []
        if platform.system() == "Linux" or platform.system() == "Darwin":
            if self.bracketing:
                self.exposures = [(1.0 / (1 << 7), "a"), (1.0 / (1 << 8), "b"), (1.0 / (1 << 6), "c")]
            else:
                self.exposures = [(1.0 / (1 << 7), "a")]
        else:
            if self.bracketing:
                self.exposures = [(-7, "a"), (-8, "b"), (-6, "c")]
            else:
                self.exposures = [(-7, "a")]

    def initialize_camera(self) -> None:
        """Open the camera and configure resolution / properties."""
        api: cython.int = cv2.CAP_ANY
        if platform.system() == "Windows":
            api = cv2.CAP_DSHOW if self.settings else cv2.CAP_MSMF
        elif platform.system() == "Linux":
            api = cv2.CAP_V4L2
        elif platform.system() == "Darwin":
            api = cv2.CAP_AVFOUNDATION

        try:
            self.cap = cv2.VideoCapture(self.device_number, api, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        except Exception as e:
            self.logger.error(f"DigitizeVideo.initialize_camera: could not find a camera: {e}")

        time.sleep(1)
        # set preferred resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # warm up
        _, _ = self.cap.read()
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        # various capture tuning
        self.cap.set(cv2.CAP_PROP_FORMAT, -1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # reset exposure state (works around driver quirks)
        if platform.system() == "Windows":
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto
            time.sleep(0.5)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # auto
            time.sleep(0.5)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # manual

        if platform.system() == "Windows":
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        else:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 1.0 / (1 << 7))

        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        time.sleep(1)

        self.compression_level = 1 # relevant for writing PNG-files

        # debug/log some properties
        self.logger.debug(f"Camera properties: width={self.img_width} height={self.img_height} fps={self.cap.get(cv2.CAP_PROP_FPS)}")

    def initialize_grabber(self) -> None:
        """
        Start the thread that reads the camera continuously.

        Every frame is stamped with time.perf_counter() right after cap.grab() returned
        (same clock as the trigger timestamp of the FT232H poller) and kept in a short ring.
        A trigger then picks the first frame that arrived after it, see _wait_frame().
        Because the camera is drained all the time, there is no stale driver queue and no
        long first read. The method returns after a few warm-up frames have arrived.
        """
        warmup_frames = 5

        self._frame_ring = deque(maxlen=max(4, self.ring_size))
        self._frame_cond = threading.Condition()
        self._cap_lock = threading.Lock()
        self._grab_stop = threading.Event()
        self._grab_seq = 0

        self._grab_thread = threading.Thread(target=self._grab_loop, name="grabber", daemon=True)
        self._grab_thread.start()

        deadline = time.perf_counter() + 5.0
        with self._frame_cond:
            while self._grab_seq < warmup_frames:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._frame_cond.wait(remaining)

        if self._grab_seq < warmup_frames:
            self.logger.error("Grabber: camera delivers no frames (check camera / settings)")
        else:
            self.logger.info(
                f"Grabber running; d_min {self.capture_min_delay * 1000:.0f} ms, "
                f"window after d_min {self.capture_window * 1000:.0f} ms, "
                f"timeout {self.capture_timeout * 1000:.0f} ms, ring of {self._frame_ring.maxlen} frames"
            )
            if self._min_delay_is_default:
                self.logger.warning(
                    f"d_min = {self.capture_min_delay * 1000:.0f} ms is the uncalibrated default - "
                    f"determine the real value with d_min_from_images.py and pass capture_min_delay_ms"
                )

    def _grab_loop(self) -> None:
        """Grabber thread: read the camera continuously and keep the newest frames with their arrival time."""
        cap = self.cap
        cond = self._frame_cond
        ring = self._frame_ring
        failures = 0

        while not self._grab_stop.is_set():
            ok = False
            frame = None
            ts = 0.0
            try:
                with self._cap_lock:
                    ok = cap.grab()
                    ts = time.perf_counter()          # arrival time of the frame
                    if ok:
                        ok, frame = cap.retrieve()    # decode after the timestamp was taken
            except Exception as e:
                ok = False
                if failures % 100 == 0:
                    self.logger.error(f"Grabber: camera read raised {e}")

            if not ok or frame is None:
                failures += 1
                if failures == 1 or failures % 200 == 0:
                    self.logger.warning(f"Grabber: camera read failed ({failures} in a row)")
                time.sleep(0.005)
                continue
            failures = 0

            if not frame.flags.c_contiguous or frame.dtype != np.uint8:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)

            with cond:
                self._grab_seq += 1
                ring.append((self._grab_seq, ts, frame))
                cond.notify_all()

    def _wait_frame(self, double min_ts, long min_seq, double deadline):
        """
        Return the oldest ring entry (seq, arrival_time, frame) that arrived at or after min_ts and
        has a sequence number >= min_seq. Waits until `deadline` (perf_counter time) for it.
        A frame that is already in the ring is returned immediately, also when the capture thread
        runs late. Returns None if no such frame exists in time.
        """
        with self._frame_cond:
            while True:
                for entry in self._frame_ring:          # oldest first
                    if entry[0] >= min_seq and entry[1] >= min_ts:
                        return entry
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._frame_cond.wait(remaining)

    def _latest_frame(self):
        """Newest ring entry or None."""
        with self._frame_cond:
            # no negative index: wraparound=False is set for this module
            n = len(self._frame_ring)
            return self._frame_ring[n - 1] if n > 0 else None

    def log_camera_properties(self) -> None:
        """
        Log all relevant OpenCV VideoCapture properties.
        Useful for debugging backend- and driver-specific behavior.
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.warning("Camera not opened – cannot log properties")
            return

        # Commonly relevant CAP_PROP_* attributes
        properties = [
            ("FRAME_WIDTH",        cv2.CAP_PROP_FRAME_WIDTH),
            ("FRAME_HEIGHT",       cv2.CAP_PROP_FRAME_HEIGHT),
            ("FPS",                cv2.CAP_PROP_FPS),
            ("FOURCC",             cv2.CAP_PROP_FOURCC),
            ("FORMAT",             cv2.CAP_PROP_FORMAT),
            ("MODE",               cv2.CAP_PROP_MODE),
            ("BUFFERSIZE",         cv2.CAP_PROP_BUFFERSIZE),

            ("BRIGHTNESS",         cv2.CAP_PROP_BRIGHTNESS),
            ("CONTRAST",           cv2.CAP_PROP_CONTRAST),
            ("SATURATION",         cv2.CAP_PROP_SATURATION),
            ("HUE",                cv2.CAP_PROP_HUE),
            ("GAIN",               cv2.CAP_PROP_GAIN),

            ("EXPOSURE",           cv2.CAP_PROP_EXPOSURE),
            ("AUTO_EXPOSURE",      cv2.CAP_PROP_AUTO_EXPOSURE),

            ("AUTO_WB",            cv2.CAP_PROP_AUTO_WB),
            ("WB_TEMPERATURE",     cv2.CAP_PROP_WB_TEMPERATURE),

            ("SHARPNESS",          cv2.CAP_PROP_SHARPNESS),
            ("GAMMA",              cv2.CAP_PROP_GAMMA),

            ("ZOOM",               cv2.CAP_PROP_ZOOM),
            ("FOCUS",              cv2.CAP_PROP_FOCUS),
            ("AUTOFOCUS",          cv2.CAP_PROP_AUTOFOCUS),

            ("BACKEND",            cv2.CAP_PROP_BACKEND),
            ("HW_ACCELERATION",    cv2.CAP_PROP_HW_ACCELERATION),
        ]

        self.logger.info("===== OpenCV Camera Properties =====")

        for name, prop in properties:
            try:
                value = self.cap.get(prop)
                if value != -1:
                    if prop == cv2.CAP_PROP_FOURCC:
                        fourcc = int(value)
                        fourcc_str = "".join([
                            chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)
                        ])
                        self.logger.info(f"{name:20s}: {fourcc_str} (0x{fourcc:08X})")
                    else:
                        self.logger.info(f"{name:20s}: {value}")
            except Exception as e:
                self.logger.debug(f"{name:20s}: unsupported ({e})")

        self.logger.info("===================================")


    def initialize_threads(self) -> None:
        """Create thread pool and subscribe to the signal_subject for captures."""
        cpu_count = multiprocessing.cpu_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, cpu_count - 1))

        # SignalObserver triggers final_write_to_disk on subject completion.
        self.signal_observer = SignalObserver(self.final_write_to_disk, logging.getLogger(__name__ + ".SignalObserver"))

        # Capture runs on its own thread, so cap.read() can never block the FT232H poller.
        self.capture_scheduler = EventLoopScheduler(
            thread_factory=lambda target: threading.Thread(target=target, name="capture", daemon=True)
        )

        # Subscribe to subject:
        #  1) observe_on hands the event (and later on_completed) over to the capture thread,
        #     the poller thread only enqueues it (microseconds),
        #  2) map runs the actual capture there, one event after the other, in order.
        self.photoCellSignalDisposable = self.signal_subject.pipe(
            ops.observe_on(self.capture_scheduler),
            ops.map(self._capture_safe),
        ).subscribe(self.signal_observer)

        # Also subscribe only for completion to set a local event we can wait on.
        # Keep disposable to remove subscription during cleanup.
        def _on_completed():
            # set event for waiting thread(s)
            try:
                self._completion_event.set()
            except Exception:
                pass

    def _capture_safe(self, event):
        """
        Runs on the capture thread: capture one trigger event.
        Exceptions are logged and swallowed; an exception escaping from map() would terminate
        the whole subscription and silently stop all further captures.
        """
        try:
            self.handle_trigger(event)
        except Exception:
            self.logger.exception(f"[Capture] capture failed for event {event!r}")
        return None

    cpdef connect(self, Ft232hConnector conn):
        """
        Call this after Ft232hConnector is created.
        Stores both Python-level and C-level timing references.
        """
        try:
            # prefer Python-level safe accessor
            py_tv = get_timing()
            if py_tv is None:
                # fallback to requesting from connector instance (safer if connector created now)
                py_tv = conn.get_timing()
            self.timing_view = <CTimingResult> py_tv  # store C-typed reference
        except Exception as e:
            # keep the attribute None on failure
            self.timing_view = None
            self.logger.error(f"DigitizeVideo.connect: could not obtain timing_view: {e}")

    cpdef bint set_exposure(self, double exp_val):
        """Set camera exposure; return True on success (serialized with the grabber thread)."""
        with self._cap_lock:
            return bool(self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val))

    def shm_to_uint8_4d(self, buf, Py_ssize_t frames, int h, int w):
        """
        Return a numpy ndarray shaped (frames, h, w, 3) backed by shm.buf.
        """
        cdef np.ndarray[np.uint8_t, ndim=1] arr1d = np.frombuffer(buf, dtype=np.uint8, count=frames * h * w * 3)
        arr4d = arr1d.reshape(frames, h, w, 3)
        return arr4d

    def initialize_process_pool(self) -> None:
        """
            Create a multiprocessing pool (forkserver) and pre-allocate shared memory buffers.
        """

        self.img_bytes = self.img_width * self.img_height * 3

        if sys.platform.startswith("win"):
            # Windows only supports "spawn"
            ctx = multiprocessing.get_context("spawn")
        else:
            # Unix: forkserver is best, but use fallback to fork
            try:
                ctx = multiprocessing.get_context("forkserver")
            except ValueError:
                ctx = multiprocessing.get_context("fork")

        self.process_count = max(2, multiprocessing.cpu_count() - 1)
        self.pool = ctx.Pool(self.process_count)

        frames_per_buffer = int(self.chunk_size * self.frame_multiplier)

        # Shared memory allocation
        self.shared_buffers = [shared_memory.SharedMemory(create=True, size=(frames_per_buffer * self.img_bytes)) for _ in range(self.process_count)]

        self._shm_arrays = []
        self._desc_shms = []
        self._desc_arrays = []

        for shm in self.shared_buffers:
            arr = self.shm_to_uint8_4d(shm.buf, frames_per_buffer, self.img_height, self.img_width)
            self._shm_arrays.append(arr)

            desc_nbytes = frames_per_buffer * DESCRIPTOR_COLUMNS * np.dtype(np.uint32).itemsize
            dshm = shared_memory.SharedMemory(create=True, size=desc_nbytes)
            desc_arr = np.ndarray((frames_per_buffer, DESCRIPTOR_COLUMNS), dtype=np.uint32, buffer=dshm.buf)
            desc_arr.fill(0)
            self._desc_shms.append(dshm)
            self._desc_arrays.append(desc_arr)

        self.shared_buffers_index = 0
        self.buffers_in_use = [False] * self.process_count
        self.buffer_lock = threading.Lock()
        self.processes = []
        self._frames_per_buffer = frames_per_buffer

    # ---------------------------
    # Capture / buffer management
    # ---------------------------
    def _cleanup_finished_processes(self) -> None:
        """
        Check child processes for completion.
        Properly close/unlink SHM and clear descriptor arrays safely.
        """
        still = []

        for res, shm, desc_shm, buf_idx in self.processes:
            if res.ready():
                try:
                    # re-raise worker exceptions here so we can log them
                    res.get()
                except Exception as e:
                    # Log the worker error, but still reclaim the buffer so the pipeline continues
                    self.logger.error(f"Child process failed: {e}")

                # mark buffer as free for reuse (under lock)
                with self.buffer_lock:
                    self.buffers_in_use[buf_idx] = False

                # Clear descriptor array now that worker consumed it
                # (desc arrays are pre-allocated and reused; do NOT close/unlink here)
                try:
                    self._desc_arrays[buf_idx].fill(0)
                except Exception:
                    # best-effort only
                    pass
            else:
                still.append((res, shm, desc_shm, buf_idx))
        self.processes = still

    def handle_trigger(self, event: SubjectDescType) -> None:
        """
        Called for each optocoupler event. Grabs frames and fills the current shared buffer.
        """
        cdef double[:, :] buf

        frame_count, start_cycle = event

        # pick the frame(s) belonging to this trigger from the grabber ring
        self.take_picture(frame_count, start_cycle)

        # If we've completed a chunk, schedule writing to disk
        if (self.processed_frames % self.chunk_size) == 0:
            with self.img_desc_lock:
                descriptors = list(self.img_desc)
                self.img_desc = []

            with self.buffer_lock:
                self.buffers_in_use[self.shared_buffers_index] = True

            buffer_index = self.shared_buffers_index
            self.executor.submit(self._post_capture, buffer_index, descriptors)

        capture_duration = time.perf_counter() - start_cycle

        if (self.processed_frames % 100) == 0:
            self.logger.info(f"[Capture] Frame {frame_count} ({self.processed_frames}) took {capture_duration*1000:.2f} ms")

        # add capture duration (trigger -> image in buffer, incl. time spent in the queue)
        # to the timing statistics. The row index is the film frame number: the poller appends
        # row <count> for every trigger. (Using processed_frames here was off by one row, and
        # off by a factor of 3 with bracketing, because it is incremented in take_picture().)
        try:
            if frame_count < self.timing_view.max_frames:
                buf = self.timing_view.buf
                buf[frame_count, 3] = capture_duration
            else:
                self.logger.error(
                    f"Timing buffer overflow at frame {frame_count}, "
                    f"max={self.timing_view.max_frames}"
                )
        except Exception:
            self.logger.error(f"Could not add {capture_duration=} for frame {frame_count} to self.timing_view")
            pass


    def _post_capture(self, buffer_index: int, descriptors: List[ImgDescType]) -> None:
        """Run in a threadpool: find a free buffer and schedule worker to write it."""
        self._cleanup_finished_processes()

        next_buf = None
        while next_buf is None:
            with self.buffer_lock:
                for i in range(self.process_count):
                    cand = (buffer_index + 1 + i) % self.process_count
                    if not self.buffers_in_use[cand]:
                        next_buf = cand
                        break
            if next_buf is None:
                self._cleanup_finished_processes()
                self.logger.warning("All shared buffers busy; waiting...")
                time.sleep(0.01)

        self.shared_buffers_index = next_buf

        try:
            shm = self.shared_buffers[buffer_index]
            desc_shm = self._desc_shms[buffer_index]
            shm_name = (<str>shm.name).encode('utf-8')
            desc_name = (<str>desc_shm.name).encode('utf-8')
            frames_total = self._frames_per_buffer

            def process_error_callback(error):
                self.logger.error(f"Error in child process: {error}")

            result = self.pool.apply_async(
                write_images,
                args=(shm_name, desc_name, frames_total, self.img_width, self.img_height, self.output_path_b, self.compression_level),
                kwds={"desc_cols": DESCRIPTOR_COLUMNS},
                error_callback=process_error_callback
            )

            with self.buffer_lock:
                self.processes.append((result, shm, desc_shm, buffer_index))

        except Exception as e:
            self.logger.error(f"Error during apply_async: {e}")
            with self.buffer_lock:
                self.buffers_in_use[buffer_index] = False


    def _check_budget(self, double trigger_ts, long frame_count) -> None:
        """
        Warn (at most once per 100 film frames) if no camera frame is guaranteed to fall into the
        time the lens is open. Uses the measured projector cycle (median of the last trigger
        intervals), the shutter width and the measured camera frame period.
        The check is cautious: the lens is assumed to be open open_fraction * cycle after the
        trigger, and a frame needs d_min + one camera period (+ 3 ms jitter) to be guaranteed.
        """
        hist = self._trig_hist
        hist.append(trigger_ts)
        n = len(hist)
        if n < 6:
            return
        ts = list(hist)
        cycles = sorted([ts[i + 1] - ts[i] for i in range(n - 1)])
        cycle = cycles[(n - 1) // 2]

        with self._frame_cond:
            arr = [e[1] for e in self._frame_ring]
        if len(arr) < 4:
            return
        periods = sorted([arr[i + 1] - arr[i] for i in range(len(arr) - 1)])
        period = periods[(len(arr) - 1) // 2]

        t_open = self.open_fraction * cycle
        need = self.capture_min_delay + period + 0.003
        if need > t_open and (frame_count - self._budget_warned_at) >= 100:
            self._budget_warned_at = frame_count
            self.logger.warning(
                f"[Capture] rest window too short: cycle {cycle * 1000:.0f} ms, lens open about "
                f"{t_open * 1000:.0f} ms, but d_min {self.capture_min_delay * 1000:.0f} ms + camera period "
                f"{period * 1000:.0f} ms + 3 ms = {need * 1000:.0f} ms - no frame is guaranteed to be "
                f"captured while the film is at rest (slow the projector down, use a faster camera, "
                f"or lower exposure/d_min)"
            )

    def take_picture(self, int picture_count, double trigger_ts) -> None:
        """
        Select the frame(s) that belong to one trigger and copy them into the shared buffer.

        Bracket 0: the first camera frame that arrived at or after trigger + d_min (trigger_ts is the
        poller's perf_counter() stamp, d_min = capture_min_delay). A frame that arrives sooner was
        exposed while the shutter still covered the lens and is never taken. If the chosen frame
        arrived no later than d_min + capture_window the capture is fine. Otherwise the best
        available frame is used and flagged SUSPECT:
          - LATE   : first frame after d_min, but it arrived after d_min + window
          - STALE  : no frame after d_min within capture_timeout; newest frame from before it
          - NOFRAME: the camera delivered nothing at all; a blank image is written
        capture_info records per film frame: delay of the chosen frame after the trigger, the suspect
        flag and capture_work_ms (arrival of the chosen frame -> image in the shared buffer, i.e. the
        real processing time, as opposed to the latency trigger -> buffer in column "capture opencv").
        Further brackets: after an exposure change the next `bracket_settle_frames` frames are
        skipped (the new exposure needs a frame or two to take effect). They are judged against the
        same latest-arrival limit (d_min + window), which is usually missed - the film moves on.
        """
        frame_count = picture_count

        frame_multiplier = self.frame_multiplier
        exposures = self.exposures
        buf_idx = self.shared_buffers_index
        shm_frames = self._shm_arrays[buf_idx]
        desc_arr = self._desc_arrays[buf_idx]
        frames_per_buffer = self._frames_per_buffer
        img_bytes = self.img_bytes
        min_delay = self.capture_min_delay
        late_limit = min_delay + self.capture_window     # latest acceptable arrival after the trigger
        timeout = self.capture_timeout
        n_brackets = len(exposures)

        self._check_budget(trigger_ts, frame_count)

        cdef np.uint8_t[:, :, :] src_mv
        cdef np.uint8_t[:, :, :] dst_mv
        cdef unsigned char* src_ptr
        cdef unsigned char* dst_ptr

        min_ts = trigger_ts + min_delay
        min_seq = 0
        # deadline for the first bracket is relative to d_min; if the capture thread runs late
        # the frame is usually already in the ring and returned at once
        deadline = min_ts + timeout
        first_delay_ms = None
        first_work_ms = None
        any_suspect = False

        # Is the capture thread behind? Waiting for the camera frame is normal and does not count;
        # only a capture that STARTS after trigger + d_min is late.
        start_lag = time.perf_counter() - min_ts
        if start_lag > CAPTURE_START_LAG_WARN_S:
            self.logger.warning(
                f"[Capture] capture of frame {frame_count} started {start_lag * 1000:.0f} ms after "
                f"trigger + d_min - the capture thread is behind"
            )

        for bracket_index, (exp_val, suffix) in enumerate(exposures):
            entry = self._wait_frame(min_ts, min_seq, deadline)

            status = "ok"
            if entry is not None:
                frame_data = entry[2]
                frame_ts = entry[1]
                delay = frame_ts - trigger_ts
                if delay > late_limit:
                    status = "LATE"
            else:
                latest = self._latest_frame()
                if latest is not None:
                    frame_data = latest[2]
                    frame_ts = latest[1]
                    delay = frame_ts - trigger_ts           # negative: frame is older than the trigger
                    status = "STALE"
                else:
                    frame_data = None
                    frame_ts = float("nan")
                    delay = float("nan")
                    status = "NOFRAME"

            suspect = status != "ok"
            if first_delay_ms is None:
                first_delay_ms = delay * 1000.0

            if suspect:
                any_suspect = True
                self.suspect_count += 1
                self.logger.warning(
                    f"[Capture] SUSPECT frame {frame_count}{suffix}: {status}, "
                    f"delay {delay * 1000:.1f} ms (d_min {min_delay * 1000:.0f} ms, latest {late_limit * 1000:.0f} ms), "
                    f"exposure {exp_val}"
                )

            if self.bracketing and (bracket_index < (n_brackets - 1)):
                # change the exposure for the next bracket; the next usable frame is the first one
                # that arrives after `settle` further frames
                next_exp, _ = exposures[bracket_index + 1]
                if not self.set_exposure(next_exp):
                    self.logger.error(f"Could not set exposure to {next_exp} for frame {frame_count}{suffix}")
                latest = self._latest_frame()
                latest_seq = latest[0] if latest is not None else 0
                min_seq = latest_seq + 1 + self.bracket_settle_frames
                min_ts = time.perf_counter()
                deadline = min_ts + timeout

            logical_index = frame_count * frame_multiplier
            frame_index = (logical_index + bracket_index) % frames_per_buffer

            if frame_data is not None:
                src_mv = frame_data
            else:
                self.logger.error(f"No camera frame for frame {frame_count}{suffix}, writing blank image")
                src_mv = self.blank_frame

            dst_mv = shm_frames[frame_index]
            src_ptr = &src_mv[0, 0, 0]
            dst_ptr = &dst_mv[0, 0, 0]

            memcpy(dst_ptr, src_ptr, img_bytes)

            if first_work_ms is None:
                # processing time: arrival of the chosen camera frame -> image in the shared buffer
                first_work_ms = (time.perf_counter() - frame_ts) * 1000.0

            desc_arr[frame_index, 0] = np.uint32(img_bytes)
            desc_arr[frame_index, 1] = np.uint32(frame_count)
            desc_arr[frame_index, 2] = np.uint32(bracket_index)
            if DESCRIPTOR_COLUMNS > 3:
                desc_arr[frame_index, 3] = np.uint32(1 if suspect else 0)

            with self.img_desc_lock:
                self.img_desc.append((img_bytes, frame_count, suffix))

            self.processed_frames += 1

        # delay (first bracket), suspect flag and processing time per film frame for the timing CSV
        self.capture_info[frame_count] = (first_delay_ms, 1 if any_suspect else 0, first_work_ms)

        if self.bracketing:
            first_exp, _ = exposures[0]
            self.set_exposure(first_exp)

    def final_write_to_disk(self) -> None:
        """
        Trigger final write for remaining images and wait for workers to finish.
        This is idempotent and safe to call multiple times.
        """

        if self._final_write_done:
            return

        self._final_write_done = True

        if self.img_desc:
            current_buf = self.shared_buffers_index
            with self.buffer_lock:
                self.buffers_in_use[current_buf] = True

            try:
                shm = self.shared_buffers[current_buf]
                desc_shm = self._desc_shms[current_buf]
                shm_name = (<str>shm.name).encode('utf-8')
                desc_name = (<str>desc_shm.name).encode('utf-8')
                frames_total = self._frames_per_buffer

                write_images(shm_name, desc_name, frames_total, self.img_width, self.img_height, self.output_path_b, self.compression_level,
                             desc_cols=DESCRIPTOR_COLUMNS)

                # clear descriptor buffer
                desc_arr = self._desc_arrays[current_buf]
                desc_arr.fill(0)
            except Exception as e:
                self.logger.error(f"Error during final apply_async: {e}")
                with self.buffer_lock:
                    self.buffers_in_use[current_buf] = False

            self.img_desc = []

            time.sleep(3.0)

        # log statistics and save timing
        elapsed_time = time.perf_counter() - self.start_time
        average_fps = (self.processed_frames / elapsed_time) if elapsed_time > 0 else 0.0
        self.logger.info("------- End of Film ---------")
        self.logger.info(f"Total saved images (incl. exposures): {self.processed_frames}")
        self.logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Average FPS: {average_fps:.2f}")
        if self.suspect_count:
            self.logger.warning(f"SUSPECT captures: {self.suspect_count} (see log and CSV column 'suspect')")
        else:
            self.logger.info("SUSPECT captures: 0")

        time_now  = datetime.now()
        fname = str(self.output_path / f"timing_{time_now.strftime('%Y_%m_%d_%H_%M_%S')}_{self.processed_frames:05d}.csv")
        self.logger.info(f"CSV file to be opened: {fname}")
        self.write_timing_csv(fname)

        # final cleanup
        self.cleanup()

        # signal completion to main
        self.wait_subject.on_completed()

    cpdef write_timing_csv(self, str filename):
        cdef int i
        cdef int count = 0
        cdef int cycle = 1
        cdef int work = 2
        cdef int capture = 3
        cdef int latency = 4
        cdef int wait_time = 5
        cdef int total_work = 6

        buf = self.timing_view.buf
        rows = self.timing_view.size

        with open(filename, "w") as f:
            f.write("frame,capture opencv,work,cycle (work + latency),latency,wait_time,total_work,"
                    "frame_delay_ms,suspect,capture_work_ms\n")

            for i in range(rows):
                info = self.capture_info.get(int(buf[i, count]))
                if info is not None:
                    delay_ms = "" if info[0] != info[0] else f"{info[0]:.3f}"   # nan -> empty
                    suspect = info[1]
                    work_ms = "" if (info[2] is None or info[2] != info[2]) else f"{info[2]:.3f}"
                else:
                    delay_ms = ""      # no capture for this row (e.g. trigger not processed)
                    suspect = ""
                    work_ms = ""
                f.write(
                    f"{buf[i, count]},{buf[i, capture]},{buf[i, work]},{buf[i, cycle]},"
                    f"{buf[i, latency]},{buf[i, wait_time]},{buf[i, total_work]},"
                    f"{delay_ms},{suspect},{work_ms}\n"
                )

    def wait_for_completion(self, timeout: float = None) -> bool:
        """
        Block until the FT232H signals completion (subject.on_completed) or until `timeout` seconds.
        Returns True if completion event was seen, False if timed out.
        """
        return self._completion_event.wait(timeout)

    def cleanup(self, timeout_wait_for_workers: float = 2.0) -> None:
        """
        Simplified, idempotent cleanup.
        - stop ft232h poller (if present)
        - shutdown executor (if present)
        - reap finished pool tasks non-blocking for up to timeout_wait_for_workers
        - close/join pool
        - dispose Rx disposables, release camera
        - release memoryviews, close+unlink shared memory
        - final gc and close FTDI
        """
        if getattr(self, "_cleaned_up", False):
            self.logger.info("cleanup() already ran — skipping")
            return
        self._cleaned_up = True

        self.logger.info("Cleaning up ...")

        # 0) Stop FTDI poller early to avoid new events
        try:
            ft = getattr(self, "ft232h", None)
            if ft is not None:
                stop = getattr(ft, "stop", None)
                if callable(stop):
                    stop()
        except Exception:
            self.logger.exception("Error stopping ft232h in cleanup")

        # 1) Shutdown thread executor (if any)
        try:
            executor = getattr(self, "executor", None)
            if executor is not None:
                executor.shutdown(wait=True)
        except Exception:
            self.logger.exception("Error shutting down executor")

        # 2) Reap finished worker results (non-blocking), wait briefly for outstanding ones
        t0 = time.time()
        try:
            while self.processes:
                try:
                    self._final_cleanup_finished_processes()
                except Exception:
                    self.logger.exception("_final_cleanup_finished_processes failed")
                if time.time() - t0 > timeout_wait_for_workers:
                    self.logger.warning("Timeout waiting for worker processes to finish")
                    break
                # short sleep to avoid busy loop; small so we remain responsive
                time.sleep(0.05)
        except Exception:
            self.logger.exception("Error while waiting for child processes")

        # 3) Close & join multiprocessing pool (if any)
        try:
            pool = getattr(self, "pool", None)
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    self.logger.debug("pool.close() raised exception (continuing to join/terminate)")

                try:
                    pool.join()
                except Exception:
                    try:
                        pool.terminate()
                        pool.join()
                    except Exception:
                        self.logger.exception("Could not terminate/join multiprocessing pool")
        except Exception:
            self.logger.exception("Error cleaning up pool")

        # 4) Dispose Rx subscriptions (if any)
        for attr in ("photoCellSignalDisposable", "completion_disposable"):
            try:
                disp = getattr(self, attr, None)
                if disp is not None:
                    try:
                        # many disposables implement dispose; if not, ignore errors
                        disp.dispose()
                    except Exception:
                        pass
            except Exception:
                self.logger.exception(f"Error disposing {attr}")

        # 4b) Stop the capture thread's event loop (cleanup() normally runs on that very thread,
        #     dispose() only flags it for shutdown and does not join)
        try:
            sched = getattr(self, "capture_scheduler", None)
            if sched is not None:
                sched.dispose()
        except Exception:
            self.logger.debug("capture_scheduler.dispose() raised (ignored)")

        # 4c) Stop the grabber before the camera is released
        try:
            stop_ev = getattr(self, "_grab_stop", None)
            if stop_ev is not None:
                stop_ev.set()
            gt = getattr(self, "_grab_thread", None)
            if gt is not None and gt is not threading.current_thread():
                gt.join(2.0)
            ring = getattr(self, "_frame_ring", None)
            if ring is not None:
                with self._frame_cond:
                    ring.clear()
        except Exception:
            self.logger.exception("Error stopping grabber")

        # 5) Release camera (if present)
        try:
            cap = getattr(self, "cap", None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    self.logger.exception("Error releasing camera")
        except Exception:
            self.logger.exception("Error releasing camera (outer)")

        # 6) Drop memoryview references in parent (allow unlink)
        try:
            # clear lists that hold numpy/memoryview references
            try:
                self._shm_arrays[:] = []
            except Exception:
                self._shm_arrays = []
            try:
                self._desc_arrays[:] = []
            except Exception:
                self._desc_arrays = []

            try:
                self.img_desc = []
            except Exception:
                pass

            # ensure processes emptied (we reaped above), drop references anyway
            try:
                self.processes = []
            except Exception:
                pass
        except Exception:
            self.logger.exception("Error clearing local memoryview references")

        # 7) Force GC now to free exported views (one short collect)
        try:
            import gc
            gc.collect()
            time.sleep(0.02)
        except Exception:
            pass

        # 8) Close & unlink all SharedMemory objects with retries
        for list_name in ("shared_buffers", "_desc_shms"):
            lst = getattr(self, list_name, None)
            if not lst:
                continue
            # copy to avoid concurrent mutation
            try:
                items = list(lst)
            except Exception:
                items = lst[:]
            for shm in items:
                try:
                    self._close_and_unlink_shm_with_retries(shm, retries=5, sleep=0.02)
                except Exception:
                    self.logger.exception(f"Failed close/unlink for an entry in {list_name}")

        # 9) Clear those lists (remove references to SharedMemory objects)
        try:
            self.shared_buffers = []
        except Exception:
            pass
        try:
            self._desc_shms = []
        except Exception:
            pass

        # 10) Final GC + short sleep to let runtime finalize destructors
        try:
            import gc
            gc.collect()
            time.sleep(0.02)
        except Exception:
            pass

        # 11) Close FTDI connector last (best-effort)
        try:
            ft = getattr(self, "ft232h", None)
            if ft is not None:
                try:
                    close = getattr(ft, "close", None)
                    if callable(close):
                        close()
                    else:
                        stop = getattr(ft, "stop", None)
                        if callable(stop):
                            stop()
                except Exception:
                    self.logger.exception("Error closing ft232h at end of cleanup")
        except Exception:
            pass

        self.logger.info("Cleanup complete.")


    def _final_cleanup_finished_processes(self) -> None:
        """
        Reap finished worker ApplyResult entries in self.processes.

        Each entry is expected to be a tuple:
            (apply_result, shm_frame, desc_shm, buf_idx)

        - Non-blocking: only process items where apply_result.ready() == True
        - Close/unlink shared memory and mark buffer free under buffer_lock
        - Remove reaped entries from self.processes
        """
        try:
            procs_copy = list(self.processes)
        except Exception:
            # nothing we can do
            return

        for entry in procs_copy:
            # Validate tuple shape
            if not isinstance(entry, tuple) or len(entry) < 1:
                # unexpected; remove defensively
                try:
                    self.processes.remove(entry)
                except Exception:
                    pass
                continue

            apply_res = entry[0]
            shm_frame = entry[1] if len(entry) > 1 else None
            desc_shm  = entry[2] if len(entry) > 2 else None
            buf_idx   = entry[3] if len(entry) > 3 else None

            # Only touch finished results
            try:
                ready_callable = getattr(apply_res, "ready", None)
                finished = bool(ready_callable()) if callable(ready_callable) else False
            except Exception:
                finished = False

            if not finished:
                continue

            # collect/raise worker exceptions (non-blocking since ready() was True)
            try:
                apply_res.get()
            except Exception as e:
                self.logger.error(f"Worker raised exception: {e}")

            # mark buffer free for reuse
            if buf_idx is not None:
                try:
                    with self.buffer_lock:
                        if 0 <= buf_idx < len(self.buffers_in_use):
                            self.buffers_in_use[buf_idx] = False
                except Exception:
                    self.logger.debug(f"Could not mark buffer {buf_idx} free")

            # clear descriptor array if present
            try:
                if buf_idx is not None and 0 <= buf_idx < len(self._desc_arrays):
                    try:
                        self._desc_arrays[buf_idx].fill(0)
                    except Exception:
                        self.logger.debug("Could not clear descriptor array (likely already released)")
            except Exception:
                pass

            # close+unlink shared memory objects (best-effort)
            for shm_obj in (shm_frame, desc_shm):
                if shm_obj is None:
                    continue
                try:
                    self._close_and_unlink_shm_with_retries(shm_obj, retries=3, sleep=0.02)
                except Exception:
                    self.logger.exception("Error closing/unlinking a shared memory object")

                # remove from global lists if present
                try:
                    if hasattr(self, "shared_buffers") and shm_obj in self.shared_buffers:
                        try:
                            self.shared_buffers.remove(shm_obj)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    if hasattr(self, "_desc_shms") and shm_obj in self._desc_shms:
                        try:
                            self._desc_shms.remove(shm_obj)
                        except Exception:
                            pass
                except Exception:
                    pass

            # finally remove entry from live processes
            try:
                if entry in self.processes:
                    self.processes.remove(entry)
            except Exception:
                pass

        # log remaining
        try:
            self.logger.info(f"_cleanup_finished_processes: remaining processes: {len(self.processes)}")
        except Exception:
            pass


    def _close_and_unlink_shm_with_retries(self, shm_obj, retries=5, sleep=0.05):
        """
        Try to close a SharedMemory object and unlink it. Retry on BufferError.
        """
        name = getattr(shm_obj, "name", "<unknown>")
        for attempt in range(retries):
            try:
                try:
                    shm_obj.close()
                except BufferError:
                    # memoryview still exported in this process; try GC and retry
                    import gc
                    gc.collect()
                    time.sleep(sleep)
                    continue
                except Exception:
                    # other close errors — log and continue to unlink attempt
                    self.logger.debug(f"_close_and_unlink_shm_with_retries: close() raised for {name}")
                # try unlink (may raise FileNotFoundError if another process already unlinked)
                try:
                    shm_obj.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    self.logger.debug(f"_close_and_unlink_shm_with_retries: unlink() raised for {name}")
                return
            except Exception as e:
                self.logger.debug(f"_close_and_unlink_shm_with_retries: unexpected error for {name}: {e}")
        # final attempt to unlink, best-effort
        try:
            shm_obj.unlink()
        except Exception:
            self.logger.debug(f"_close_and_unlink_shm_with_retries: final unlink failed for {name}")


    def _on_signal(self, signum: int, frame) -> None:
        """Process-level signal handler for graceful shutdown (SIGINT/SIGTERM)."""
        name = signal.Signals(signum).name
        self.logger.warning(f"Signal {name} received, stopping...")
        try:
            # ensure final frames are written and connector stopped
            self.final_write_to_disk()
        except Exception:
            pass
        # wait a moment and exit
        try:
            if hasattr(self, 'ft232h') and self.ft232h is not None and hasattr(self.ft232h, "close"):
                self.ft232h.close()
        except Exception:
            pass
        sys.exit(1)


class SignalObserver(Observer):
    """Small Observer adapter to bind final_write_to_disk as completion handler."""

    def __init__(self, final_write_to_disk, logger) -> None:
        super().__init__()
        self.final_write_to_disk = final_write_to_disk
        self.logger = logger

    def on_next(self, value) -> None:
        # No-op: main pipeline already calls DigitizeVideo.handle_trigger
        return None

    def on_error(self, error) -> None:
        self.logger.error(f"Signal observer error: {error}")

    def on_completed(self) -> None:
        # Called when FT232H connector signals EOF via subject.on_completed()
        try:
            self.final_write_to_disk()
        except Exception:
            pass
