# DigitizeVideo.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
DigitizeVideo - top-level capture & orchestration for beck-view-digitize.

Design notes (Option B):
- Ft232hConnector owns the FTDI device and the poller thread. After the connector has been
  created, call DigitizeVideo.connect(connector): that gives DigitizeVideo the timing data and
  the handle it needs to stop and close the connector during shutdown.

- DigitizeVideo subscribes to the signal_subject for incoming frame events, captures every
  frame into a shared memory buffer and hands full buffers to a pool of worker processes
  (WriteImages.write_images).

- Buffer hand-over (all on the poller thread, so it is race free):
    * handle_trigger() runs synchronously inside subject.on_next(), i.e. on the poller thread.
    * When a buffer is full it is marked busy, the next free buffer is selected and made the
      capture target *before* the full buffer is submitted to the pool. A buffer that is in
      use by a worker is never written to.
    * If no buffer is free the capture waits (and logs) instead of overwriting data.

- Shutdown sequence:
    1) The poller signals EOF (or an error) by calling subject.on_completed()
    2) final_write_to_disk() writes the remaining frames, the timing CSV and runs cleanup()
    3) cleanup() stops the connector, waits for the workers, closes the pool, releases the
       camera and unlinks the shared memory. Finally wait_subject completes.
  The same path is used for SIGINT/SIGTERM (SIGBREAK on Windows).
"""

import cython
from cython.view cimport array

cimport numpy as np
import numpy as np

from datetime import datetime

from libc.string cimport memcpy

import gc
import platform
import logging
import multiprocessing
from multiprocessing import shared_memory
import threading
import time
import signal
import sys
from argparse import Namespace
from pathlib import Path
from typing import List

import cv2
from reactivex import operators as ops, Observer
from reactivex.subject import Subject

from TypeDefinitions import ImgDescType, SubjectDescType
from WriteImages import write_images

from Ft232hConnector cimport Ft232hConnector
from Ft232hConnector import get_timing, get_timing_view

from TimingResult cimport TimingResult as CTimingResult

# Time the camera needs after a changed exposure before the next frame is read. The value is
# a guess; verify with a test film that frames a/b/c of a bracketing series really differ in
# brightness by about one stop each (increase the value if not).
EXPOSURE_SETTLE_SECONDS = 0.01

# While the capture waits for a free buffer, an error is logged this often (seconds)
BUFFER_WAIT_LOG_INTERVAL = 5.0

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


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

        # capture state
        object capture_lock         # held while a frame is captured; final write waits for it
        long _pending_images        # images in the current buffer that were not handed to a worker yet

        # blank frame to copy on read errors
        np.ndarray blank_frame

        # Rx objects
        object signal_observer          # SignalObserver instance
        object photoCellSignalDisposable  # subscription disposable
        object completion_disposable     # kept for compatibility, may be None
        object wait_subject             # completed when the shutdown is finished

        # shared memory buffers for workers
        list shared_buffers            # list of shared_memory.SharedMemory objects
        list _shm_arrays               # numpy views (per shared buffer)
        list _desc_shms                # descriptor shared_memory objects
        list _desc_arrays              # numpy descriptor arrays (per buffer)

        int shared_buffers_index       # buffer the capture currently writes into
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

        This constructor creates the camera capture object, the shared memory buffers, the
        worker pool and subscribes to the signal subject. The FT232H poller itself belongs to
        the Ft232hConnector; call connect() after the connector has been created.
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
        # a chunk holds complete exposure series; at least one series (avoids chunk_size == 0)
        self.chunk_size = max(self.frame_multiplier,
                              args.chunk_size - (args.chunk_size % self.frame_multiplier))
        self.settings = bool(args.settings)
        self.gui = bool(args.gui)

        # subject to receive signals from FT232H connector / completed when we are done
        self.signal_subject = signal_subject
        self.wait_subject = wait_subject

        # administration (before anything can fail or any signal can arrive)
        self._completion_event = threading.Event()
        self.capture_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.processes = []
        self._pending_images = 0
        self.ft232h = None
        self.timing = None
        self.timing_view = None
        self._final_write_done = False
        self._cleaned_up = False
        self.processed_frames = 0

        # logging
        self.initialize_logging()

        # the GUI allows to choose a directory that does not exist yet
        self.output_path.mkdir(parents=True, exist_ok=True)

        # capture settings & buffers
        self.initialize_bracketing()
        self.initialize_camera()
        self.log_camera_properties()

        try:
            self.initialize_process_pool()
            self.blank_frame = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

            # stats
            self.start_time = time.perf_counter()
            self.new_tick = self.start_time

            # everything is ready: now accept signals and frame events
            self.register_signal_handlers()
            self.initialize_threads()
        except Exception:
            self.logger.exception("Initialization failed - releasing resources")
            try:
                self.cleanup()
            except Exception:
                pass
            raise

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
            self.logger.error(f"DigitizeVideo.initialize_camera: could not open camera {self.device_number}: {e}")
            raise

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Camera with device number {self.device_number} could not be opened "
                               f"(capture backend {api})")

        time.sleep(1)
        # set preferred resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # warm up
        ok, _ = self.cap.read()
        if not ok:
            self.logger.warning("DigitizeVideo.initialize_camera: warm-up read failed")
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        if self.img_width <= 0 or self.img_height <= 0:
            raise RuntimeError(f"Camera reports an invalid resolution {self.img_width}x{self.img_height}")
        if self.img_width != self.width or self.img_height != self.height:
            self.logger.warning(f"Camera delivers {self.img_width}x{self.img_height} instead of the "
                                f"requested {self.width}x{self.height}")

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
        """
        Subscribe to the signal_subject.

        The pipeline calls handle_trigger() for every event on the poller thread. The observer
        starts the final write when the subject completes (EOF) or fails.
        """
        self.signal_observer = SignalObserver(self.final_write_to_disk,
                                              logging.getLogger(__name__ + ".SignalObserver"))

        self.photoCellSignalDisposable = self.signal_subject.pipe(
            ops.map(lambda event: self.handle_trigger(event))
        ).subscribe(self.signal_observer)

    cpdef connect(self, Ft232hConnector conn):
        """
        Call this after Ft232hConnector is created.

        Remembers the connector (it is stopped and closed during cleanup) and stores the
        C-level timing reference.
        """
        self.ft232h = conn
        try:
            py_tv = get_timing()
            if py_tv is None:
                raise RuntimeError("the timing singleton has not been created yet")
            self.timing = py_tv
            self.timing_view = <CTimingResult?> py_tv  # checked cast
        except Exception as e:
            # keep the attribute None on failure
            self.timing_view = None
            self.logger.error(f"DigitizeVideo.connect: could not obtain timing_view: {e}")

    cpdef bint set_exposure(self, double exp_val):
        """Set camera exposure; return True on success."""
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
        Create a multiprocessing pool and pre-allocate the shared memory buffers.

        One buffer holds exactly `chunk_size` images, the number of images that are handed to a
        worker in one task (this is the "Anzahl Bilder pro Prozess" setting of the GUI).
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

        frames_per_buffer = int(self.chunk_size)

        # Shared memory allocation
        self.shared_buffers = [shared_memory.SharedMemory(create=True, size=(frames_per_buffer * self.img_bytes)) for _ in range(self.process_count)]

        self._shm_arrays = []
        self._desc_shms = []
        self._desc_arrays = []

        for shm in self.shared_buffers:
            arr = self.shm_to_uint8_4d(shm.buf, frames_per_buffer, self.img_height, self.img_width)
            self._shm_arrays.append(arr)

            desc_nbytes = frames_per_buffer * 3 * np.dtype(np.uint32).itemsize
            dshm = shared_memory.SharedMemory(create=True, size=desc_nbytes)
            desc_arr = np.ndarray((frames_per_buffer, 3), dtype=np.uint32, buffer=dshm.buf)
            desc_arr.fill(0)
            self._desc_shms.append(dshm)
            self._desc_arrays.append(desc_arr)

        self.shared_buffers_index = 0
        self.buffers_in_use = [False] * self.process_count
        self.processes = []
        self._frames_per_buffer = frames_per_buffer

    # ---------------------------
    # Capture / buffer management
    # ---------------------------
    def _cleanup_finished_processes(self) -> None:
        """
        Reap finished worker tasks and give their buffers back.

        Every finished entry is removed from self.processes under the lock, so exactly one
        caller handles it (this runs on the poller thread and, during shutdown, possibly on
        another thread). The descriptors are cleared *before* the buffer is released: once it
        is released the capture may write fresh descriptors into it.
        """
        ready = []
        still = []
        with self.buffer_lock:
            for entry in self.processes:
                if entry[0].ready():
                    ready.append(entry)
                else:
                    still.append(entry)
            self.processes = still

        for res, shm, desc_shm, buf_idx in ready:
            try:
                # re-raise worker exceptions here so we can log them
                res.get()
            except Exception as e:
                # Log the worker error, but still reclaim the buffer so the pipeline continues
                self.logger.error(f"Child process failed: {e}")

            try:
                self._desc_arrays[buf_idx].fill(0)
            except Exception:
                # best-effort only (arrays are released during cleanup)
                pass

            with self.buffer_lock:
                self.buffers_in_use[buf_idx] = False

    def _acquire_free_buffer(self, int current):
        """
        Return the index of a free buffer (not `current`), waiting for the workers if necessary.
        Returns None when the program is shutting down.
        """
        cdef int i, cand
        waited_since = None
        last_log = 0.0

        while True:
            self._cleanup_finished_processes()

            with self.buffer_lock:
                for i in range(1, self.process_count):
                    cand = (current + i) % self.process_count
                    if not self.buffers_in_use[cand]:
                        return cand

            if self._final_write_done:
                return None

            now = time.perf_counter()
            if waited_since is None:
                waited_since = now
                last_log = now
                self.logger.warning("All shared buffers busy; capture waits for the writer processes "
                                    "(frames may be missed - consider a larger/smaller chunk size)")
            elif now - last_log > BUFFER_WAIT_LOG_INTERVAL:
                last_log = now
                self.logger.error(f"Still waiting for a free buffer after {now - waited_since:.1f}s - "
                                  f"are the writer processes alive and is the disk writable?")
            time.sleep(0.005)

    def _write_buffer_sync(self, int buffer_index) -> None:
        """Write one buffer in the calling process (used for the final partial buffer and as fallback)."""
        shm = self.shared_buffers[buffer_index]
        desc_shm = self._desc_shms[buffer_index]
        write_images(shm.name.encode('utf-8'), desc_shm.name.encode('utf-8'),
                     self._frames_per_buffer, self.img_width, self.img_height,
                     self.output_path_b, self.compression_level)
        self._desc_arrays[buffer_index].fill(0)

    def _flush_current_buffer(self):
        """
        The current buffer is full: switch capture to the next free buffer and hand the full
        one to the worker pool. Runs on the poller thread (the only one that captures).

        Order matters: the capture target is switched *before* the full buffer is submitted,
        and the full buffer is marked busy before the switch.
        """
        cdef int buffer_index = self.shared_buffers_index

        with self.buffer_lock:
            self.buffers_in_use[buffer_index] = True

        next_buf = self._acquire_free_buffer(buffer_index)
        if next_buf is None:
            # shutting down: the images stay pending, final_write_to_disk() writes this buffer
            return False

        self.shared_buffers_index = next_buf
        self._pending_images = 0

        try:
            shm = self.shared_buffers[buffer_index]
            desc_shm = self._desc_shms[buffer_index]

            def process_error_callback(error):
                self.logger.error(f"Error in child process: {error}")

            result = self.pool.apply_async(
                write_images,
                args=(shm.name.encode('utf-8'), desc_shm.name.encode('utf-8'), self._frames_per_buffer,
                      self.img_width, self.img_height, self.output_path_b, self.compression_level),
                error_callback=process_error_callback
            )

            with self.buffer_lock:
                self.processes.append((result, shm, desc_shm, buffer_index))
            return True

        except Exception as e:
            self.logger.error(f"Error during apply_async ({e}); writing the buffer in this process")
            try:
                self._write_buffer_sync(buffer_index)
            except Exception:
                self.logger.exception("Synchronous write failed, images of this chunk are lost")
            finally:
                with self.buffer_lock:
                    self.buffers_in_use[buffer_index] = False
            return False

    def handle_trigger(self, event: SubjectDescType) -> None:
        """
        Called for each optocoupler event (synchronously on the poller thread).

        An exception here must never escape: reactivex would end the subscription and silently
        drop all further events (including the completion that starts the final write).
        """
        if self._final_write_done:
            return
        try:
            with self.capture_lock:
                # re-check: the final write may have started while we waited for the lock
                if self._final_write_done:
                    return
                self._capture_frame(event)
        except Exception:
            self.logger.exception(f"[Capture] unhandled error while capturing {event!r}")

    def _capture_frame(self, event) -> None:
        frame_count, start_cycle = event

        # retrieve frame(s) from camera
        self.take_picture(frame_count)

        # If we've completed a chunk, schedule writing to disk
        if (self.processed_frames % self.chunk_size) == 0:
            self._flush_current_buffer()

        if (frame_count % 100) == 0:
            capture_duration = time.perf_counter() - start_cycle
            self.logger.info(f"[Capture] Frame {frame_count} ({self.processed_frames}) took {capture_duration*1000:.2f} ms")

    def take_picture(self, int picture_count) -> None:
        """Capture one (or multiple if bracketing) frames and copy into shared buffer."""
        cdef np.uint8_t[:, :, :] src_mv
        cdef np.uint8_t[:, :, :] dst_mv
        cdef np.uint32_t[:, :] desc_mv
        cdef unsigned char* src_ptr
        cdef unsigned char* dst_ptr
        cdef Py_ssize_t frame_index
        cdef int img_bytes = self.img_bytes
        cdef int frames_per_buffer = self._frames_per_buffer

        frame_count = picture_count
        exposures = self.exposures
        n_exposures = len(exposures)
        shm_frames = self._shm_arrays[self.shared_buffers_index]
        desc_mv = self._desc_arrays[self.shared_buffers_index]
        expected_shape = (self.img_height, self.img_width, 3)

        for bracket_index, (exp_val, suffix) in enumerate(exposures):
            try:
                success, frame_data = self.cap.read()
            except Exception:
                # treated like success == False: a blank frame keeps the numbering free of gaps
                self.logger.exception(f"cap.read() raised at frame {frame_count}{suffix}")
                success, frame_data = False, None

            if self.bracketing and (bracket_index < (n_exposures - 1)):
                # set exposure value for next frame read
                next_exp = exposures[bracket_index + 1][0]
                if not self.set_exposure(next_exp):
                    self.logger.error(f"Could not set exposure to {next_exp} for frame {frame_count}{suffix}")
                time.sleep(EXPOSURE_SETTLE_SECONDS)

            # position inside the buffer: images are written one after the other and the buffer
            # is handed over after exactly frames_per_buffer images
            frame_index = self.processed_frames % frames_per_buffer

            # the size is checked: a frame with another size would make memcpy read past its end
            if success and frame_data is not None and frame_data.shape == expected_shape:
                if frame_data.dtype != np.uint8 or not frame_data.flags.c_contiguous:
                    frame_data = np.ascontiguousarray(frame_data, dtype=np.uint8)
                src_mv = frame_data
            else:
                got = None if frame_data is None else frame_data.shape
                self.logger.error(f"Read error at frame {frame_count}{suffix}, exposure {exp_val} "
                                  f"(success={success}, shape={got}, expected {expected_shape})")
                src_mv = self.blank_frame

            dst_mv = shm_frames[frame_index]
            src_ptr = &src_mv[0, 0, 0]
            dst_ptr = &dst_mv[0, 0, 0]

            memcpy(dst_ptr, src_ptr, img_bytes)

            desc_mv[frame_index, 0] = img_bytes
            desc_mv[frame_index, 1] = frame_count
            desc_mv[frame_index, 2] = bracket_index

            self.processed_frames += 1
            self._pending_images += 1

        if self.bracketing:
            first_exp = exposures[0][0]
            self.set_exposure(first_exp)

    # ---------------------------
    # Shutdown
    # ---------------------------
    def _write_remaining_images(self) -> None:
        """Write the images of the last, partially filled buffer."""
        if self._pending_images <= 0:
            return
        self.logger.info(f"Writing the remaining {self._pending_images} images")
        self._write_buffer_sync(self.shared_buffers_index)
        self._pending_images = 0

    def final_write_to_disk(self) -> None:
        """
        Write the remaining images and the timing CSV, then clean up and complete wait_subject.

        Idempotent: only the first caller does the work; later callers return at once (use
        wait_for_completion() to wait for the end). Every step is protected so that
        wait_subject is always completed and the main program cannot hang.
        """
        if self._final_write_done:
            return

        # no new captures from now on
        self._final_write_done = True

        try:
            # wait for a capture that is in progress, then write what is left in the buffer
            with self.capture_lock:
                try:
                    self._write_remaining_images()
                except Exception:
                    self.logger.exception("Error while writing the remaining images")

            # log statistics and save timing
            elapsed_time = time.perf_counter() - self.start_time
            average_fps = (self.processed_frames / elapsed_time) if elapsed_time > 0 else 0.0
            self.logger.info("------- End of Film ---------")
            self.logger.info(f"Total saved images (incl. exposures): {self.processed_frames}")
            self.logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
            self.logger.info(f"Average FPS: {average_fps:.2f}")

            time_now = datetime.now()
            fname = str(self.output_path / f"timing_{time_now.strftime('%Y_%m_%d_%H_%M_%S')}_{self.processed_frames:05d}.csv")
            self.logger.info(f"CSV file to be opened: {fname}")
            try:
                self.write_timing_csv(fname)
            except Exception:
                self.logger.exception("Could not write the timing CSV")
        except Exception:
            self.logger.exception("Error in final_write_to_disk")
        finally:
            try:
                self.cleanup()
            except Exception:
                self.logger.exception("Error during cleanup")
            finally:
                self._completion_event.set()
                # signal completion to main
                try:
                    self.wait_subject.on_completed()
                except Exception:
                    self.logger.exception("wait_subject.on_completed() failed")

    cpdef write_timing_csv(self, str filename):
        cdef int i
        cdef int rows
        cdef int count = 0
        cdef int cycle = 1
        cdef int work = 2
        cdef int read = 3
        cdef int latency = 4
        cdef int wait_time = 5
        cdef int total_work = 6
        cdef double[:, :] buf
        cdef CTimingResult tv = self.timing_view

        if tv is None:
            tv = get_timing_view()
        if tv is None:
            self.logger.warning("No timing data available - CSV file not written")
            return

        buf = tv.buf
        rows = tv.size

        with open(filename, "w") as f:
            f.write("frame,read,cycle,work,latency,wait_time,total_work\n")

            for i in range(rows):
                f.write(
                    f"{buf[i, count]},{buf[i, read]},{buf[i, cycle]},"
                    f"{buf[i, work]},{buf[i, latency]},{buf[i, wait_time]},{buf[i, total_work]}\n"
                )

    def wait_for_completion(self, timeout: float = None) -> bool:
        """
        Block until the final write and the cleanup are finished or until `timeout` seconds.
        Returns True if the shutdown finished, False if timed out.
        """
        return self._completion_event.wait(timeout)

    def cleanup(self, timeout_wait_for_workers: float = 2.0) -> None:
        """
        Idempotent cleanup.
        - stop the ft232h poller (if connected)
        - reap finished pool tasks, wait briefly for outstanding ones
        - close/join pool (waits for the remaining tasks)
        - dispose Rx subscriptions, release camera
        - release memoryviews, close+unlink shared memory
        - final gc and close the FTDI connector
        """
        if self._cleaned_up:
            self.logger.info("cleanup() already ran — skipping")
            return
        self._cleaned_up = True

        self.logger.info("Cleaning up ...")

        # 0) Stop FTDI poller early to avoid new events
        ft = self.ft232h
        if ft is not None:
            try:
                ft.stop()
            except Exception:
                self.logger.exception("Error stopping ft232h in cleanup")

        # 1) Reap finished worker results, wait briefly for outstanding ones
        t0 = time.time()
        remaining = 0
        try:
            while True:
                self._cleanup_finished_processes()
                with self.buffer_lock:
                    remaining = len(self.processes)
                if remaining == 0:
                    break
                if time.time() - t0 > timeout_wait_for_workers:
                    self.logger.warning(f"{remaining} worker task(s) still running - waiting in pool.join()")
                    break
                time.sleep(0.05)
        except Exception:
            self.logger.exception("Error while waiting for child processes")

        # 2) Close & join multiprocessing pool (waits for the remaining tasks)
        pool = self.pool
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
            # log the result of tasks that finished while joining
            try:
                self._cleanup_finished_processes()
            except Exception:
                self.logger.exception("Error reaping worker results")

        # 3) Dispose Rx subscriptions
        for disp in (self.photoCellSignalDisposable, self.completion_disposable):
            if disp is not None:
                try:
                    disp.dispose()
                except Exception:
                    pass

        # 4) Release camera
        cap = self.cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                self.logger.exception("Error releasing camera")

        # 5) Drop memoryview references in parent (allow unlink)
        self._shm_arrays = []
        self._desc_arrays = []
        self.processes = []

        # 6) Force GC now to free exported views
        gc.collect()
        time.sleep(0.02)

        # 7) Close & unlink all SharedMemory objects with retries
        for list_name in ("shared_buffers", "_desc_shms"):
            lst = self.shared_buffers if list_name == "shared_buffers" else self._desc_shms
            if not lst:
                continue
            for shm in list(lst):
                try:
                    self._close_and_unlink_shm_with_retries(shm, retries=5, sleep=0.02)
                except Exception:
                    self.logger.exception(f"Failed close/unlink for an entry in {list_name}")

        # 8) Clear those lists (remove references to SharedMemory objects)
        self.shared_buffers = []
        self._desc_shms = []

        # 9) Final GC + short sleep to let runtime finalize destructors
        gc.collect()
        time.sleep(0.02)

        # 10) Close FTDI connector last (best-effort)
        if ft is not None:
            try:
                ft.close()
            except Exception:
                self.logger.exception("Error closing ft232h at end of cleanup")

        self.logger.info("Cleanup complete.")

    def _final_cleanup_finished_processes(self) -> None:
        """Kept for compatibility: reaping is done by _cleanup_finished_processes()."""
        self._cleanup_finished_processes()

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
        """Process-level signal handler for graceful shutdown (SIGINT/SIGTERM/SIGBREAK)."""
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        self.logger.warning(f"Signal {name} received, stopping...")
        try:
            # writes the remaining frames, stops and closes the connector, cleans up
            self.final_write_to_disk()
            # if another thread (e.g. the poller at EOF) is already doing the final write, wait for it
            self._completion_event.wait(60.0)
        except Exception:
            self.logger.exception("Error during shutdown after signal")
        sys.exit(1)


class SignalObserver(Observer):
    """Small Observer adapter to bind final_write_to_disk as completion/error handler."""

    def __init__(self, final_write_to_disk, logger) -> None:
        super().__init__()
        self.final_write_to_disk = final_write_to_disk
        self.logger = logger

    def on_next(self, value) -> None:
        # No-op: main pipeline already calls DigitizeVideo.handle_trigger
        return None

    def on_error(self, error) -> None:
        # whatever went wrong upstream: save what has been captured so far
        self.logger.error(f"Signal observer error: {error!r}")
        self._finish()

    def on_completed(self) -> None:
        # Called when the FT232H connector signals EOF via subject.on_completed()
        self._finish()

    def _finish(self) -> None:
        try:
            self.final_write_to_disk()
        except Exception:
            self.logger.exception("final_write_to_disk failed")
