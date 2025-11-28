# DigitizeVideo.pyx
# Cython module for digitizing Super8 frames (capture + background writer)
# Focus: cooperative, clean shutdown that integrates with an external shutdown Event.

import cython
from cython.view cimport array  # Import for memory views

cimport numpy as np
import numpy as np  # runtime numpy

from libc.string cimport memcpy

import logging
import multiprocessing
from multiprocessing import shared_memory, set_start_method
import concurrent.futures
import threading

import platform
import signal
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
from reactivex import operators as ops, Observer
from reactivex.subject import Subject

from typing import List

from TypeDefinitions import ImgDescType, ProcessType, SubjectDescType
from WriteImages import write_images

from Ft232hConnector import get_timing

cdef class DigitizeVideo:
    """
    Cython extension class for digitizing Super8 frames.

    Changes for clean shutdown:
    - Accept an externally provided threading.Event (`shutdown_event`) to coordinate shutdown
      between main and this class (safe in signal handlers in main).
    - Expose `request_shutdown()` to programmatically request shutdown.
    - Provide `run_main_loop()` which blocks until shutdown is requested and then performs
      finalization (final_write_to_disk + cleanup).
    """

    cdef:
        # basic configuration & command-line values
        int device_number
        object output_path          # pathlib.Path at runtime (store as object)
        object exposures            # list of (exposure, suffix)
        object timing               # TimingResult singleton (Python object)

        # camera and capture geometry
        int width
        int height
        bint bracketing
        int frame_multiplier
        int chunk_size
        bint settings
        bint gui

        # subject / logging / sync
        object signal_subject       # reactivex Subject (Python)
        object logger               # Python logger

        # camera runtime parameters (after open)
        int img_width
        int img_height
        int img_bytes               # bytes per frame (w*h*3)

        # small per-frame bookkeeping
        list img_desc               # light-weight Python descriptors for logging (list of tuples)
        object img_desc_lock        # threading.Lock

        # prebuilt blank frame to use on read errors
        np.ndarray blank_frame      # numpy ndarray

        # timing / stats
        long processed_frames
        double start_time
        double new_tick

        # thread/process pools and related objects
        object executor                 # ThreadPoolExecutor (python object)
        object signal_observer          # SignalObserver instance
        object photoCellSignalDisposable  # subscription disposable

        # multiprocessing shared buffers
        list shared_buffers            # list of shared_memory.SharedMemory objects
        list _shm_arrays               # numpy views (per shared buffer)
        list _desc_shms                # descriptor shared_memory objects
        list _desc_arrays              # numpy descriptor arrays (per buffer)

        int shared_buffers_index
        list buffers_in_use
        object buffer_lock             # threading.Lock
        list processes                 # list of (AsyncResult, shm, buf_idx)
        int process_count
        int _frames_per_buffer

        # camera handle and settings
        object cap
        int compression_level
        object pool                    # multiprocessing.Pool (worker processes)

        # external shutdown coordination (threading.Event)
        object _shutdown_event

    def __cinit__(self):
        # ensure numpy C-API initialized for typed arrays if needed
        # (np.import_array must be called from Python init path; we also call it in __init__ to be safe)
        pass

    def __init__(self, args: Namespace, signal_subject: Subject, object shutdown_event=None) -> None:
        """
        Initialize the DigitizeVideo instance.

        Parameters:
            args: parsed CLI Namespace
            signal_subject: reactivex Subject used to receive frame triggers
            shutdown_event: optional threading.Event used to coordinate shutdown
                            (If omitted DigitizeVideo creates its own Event)
        """
        # Ensure NumPy C-API available for typed memoryviews/ndarrays used later
        np.import_array()

        # basic fields from args
        print(f"{args=}")
        self.device_number = args.device
        self.output_path = Path(args.output_path)
        self.width = args.width
        self.height = args.height
        self.bracketing = bool(args.bracketing)
        self.frame_multiplier = 3 if self.bracketing else 1
        # ensure chunk_size is multiple of frame_multiplier
        self.chunk_size = args.chunk_size - (args.chunk_size % self.frame_multiplier)
        self.settings = bool(args.settings)   # display direct show settings menu
        self.gui = bool(args.gui)

        # store external subject (this object does not register signal handlers itself)
        # Signal handling must be handled in main (main thread) and set the shutdown_event.
        self.signal_subject = signal_subject  # a reactivex subject emitting photo cell signals.

        # Accept an external shutdown event, otherwise create our own.
        # The external event should be created and set by main's signal handler.
        if shutdown_event is None:
            self._shutdown_event = threading.Event()
        else:
            self._shutdown_event = shutdown_event

        # register signal handlers should NOT be done here (must be done in main thread).
        # removed: signal.signal(signal.SIGINT, self._on_signal)

        # logging + subsystems
        self.initialize_logging()
        self.initialize_bracketing()
        self.initialize_camera()
        self.initialize_threads()
        self.initialize_process_pool()

        # descriptors and thread-safety
        self.img_desc = []               # kind of meta data (Python list)
        self.img_desc_lock = threading.Lock()

        # prebuilt blank frame (numpy) to memcpy on errors
        # This uses img dimensions determined in initialize_camera()
        self.blank_frame = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # get timing singleton from FT232H connector module
        self.timing = get_timing()

        # stats
        self.processed_frames = 0
        self.start_time = time.perf_counter()
        self.new_tick = self.start_time

    # -------------------------
    # Lifecycle helpers
    # -------------------------
    def request_shutdown(self) -> None:
        """
        Programmatically request a graceful shutdown. This simply sets the internal
        shutdown Event so run_main_loop() (or other loops) can finish cooperatively.

        This method is safe to call from any thread (it just sets threading.Event).
        """
        try:
            self._shutdown_event.set()
        except Exception:
            # defensive: ignore if event object is invalid
            pass

    def run_main_loop(self, double poll_interval=0.1) -> None:
        """
        Blocks until shutdown is requested.

        When the event is set, perform final write and cleanup before returning.
        This centralises the cleanup sequence and keeps the signal-handler itself tiny.
        """
        # Wait until shutdown event is set (set by main's signal handler or elsewhere)
        try:
            while not self._shutdown_event.is_set():
                # Poll periodically — this keeps the thread responsive to shutdown
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            # As an additional guard allow KeyboardInterrupt to request shutdown
            self.request_shutdown()

        # Once shutdown requested, ensure frames are flushed to disk and resources cleaned up
        try:
            self.final_write_to_disk()
        except Exception:
            # best-effort: log but don't re-raise (we are shutting down)
            try:
                self.logger.exception("Error during final_write_to_disk()")
            except Exception:
                pass

        # perform final cleanup (this will try to close pools and shared memory)
        try:
            self.cleanup()
        except Exception:
            try:
                self.logger.exception("Error during cleanup()")
            except Exception:
                pass

    # -------------------------
    # The rest of the class is unchanged (capture, pool creation, etc.)
    # Keep original behaviour for capture and writing.
    # -------------------------

    def initialize_logging(self) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # if program was started from beck-view-gui
        if self.gui:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)

    def initialize_bracketing(self) -> None:
        # Define exposure settings
        # ensure exposures exist as attribute
        self.exposures = []

        if platform.system() == "Linux":
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
        # self.logger.info(f"Build details: {cv2.getBuildInformation()}")
        """
        Initialize the camera for video capturing based on the device number.
        """
        api: cython.int = cv2.CAP_ANY
        if platform.system() == "Windows":
            api = cv2.CAP_DSHOW if self.settings else cv2.CAP_MSMF
        elif platform.system() == "Linux":
            api = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(self.device_number, api, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        time.sleep(1) # Windows needs some time to initialize the camera

        # Set camera resolution to HDMI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # warm up camera before setting properties
        _, _ = self.cap.read()

        # Retrieve video frame properties
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.compression_level = 3

        # Format of the Mat objects (see Mat::type()) returned by VideoCapture::retrieve().
        # Set value -1 to fetch undecoded RAW video streams (as Mat 8UC1).
        self.cap.set(cv2.CAP_PROP_FORMAT, -1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # CAP_PROP_AUTO_EXPOSURE (https://github.com/opencv/opencv/issues/9738)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # automode
        time.sleep(1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual

        if platform.system() == "Windows":
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        else:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 1.0 / (1 << 7))

        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        time.sleep(1)

        if platform.system() == "Windows" and self.settings:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 0) # launches DirectShow menu for ELP camera

        self.logger.info(f"Camera properties:")
        self.logger.info(f"   frame width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.logger.info(f"   frame height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f"   fps = {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.logger.info(f"   gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f"   auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f"   exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f"   format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f"   fourcc = {self.cap.get(cv2.CAP_PROP_FOURCC)}")
        self.logger.info(f"   mode = {self.cap.get(cv2.CAP_PROP_MODE)}")
        self.logger.info(f"   buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
        self.logger.info(f"   hardware acceleration support = {cv2.checkHardwareSupport(cv2.CAP_PROP_HW_ACCELERATION)}")
        self.logger.info(f"   video acceleration support = {cv2.checkHardwareSupport(cv2.VIDEO_ACCELERATION_ANY)}")
        self.logger.info(f"   fps support = {cv2.checkHardwareSupport(cv2.CAP_PROP_FPS)}")
        self.logger.info(f"   gain hardware support = {cv2.checkHardwareSupport(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f"   auto exposure hardware support = {cv2.checkHardwareSupport(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f"   exposure hardware support = {cv2.checkHardwareSupport(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f"   format hardware support = {cv2.checkHardwareSupport(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f"   mode hardware support = {cv2.checkHardwareSupport(cv2.CAP_PROP_MODE)}")
        self.logger.info(f"   buffersize hardware support = {cv2.checkHardwareSupport(cv2.CAP_PROP_BUFFERSIZE)}")

    def initialize_threads(self) -> None:
        """
        Initialize threads, subjects, and subscriptions for multithreaded processing.
        """

        cpu_count = multiprocessing.cpu_count()
        self.logger.info(f"CPU count: {cpu_count}")
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # Create an observer for processing photo cell signals
        self.signal_observer = SignalObserver(self.final_write_to_disk)

        # Subscribe to signal
        # The observer will call final_write_to_disk() when it gets an on_completed().
        self.photoCellSignalDisposable = self.signal_subject.pipe(
            # Capture immediately upon signal
            ops.map(lambda event: self.handle_trigger(event))
        ).subscribe(self.signal_observer)

    cpdef bint set_exposure(self, double exp_val):
        """C callable exposure setter returning bool"""
        return bool(self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val))

    def shm_to_uint8_4d(self, buf, Py_ssize_t frames, int h, int w):
        """
        Return a numpy ndarray shaped (frames, h, w, 3) backed by shm.buf.
        Caller must keep ndarray alive; we store references in self._shm_arrays.
        """
        cdef np.ndarray[np.uint8_t, ndim=1] arr1d = np.frombuffer(buf, dtype=np.uint8, count=frames * h * w * 3)
        arr4d = arr1d.reshape(frames, h, w, 3)
        return arr4d

    def initialize_process_pool(self) -> None:
        """
        Create a pool of worker processes for parallel processing.
        """
        # bytes per image in the configured capture resolution
        self.img_bytes = self.img_width * self.img_height * 3  # Calculate bytes in a single frame

        ctx = multiprocessing.get_context("forkserver")   # explicit context
        # Calculate the optimal number of processes
        self.process_count = max(2, multiprocessing.cpu_count() - 1)
        # Create pool — store as attribute so finalize/wait logic can close it later
        self.pool = ctx.Pool(self.process_count)

        # frames per buffer: chunk_size * frame_multiplier (frames stored physically per buffer)
        frames_per_buffer = int(self.chunk_size * self.frame_multiplier)

        # pre-allocate shared memory buffers (one per worker buffer)
        self.shared_buffers = [shared_memory.SharedMemory(create=True, size=(frames_per_buffer * self.img_bytes)) for _ in range(self.process_count)]

        # Create per-buffer numpy arrays backed by SHM for frames and descriptors
        self._shm_arrays = []
        self._desc_shms = []
        self._desc_arrays = []

        for shm in self.shared_buffers:
            # frames ndarray (frames_per_buffer, h, w, 3)
            arr = self.shm_to_uint8_4d(shm.buf, frames_per_buffer, self.img_height, self.img_width)
            self._shm_arrays.append(arr)

            # descriptor SHM for that buffer: dtype uint32, shape (frames_per_buffer, 3)
            desc_nbytes = frames_per_buffer * 3 * np.dtype(np.uint32).itemsize
            dshm = shared_memory.SharedMemory(create=True, size=desc_nbytes)
            desc_arr = np.ndarray((frames_per_buffer, 3), dtype=np.uint32, buffer=dshm.buf)
            # initialize to zeros
            desc_arr.fill(0)
            self._desc_shms.append(dshm)
            self._desc_arrays.append(desc_arr)

        # bookkeeping
        self.shared_buffers_index = 0
        self.buffers_in_use = [False] * self.process_count
        self.buffer_lock = threading.Lock()
        self.processes = []

        # store these for worker calls (Python-level)
        self._frames_per_buffer = frames_per_buffer

    def _release_views(self) -> None:
        # release references so numpy views are GC'able
        self._shm_arrays = []
        self._desc_arrays = []
        # don't unlink here; cleanup() will close and unlink SHM objects

    def _cleanup_finished_processes(self) -> None:
        still = []
        for res, shm, buf_idx in self.processes:
            if res.ready():
                try:
                    res.get()
                except Exception as e:
                    self.logger.error(f"Child process failed: {e}")

                # mark buffer free and close the SHM handle in the parent (we still keep it in self.shared_buffers for reuse)
                with self.buffer_lock:
                    self.buffers_in_use[buf_idx] = False

                # If you want to reduce handles, close the shm handle here (main keeps the SharedMemory object but close/unlink in final cleanup)
                try:
                    shm.close()
                except Exception:
                    pass
            else:
                still.append((res, shm, buf_idx))
        self.processes = still


    def handle_trigger(self, event: SubjectDescType) -> None:
        frame_count, start_cycle = event

        # Capture frame immediately (blocking, but fast if camera is primed)
        read_time_start = time.perf_counter()
        self.take_picture(event)

        # slow path (chunk rollover, scheduling) in executor
        if (self.processed_frames % self.chunk_size) == 0:
            # Write frames to disk in background

            # snapshot descriptors
            with self.img_desc_lock:
                descriptors = list(self.img_desc)   # keep for logging; worker uses SHM
                self.img_desc = []

            with self.buffer_lock:
                self.buffers_in_use[self.shared_buffers_index] = True

            # current buffer holds the just-filled chunk
            buffer_index = self.shared_buffers_index

            # Schedule async housekeeping
            self.executor.submit(self._post_capture, buffer_index, descriptors)

        if self.gui and self.processed_frames % 100 == 0:
            capture_duration = time.perf_counter() - read_time_start
            # timing supports item assignment / slicing — keep as-is
            try:
                self.timing[self.processed_frames:3] = capture_duration
            except Exception:
                pass
            self.logger.info(f"[Capture] Frame {frame_count} ({self.processed_frames}) took {capture_duration*1000:.2f} ms")

    def _post_capture(self, buffer_index: int, descriptors: List[ImgDescType]) -> None:
        """Executed in thread pool — can block without stalling next frame capture."""

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

        # advance capture target
        self.shared_buffers_index = next_buf

        # schedule write to disk — pass SHM names and frames-per-buffer (no Python descriptors)
        try:
            shm = self.shared_buffers[buffer_index]
            desc_shm = self._desc_shms[buffer_index]
            shm_name = shm.name
            desc_name = desc_shm.name
            frames_total = self._frames_per_buffer

            def process_error_callback(error):
                self.logger.error(f"Error in child process: {error}")

            result = self.pool.apply_async(
                write_images,
                args=(shm_name, desc_name, frames_total, self.img_width, self.img_height, self.output_path, self.compression_level),
                error_callback=process_error_callback
            )

            with self.buffer_lock:
                self.processes.append((result, shm, buffer_index))

            # clear descriptor shm for next use
            desc_arr = self._desc_arrays[buffer_index]
            desc_arr.fill(0)
        except Exception as e:
            self.logger.error(f"Error during apply_async: {e}")
            with self.buffer_lock:
                self.buffers_in_use[buffer_index] = False
        finally:
            self._cleanup_finished_processes()

    def take_picture(self, descriptor: SubjectDescType) -> None:
        """
        Capture frame(s) from the camera. If bracketing is enabled, capture three exposures:
        - standard (-7, suffix 'a'), short (-8, suffix 'b'), long (-6, suffix 'c').
        Returns None.
        """

        frame_count, signal_time = descriptor

        frame_multiplier = self.frame_multiplier
        chunk_size = self.chunk_size
        img_bytes = self.img_bytes
        exposures = self.exposures
        buf_idx = self.shared_buffers_index
        frames_per_buffer = self._frames_per_buffer

        cdef np.ndarray[np.uint8_t, ndim=4] shm_view
        cdef np.ndarray[np.uint8_t, ndim=3] shm_frame
        cdef np.ndarray[np.uint8_t, ndim=3] fmv

        # declare C-typed memoryviews
        cdef np.uint8_t[:, :, :] src_mv
        cdef np.uint8_t[:, :, :] dst_mv

        # declare C-typed memoryviews
        cdef unsigned char* src_ptr
        cdef unsigned char* dst_ptr

        for bracket_index, (exp_val, suffix) in enumerate(exposures):
            success, frame_data = self.cap.read()

            if self.bracketing:
                if bracket_index < (len(exposures) - 1):
                    next_exp, _ = exposures[bracket_index + 1]
                    if not self.set_exposure(next_exp):
                        self.logger.error(f"Could not set exposure to {next_exp} for frame {frame_count}{suffix}")
                    time.sleep(0.01)

            # logical -> physical index within buffer (account bracket exposures)
            logical_index = frame_count * frame_multiplier
            frame_index = (logical_index + bracket_index) % frames_per_buffer

            # get numpy views for this buffer
            shm_frames = self._shm_arrays[buf_idx]       # ndarray shape (frames_per_buffer, h, w, 3)
            desc_arr = self._desc_arrays[buf_idx]        # ndarray shape (frames_per_buffer, 3) uint32

            # pointer target: shm_frames[frame_index] is an ndarray view shape (h,w,3)
            if success:
                # ensure frame_data is contiguous
                if not frame_data.flags.c_contiguous or frame_data.dtype != np.uint8:
                    frame_data = np.ascontiguousarray(frame_data, dtype=np.uint8)

                # cast numpy → typed view and shm view → typed view
                src_mv = frame_data
            else:
                self.logger.error(f"Read error at frame {frame_count}{suffix}, exposure {exp_val}")
                src_mv = self.blank_frame

            dst_mv = shm_frames[frame_index]

            # obtain raw C pointers
            src_ptr = &src_mv[0, 0, 0]
            dst_ptr = &dst_mv[0, 0, 0]

            memcpy(dst_ptr, src_ptr, img_bytes)

            # write descriptor (uint32: img_bytes, frame_count, bracket_index)
            desc_arr[frame_index, 0] = np.uint32(img_bytes)
            desc_arr[frame_index, 1] = np.uint32(frame_count)
            desc_arr[frame_index, 2] = np.uint32(bracket_index)

            # maintain a lightweight Python descriptor list for optional logging/legacy code
            with self.img_desc_lock:
                self.img_desc.append((img_bytes, frame_count, suffix))

            self.processed_frames += 1

        # reset to first exposure after loop
        if self.bracketing:
            first_exp, _ = exposures[0]
            self.set_exposure(first_exp)


    def final_write_to_disk(self) -> None:
        """
        Write final images to disk and ensure all processes have finished.
        """
        if self.img_desc:
            current_buf = self.shared_buffers_index

            with self.buffer_lock:
                self.buffers_in_use[current_buf] = True

            # schedule final write using SHM
            try:
                shm = self.shared_buffers[current_buf]
                desc_shm = self._desc_shms[current_buf]

                shm_name = shm.name
                desc_name = desc_shm.name
                frames_total = self._frames_per_buffer

                def process_error_callback(error):
                    self.logger.error(f"Error in child process: {error}")

                result = self.pool.apply_async(
                    write_images,
                    args=(
                        shm_name,
                        desc_name,
                        frames_total,
                        self.img_width,
                        self.img_height,
                        self.output_path,
                        self.compression_level
                    ),
                    error_callback=process_error_callback
                )

                with self.buffer_lock:
                    self.processes.append((result, shm, current_buf))

                # clear descriptor shm for next use
                desc_arr = self._desc_arrays[current_buf]
                desc_arr.fill(0)
            except Exception as e:
                self.logger.error(f"Error during final apply_async: {e}")
                with self.buffer_lock:
                    self.buffers_in_use[current_buf] = False

            self.img_desc = []

        # wait for pool tasks
        while self.processes:
            self._cleanup_finished_processes()
            time.sleep(0.05)

        # close & join pool
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.close()
                try:
                    self.pool.join()
                except Exception as e:
                    self.logger.error(f"Error joining pool: {e}")
        except Exception:
            # ignore pool cleanup errors
            pass

        # Calculate elapsed time and log statistics
        elapsed_time = time.perf_counter() - self.start_time
        average_fps = (self.processed_frames / elapsed_time) if elapsed_time > 0 else 0.0

        self.logger.info("------- End of Film ---------")
        self.logger.info(f"Total saved images (incl. exposures): {self.processed_frames}")
        self.logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Average FPS: {average_fps:.2f}")

        # get timing array as numpy (copy)
        try:
            tim = self.timing.to_numpy()
        except Exception:
            tim = np.empty((0, 7))

        if tim.size > 0:
            wait_time        = tim[:, 5]
            work_time        = tim[:, 2]
            read_time        = tim[:, 3]
            total_work_time  = tim[:, 6]
            latency_time     = tim[:, 4]

            self.logger.info(f"Average wait time = {wait_time.mean():.5f}")
            self.logger.info(f"Min wait time     = {wait_time.min():.5f}")
            self.logger.info(f"Max wait time     = {wait_time.max():.5f}")

            self.logger.info(f"Average work time = {work_time.mean():.5f}")
            self.logger.info(f"Min work time     = {work_time.min():.5f}")
            self.logger.info(f"Max work time     = {work_time.max():.5f}")

            self.logger.info(f"Average read time = {read_time.mean():.5f}")
            self.logger.info(f"Min read time     = {read_time.min():.5f}")
            self.logger.info(f"Max read time     = {read_time.max():.5f}")

            self.logger.info(f"Average total     = {total_work_time.mean():.5f}")
            self.logger.info(f"Min total         = {total_work_time.min():.5f}")
            self.logger.info(f"Max total         = {total_work_time.max():.5f}")

            # sort by total_work descending
            order = total_work_time.argsort()[::-1]

            for rank in range(min(25, len(order))):
                idx = order[rank]
                self.logger.info(
                    f"Longest {rank}: frame={int(tim[idx,0])} total={tim[idx,6]:.6f}"
                )

            # reverse for shortest
            order = order[::-1]
            for rank in range(min(25, len(order))):
                idx = order[rank]
                self.logger.info(
                    f"Shortest {rank}: frame={int(tim[idx,0])} total={tim[idx,6]:.6f}"
                )

            timing_log = str(self.output_path / f"timing_{self.processed_frames:05d}.csv").encode('utf-8')
            np.savetxt(timing_log, self.timing[:self.processed_frames+1], delimiter=",")

        self.cleanup()

    def release_camera(self) -> None:
        """
        Release the camera.

        Returns:
            None
        """

        try:
            self.cap.release()
        except Exception:
            pass

    def cleanup(self) -> None:
        """
        Clean up resources and log statistics upon instance destruction.

        Returns:
            None
        """

        self.logger.info("Cleaning up ...")

        # shutdown executor threads (they schedule tasks synchronously to pool)
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'cap'):
            self.release_camera()

        # dispose subscription
        try:
            self.photoCellSignalDisposable.dispose()
        except Exception:
            pass

        # release references
        self._release_views()

        # close & unlink shared memory (frames and descriptors)
        for shm in getattr(self, "shared_buffers", []):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                # log unexpected errors
                try:
                    self.logger.exception("Error unlinking shm")
                except Exception:
                    pass

        for dshm in getattr(self, "_desc_shms", []):
            try:
                dshm.close()
            except Exception:
                pass
            try:
                dshm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                try:
                    self.logger.exception("Error unlinking desc shm")
                except Exception:
                    pass

        # Additional aggressive cleanup: ensure pool is closed/joined
        if hasattr(self, "pool") and self.pool is not None:
            try:
                self.pool.close()
            except Exception:
                pass
            try:
                self.pool.join()
            except Exception:
                # try terminating if join hangs
                try:
                    self.pool.terminate()
                    self.pool.join()
                except Exception:
                    pass

    def _on_signal(self, signum: int, frame) -> None:
        """
        Kept for compatibility but not used for signal registration.
        Prefer using a main-level signal handler that sets external shutdown_event.
        """
        name = signal.Signals(signum).name
        self.logger.warning(f"Signal {name} received, requesting shutdown...")
        self.request_shutdown()


class SignalObserver(Observer):
    """
    Observer that calls final_write_to_disk() on completion.
    """

    def __init__(self, final_write_to_disk) -> None:
        super().__init__()
        self.final_write_to_disk = final_write_to_disk
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def on_next(self, value) -> None:
        # no action required — the pipeline calls DigitizeVideo.handle_trigger()
        pass

    def on_error(self, error) -> None:
        self.logger.error(f"Signal observer error: {error}")

    def on_completed(self) -> None:
        # when the subject completes it signals end-of-film: flush and cleanup
        try:
            self.final_write_to_disk()
        except Exception:
            try:
                self.logger.exception("Error in final_write_to_disk during on_completed")
            except Exception:
                pass
