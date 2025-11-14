import cython
from cython.view cimport array  # Import for memory views

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is distributed with Numpy).
# Here we've used the name "cnp" to make it easier to understand what comes from the cimported module and
# what comes from the imported module, however you can use the same name for both if you wish.
cimport numpy as np # Import numpy for uint8 type

from libc.string cimport memcpy

import logging
import multiprocessing
from multiprocessing import shared_memory, set_start_method
import concurrent.futures
import threading

import platform
import signal
import types
import sys
import time
from argparse import Namespace

from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops, Observer
from reactivex.subject import Subject

from typing import List

from Timing import timing
from TypeDefinitions import ImgDescType, ProcessType, SubjectDescType
from WriteImages import write_images

class DigitizeVideo:
    """
    Class for digitizing analog super 8 film frames.

    This class initializes a video capturing process and provides methods to process frames using reactive programming.
    """

    def __init__(self, args: Namespace, signal_subject: Subject) -> None:
        """
        Initialize the DigitizeVideo instance with provided arguments and a signal subject.

        Args:
            args: Namespace containing command line arguments.
            signal_subject: Subject emitting photo cell signals.
        """
        # Initialize instance attributes
        self.device_number: cython.int = args.device  # device number of camera
        self.output_path: Path = args.output_path  # The directory for dumping digitised frames into
        self.width: cython.int = args.width
        self.height: cython.int = args.height
        self.bracketing: cython.bint = args.bracketing  # Use exposure bracketing
        self.frame_multiplier: cython.int = 3 if self.bracketing else 1
        self.chunk_size: cython.int = args.chunk_size - (
                args.chunk_size % self.frame_multiplier)  # Quantity of frames (images) passed to a process
        self.settings: cython.bint = args.settings  # Display direct show settings menu
        self.gui: cython.bint = args.gui

        self.signal_subject: Subject = signal_subject  # A reactivex subject emitting photo cell signals.

        # Signal handler is called on interrupt (ctrl-c) and terminate
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        # It's necessary to call "import_array" as we use part of the numpy PyArray_* API.
        # From Cython 3, accessing attributes like ".shape" on a typed Numpy array use this API.
        # Therefore it is recommend to always calling "import_array" whenever "cimport numpy" is used.
        np.import_array()

        # Set up logging, camera, threading and process pool
        self.initialize_logging()
        self.initialize_bracketing()
        self.initialize_camera()
        self.initialize_threads()
        self.initialize_process_pool()

        self.img_desc: [ImgDescType] = []  # kind of meta data
        self.img_desc_lock = threading.Lock() # for thread safety

        # Initialize counters and timing
        self.processed_frames: cython.int = 0
        self.start_time: cython.double = time.perf_counter()
        self.new_tick: cython.double = self.start_time

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
            if self.settings:
                api = cv2.CAP_DSHOW
            else:
                api = cv2.CAP_MSMF
        elif platform.system() == "Linux":
            api = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(self.device_number,
                                    api,
                                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        time.sleep(1)  # Windows needs some time to initialize the camera

        # Set camera resolution to HDMI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # warm up camera before setting properties
        _, _ = self.cap.read()

        # Retrieve video frame properties
        self.img_width: cython.int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: cython.int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        # Format of the Mat objects (see Mat::type()) returned by VideoCapture::retrieve().
        # Set value -1 to fetch undecoded RAW video streams (as Mat 8UC1).
        self.cap.set(cv2.CAP_PROP_FORMAT, -1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # CAP_PROP_AUTO_EXPOSURE (https://github.com/opencv/opencv/issues/9738)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # automode
        time.sleep(1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual

        if platform.system() == "Windows":
            # EXP_TIME = 2^(-EXP_VAL)  (https://www.kurokesu.com/main/2020/05/22/uvc-camera-exposure-timing-in-opencv/)
            # CAP_PROP_EXPOSURE  Actual exposure time
            #     0                    1s
            #    -1                    500ms
            #    -2                    250ms
            #    -3                    125ms
            #    -4                    62.5ms
            #    -5                    31.3ms
            #    -6                    15.6ms
            #    -7                     7.8ms
            #    -8                     3.9ms
            #    ...
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        else:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 1.0 / (1 << 7))

        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        time.sleep(1)

        if platform.system() == "Windows" and self.settings:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 0)  # launches DirectShow menu for ELP camera

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
        # self.logger.info(f"   backend = {self.cap.getBackendName()}")
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
        # Default value of max_workers is changed to min(32, (os.process_cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # Create an observer for processing photo cell signals
        self.signal_observer = SignalObserver(self.final_write_to_disk)

        # Subscribe to signal
        self.photoCellSignalDisposable = self.signal_subject.pipe(
            # Capture immediately upon signal
            ops.map(lambda event: self.handle_trigger(event))
        ).subscribe(self.signal_observer)

    def shm_to_uint8_1d(buf, Py_ssize_t total):
        """
        Convert a shared memory buffer (or any buffer) to a 1D uint8 image:
        """

        # interpret buffer as a 1D uint8 numpy array
        cdef np.ndarray[np.uint8_t, ndim=1] arr1d
        arr1d = np.frombuffer(buf, dtype=np.uint8, count=total)

        cdef unsigned char[:] mv = arr1d
        return mv

    def initialize_process_pool(self) -> None:
        """
        Create a pool of worker processes for parallel processing.
        """
        self.img_bytes: cython.int = self.img_width * self.img_height * 3  # Calculate bytes in a single frame

        # Calculate the optimal number of processes
        self.process_count = max(2, multiprocessing.cpu_count() - 1)
        # Create a pool of worker processes
        self.pool = multiprocessing.Pool(self.process_count)

        try:
            if multiprocessing.get_start_method(allow_none=True) != "forkserver":
                set_start_method("forkserver")
        except RuntimeError:
            # already set; ignore
            pass

        # Pre-allocate shared memory buffers
        self.shared_buffers = [
            shared_memory.SharedMemory(create=True, size=(self.chunk_size * self.img_bytes)) for _ in range(self.process_count)
        ]

        self._shm_views = []
        for shm in self.shared_buffers:
            self._shm_views.append(self.shm_to_uint8_1d(shm.buf, self.chunk_size * self.img_bytes))

        # bookkeeping for round-robin buffers
        self.shared_buffers_index: cython.int = 0
        self.buffers_in_use: List[bool] = [False] * self.process_count

        self.buffer_lock = threading.Lock()

        # Initialize list of processes
        self.processes: [ProcessType] = []

    def _release_views(self) -> None:
        self._shm_views = []

    def _cleanup_finished_processes(self) -> None:
        """Poll finished child processes, close their SharedMemory and mark buffer free."""
        still = []
        for res, shm, buf_idx in self.processes:
            if res.ready():
                try:
                    res.get()  # propagate exceptions from child
                except Exception as e:
                    self.logger.error(f"Child process failed: {e}")

                # mark buffer free
                with self.buffer_lock:
                    self.buffers_in_use[buf_idx] = False
            else:
                still.append((res, shm, buf_idx))
        self.processes = still

    def handle_trigger(self, event: SubjectDescType) -> None:
        frame_count, start_cycle = event

        # Capture frame immediately (blocking, but fast if camera is primed)
        read_time_start = time.perf_counter()
        self.take_picture(event)
        capture_duration = time.perf_counter() - read_time_start
        if self.gui and self.processed_frames % 100 == 0:
            self.logger.info(f"[Capture] Frame {frame_count} ({self.processed_frames}) took {capture_duration*1000:.2f} ms")

        # Slow path (chunk rollover, scheduling) in executor
        if self.processed_frames % self.chunk_size == 0:
            # Write frames to disk in background

            # snapshot descriptors for this chunk
            with self.img_desc_lock:
                descriptors = list(self.img_desc)
                self.img_desc = []

            with self.buffer_lock:
                self.buffers_in_use[self.shared_buffers_index] = True

            # current buffer holds the just-filled chunk
            buffer_index = self.shared_buffers_index

            # Schedule async housekeeping
            future = self.executor.submit(self._post_capture, buffer_index, descriptors)


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

        # Switch capture target
        self.shared_buffers_index = next_buf

        # Schedule write to disk (still async to process pool)
        self.write_to_disk(buffer_index, descriptors)

    def take_picture(self, descriptor: SubjectDescType) -> None:
        """
        Capture frame(s) from the camera. If bracketing is enabled, capture three exposures:
        - standard (-7, suffix 'a'), short (-8, suffix 'b'), long (-6, suffix 'c').
        Returns None.

        Assuming the digitalizing process does 5 frames per second, this is 0.2 seconds per frame.
        About 30% of this time belongs to the frame advance process. Therefore, we have about 0.14 seconds left
        for the current frame to be at rest in front of the projector lens. This time is split up into the following
        segments.

        0.14 seconds
                        ca. 0.008 seconds 1/128 seconds for the first exposure
                            0.03 seconds sleep to give the camera enough time to adjust for the change in exposure time
                        ca. 0.004 seconds 1/256 seconds for the second exposure
                            0.03 seconds sleep to give the camera enough time to adjust for the change in exposure time
                        ca. 0.016 seconds 1/64 seconds for the third exposure
                        sum 0.088 seconds, this means still some spare time of about 0.05 seconds for the program to do
                            some work, e.g. take care of the digitised frames and pass them batch-wise to the writing
                            processes.
        """
        cdef int frame_count
        cdef double signal_time

        cdef int bracket_index
        cdef int start_index
        cdef int frame_multiplier = self.frame_multiplier
        cdef int chunk_size = self.chunk_size
        cdef int img_bytes = self.img_bytes

        cdef unsigned char[:] shm_view
        cdef unsigned char* dst
        cdef np.uint8_t[:, :, :] fmv
        cdef np.uint8_t[:] blank

        frame_count, signal_time = descriptor

        cdef bint success

        # ChatGPT points out that self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) ensures that the latest frame is read.
        # Therefore, no discarding of a stale frame might be necessary any more. Clearly has to be tested !!!
        # if os.name == "nt" and self.settings:
        #
        # Gerald added delay to the triggering of the optocoupler. Due to this adjustment we do not have to take a dummy picture and throw it away!
        # _ = self.cap.read()  # discard stale frame

        for bracket_index, (exp_val, suffix) in enumerate(self.exposures):
            success, frame_data = self.cap.read()

            if self.bracketing:
                if bracket_index < 2:
                    (exp_val, _) = self.exposures[bracket_index + 1]
                    result = self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
                    if not result:
                        self.logger.error(f"Could not set exposure to {exp_val} working on frame {frame_count}{suffix}")
                    time.sleep(0.01)  # brief pause to let exposure apply

            # Calculate the index for this frame in the pre-allocated shared buffer slice
            start_index: cython.int = ((frame_count * frame_multiplier + bracket_index) % chunk_size) * img_bytes

            shm_view = self._shm_views[self.shared_buffers_index]

            if success:
                # Use memcpy to store frame data in shared memory
                dst = & shm_view[start_index]
                if not frame_data.flags.c_contiguous or frame_data.dtype != np.uint8:
                    frame_data = np.ascontiguousarray(frame_data, dtype=np.uint8)
                fmv = frame_data
                # fmv ist np.uint8_t[:, :, :] (memoryview / ndarray view)
                # dst ist unsigned char* (Pointer in shared memory)
                memcpy(dst, &fmv[0, 0, 0], self.img_bytes)

                self.logger.debug(f"Frame {frame_count} exposure {suffix} stored at index {start_index}")
            else:
                self.logger.error(f"Read error at frame {frame_count}{suffix}, exposure {exp_val}")
                blank_data = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
                blank = blank_data.ravel()
                shm_view[start_index : start_index + img_bytes] = blank

            # Create a frame description tuple and append it to the list
            frame_item: ImgDescType = (img_bytes, frame_count, suffix)

            with self.img_desc_lock:
                self.img_desc.append(frame_item)

            # Increment the processed frame count
            self.processed_frames += 1

        # reset to standard exposure after end of loop - this avoids one time.sleep(0.05)
        if self.bracketing:
            (exp_val, _) = self.exposures[0]
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)

    def write_to_disk(self, buffer_index: int, descriptors: List[ImgDescType]) -> None:
        """
        Write a full chunk of image data and associated descriptors to disk.
        This function supports exposure bracketing: multiple exposures per logical frame
        (e.g., normal, short, long) are handled transparently.

        Each descriptor includes (img_bytes, frame_count, suffix), allowing the downstream
        write_images function to name files accordingly (e.g., frame1234a.png, frame1234b.png, etc.).

        Returns:
            None
        """

        def process_error_callback(error):
            self.logger.error(f"Error in child process: {error}")

        # Asynchronous application of `write_images` function with appropriate arguments to work on chunk of frames
        try:
            shm = self.shared_buffers[buffer_index]
            shm_name = shm.name
            result = self.pool.apply_async(
                write_images,
                args=(shm_name,
                      descriptors,
                      self.img_width,
                      self.img_height,
                      self.output_path),
                error_callback=process_error_callback
            )

            # Keep track of the processes and its shared memory object
            # It's a Windows requirement to hold a reference to shared memory to prevent its premature release
            with self.buffer_lock:
                self.processes.append((result, shm, buffer_index))
        except Exception as e:
            self.logger.error(f"Error during `apply_async`: {e}")
            with self.buffer_lock:
                self.buffers_in_use[buffer_index] = False
            return
        finally:
            self._cleanup_finished_processes()

    def final_write_to_disk(self) -> None:
        """
        Write final images to disk and ensure all processes have finished.

        Returns:
            None
        """

        if self.img_desc:
            # the buffer containing the incomplete chunk is the buffer currently used for capture
            current_buf = self.shared_buffers_index
            # reserve it for writing
            with self.buffer_lock:
                self.buffers_in_use[current_buf] = True

            descriptors: List[ImgDescType] = list(self.img_desc)
            # schedule write
            self.write_to_disk(current_buf, descriptors)
            self.img_desc = []

        # wait for all pool jobs to finish; cleanup shared memory handles
        while self.processes:
            self._cleanup_finished_processes()
            time.sleep(0.05)

        # Close and join the pool to properly manage resources
        # Due to Windows pool closing and joining can not be shifted to __del__.
        self.pool.close()
        try:
            self.pool.join()
        except Exception as e:
            self.logger.error(f"Error while joining pool: {e}")

        # Calculate elapsed time and log statistics
        elapsed_time: cython.double = time.perf_counter() - self.start_time
        average_fps: cython.double = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("------- End of Film ---------")
        self.logger.info(f"Total saved images (incl. exposures): {self.processed_frames}")
        self.logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Average FPS: {average_fps:.2f}")

        wait_time = np.asarray([x["wait_time"] for x in timing])
        work_time = np.asarray([x["work"] for x in timing])
        total_work_time = np.asarray([x["total_work"] for x in timing])
        latency_time = np.asarray([x["latency"] for x in timing])



        if len(work_time) > 0:
            self.logger.info(f"Average wait time = {(np.average(wait_time)):.5f} seconds")
            self.logger.info(f"Minimum wait time = {(np.min(wait_time)):.5f}")
            self.logger.info(f"Maximum wait time = {(np.max(wait_time)):.5f}")

            self.logger.info(f"Average work time = {(np.average(work_time)):.5f} seconds")
            self.logger.info(f"Minimum work time = {(np.min(work_time)):.5f}")
            self.logger.info(f"Maximum work time = {(np.max(work_time)):.5f}")

            self.logger.info(f"Average Total work time = {(np.average(total_work_time)):.5f} seconds")
            self.logger.info(f"Minimum Total work time = {(np.min(total_work_time)):.5f}")
            self.logger.info(f"Maximum Total work time = {(np.max(total_work_time)):.5f}")

            self.logger.info(f"Average latency time = {(np.average(latency_time)):.5f} seconds")
            self.logger.info(f"Minimum latency time = {(np.min(latency_time)):.5f}")
            self.logger.info(f"Maximum latency time = {(np.max(latency_time)):.5f}")

        timing.sort(key=lambda x: x["total_work"], reverse=True)

        l = len(timing)
        for i in range(min(25, l)):
            self.logger.info(f"longest total work time for {i} {timing[i]}")

        timing.reverse()
        for i in range(min(25, l)):
            self.logger.info(f"shortest total work time for {i} {timing[i]}")

        self.cleanup()

    def release_camera(self) -> None:
        """
        Release the camera.

        Returns:
            None
        """
        self.cap.release()

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

        # Dispose of subscriptions
        for disposable_name in [
            "photoCellSignalDisposable"
        ]:
            try:
                getattr(self, disposable_name).dispose()
            except Exception as e:
                self.logger.warning(f"Failed to dispose {disposable_name}: {e}")

        self._release_views()

        for shm in getattr(self, "shared_buffers", []):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        """
        Handle interrupt signals.
        """
        name = signal.Signals(signum).name
        self.logger.warning(f"Signal {name} received, stopping...")
        if frame != None:
            self.logger.warning(f"{frame}")

        self.final_write_to_disk()
        sys.exit(1)

class SignalObserver(Observer):
    """
    Custom observer for handling photo cell signals.

    Attributes:
        final_write_to_disk: Function to write final images to disk.

    Methods:
        on_next(value): Handle the next emitted value.
        on_error(error): Handle errors during signal processing.
        on_completed(): Handle completion of signal processing.
    """

    def __init__(self, final_write_to_disk) -> None:
        """
        Initialize the SignalObserver instance.

        Args:
            final_write_to_disk: Function to write final images to shared memory.
        """
        super().__init__()
        self.final_write_to_disk = final_write_to_disk
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def on_next(self, value) -> None:
        """
        Handle the next emitted value from the signal subject.

        Args:
            value: The next emitted value.

        Returns:
            None
        """
        pass

    def on_error(self, error) -> None:
        """
        Handle errors during signal processing.

        Args:
            error: The error encountered.

        Returns:
            None
        """
        self.logger.error(f"Signal observer error: {error}")

    def on_completed(self) -> None:
        """
        Handle completion of signal processing.

        Returns:
            None
        """
        self.final_write_to_disk()
