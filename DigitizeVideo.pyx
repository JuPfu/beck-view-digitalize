# cython: language_level=3
# cython.infer_types(True)
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from typing import List

cimport numpy as np  # Import numpy for uint8 type
import cython
import numpy as np  # Import numpy for Python-level use

np.import_array()

import logging
import multiprocessing
import os
import signal
import sys
import time
from argparse import Namespace
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops, Observer
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from SignalHandler import signal_handler
from Timing import timing
from TypeDefinitions import ImgDescType, FrameDescType, ProcessType, SubjectDescType, RGBImageArray
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
        self.width: cython.int = args.width_height.split()[0]
        self.height: cython.int = args.width_height.split()[1]
        self.monitoring: cython.bint = args.monitor  # Display monitoring window
        self.chunk_size: cython.int = args.chunk_size  # Quantity of frames (images) passed to a process
        self.settings: cython.bint = args.settings  # Display direct show settings menu
        self.bracketing: cython.bint = args.bracketing  # Use exposure bracketing
        self.frame_multiplier: cython.int = 3 if self.bracketing else 1

        self.signal_subject: Subject = signal_subject  # A reactivex subject emitting photo cell signals.

        # Signal handler is called on interrupt (ctrl-c) and terminate
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Set up logging, camera, threading and process pool
        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()
        self.initialize_process_pool()

        # Retrieve video frame properties
        self.img_width: cython.int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: cython.int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.img_bytes: cython.int = self.img_width * self.img_height * 3  # Calculate bytes in a single frame

        # Pre-allocate image data buffer and initialize frame descriptions list
        self.image_data = np.zeros(self.chunk_size * self.img_bytes, dtype=np.uint8)
        self.img_desc: [ImgDescType] = []  # kind of meta data

        # Initialize list of processes
        self.processes: [ProcessType] = []

        # Create monitoring window if needed
        self.create_monitoring_window()

        # Initialize counters and timing
        self.processed_frames: cython.int = 0
        self.start_time: cython.double = time.perf_counter()
        self.new_tick: cython.double = self.start_time

        self.time_read: cython.list[(cython.int, cython.double, cython.p_char)] = []
        self.time_roundtrip: cython.list[(cython.int, cython.double)] = []

    def initialize_logging(self) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)

    def initialize_camera(self) -> None:
        # self.logger.info(f"Build details: {cv2.getBuildInformation()}")
        """
        Initialize the camera for video capturing based on the device number.
        """
        api: cython.int = cv2.CAP_ANY
        if os.name == "nt":
            if self.settings:
                api = cv2.CAP_DSHOW
            else:
                api = cv2.CAP_MSMF

        self.cap = cv2.VideoCapture(self.device_number,
                                    api,
                                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        time.sleep(1)  # Windows needs some time to initialize the camera

        # Set camera resolution to HDMI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # warm up camera before setting properties
        _, _ = self.cap.read()

        self.cap.set(cv2.CAP_PROP_FORMAT, -1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        # CAP_PROP_AUTO_EXPOSURE (https://github.com/opencv/opencv/issues/9738)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # automode
        time.sleep(1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual
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
        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        time.sleep(1)

        if os.name == "nt" and self.settings:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 0)  # launches DirectShow menu for ELP camera

        self.logger.info(f"Camera properties:")
        self.logger.info(f"   frame width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.logger.info(f"   frame height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f"   fps = {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.logger.info(f"   gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f"   auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f"   exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f"   format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f"   mode = {self.cap.get(cv2.CAP_PROP_MODE)}")
        self.logger.info(f"   buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
        self.logger.info(f"   backend = {self.cap.getBackendName()}")
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
        # Create thread pool scheduler with optimal number of threads
        optimal_thread_count = multiprocessing.cpu_count()
        self.thread_pool_scheduler = ThreadPoolScheduler(optimal_thread_count)
        self.logger.info(f"CPU count: {optimal_thread_count}")

        # Create reactivex subjects for frame monitoring and writing
        self.monitorFrameSubject = Subject()
        self.writeFrameSubject = Subject()

        # Determine function for monitoring frames based on the monitoring flag
        monitor_frame_function = self.monitor_picture if self.monitoring else lambda state: None

        # Subscribe to frame monitoring and writing
        self.monitorFrameDisposable = self.monitorFrameSubject.pipe(
            ops.map(monitor_frame_function),
        ).subscribe(on_error=lambda e: self.logger.error(e))

        self.writeFrameDisposable = self.writeFrameSubject.pipe(
            ops.map(self.memory_write_picture)
        ).subscribe(on_error=lambda e: self.logger.error(e))

        # Create an observer for processing photo cell signals
        self.signal_observer = SignalObserver(self.final_write_to_shared_memory)

        # Subscribe to photo cell signals and handle emitted values
        self.photoCellSignalDisposable = self.signal_subject.pipe(
            ops.map(self.take_picture),
            ops.do_action(self.writeFrameSubject.on_next),
            ops.do_action(self.monitorFrameSubject.on_next),
        ).subscribe(self.signal_observer)

    def initialize_process_pool(self) -> None:
        """
        Create a pool of worker processes for parallel processing.
        """

        # Calculate the optimal number of processes
        process_count = multiprocessing.cpu_count()
        # Create a pool of worker processes
        self.pool = multiprocessing.Pool(process_count)

    def take_picture(self, descriptor: SubjectDescType) -> List[FrameDescType]:
        """
        Capture frame(s) from the camera. If bracketing is enabled, capture three exposures:
        - standard (-7), short (-8, suffix 's'), long (-6, suffix 'l').
        Returns a list of tuples: List[(frame_image, count, suffix)].
        """
        cdef int count
        cdef double signal_time
        count, signal_time = descriptor

        frames = []

        if os.name == "nt" and self.settings:
            self.cap.retrieve()  # discard stale frame

        # Define exposure settings
        exposures = [(-7, "n"), (-8, "s"), (-6, "l")] if self.bracketing else [(-7, "n")]
        for exp_val, suffix in exposures:
            if self.bracketing:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
                time.sleep(0.05)  # brief pause to let exposure apply

            success, frame = self.cap.read()
            ts = time.perf_counter() - signal_time
            self.time_read.append((count, ts, suffix))

            if not self.monitoring and count % 100 == 0:
                self.logger.info(f"Working on Frame {count} exposure {exp_val} ({suffix})")

            if success:
                frames.append((frame, count, suffix))
            else:
                self.logger.error(f"Read error at frame {count}, exposure {exp_val}")
                blank = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
                frames.append((blank, count, suffix))

        return frames

    @staticmethod
    def monitor_picture(frames: List[FrameDescType]) -> None:
        """
        Display one frame in monitor window with added tag (image count) in the upper left corner.

        Args:
            frames: List of tuples each containing an RGB image array, an integer (frame count) and one of the suffix strings 'n', 'h' or 's'.

        Returns:
            None
        """

        # use first image of the list of frames - it should be the "standard" exposure - for monitoring
        frame_data, frame_count = frames[0]
        # The elp camera is mounted upside down - no flipping of image required.
        # Adjust to your needs, e.g. add vertical flip
        # monitor_frame = cv2.flip(frame_data.copy(), 0)
        monitor_frame: RGBImageArray = frame_data.copy()
        # Add image count tag to the upper left corner of the image
        cv2.putText(monitor_frame, text=f"Frame {frame_count}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', monitor_frame)
        cv2.waitKey(1)

    def memory_write_picture(self, frames: List[FrameDescType]) -> List[FrameDescType]:
        """
        Write captured image data to a buffer in memory and keep track of processed frames.

        This function is called when a new frame (image) is captured.
        It flattens the image data into a one-dimensional array and appends it to a buffer containing
        previously captured frames. It also keeps track of the number of processed frames and writes
        the accumulated chunk of frames to shared memory when the chunk size is reached.

        Args:
            frames: List of tuples each containing an RGB image array, an integer (frame count) and one of the suffix strings 'n', 'h' or 's'.

        Returns:
            None
        """

        for bracket_index, (frame_data, frame_count, suffix) in enumerate(frames):
            # Calculate the index for this frame in the pre-allocated image_data array
            start_index: cython.int = ((frame_count * self.frame_multiplier + bracket_index) % self.chunk_size) * self.img_bytes

            # Use NumPy vectorized operations to flatten image data and insert it into the buffer
            self.image_data[start_index:start_index + self.img_bytes] = frame_data.ravel()

            self.logger.debug(f"Frame {frame_count} exposure {suffix} stored at index {start_index}")

            # Create a frame description tuple and append it to the list
            frame_item: ImgDescType = (self.img_bytes, self.processed_frames, suffix)
            self.img_desc.append(frame_item)

            # Check if the chunk size has been reached
            if (self.processed_frames * self.frame_multiplier + bracket_index + 1) % self.chunk_size == 0:
                self.write_to_shared_memory()

        # Increment the processed frame count
        self.processed_frames += 1

        return frames

    def write_to_shared_memory(self) -> None:
        """
        Write a full chunk of image data and associated descriptors to shared memory.
        This function supports exposure bracketing: multiple exposures per logical frame
        (e.g., normal, short, long) are h   andled transparently.

        Each descriptor includes (img_bytes, frame_count, suffix), allowing the downstream
        write_images function to name files accordingly (e.g., frame1234n.png, frame1234s.png, etc.).

        Returns:
            None
        """

        import uuid

        # Create a unique name for the shared memory block
        shm_name = f"img_chunk_{uuid.uuid4().hex}"

        shm: SharedMemory = shared_memory.SharedMemory(name=shm_name, create=True,
                                                       size=self.chunk_size * self.img_bytes)

        # Use memory view to copy data into shared memory
        shm.buf[:] = self.image_data.view().reshape(-1)

        # Copy descriptors so they don’t mutate during the async operation
        descriptors = list(self.img_desc)

        def process_error_callback(error):
            self.logger.error(f"Error in child process: {error}")

        # Asynchronous application of `write_images` function with appropriate arguments to work on chunk of frames
        try:
            result = self.pool.apply_async(
                write_images,
                args=(shm.name, descriptors, self.img_width, self.img_height, self.output_path),
                error_callback=process_error_callback
            )

            # Keep track of the processes and its shared memory object
            # It's a Windows requirement to hold a reference to shared memory to prevent its premature release
            self.processes.append((result, shm))
        except Exception as e:
            self.logger.error(f"Error during `apply_async`: {e}")
        finally:
            # Clear frame descriptions and remove finished processes from list of processes
            self.img_desc = []
            self.processes = [process for process in self.processes if not process[0].ready()]

    def final_write_to_shared_memory(self) -> None:
        """
        Write final images to shared memory and ensure all processes have finished.

        Returns:
            None
        """
        # Ensure there are images left to write
        if len(self.img_desc) > 0:
            self.write_to_shared_memory()

        # Close and join the pool to properly manage resources
        # Due to Windows pool closing and joining can not be shifted to __del__.
        if len(self.img_desc) > 0:
            self.write_to_shared_memory()

        self.pool.close()
        try:
            self.pool.join()
        except Exception as e:
            self.logger.error(f"Error while joining pool: {e}")

        # Calculate elapsed time and log statistics
        elapsed_time: cython.double = time.perf_counter() - self.start_time
        average_fps: cython.double = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        total_images = self.processed_frames * self.frame_multiplier

        self.logger.info("------- End of Film ---------")
        self.logger.info(f"Total logical frames: {self.processed_frames}")
        self.logger.info(f"Total saved images (incl. exposures): {total_images}")
        self.logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Average FPS: {average_fps:.2f}")

        for i in range(len(timing)):
            timing[i]["read"] = self.time_read[i][1]

        read_time = np.asarray([[*x] for x in self.time_read])

        self.logger.info(f"Average read time = {np.average(read_time[:, 1]):.5f} seconds")
        self.logger.info(f"Variance of read time = {np.var(read_time[:, 1]):.5f}")
        self.logger.info(f"Standard deviation of read time = {np.std(read_time[:, 1]):.5f}")
        self.logger.info(f"Minimum read time = {np.min(read_time[:, 1]):.5f}")
        self.logger.info(f"Maximum read time = {np.max(read_time[:, 1]):.5f}")

        timing.sort(key=lambda x: x["cycle"], reverse=True)

        l = len(timing)
        for i in range(min(25, l)):
            self.logger.info(f"longest cycle time for {i} {timing[i]}")

        timing.reverse()
        for i in range(min(25, l)):
            self.logger.info(f"shortest cycle time for {i} {timing[i]}")

        self.cleanup()

    def create_monitoring_window(self) -> None:
        """
        Create monitoring window for displaying all digitized images.

        Returns:
            None
        """
        if self.monitoring:
            cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    def delete_monitoring_window(self) -> None:
        """
        Destroy all monitoring windows created.

        Returns:
            None
        """
        if self.monitoring:
            cv2.destroyAllWindows()

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

        if hasattr(self, 'thread_pool_scheduler'):
            self.thread_pool_scheduler.executor.shutdown()

        if hasattr(self, 'cap'):
            self.release_camera()

        # Delete monitoring window
        self.delete_monitoring_window()

        # Dispose of subscriptions
        for disposable_name in [
            "monitorFrameDisposable",
            "writeFrameDisposable",
            "photoCellSignalDisposable"
        ]:
            try:
                getattr(self, disposable_name).dispose()
            except Exception as e:
                self.logger.warning(f"Failed to dispose {disposable_name}: {e}")


class SignalObserver(Observer):
    """
    Custom observer for handling photo cell signals.

    Attributes:
        final_write_to_shared_memory: Function to write final images to shared memory.

    Methods:
        on_next(value): Handle the next emitted value.
        on_error(error): Handle errors during signal processing.
        on_completed(): Handle completion of signal processing.
    """

    def __init__(self, final_write_to_shared_memory) -> None:
        """
        Initialize the SignalObserver instance.

        Args:
            final_write_to_shared_memory: Function to write final images to shared memory.
        """
        super().__init__()
        self.final_write_to_shared_memory = final_write_to_shared_memory
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
        self.final_write_to_shared_memory()
