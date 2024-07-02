import logging
import multiprocessing
import signal
import sys
import time
from argparse import Namespace
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops, Observer
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from SignalHandler import signal_handler
from TypeDefinitions import ImgDescType, StateType, ProcessType, SubjectDescType
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
        self.device_number: int = args.device  # device number of camera
        self.output_path: Path = args.output_path  # The directory for dumping digitised frames into
        self.monitoring: bool = args.monitor  # Display monitoring window
        self.chunk_size: int = args.chunk_size  # Quantity of frames (images) passed to a process

        self.signal_subject = signal_subject  # A reactivex subject emitting photo cell signals.

        # Signal handler is called on interrupt (ctrl-c) and terminate
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Set up logging, camera, threading and process pool
        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()
        self.initialize_process_pool()

        # Retrieve video frame properties
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.img_bytes = self.img_width * self.img_height * 3  # Calculate bytes in a single frame

        # Pre-allocate image data buffer and initialize frame descriptions list
        self.image_data = np.zeros(self.chunk_size * self.img_bytes, dtype=np.uint8)
        self.img_desc: [ImgDescType] = []  # kind of meta data

        # Initialize list of processes
        self.processes: [ProcessType] = []

        # Create monitoring window if needed
        self.create_monitoring_window()

        # Initialize counters and timing
        self.processed_frames: int = 0
        self.start_time: float = time.perf_counter()
        self.last_tick: float = self.start_time
        self.new_tick: float = self.start_time

        self.time_read: list[float] = []

    def initialize_logging(self) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)

    def initialize_camera(self) -> None:
        """
        Initialize the camera for video capturing based on the device number.
        """
        self.cap = cv2.VideoCapture(self.device_number,
                                    cv2.CAP_ANY,
                                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        # Set camera resolution to HDMI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

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
        #    ...
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        time.sleep(1)

        self.logger.info(f"Camera properties:")
        self.logger.info(f"   frame width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.logger.info(f"   frame height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f"   fps = {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.logger.info(f"   gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f"   auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f"   exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f"   format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f"   buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

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
            ops.map(monitor_frame_function)
        ).subscribe(on_error=lambda e: self.logger.error(e))

        self.writeFrameDisposable = self.writeFrameSubject.pipe(
            ops.map(self.memory_write_picture)
        ).subscribe(on_error=lambda e: self.logger.error(e))

        # Create an observer for processing photo cell signals
        self.signal_observer = SignalObserver(self.final_write_to_shared_memory)

        # Subscribe to photo cell signals and handle emitted values
        self.photoCellSignalDisposable = self.signal_subject.pipe(
            ops.map(self.take_picture),
            ops.do_action(self.monitorFrameSubject.on_next),
            ops.observe_on(self.thread_pool_scheduler),
            ops.do_action(self.writeFrameSubject.on_next)
        ).subscribe(self.signal_observer)

    def initialize_process_pool(self) -> None:
        """
        Create a pool of worker processes for parallel processing.
        """

        # Calculate the optimal number of processes
        process_count = multiprocessing.cpu_count()
        # Create a pool of worker processes
        self.pool = multiprocessing.Pool(process_count)

    def take_picture(self, descriptor: SubjectDescType) -> StateType:
        """
        Capture and retrieve an image frame from the camera.

        Args:
            descriptor: Tuple containing the frame count and signal time.

        Returns:
            Tuple containing the captured image data and frame count.
        """
        count, signal_time = descriptor

        success, frame = self.cap.read()
        if success:
            self.hint(count, signal_time)
            if not self.monitoring and count % 100 == 0:
                self.logger.info(f"Working on Frame {count} ...")
            return frame, count
        else:
            self.logger.error(f"Read error at frame {count}")
            # Return blank image in case of read error
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8), count

    def hint(self, count, signal_time) -> None:
        """
        Calculate and log processing statistics such as FPS and potential late reads.

        Args:
            count: Current frame count.
            signal_time: Time the signal was received.

        Returns:
            None
        """
        # Calculate elapsed time and intervals
        self.last_tick = self.new_tick
        self.new_tick = time.perf_counter()
        elapsed_time = self.new_tick - self.start_time
        time_for_read = self.new_tick - signal_time
        round_trip_time = self.new_tick - self.last_tick

        self.time_read.append(time_for_read)

        # Calculate FPS and timing constraints
        if count > 0:
            fps = count / elapsed_time
            upper_limit = (1.0 / fps) * 0.75

            # Check if the time taken to read a frame exceeds the upper limit
            if time_for_read >= upper_limit:
                percent = (time_for_read / upper_limit) * 100.0
                self.logger.warning(
                    f"Frame {count}: Read time {time_for_read:.4f}s exceeded upper limit {upper_limit:.4f}s, {percent - 100:.2f}% more than expected. Current FPS: {fps:.2f}, Round trip time: {round_trip_time:.4f}s.")
                if time_for_read >= round_trip_time:
                    self.logger.error(f"Round trip time exceeded by frame {count}")

    @staticmethod
    def monitor_picture(state: StateType) -> None:
        """
        Display image in monitor window with added tag (image count) in the upper left corner.

        Args:
            state: Current image data and image count.

        Returns:
            None
        """
        frame_data, frame_count = state
        # The elp camera is mounted upside down - no flipping of image required.
        # Adjust to your needs, e.g. add vertical flip
        # monitor_frame = cv2.flip(frame_data.copy(), 0)
        monitor_frame = frame_data.copy()
        # Add image count tag to the upper left corner of the image
        cv2.putText(monitor_frame, text=f"Frame {frame_count}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', monitor_frame)
        cv2.waitKey(1)

    def memory_write_picture(self, state: StateType) -> None:
        """
        Write captured image data to a buffer in memory and keep track of processed frames.

        This function is called when a new frame (image) is captured.
        It flattens the image data into a one-dimensional array and appends it to a buffer containing
        previously captured frames. It also keeps track of the number of processed frames and writes
        the accumulated chunk of frames to shared memory when the chunk size is reached.

        Args:
            state: Tuple containing the captured image data and frame count.

        Returns:
            None
        """
        frame_data, frame_count = state

        # Calculate the index for this frame in the pre-allocated image_data array
        start_index = (frame_count % self.chunk_size) * self.img_bytes

        # Use NumPy vectorized operations to flatten image data and insert it into the buffer
        self.image_data[start_index:start_index + self.img_bytes] = frame_data.ravel()

        # Create a frame description tuple and append it to the list
        frame_item: ImgDescType = (self.img_bytes, self.processed_frames)
        self.img_desc.append(frame_item)

        # Increment the processed frame count
        self.processed_frames += 1

        # Check if the chunk size has been reached
        if self.processed_frames % self.chunk_size == 0:
            self.write_to_shared_memory()

    def write_to_shared_memory(self) -> None:
        """
        Write a chunk of images to shared memory and start a separate process to emit images to persistent storage.

        Returns:
            None
        """
        # Calculate total size of shared memory for the current chunk
        shm = shared_memory.SharedMemory(create=True, size=self.chunk_size * self.img_bytes)
        shm.buf[:] = self.image_data[:]

        def process_error_callback(error):
            self.logger.error(f"Error in child process: {error}")

        # Asynchronous application of `write_images` function with appropriate arguments to work on chunk of frames
        try:
            result = self.pool.apply_async(
                write_images,
                args=(shm.name, self.img_desc, self.img_width, self.img_height, self.output_path),
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
        self.pool.close()
        self.pool.join()

        # Calculate elapsed time and log statistics
        elapsed_time = time.perf_counter() - self.start_time
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("------- End of Film ---------")
        self.logger.info(f"Total processed frames: {self.processed_frames}")
        self.logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Average FPS: {average_fps:.2f}\n")

        self.logger.info(f"Average read time = {np.average(self.time_read):.5f} seconds")
        self.logger.info(f"Variance of read time = {np.var(self.time_read):.5f}")
        self.logger.info(f"Standard deviation of read time = {np.std(self.time_read):.5f}")

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

    def __del__(self) -> None:
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
        self.monitorFrameDisposable.dispose()
        self.writeFrameDisposable.dispose()
        self.photoCellSignalDisposable.dispose()


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
