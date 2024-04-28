import logging
import multiprocessing
import time
from argparse import Namespace
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops, Observer
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from TypeDefinitions import ImgDescType, StateType, ProcessType
from WriteImages import write_images


class DigitizeVideo:
    """
    DigitizeVideo class for processing and digitalizing video frames.

    This class provides methods to initialize the video capturing process,
    process frames using reactive programming, monitor frames, and more.

    Attributes:
        device_number (int): The device number of the camera.
        signal_subject (Subject): A reactivex subject emitting photo cell signals.
        output_path (Path): The directory for dumping digitized frames.
        monitoring (bool): Whether to display monitoring window.
        chunk_size (int): Number of frames passed to a process.
        img_width (int): Width of the video frames.
        img_height (int): Height of the video frames.
        img_bytes (int): Number of bytes in a single frame.
        img_desc (List[ImgDescType]): List of frame descriptions.
        image_data (np.array): Buffer for storing image data.
        processes (List[ProcessType]): List of processes for writing frames.
        processed_frames (int): Number of frames processed.
        start_time (int): Start time of processing in nanoseconds.
        new_tick (int): Latest time of processing in nanoseconds.
        thread_pool_scheduler (ThreadPoolScheduler): Scheduler for managing threads.
        monitorFrameSubject (Subject): Subject for emitting frames to be monitored.
        writeFrameSubject (Subject): Subject for emitting frames to be written to storage.
        signal_observer (Observer): Observer for processing photo cell signals.
    """

    def __init__(self, args: Namespace, signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        :parameter
            args: Namespace -- Command line arguments.

            signal_subject: Subject -- A reactivex subject emitting photo cell signals.
        :return None
        """

        # Initialize class attributes
        self.device_number: int = args.device  # device number of camera
        self.output_path: Path = args.output_path  # The directory for dumping digitised frames into
        self.monitoring: bool = args.monitor  # Display monitoring window
        self.chunk_size: int = args.chunk_size  # number of frames (images) passed to a process

        self.signal_subject = signal_subject  # A reactivex subject emitting photo cell signals.

        # Initialize logging, camera, threading and process pool
        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()
        self.initialize_process_pool()

        # Get video frame properties
        self.img_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.img_bytes: int = self.img_width * self.img_height * 3

        # Initialize image data buffer and frame descriptions list
        self.img_desc: [ImgDescType] = []
        self.image_data = np.zeros(self.img_bytes * self.chunk_size, dtype=np.uint8)

        # Initialize the list of processes
        self.processes: [ProcessType] = []

        # create monitoring window if necessary
        self.create_monitoring_window()

        # Initialize frame processing counters and timing
        self.processed_frames: int = 0
        self.start_time: int = time.perf_counter()
        self.new_tick: int = self.start_time

    def initialize_logging(self) -> None:
        """
        Initialize logging configuration for the application.

        :return None
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def initialize_camera(self) -> None:
        """
        Initialize the camera for video capturing based on the specified device number.

        :return None
        """
        self.cap = cv2.VideoCapture(self.device_number, cv2.CAP_ANY,
                                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        # Set camera resolution to HDMI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Log camera properties
        self.logger.info(f"Camera properties:")
        self.logger.info(f"   frame width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.logger.info(f"   frame height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f"   fps = {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.logger.info(f"   height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f"   gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f"   auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f"   exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f"   format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f"   buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

    def initialize_threads(self) -> None:
        """
        Initializes threads, subjects, and subscriptions for multithreading processing.

        :return None
        """

        # Calculate the optimal number of threads based on available CPU cores
        optimal_thread_count: int = multiprocessing.cpu_count()
        # Create a ThreadPoolScheduler to manage threads
        self.thread_pool_scheduler = ThreadPoolScheduler(optimal_thread_count)
        self.logger.info("CPU count is: %d", optimal_thread_count)

        # Create subjects for frame monitoring and writing
        self.monitorFrameSubject = Subject()  # Subject for emitting frames to be monitored
        self.writeFrameSubject = Subject()  # Subject for emitting frames to be written to storage

        # Determine the function for monitoring frames based on the monitoring flag
        monitor_frame_function = DigitizeVideo.monitor_picture if self.monitoring else lambda state: None

        # Subscription for monitoring and displaying frames
        self.monitorFrameDisposable = self.monitorFrameSubject.pipe(
            ops.map(monitor_frame_function)  # Map frames to the monitor_picture function
        ).subscribe(
            on_error=lambda e: self.logger.error(e)  # Handle errors during monitoring
        )

        # Subscription for writing frames to storage
        self.writeFrameDisposable = self.writeFrameSubject.pipe(
            ops.map(self.memory_write_picture),  # Map frames to the memory_write_picture function
        ).subscribe(
            on_error=lambda e: self.logger.error(e)  # Handle errors during writing
        )

        # Create an observer for processing photo cell signals
        self.signal_observer = self.SignalObserver(self.final_write_to_shared_memory)

        # Subscription for processing photo cell signals
        self.photoCellSignalDisposable = self.signal_subject.pipe(
            ops.map(self.take_picture),  # Get picture from camera
            ops.do_action(self.monitorFrameSubject.on_next),  # Emit frame for monitoring
            ops.observe_on(self.thread_pool_scheduler),  # Switch to thread pool for subsequent operations
            ops.do_action(lambda state: self.writeFrameSubject.on_next(state)),  # Emit frame for writing
        ).subscribe(self.signal_observer)

    def initialize_process_pool(self) -> None:
        """
        Create a pool of worker processes with the optimal number of processes.

        :return None
        """
        # Calculate the optimal number of processes
        process_count = multiprocessing.cpu_count()
        # Create a pool of worker processes
        self.pool = multiprocessing.Pool(process_count)

    class SignalObserver(Observer):
        """
        Custom observer for handling photo cell signals.

        Attributes:
            final_write_to_shared_memory (function): Function to write final images to shared memory.

        Methods:
            on_next(value): Handle the next emitted value.
            on_error(error): Handle errors during signal processing.
            on_completed(): Handle completion of signal processing.
        """

        def __init__(self, final_write_to_shared_memory) -> None:
            """
            Initialize the SignalObserver instance.

            :parameter
                final_write_to_shared_memory: function -- Function to write final images to shared memory.

            :return None
            """
            super().__init__()
            self.final_write_to_shared_memory = final_write_to_shared_memory

            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

        def on_next(self, value) -> None:
            """
            Handle the next emitted value from the signal subject.

            :parameter
                value: The next emitted value.

            :return: None
            """
            pass

        def on_error(self, error) -> None:
            """
            Handle errors during signal processing.

            :parameter
                error: The error encountered.

            :return: None
            """
            self.logger.error(f"{error}")

        def on_completed(self) -> None:
            """
            Handle completion of signal processing.

            :return None
            """
            self.final_write_to_shared_memory()

    def take_picture(self, descriptor: ImgDescType) -> StateType:
        """
        Grab and retrieve image from camera

        :parameter
             descriptor: ImgDescType -- A tuple containing the frame count and signal time
        :returns
            StateType -- A tuple containing the image data read and the frame count.
        """

        count, signal_time = descriptor

        # Read an image frame from the camera
        success, frame = self.cap.read()
        if success:
            # Process the frame data and timing
            self.hint(count, signal_time)
            return frame, count
        else:
            # Log an error if reading fails
            self.logger.error(f"Read error at frame {count}")

        return np.zeros([self.img_height, self.img_width, 3], np.uint8), count

    def hint(self, count, signal_time):
        """
        Calculate and log processing statistics such as FPS and potential late reads.

        :parameter
            count (int): The current frame count.
            signal_time (int): The signal time in nanoseconds.

        :returns: None
        """
        # Ensure count is greater than 1 before calculating
        if count <= 1:
            return

        # Calculate time intervals
        self.last_tick = self.new_tick
        self.new_tick = time.perf_counter()
        elapsed_time = (self.new_tick - self.start_time)
        time_for_read = (self.new_tick - signal_time)
        round_trip_time = (self.new_tick - self.last_tick)

        # Calculate FPS and update timing
        fps = count / elapsed_time if elapsed_time > 0.0 else 0.0
        upper_limit = (1.0 / fps) * 0.75

        # Check if time for read exceeds upper limit
        if time_for_read >= upper_limit:
            percent = (time_for_read / upper_limit) * 100.0 - 100.0
            self.logger.warning(
                f"Frame {count}: Read time {time_for_read:.4f}s exceeded upper limit "
                f"{upper_limit:.4f}s, {percent:.2f}%. "
                f"Current FPS: {fps:.2f}, Round trip time: {round_trip_time:.4f}s."
            )
            if time_for_read >= round_trip_time:
                self.logger.error(f"Round trip time exceeded by frame {count}")

    @staticmethod
    def monitor_picture(state: StateType) -> None:
        """
        Display image in monitor window with added tag (image count) in upper left corner

        :parameter
            state: StateType -- current image data and image count
        :returns
            None
        """
        frame_data, frame_count = state

        monitor_frame = frame_data.copy()  # make copy of image
        # add image count tag to upper left corner of image
        cv2.putText(img=monitor_frame, text=f"frame{frame_count}", org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', monitor_frame)  # display image in monitor window
        cv2.waitKey(1) & 0XFF  # Process key events (pause briefly between frames)

    def memory_write_picture(self, state: StateType) -> None:
        """
        Write captured image data to a buffer in memory and keep track of processed frames.

        This function is called when a new frame (image) is captured.
        It flattens the image data into a one-dimensional array and appends it to a buffer containing
        previously captured frames. It also keeps track of the number of processed frames and writes
        the accumulated chunk of frames to shared memory when the chunk size is reached.

        :param
            state: StateType -- A tuple containing the captured image data and the frame count
        :returns None
        """

        frame_data, frame_count = state

        # Calculate the index for this frame in the pre-allocated image_data array
        start_index = (frame_count % self.chunk_size) * self.img_bytes

        # Flatten the image data and insert it into the image_data array at the calculated index
        self.image_data[start_index:start_index + self.img_bytes] = frame_data.ravel()

        # Create a frame description tuple
        frame_item: ImgDescType = self.img_bytes, self.processed_frames
        self.img_desc.append(frame_item)  # Add the frame description to the list

        # Increment the processed frame count
        self.processed_frames += 1

        # Check if the chunk size has been reached
        if self.processed_frames % self.chunk_size == 0:
            self.write_to_shared_memory()  # Write the accumulated chunk to shared memory

    def write_to_shared_memory(self) -> None:
        """
        Write chunk of images to shared memory and start a separate process to emit the images to persistent storage

        :returns None
        """

        # Create a shared memory object with appropriate size to accommodate the current chunk of images
        shm = shared_memory.SharedMemory(create=True, size=(self.chunk_size * self.img_bytes))
        shm.buf[:] = self.image_data[:]  # Copy the image data to the shared memory buffer

        # Define a callback function for handling errors in the generated process
        def process_error_callback(error):
            # Log error
            self.logger.error(f"{error}")

        try:
            # Use the pool to apply the write_images function with the appropriate arguments
            result = self.pool.apply_async(write_images,
                                           args=(
                                               shm.name,
                                               self.img_desc,
                                               self.img_width,
                                               self.img_height,
                                               self.output_path),
                                           error_callback=process_error_callback
                                           )
            # Only Windows needs a reference to shared memory to not prematurely free it
            self.processes.append((result, shm))
        except Exception as e:
            self.logger.error(f"{e}")
        finally:
            # Cleanup for the next chunk:
            # - Clear frame descriptions
            self.img_desc = []
            # - remove finished processes from processes array
            self.processes = list(filter(self.filter_finished_processes, self.processes))

    def final_write_to_shared_memory(self):
        """
        Write final images to shared memory and ensure all processes have finished.

        :returns None
        """

        # If there are images left to write, write them to shared memory
        if len(self.img_desc) > 0:
            self.write_to_shared_memory()

        # Close the pool of worker processes
        # Pool closing and joining must be done here due to Windows. Shifting the two statements to __del__ does
        # not work.
        self.pool.close()
        self.pool.join()

    def filter_finished_processes(self, item: ProcessType) -> bool:
        """
        Filter and return only processes that are still running.
        Free shared memory of finished processes.

        :parameter
            item (ProcessType): A tuple containing the process and its shared memory object.

        :returns
            bool: True if the process is still running, False otherwise.
        """
        process, shm = item
        ready = process.ready()
        if ready:
            shm.unlink()
        return not ready

    def create_monitoring_window(self) -> None:
        """
        Create monitoring window which displays all digitized images.

        :returns None
        """
        if self.monitoring is True: cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    def delete_monitoring_window(self) -> None:
        """
        Destroy all windows created.

        :returns None
        """
        if self.monitoring is True: cv2.destroyAllWindows()

    def release_camera(self) -> None:
        """
        Release camera.

        :returns None
        """
        self.cap.release()

    def __del__(self) -> None:
        """
        Clean up resources and log statistics upon instance destruction.

        :returns None
        """

        if hasattr(self, 'thread_pool_scheduler'):
            self.thread_pool_scheduler.executor.shutdown()

        if hasattr(self, 'cap'):
            self.release_camera()

        # delete monitoring window
        self.delete_monitoring_window()

        elapsed_time = (time.perf_counter() - self.start_time)
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("-------End Of Film---------")
        self.logger.info("Total processed frames: %d", self.processed_frames)
        self.logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        self.logger.info("Average FPS: %.2f", average_fps)

        self.monitorFrameDisposable.dispose()
        self.writeFrameDisposable.dispose()
        self.photoCellSignalDisposable.dispose()
