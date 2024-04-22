import logging
import multiprocessing
import time
from argparse import Namespace
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from TypeDefinitions import ImgDescType, StateType, ProcessType
from WriteImages import write_images


class DigitizeVideo:
    """
    DigitizeVideo class for processing and digitalizing video frames.

    This class provides methods to initialize the video capturing process,
    process frames using reactive programming, monitor frames, and more.

    :parameter
        device_number: int -- The device number of the camera.

        photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
    """

    def __init__(self, args: Namespace,
                 photo_cell_signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        :parameter
            args: Namespace -- Command line arguments.

            photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
        :return: None
        """

        self.device_number: int = args.device  # device number of camera
        self.output_path: Path = args.output_path  # The directory for dumping digitised frames into
        self.monitoring: bool = args.monitor  # Display monitoring window
        self.chunk_size: int = args.chunk_size  # number of frames (images) passed to a process

        self.photo_cell_signal_subject = photo_cell_signal_subject

        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()
        self.initialize_processes()

        self.img_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.img_nbytes: int = self.img_width * self.img_height * 3

        self.img_desc: [ImgDescType] = []

        self.image_data = np.zeros(self.img_nbytes * self.chunk_size, dtype=np.uint8)

        self.processes: [ProcessType] = []

        # create monitoring window
        self.create_monitoring_window()

        self.processed_frames: int = 0
        self.start_time: int = time.time_ns()
        self.new_tick: int = self.start_time

    def initialize_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def initialize_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.device_number, cv2.CAP_ANY,
                                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

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
        """

        # Calculate the optimal number of threads based on available CPU cores
        optimal_thread_count: int = multiprocessing.cpu_count()
        # Create a ThreadPoolScheduler to manage threads
        self.thread_pool_scheduler = ThreadPoolScheduler(optimal_thread_count)
        self.logger.info("CPU count is: %d", optimal_thread_count)

        # Create subjects for frame monitoring and writing
        self.monitorFrameSubject = Subject()  # Subject for emitting frames to be monitored
        self.writeFrameSubject = Subject()  # Subject for emitting frames to be written to storage

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

        # Subscription for processing photo cell signals
        self.photoCellSignalDisposable = self.photo_cell_signal_subject.pipe(
            ops.map(self.take_picture),  # Get picture from camera
            ops.do_action(lambda state: self.monitorFrameSubject.on_next(state)),  # Emit frame for monitoring
            ops.observe_on(self.thread_pool_scheduler),  # Switch to thread pool for subsequent operations
            ops.do_action(lambda state: self.writeFrameSubject.on_next(state)),  # Emit frame for writing
        ).subscribe(
            on_completed=self.write_to_shared_memory,  # Write any remaining frames to persistent storage
            on_error=lambda e: self.logger.error(e)  # Handle errors during signal processing
        )

    def initialize_processes(self) -> None:
        # Create a pool of worker processes with the optimal number of processes
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())

    def take_picture(self, descriptor: tuple[int, int]) -> StateType:
        """
        Grab and retrieve image from camera

        :parameter
            count: int -- number to be assigned to picture being read
        :returns
            StateType -- image data read and image number assigned
        """

        count, signal_time = descriptor

        success, frame = self.cap.read()
        if success:
            self.hint(count, signal_time)
            return frame, count
        else:
            self.logger.error(f"Read error at frame {count}")

        return np.zeros([self.img_height, self.img_width, 3], np.uint8), count

    def hint(self, count, signal_time):
        # Ensure count is greater than 1 before calculating
        if count <= 1:
            return

        # Calculate time intervals
        after_read = time.time_ns()
        self.last_tick = self.new_tick
        self.new_tick = time.time_ns()
        elapsed_time = (self.new_tick - self.start_time) * 1e-9
        time_for_read = (after_read - signal_time) * 1e-9
        round_trip_time = (self.new_tick - self.last_tick) * 1e-9

        # Calculate FPS and update timing
        fps = count / elapsed_time if elapsed_time > 0 else 0
        upper_limit = (1 / fps) * 0.75

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

        monitor_frame = state[0].copy()  # make copy of image
        # add image count tag to upper left corner of image
        cv2.putText(img=monitor_frame, text=f"frame{state[1]}", org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', monitor_frame)  # display image in monitor window
        cv2.waitKey(1) & 0XFF

    def memory_write_picture(self, state: StateType) -> None:
        """
        Write captured image data to a buffer in memory and keep track of processed frames.

        This function is called when a new frame (image) is captured.
        It flattens the image data into a one-dimensional array and appends it to a buffer containing
        previously captured frames. It also keeps track of the number of processed frames and writes
        the accumulated batch of frames to shared memory when the batch size is reached.

        :param state: StateType -- A tuple containing the captured image data and the frame count
        :returns: None
        """

        frame_data, frame_count = state

        # Calculate the index for this frame in the pre-allocated image_data array
        start_index = (frame_count % self.chunk_size) * self.img_nbytes

        # Flatten the image data and insert it into the image_data array at the calculated index
        self.image_data[start_index:start_index + self.img_nbytes] = frame_data.flatten()

        # Create a frame description tuple
        frame_item: ImgDescType = self.img_nbytes, self.processed_frames
        self.img_desc.append(frame_item)  # Add the frame description to the list

        # Increment the processed frame count
        self.processed_frames += 1

        # Check if the batch size has been reached
        if self.processed_frames % self.chunk_size == 0:
            self.write_to_shared_memory()  # Write the accumulated batch to shared memory

    def write_to_shared_memory(self) -> None:
        """
        Write chunk of images to shared memory and start a separate process to emit the images to persistent storage

        :returns: None
        """

        # Create a shared memory object with appropriate size to accommodate the current batch of images
        shm = shared_memory.SharedMemory(create=True, size=(self.chunk_size * self.img_nbytes))
        shm.buf[:] = self.image_data[:]  # Copy the image data to the shared memory buffer

        # Use the pool to apply the write_images function with a callback for handling results
        def process_callback(result):
            # Handle result or cleanup after the process is done
            if result:
                self.logger.info(f"Process completed successfully: {result}")
            else:
                self.logger.error("Process failed")

        # Use the pool to apply the write_images function with the appropriate arguments
        # self.pool.imap_unordered(write_images, args=(
        self.pool.apply_async(write_images,
                              args=(
                                  shm.name,
                                  self.img_desc,
                                  self.img_width,
                                  self.img_height,
                                  self.output_path),
                              callback=process_callback
                              )

        # Cleanup for the next batch:
        # - Clear frame descriptions
        self.img_desc = []
        # - Reset image data buffer
        # self.image_data = np.zeros(self.img_nbytes * self.chunk_size, dtype=np.uint8)

    @staticmethod
    def filter_stopped_processes(item: ProcessType) -> bool:
        process, _ = item
        return process.is_alive()

    def create_monitoring_window(self) -> None:
        """
        Create monitoring window which displays all digitized images.

        :returns: None
        """
        if self.monitoring is True: cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    def delete_monitoring_window(self) -> None:
        """
        Destroy all windows created.

        :returns: None
        """
        if self.monitoring is True: cv2.destroyAllWindows()

    def release_camera(self) -> None:
        """
        Release camera.

        :returns: None
        """
        self.cap.release()

    def __del__(self) -> None:
        """
        Clean up resources and log statistics upon instance destruction.

        :returns: None
        """

        if hasattr(self, 'thread_pool_scheduler'):
            self.thread_pool_scheduler.executor.shutdown()

        # Close the pool of worker processes when done
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

        if hasattr(self, 'cap'):
            self.release_camera()

        # delete monitoring window
        self.delete_monitoring_window()

        elapsed_time = (time.time_ns() - self.start_time) * 1e-9
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("-------End Of Film---------")
        self.logger.info("Total processed frames: %d", self.processed_frames)
        self.logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        self.logger.info("Average FPS: %.2f", average_fps)

        self.monitorFrameDisposable.dispose()
        self.writeFrameDisposable.dispose()
        self.photoCellSignalDisposable.dispose()
