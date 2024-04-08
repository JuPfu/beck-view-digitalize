import copy
import logging
import multiprocessing
import time
from multiprocessing import shared_memory, Process

import cv2
import numpy as np
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from TypeDefinitions import ImgDescType, StateType
from WriteImages import write_images


class DigitalizeVideo:
    """
    DigitalizeVideo class for processing and digitalizing video frames.

    This class provides methods to initialize the video capturing process,
    process frames using reactive programming, monitor frames, and more.

    :parameter
        device_number: int -- The device number of the camera.

        photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
    """

    logger = None

    thread_pool_scheduler = None
    writeFrameSubject: Subject = None
    writeFrameDisposable = None
    monitorFrameSubject = None
    monitorFrameDisposable = None
    photoCellSignalDisposable = None

    cap = None

    def __init__(self, device_number: int, photo_cell_signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        :parameter
            device_number: int -- The device number of the camera.

            photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
        :return: None
        """

        # batch size is the number of images worked on in a process
        self.batch_size: int = 15

        self.frame_desc: [ImgDescType] = []

        self.image_data = np.array([], dtype=np.uint8)

        self.device_number: int = device_number
        self.photo_cell_signal_subject = photo_cell_signal_subject

        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()

        self.img_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.img_nbytes: int = self.img_width * self.img_height * 3

        # create monitoring window
        DigitalizeVideo.create_monitoring_window()

        self.processed_frames: int = 0
        self.start_time: float = time.time()

    def initialize_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def initialize_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.device_number, cv2.CAP_ANY,
                                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.logger.info(f"width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.logger.info(f"height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f"gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f"auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f"exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f"format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f"buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

    def initialize_threads(self) -> None:
        """
        Initializes threads, subjects, and subscriptions for multithreading processing.
        """

        # Calculate the optimal number of threads based on available CPU cores
        optimal_thread_count: int = multiprocessing.cpu_count()
        # Create a ThreadPoolScheduler to manage threads
        self.thread_pool_scheduler = ThreadPoolScheduler()
        self.logger.info("CPU count is: %d", optimal_thread_count)

        # Create subjects for frame monitoring and writing
        self.monitorFrameSubject = Subject()  # Subject for emitting frames to be monitored
        self.writeFrameSubject = Subject()  # Subject for emitting frames to be written to storage

        # Subscription for monitoring and displaying frames
        self.monitorFrameDisposable = self.monitorFrameSubject.pipe(
            ops.map(lambda x: DigitalizeVideo.monitor_picture(x))  # Map frames to the monitor_picture function
        ).subscribe(
            on_error=lambda e: self.logger.error(e)  # Handle errors during monitoring
        )

        # Subscription for writing frames to storage
        self.writeFrameDisposable = self.writeFrameSubject.pipe(
            ops.map(lambda x: self.memory_write_picture(x))  # Map frames to the memory_write_picture function
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
            on_completed=lambda: self.write_to_shared_memory(),  # Write any remaining frames to persistent storage
            on_error=lambda e: self.logger.error(e)  # Handle errors during signal processing
        )

    def take_picture(self, count) -> StateType:
        grabbed: bool = self.cap.grab()
        if grabbed:
            ret, frame = self.cap.retrieve()
            if ret:
                return {"img": frame, "img_count": count}
            else:
                self.logger.error(f"Retrieve error at frame {count}")
        else:
            self.logger.error(f"Grab error at frame {count}")

        return {"img": np.zeros([self.img_height, self.img_width, 3], np.uint8), "img_count": count}

    @staticmethod
    def monitor_picture(state: StateType) -> None:
        """
        Display image in monitor window with added tag (image count)

        :parameter
            state: StateType -- current image data and image count
        :returns
            None
        """

        monitor_frame = state['img'].copy()  # make copy of image
        # add image count tag to upper left corner of image
        cv2.putText(img=monitor_frame, text=f"frame{state['img_count']}", org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', monitor_frame)  # display image in monitor window
        cv2.waitKey(3) & 0XFF

    def memory_write_picture(self, state: StateType) -> None:
        """
        Write captured image data to a buffer in memory and keep track of processed frames.

        This function is called when a new frame (image) is captured.
        It flattens the image data into a one-dimensional array and appends it to a buffer containing
        previously captured frames. It also keeps track of the number of processed frames and writes
        the accumulated batch of frames to shared memory when the batch size is reached.

        :param state: StateType -- A dictionary containing the captured image data (`img`)
        :returns: None
        """

        # Flatten the image data from a 2D array to a 1D array
        flattened_image = state['img'].flatten()

        # Concatenate the flattened image data to the existing image data buffer
        self.image_data = np.concatenate((self.image_data, flattened_image))

        # Create a frame description dictionary containing the current processed frame count
        frame_desc = {'number_of_data_bytes': flattened_image.size, 'img_count': self.processed_frames}
        self.frame_desc.append(frame_desc)  # Add the frame description to the list

        # Increment the processed frame count
        self.processed_frames += 1

        # Check if the batch size has been reached
        if self.processed_frames % self.batch_size == 0:
            self.write_to_shared_memory()  # Write the accumulated batch to shared memory

    def write_to_shared_memory(self) -> None:
        """
        Write batch of images to shared memory and start a separate process to emit the images to persistent storage

        :returns: None
        """

        # Create a shared memory object with size to accommodate the current batch of images
        shm = shared_memory.SharedMemory(create=True, size=(len(self.frame_desc) * self.img_nbytes))
        # Copy the image data to the shared memory buffer
        shm.buf[:len(self.image_data)] = copy.copy(self.image_data)

        try:
            # Create a new process to write images from shared memory
            proc = Process(target=write_images,
                           args=(shm.name, copy.copy(self.frame_desc), self.img_width, self.img_height))
            # Start the process
            proc.start()
        finally:
            # Cleanup for the next batch:
            # - Clear frame descriptions
            self.frame_desc = []
            # - Reset image data buffer
            self.image_data = np.array([], dtype=np.uint8)

    @staticmethod
    def create_monitoring_window() -> None:
        """
        Create monitoring window which displays all digitized images.

        :returns: None
        """
        cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    @staticmethod
    def delete_monitoring_window() -> None:
        """
        Destroy all windows created.

        :returns: None
        """
        cv2.destroyAllWindows()

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

        if hasattr(self, 'cap'):
            self.release_camera()

        # delete monitoring window
        DigitalizeVideo.delete_monitoring_window()

        elapsed_time = time.time() - self.start_time
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("-------End Of Film---------")
        self.logger.info("Total processed frames: %d", self.processed_frames)
        self.logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        self.logger.info("Average FPS: %.2f", average_fps)

        self.monitorFrameDisposable.dispose()
        self.writeFrameDisposable.dispose()
        self.photoCellSignalDisposable.dispose()
