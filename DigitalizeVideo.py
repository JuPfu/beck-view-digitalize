import copy
import logging
import multiprocessing
import time
from multiprocessing import shared_memory, Process
from typing import TypedDict

import cv2
import numpy as np
import numpy.typing as npt
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

StateType = TypedDict('StateType', {'img': npt.NDArray, 'img_count': int})
ImgDescType = TypedDict('ImgDescType', {'img_count': int})

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def write_images(shared_memory_buffer_name: str, img_desc: [ImgDescType], img_width: int, img_height: int) -> None:
    """
    Write batch of images to persistent storage.
    Images are delivered via shared memory

    :parameter
        shared_memory_buffer_name: str -- Reference to shared memory

        img_desc: [ImgDescType] -- Array containing the names (count) for each image of the batch

        img_width: int -- Width of an image

        img_height: int -- Height of an image
    :returns
        None
    """

    # get access to shared memory
    shm = shared_memory.SharedMemory(shared_memory_buffer_name)
    # number of images in shared buffer is deduced from length of img_desc passed as second parameter to write images
    # re-shape bytes from shared buffer into ndarray type with data type uint8
    data = np.ndarray((len(img_desc) * img_height * img_width * 3,), dtype=np.uint8, buffer=shm.buf)

    end: int = 0

    # write all images to persistent storage
    for img in img_desc:
        start = end
        end += img['number_of_data_bytes']   # Assuming 'number_of_data_bytes' is present in ImgDescType

        filename: str = f"frame{img['img_count']}.png"
        success: bool = cv2.imwrite(filename, data[start:end].reshape((img_height, img_width, 3)))

        if not success:
            logger.error(f"Could not write {filename=}")

    shm.close()
    shm.unlink()


class DigitalizeVideo:
    """
    DigitalizeVideo class for processing and digitalizing video frames.

    This class provides methods to initialize the video capturing process,
    process frames using reactive programming, monitor frames, and more.

    :parameter
        device_number: int -- The device number of the camera.

        photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
    """

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
        logger.info(f"width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        logger.info(f"height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        logger.info(f"gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        logger.info(f"auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        logger.info(f"exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        logger.info(f"format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        logger.info(f"buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

    def initialize_threads(self) -> None:
        """
        Initializes threads, subjects, and subscriptions for multithreaded processing.
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
            on_completed=lambda: self.write_to_shared_memory(),  # Write any remaining frames in shared memory
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

        monitor_frame = state['img'].copy() # make copy of image
        # add image count tag to upper left corner of image
        cv2.putText(img=monitor_frame, text=f"frame{state['img_count']}", org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', monitor_frame)  # display image in monitor window
        cv2.waitKey(3) & 0XFF

    def memory_write_picture(self, state: StateType) -> None:
        self.image_data = np.concatenate((self.image_data, state['img'].flatten()))
        self.frame_desc.append({'img_count': self.processed_frames})

        self.processed_frames += 1

        if self.processed_frames % self.batch_size == 0:
            self.write_to_shared_memory()

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

        logger.info("-------End Of Film---------")
        logger.info("Total processed frames: %d", self.processed_frames)
        logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        logger.info("Average FPS: %.2f", average_fps)

        self.monitorFrameDisposable.dispose()
        self.writeFrameDisposable.dispose()
        self.photoCellSignalDisposable.dispose()
