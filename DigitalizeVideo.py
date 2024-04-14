import logging
import multiprocessing
import time
from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops
from reactivex.abc import DisposableBase
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from TypeDefinitions import StateType


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
    writeFrameDisposable: DisposableBase = None
    monitorFrameSubject = None
    monitorFrameDisposable: DisposableBase = None
    photoCellSignalSubject: Subject = None
    photoCellSignalDisposable: DisposableBase = None

    start_time: float = 0.0

    cap = None

    def __init__(self, device_number: int, output_path: Path, photo_cell_signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        :parameter
            device_number: int -- The device number of the camera.

            output_path: Path -- The directory for dumping digitised frames into.

            photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
        :return: None
        """

        self.device_number: int = device_number
        self.output_path: Path = output_path
        self.photoCellSignalSubject = photo_cell_signal_subject

        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()

        self.img_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        self.__state: StateType = {"img": np.array([], np.uint8), "img_count": 0}

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

        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        self.logger.info(f"Camera properties:")
        self.logger.info(f" - frame width = {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.logger.info(f" - frame height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f" - fps = {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.logger.info(f" - height = {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.logger.info(f" - gain = {self.cap.get(cv2.CAP_PROP_GAIN)}")
        self.logger.info(f" - auto exposure = {self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        self.logger.info(f" - exposure = {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        self.logger.info(f" - format = {self.cap.get(cv2.CAP_PROP_FORMAT)}")
        self.logger.info(f" - buffersize = {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

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
            ops.map(lambda x: self.write_picture(x))  # Map frames to the write_picture function
        ).subscribe(
            on_error=lambda e: self.logger.error(e)  # Handle errors during writing
        )

        # Subscription for processing photo cell signals
        self.photoCellSignalDisposable = self.photoCellSignalSubject.pipe(
            ops.map(self.take_picture),  # Get picture from camera
            ops.do_action(lambda state: self.monitorFrameSubject.on_next(state)),  # Emit frame for monitoring
            ops.observe_on(self.thread_pool_scheduler),  # Switch to thread pool for subsequent operations
            ops.do_action(lambda state: self.writeFrameSubject.on_next(state)),  # Emit frame for writing
        ).subscribe(
            on_completed=lambda: self.logger.info(f"digitization completed"),  # Log end of digitization
            on_error=lambda e: self.logger.error(e)  # Handle errors during signal processing
        )

    def take_picture(self, count) -> StateType:
        ret, frame = self.cap.read()  # Combined grab and retrieve
        if ret:
            return {"img": frame, "img_count": count}
        else:
            self.logger.error(f"Error capturing frame {count}")
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
        cv2.putText(monitor_frame, f"frame{state['img_count']}", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('Monitor', monitor_frame)
        cv2.waitKey(1) & 0XFF  # wait time

    def write_picture(self, state: StateType) -> int:
        filename = self.output_path / f"frame{state['img_count']}.png"
        success = cv2.imwrite(str(filename), state["img"])
        self.processed_frames += 1 if success else 0  # Count only successful writes
        return self.processed_frames

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
        Release the camera capture device.

        :returns: None
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released.")
        else:
            self.logger.warning("Camera is not opened.")

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
