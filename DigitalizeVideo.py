import logging
import multiprocessing
import time
from typing import TypedDict

import cv2
import numpy as np
from numpy._typing import NDArray
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DigitalizeVideo:
    """
    DigitalizeVideo class for processing and digitalizing video frames.

    This class provides methods to initialize the video capturing process,
    process frames using reactive programming, monitor frames, and more.

    Args:
        device_number (int): The device number of the camera.
        photo_cell_signal_subject (Subject): A subject emitting photo cell signals.
    """
    StateType = TypedDict('StateType', {'img': NDArray[np.uint8], 'count': int})

    def __init__(self, device_number: int, photo_cell_signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        Args:
            device_number (int): The device number of the camera.
            photo_cell_signal_subject (Subject): A subject emitting photo cell signals.
        """
        self.__photoCellSignalSubject = photo_cell_signal_subject

        self.__state: DigitalizeVideo.StateType = {"img": NDArray[np.uint8], "count": 0}

        # calculate cpu count which will be used to create a ThreadPoolScheduler
        optimal_thread_count = multiprocessing.cpu_count()
        self.__thread_pool_scheduler = ThreadPoolScheduler()

        print("Cpu count is : {0}".format(optimal_thread_count))

        logger.info("Cpu count is : %d", optimal_thread_count)

        self.processed_frames = 0

        # Subject for monitoring frames
        self.__monitorFrameSubject: Subject = Subject()

        # Subscription to monitor frames and handle errors
        self.__monitorFrameDisposable = self.__monitorFrameSubject.pipe(
            ops.map(lambda x: self.monitor_picture(x)),
        ).subscribe(
            # on_next=lambda i: print(
            #     f"VIEW PROCESS monitorFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        self.__writeFrameSubject: Subject = Subject()

        self.__writeFrameDisposable = self.__writeFrameSubject.pipe(
            ops.map(lambda x: self.write_picture(x)),
        ).subscribe(
            # on_next=lambda i: print(
            #     f"VIEW PROCESS monitorFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        # Subscription to process photo cell signals
        self.__photoCellSignalDisposable = self.__photoCellSignalSubject.pipe(
            # get picture from camera
            ops.map(self.take_picture),
            #  display picture in monitor window
            ops.do_action(lambda state: self.__monitorFrameSubject.on_next(state)),
            ops.observe_on(self.__thread_pool_scheduler),
            # write picture to storage
            ops.map(lambda state: self.__writeFrameSubject.on_next(state)),
        ).subscribe(
            on_completed=logger.info(f"digitization completed"),
            on_error=lambda e: logger.error(e)
        )

        # Initialize camera and start time
        self.__cap = cv2.VideoCapture(device_number, cv2.CAP_ANY,
                                      [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        self.initialize_camera(self.__cap)

        self.start_time = time.time()

    # initialize usb camera
    def initialize_camera(self, cap) -> None:
        # Set camera properties
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info(f"frame width = {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        logger.info(f"frame height = {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        logger.info(f"fps = {cap.get(cv2.CAP_PROP_FPS)}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        logger.info(f"gain = {cap.get(cv2.CAP_PROP_GAIN)}")
        logger.info(f"auto exposure = {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        logger.info(f"exposure = {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        logger.info(f"format = {cap.get(cv2.CAP_PROP_FORMAT)}")
        logger.info(f"buffersize = {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

    def take_picture(self, count) -> StateType:
        # Grab and retrieve a frame from the camera
        grabbed = self.__cap.grab()
        if grabbed:
            ret, frame = self.__cap.retrieve()
            if ret is False:
                logger.error(f"take_picture retrieve error at frame {count}")
            return {"img": frame, "count": count} if ret else {"img": NDArray[np.uint8], "count": count}
        else:
            logger.error(f"take_picture grab error at frame {count}")
        return {"img": NDArray[np.uint8], "count": count}

    def monitor_picture(self, state: StateType) -> StateType:
        # Display the frame with added text
        cv2.putText(img=state['img'], text=f"frame{state['count']}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', state['img'])
        cv2.waitKey(3) & 0XFF
        return state

    def write_picture(self, state: StateType):
        cv2.imwrite(f"frame{state['count']}.png", state["img"])
        self.processed_frames += 1

    def create_monitoring_window(self) -> None:
        cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    @staticmethod
    def delete_monitoring_window() -> None:
        # destroy all windows created
        cv2.destroyAllWindows()

    def release_camera(self) -> None:
        self.__cap.release()

    def __del__(self) -> None:
        """
        Clean up resources and log statistics upon instance destruction.
        """
        self.__thread_pool_scheduler.executor.shutdown(wait=True, cancel_futures=False)

        elapsed_time = time.time() - self.start_time
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        logger.info("-------End Of Film---------")
        logger.info("Total processed frames: %d", self.processed_frames)
        logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        logger.info("Average FPS: %.2f", average_fps)

        self.__monitorFrameDisposable.dispose()
        self.__writeFrameDisposable.dispose()
        self.__photoCellSignalDisposable.dispose()
