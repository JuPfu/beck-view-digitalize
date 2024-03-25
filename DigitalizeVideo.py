import logging
import multiprocessing
import time
from multiprocessing import shared_memory, Process
from typing import TypedDict

import cv2
import numpy as np
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject


def write_images(shared_memory_buffer_name, img_desc):
    logger.info(f">>>Write {shared_memory_buffer_name=}")
    logger.info(f">>>Write {img_desc=}")

    shm = shared_memory.SharedMemory(shared_memory_buffer_name)

    start = 0
    end = 0

    buf = np.frombuffer(shm.buf, dtype=np.uint8)
    for img in img_desc:
        start = end
        logger.info(f"===Write {start=}")
        end += img['data_bytes']
        logger.info(f"===Write {end=}")
        reshaped = np.reshape(buf[start:end], (-1, 1280, 3))

        filename = f"frame{img['count']}.png"
        success = cv2.imwrite(filename, reshaped)

        if success:
            logger.info(f"<<<Write {filename=}")
            del reshaped

    del buf
    shm.close()
    shm.unlink()

def write_image(shared_memory_buffer_name, filename):
    logger.info(f">>>Write {shared_memory_buffer_name=}")
    logger.info(f">>>Write {filename=}")
    shm = shared_memory.SharedMemory(shared_memory_buffer_name)
    reshaped = np.reshape(np.frombuffer(shm.buf, dtype=np.uint8), (-1, 1280, 3))

    success = cv2.imwrite(filename, reshaped)

    if success:
        logger.info(f"<<<Write {filename=}")
        del reshaped
        shm.close()
        shm.unlink()

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
    StateType = TypedDict('StateType', {'img': np.ndarray[np.uint8], 'count': int})

    ImgDescType = TypedDict('ImgDescType', {'data_bytes': int, 'img_count': int})

    frame_desc: [ImgDescType] = []
    db = np.reshape(np.array([], np.uint8), (-1, 1280,3))

    def __init__(self, device_number: int, photo_cell_signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        Args:
            device_number (int): The device number of the camera.
            photo_cell_signal_subject (Subject): A subject emitting photo cell signals.
        """
        self.__photoCellSignalSubject = photo_cell_signal_subject

        self.__state: DigitalizeVideo.StateType = {"img": np.array([], np.uint8), "count": 0}

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
            #      f"VIEW PROCESS monitorFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: logger.error(e)
        )

        self.__writeFrameSubject: Subject = Subject()

        self.__writeFrameDisposable = self.__writeFrameSubject.pipe(
            ops.map(lambda x: self.memory_write_picture(x)),
        ).subscribe(
            # on_next=lambda i: print(
            #      f"WRITE PROCESS frame: {os.getpid()} {current_thread().name} {i}"),
            on_error=lambda e: logger.error(e)
        )

        # Subscription to process photo cell signals
        self.__photoCellSignalDisposable = self.__photoCellSignalSubject.pipe(
            # get picture from camera
            ops.map(self.take_picture),
            #  display picture in monitor window
            ops.do_action(lambda state: self.__monitorFrameSubject.on_next(state)),
            ops.observe_on(self.__thread_pool_scheduler),
            # write picture to storage
            ops.do_action(lambda state: self.__writeFrameSubject.on_next(state)),
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
            return {"img": frame, "count": count} if ret else {"img": np.array([], np.uint8),
                                                               "count": count}
        else:
            logger.error(f"take_picture grab error at frame {count}")
        return {"img": np.array([], np.uint8), "count": count}

    def monitor_picture(self, state: StateType) -> StateType:
        # Display the frame with added text
        cv2.putText(img=state['img'], text=f"frame{state['count']}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', state['img'])
        cv2.waitKey(3) & 0XFF
        return state

    def write_picture(self, state: StateType) -> int:
        cv2.imwrite(f"frame{state['count']}.png", state["img"])
        self.processed_frames += 1
        return self.processed_frames

    def memory_write_picture(self, state: StateType) -> int:
        filename = f"frame{state['count']}.png"
        logger.info(f">>>memory_write_picture {filename=}")

        # data_bytes = state['img']
        # data_bytes_len = state['img'].size
        name = f"shm{state['count']}"

        # self.db = self.db + state['img']
        self.db = np.concatenate((self.db, state['img']), dtype='uint8')
        self.frame_desc.append({'data_bytes': state['img'].size, 'count': self.processed_frames})
        self.processed_frames += 1

        if  len(self.frame_desc) > 7 :
            logger.info(f">>>memory_write_picture {len(self.frame_desc)=}")
            sum = 0
            buf = np.reshape(np.array([], np.uint8), (-1, 1280,3))

            for e in self.frame_desc:
                # logger.info(f">>>memory_write_picture { e['data_bytes']=}")
                sum += e['data_bytes']
                # logger.info(f">>>memory_write_picture  === {sum=}")


            # for e in self.db:
            #     buf = np.concatenate((buf, e['img']), dtype='uint8')

            logger.info(f"===memory_write_picture {sum=}")
            logger.info(f"===memory_write_picture {buf.size=}")

            shm = shared_memory.SharedMemory(name=name, create=True, size=sum)
            logger.info(f"===memory_write_picture {len(self.db.tobytes())=}")
            shm.buf[:sum] = self.db.tobytes()
            logger.info(f"===memory_write_picture {shm.buf.nbytes=}")

            logger.info(f">>>memory_write_picture vor Process {shm.name=}")
            try:
                proc = Process(target=write_images, args=(shm.name, self.frame_desc))
                proc.start()
                # self.processed_frames += len(self.ImgDescType)
                # proc.join()

            # shm = shared_memory.SharedMemory(name=name, create=True, size=data_bytes_len)
            # shm.buf[:data_bytes_len] = data_bytes.tobytes()
            # try:
            #     proc = Process(target=write_image, args=(shm.name, filename))
            #     proc.start()
            #     self.processed_frames += 1
            #     # proc.join()
            finally:
                logger.info(f">>>memory_write_picture finally close {shm.name=}")
                self.db  = np.reshape(np.array([], np.uint8), (-1, 1280,3))
                self.frame_desc = []

    def create_monitoring_window(self) -> None:
        cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    @staticmethod
    def delete_monitoring_window() -> None:
        # destroy all windows created
        cv2.destroyAllWindows()

    def release_camera(self) -> None:
        self.__cap.release()

    def __del__(self) -> None:

        # complete the processes
        # for proc in self.processes:
        #     proc.join()

        """
        Clean up resources and log statistics upon instance destruction.
        """
        self.__thread_pool_scheduler.executor.shutdown()

        elapsed_time = time.time() - self.start_time
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        logger.info("-------End Of Film---------")
        logger.info("Total processed frames: %d", self.processed_frames)
        logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        logger.info("Average FPS: %.2f", average_fps)

        self.__monitorFrameDisposable.dispose()
        self.__writeFrameDisposable.dispose()
        self.__photoCellSignalDisposable.dispose()
