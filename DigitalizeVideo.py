import multiprocessing
import os
import time
from threading import current_thread
from typing import TypedDict

import cv2
import reactivex as rx
from numpy import uint8
from numpy.typing import NDArray
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler


class DigitalizeVideo:
    StateType = TypedDict('StateType', {'img': NDArray[uint8], 'count': int})

    def __init__(self, device_number: int, photo_cell_signal_subject: rx.Subject) -> None:
        self.__photoCellSignalSubject = photo_cell_signal_subject

        self.__state = {"img": [], "count": 0}

        # calculate cpu count which will be used to create a ThreadPoolScheduler
        optimal_thread_count = multiprocessing.cpu_count()
        self.__thread_pool_scheduler = ThreadPoolScheduler()

        print("Cpu count is : {0}".format(optimal_thread_count))

        self.__writeFrameSubject: rx.subject.Subject = rx.subject.Subject()
        self.__writeFrameDisposable = self.__writeFrameSubject.pipe(
            ops.observe_on(self.__thread_pool_scheduler),
            ops.do_action(lambda x: self.write_picture(x)),
        ).subscribe(
            # on_next=lambda i: print(f"PROCESS writeFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        self.__monitorFrameSubject: rx.subject.Subject = rx.subject.Subject()
        self.__monitorFrameDisposable = self.__monitorFrameSubject.pipe(
            ops.do_action(lambda x: self.monitor_picture(x)),
        ).subscribe(
            # on_next=lambda i: print(
            #     f"VIEW PROCESS monitorFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        self.__takePictureSubject: rx.subject.Subject = rx.subject.Subject()
        self.__takePictureDisposable = self.__takePictureSubject.pipe(
            ops.map(self.take_picture),
            ops.do_action(lambda x: self.__writeFrameSubject.on_next(x)),
            ops.do_action(lambda x: self.__monitorFrameSubject.on_next(x)),
        ).subscribe(
            # on_next=lambda i: print(f"PROCESS takePicture: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        self.__photoCellSignalDisposable = self.__photoCellSignalSubject.pipe(
            ops.subscribe_on(self.__thread_pool_scheduler),
            ops.map(lambda count: self.__takePictureSubject.on_next(count)),
        ).subscribe(
            # on_next=lambda i: print(f"VIEW PROCESS photoCellSignal: {os.getpid()} {current_thread().name}"),
            on_error=lambda e: print(e),
        )

        self.__cap = cv2.VideoCapture(device_number, cv2.CAP_ANY,
                                      [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

        self.initialize_camera(self.__cap)

        self.start_time = time.time()

    # initialize usb camera
    def initialize_camera(self, cap) -> None:
        print(f"!!!WIDTH = {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"!!!HEIGHT = {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)

    def take_picture(self, count) -> StateType:
        grabbed = self.__cap.grab()
        if grabbed:
            ret, frame = self.__cap.retrieve()
            return {"img": frame, "count": count} if ret else {"img": [], "count": count}
        return {"img": [], "count": count}

    def monitor_picture(self, state: StateType) -> None:
        cv2.putText(img=state['img'], text=f"Gerald{state['count']}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', state['img'])
        cv2.waitKey(3) & 0XFF

    def write_picture(self, state: StateType) -> None:
        filename = f"Gerald{state['count']}.png"
        cv2.imwrite(filename, state["img"])

    def create_monitoring_window(self) -> None:
        cv2.startWindowThread()
        cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    def delete_monitoring_window(self) -> None:
        # destroy all windows created
        cv2.destroyAllWindows()

    def release_camera(self) -> None:
        self.__cap.release()

    def __del__(self) -> None:
        self.__thread_pool_scheduler.executor.shutdown(wait=True, cancel_futures=False)

        # self.__writeFrameSubject.dispose()
        # self.__monitorFrameSubject.dispose()
        # self.__writeFrameDisposable.dispose()
        # self.__monitorFrameDisposable.dispose()
        # self.__photoCellSignalDisposable.dispose()

        print("-------End Of Film---------")
        print((time.time() - self.start_time))
