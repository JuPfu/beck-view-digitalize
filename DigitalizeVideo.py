import multiprocessing
import os
import time
from threading import current_thread
from typing import TypedDict

import cv2
import numpy
import reactivex as rx
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler


class DigitalizeVideo:
    StateType = TypedDict('StateType', {'img': numpy.ndarray, 'count': int})

    def __init__(self, photoCellSignalSubject) -> None:
        self.__photoCellSignalSubject = photoCellSignalSubject

        self.__count = 0
        self.__state = {"img": [], "count": 0}

        self.__optimal_thread_count = multiprocessing.cpu_count()
        self.__pool_scheduler = ThreadPoolScheduler(self.__optimal_thread_count)
        # calculate cpu count which will be used to create a ThreadPoolScheduler
        self.__thread_count = multiprocessing.cpu_count()
        self.__thread_pool_scheduler = ThreadPoolScheduler(self.__thread_count)
        print("Cpu count is : {0}".format(self.__thread_count))

        self.__photoCellSignalDisposable = self.__photoCellSignalSubject.pipe(
            ops.do_action(self.grab_image),
            ops.observe_on(self.__thread_pool_scheduler)
        ).subscribe(
            on_next=lambda i: print(f"VIEW PROCESS photoCellSignal: {os.getpid()} {current_thread().name}"),
            on_error=lambda e: print(e),
        )

        self.__writeFrameSubject: rx.subject.Subject = rx.subject.Subject()
        self.__writeFrameDisposable = self.__writeFrameSubject.pipe(
            # ops.filter(lambda state: len(state["img"]) > 0),
            ops.do_action(self.write_picture),
            ops.observe_on(self.__thread_pool_scheduler)
        ).subscribe(
            on_next=lambda i: print(f"PROCESS writeFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        self.__monitorFrameSubject: rx.subject.Subject = rx.subject.Subject()
        self.__monitorFrameDisposable = self.__monitorFrameSubject.pipe(
            # ops.filter(lambda state: len(state["img"]) > 0),
            ops.do_action(self.monitor_picture),
            ops.observe_on(self.__thread_pool_scheduler)
        ).subscribe(
            on_next=lambda i: print(
                f"VIEW PROCESS monitorFrame: {os.getpid()} {current_thread().name} {len(i['img'])}"),
            on_error=lambda e: print(e),
        )

        self.start_time = time.time()

    def initialize_camera(self, device_number: int):
        cap = cv2.VideoCapture(device_number)
        # cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return cap

    def take_picture(self, camera) -> numpy.ndarray:
        ret, frame = camera.read()
        return frame if ret else []

    def monitor_picture(self, state: StateType) -> None:
        cv2.putText(img=state['img'], text=f"Gerald{state['count']}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('Monitor', state['img'])
        cv2.waitKey(3) & 0XFF

    def write_picture(self, state: StateType) -> None:
        cv2.imwrite(f"Gerald{state['count']}.jpg", state["img"], [cv2.IMWRITE_JPEG_QUALITY, 100])

    def create_monitoring_window(self) -> None:
        cv2.startWindowThread()
        cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

    def grab_image(self, cap) -> None:
        self.__count = self.__count + 1
        self.__state = {"img": self.take_picture(cap), "count": self.__count}
        self.__writeFrameSubject.on_next(self.__state)
        self.__monitorFrameSubject.on_next(self.__state)

    def delete_monitoring_window(self) -> None:
        # destroy all windows created
        cv2.destroyAllWindows()

    def release_camera(self, camera) -> None:
        camera.release()
        
    def __del__(self) -> None:
        self.__thread_pool_scheduler.executor.shutdown(wait=True, cancel_futures=False)

        # self.__writeFrameSubject.dispose()
        # self.__monitorFrameSubject.dispose()
        # self.__writeFrameDisposable.dispose()
        # self.__monitorFrameDisposable.dispose()
        # self.__photoCellSignalDisposable.dispose()

        print("-------End Of Film---------")
        print((time.time() - self.start_time))
