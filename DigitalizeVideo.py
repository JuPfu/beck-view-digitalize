import multiprocessing
import time

import cv2
import reactivex as rx
from reactivex import Subject
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler


class DigitalizeVideo:
    State = dict[[], int]
    state: State

    # subject: Subject

    def __init__(self):
        self.state = {"img": [], "count": 0}

        self.optimal_thread_count = multiprocessing.cpu_count()
        self.pool_scheduler = ThreadPoolScheduler(self.optimal_thread_count)
        # calculate cpu count, using which will create a ThreadPoolScheduler
        self.thread_count = multiprocessing.cpu_count()
        self.thread_pool_scheduler = ThreadPoolScheduler(self.thread_count)
        print("Cpu count is : {0}".format(self.thread_count))

        self.writeFrameSubject = rx.subject.Subject()
        self.disposable = self.writeFrameSubject.pipe(
            ops.subscribe_on(self.thread_pool_scheduler),
            ops.filter(lambda state: len(state["img"]) > 0),
            ops.do_action(self.write_picture),
            ops.do_action(lambda state: print(f"write pic no: {state['count']}"))
        ).subscribe(
            # on_next=lambda i: print("PROCESS 1: {0} {1}".format(current_thread().name, {len(i["img"])})),
            on_error=lambda e: print(e),
            on_completed=lambda: print("PROCESS 1 done!"),
            scheduler=rx.scheduler.NewThreadScheduler
        )

    # asynchroner gpio eingang

    def initialize_camera(self, device_number: int):
        return cv2.VideoCapture(device_number)

    def release_camera(self, camera):
        camera.release()

    def take_picture(self, camera):
        ret, frame = camera.read()
        return frame if ret else []

    def show_picture(self, state: State):
        cv2.imshow('Display', state['img'])

    def write_picture(self, state: State):
        cv2.imwrite(f"jp{state['count']}.jpg", state["img"], [cv2.IMWRITE_JPEG_QUALITY, 100])

    def do_recording(self):
        print(">>>camera initialized")
        cap = self.initialize_camera(0)
        print("<<<camera initialized")
        print(f"cv2={cv2}")

        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(f"width={width}")
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"height={height}")

        pic_time = 0
        write_time = 0

        cv2.startWindowThread()
        cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

        for i in range(100):
            tic = time.perf_counter()
            pic = self.take_picture(cap)
            tac = time.perf_counter()
            self.writeFrameSubject.on_next({"img": pic, "count": i})
            toc = time.perf_counter()
            pic_time = pic_time + (tac - tic)
            write_time = write_time + (toc - tac)
            print(f"took pic={tac - tic}  wrote pic={toc - tac}")
            if len(pic) > 0:
                cv2.putText(img=pic, text=f"JP{i}", org=(15, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                            color=(0, 255, 0), thickness=2)
            self.show_picture({"img": pic, "count": i})
            cv2.waitKey(100) & 0XFF

        print(f"total time pic={pic_time} write={write_time} total={pic_time + write_time}")

        cv2.waitKey(0)

        # Destroys all the windows created
        cv2.destroyAllWindows()

        self.writeFrameSubject.dispose()
        self.release_camera(cap)
