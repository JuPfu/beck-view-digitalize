import logging
import multiprocessing
import time
from multiprocessing import shared_memory, Process
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.subject import Subject

from TypeDefinitions import ImgDescType, StateType, ProcessDict
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

    def __init__(self, device_number: int, output_path: Path, monitoring: bool,
                 photo_cell_signal_subject: Subject) -> None:
        """
        Initialize the DigitalizeVideo instance with the given parameters and set up necessary components.

        :parameter
            device_number: int -- The device number of the camera.

            output_path: Path -- The directory for dumping digitised frames into.

            monitoring: bool -- display monitoring window

            photo_cell_signal_subject: Subject -- A reactivex subject emitting photo cell signals.
        :return: None
        """

        # batch size is the number of images worked on in a process
        self.batch_size: int = 12

        self.frame_desc: [ImgDescType] = []

        self.image_data = np.array([], dtype=np.uint8)

        self.device_number: int = device_number
        self.output_path: Path = output_path
        self.monitoring: bool = monitoring
        print(f"Digitize {self.monitoring}")
        self.photo_cell_signal_subject = photo_cell_signal_subject

        self.initialize_logging()
        self.initialize_camera()
        self.initialize_threads()

        self.img_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.img_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.img_nbytes: int = self.img_width * self.img_height * 3

        self.process_dict: ProcessDict = {}

        # create monitoring window
        self.create_monitoring_window()

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
        self.thread_pool_scheduler = ThreadPoolScheduler(2 * optimal_thread_count)
        self.logger.info("CPU count is: %d", optimal_thread_count)

        print(f"{multiprocessing.get_all_start_methods()=}")

        # Create subjects for frame monitoring and writing
        self.monitorFrameSubject = Subject()  # Subject for emitting frames to be monitored
        self.writeFrameSubject = Subject()  # Subject for emitting frames to be written to storage

        monitor_frame = DigitalizeVideo.monitor_picture if self.monitoring else DigitalizeVideo.dummy_monitor_picture

        # Subscription for monitoring and displaying frames
        self.monitorFrameDisposable = self.monitorFrameSubject.pipe(
            ops.map(lambda x: monitor_frame(x))  # Map frames to the monitor_picture function
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
            # ops.do_action(on_next=lambda state: self.monitorFrameSubject.on_next(state)),  # Emit frame for monitoring
            ops.observe_on(self.thread_pool_scheduler),  # Switch to thread pool for subsequent operations
            ops.do_action(on_next=lambda state: self.writeFrameSubject.on_next(state)),  # Emit frame for writing
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

    @staticmethod
    def dummy_monitor_picture(state: StateType) -> None:
        """
        Do nothing - do not display image in monitor window with added tag (image count)

        :parameter
            state: StateType -- current image data and image count
        :returns
            None
        """

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
        shm.buf[:] = self.image_data[:] # Copy the image data to the shared memory buffer

        try:
            # Create a new process to write images from shared memory
            proc = Process(target=write_images,
                           args=(
                               shm.name,
                               self.frame_desc,
                               self.img_width,
                               self.img_height,
                               self.output_path)
                           )
            # Windows needs a reference to the shared memory
            self.process_dict[proc.name] = {proc: Process, shm: SharedMemory}
            # Start the process
            proc.start()
        finally:
            # Cleanup for the next batch:
            # - Clear frame descriptions
            self.frame_desc = []
            # - Reset image data buffer
            self.image_data = np.array([], dtype=np.uint8)
            # - remove stopped processes from process_dict
            self.process_dict = self.remove_stopped_processes(self.process_dict)

    def remove_stopped_processes(self, process_dict) -> ProcessDict:
        return dict(filter(self.filter_stopped_processes, process_dict.items()))

    def filter_stopped_processes(self, pair) -> bool:
        key, value = pair
        process, shared_memory = value
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

        if hasattr(self, 'cap'):
            self.release_camera()

        # delete monitoring window
        self.delete_monitoring_window()

        elapsed_time = time.time() - self.start_time
        average_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("-------End Of Film---------")
        self.logger.info("Total processed frames: %d", self.processed_frames)
        self.logger.info("Total elapsed time: %.2f seconds", elapsed_time)
        self.logger.info("Average FPS: %.2f", average_fps)

        self.monitorFrameDisposable.dispose()
        self.writeFrameDisposable.dispose()
        self.photoCellSignalDisposable.dispose()
