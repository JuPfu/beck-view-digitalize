# cython: language_level=3
# cython.infer_types(True)
import cython
import logging
import sys
import time

import usb
from pyftdi.ftdi import Ftdi, FtdiError
from pyftdi.gpio import GpioMpsseController
from reactivex import Subject

from Timing import timing


# see circuit diagram in README.md

class Ft232hConnector:
    """
    Ft232hConnector class for interfacing with the opto-coupler signals and controlling frame processing.

    This class provides the necessary functionality to control the digital signals from opto-couplers,
    manage frame processing, and handle the End Of Film (EoF) signal.

    Args:
        signal_subject: Subject -- A subject that emits signals triggered by opto-coupler OK1.

        max_count: int -- Emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
    """

    # 15-m-Cassette about 3.600 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 30-m-Cassette about 7.200 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 60-m-Cassette about 14.400 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 90-m-Cassette about 21.800 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 180-m-Cassette about 43.600 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 250-m-Cassette about 60.000 frames (±50 frames due to exposure and cut tolerance at start and end)

    # Constants
    CYCLE_SLEEP: cython.double = 0.001  # Sleep time in seconds
    LATENCY_THRESHOLD: cython.double = 0.01  # Suspicious latency threshold in seconds
    INITIAL_COUNT: cython.int = -1

    def __init__(self, ftdi: Ftdi, signal_subject: Subject, max_count: cython.int) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
            ftdi: Ftdi -- Ftdi device driver
            signal_subject: Subject -- A subject that emits signals triggered by opto-coupler OK1.
            max_count: int -- Emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
        """
        self._initialize_logging()
        self._initialize_device()  # Initialize USB device

        self.MSB: cython.uint = 8
        # Set up the LED to indicate frame processing
        # switch LED direction to output and set initial led value
        self.LED: cython.uint = ((1 << 1) << self.MSB)  # Pin 1 of MSB aka AC1
        # Set up opto-coupler OK1 to trigger frame processing
        # switch to output and set initial trigger value to false
        self.OK1: cython.uint = ((1 << 2) << self.MSB)  # Pin 2 of MSB aka AC2
        # Set up opto-coupler OK2 to trigger End Of Film (EoF)
        # switch to output and set initial eof value
        self.EOF: cython.uint = ((1 << 3) << self.MSB)  # Pin 3 of MSB aka AC3

        self.gpio: GpioMpsseController = GpioMpsseController()

        try:
            ftdi.validate_mpsse()
        except FtdiError as err:
            self.logger.error(f"Ftdi MPSSE error: {err}")

        self.gpio.configure('ftdi:///1',
                            direction=self.LED | self.OK1 | self.EOF,
                            frequency=ftdi.frequency_max,
                            initial=self.LED|self.OK1|self.EOF)

        # high latency improves performance - may be due to more work getting done asynchronously
        ftdi.set_latency_timer(128)

        # Set the frequency at which sequence of GPIO samples are read and written.
        ftdi.set_frequency(ftdi.frequency_max)

        # Set direction to input for OK1 and EOF and lED to output
        self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED)

        # initialize pins with current values
        self.pins: cython.uint = self.gpio.read()[0]
        print(f"<<<init {self.pins>>8=:08b}")

        self.signal_subject: Subject = signal_subject
        self.__max_count = max_count + 50  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2

        self.count: cython.int = self.INITIAL_COUNT  # Initialize frame count


    def _initialize_logging(self) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)

    def _initialize_device(self) -> None:
        """
        Initialize the USB device based on Vendor and Product IDs.

        Raises:
            ValueError: If the USB device is not found.
        """
        # Find the USB device with specified Vendor and Product IDs
        self.dev = usb.core.find(idVendor=0x0403, idProduct=0x6014)
        if self.dev is None:
            raise ValueError("USB device not found.")
        self.logger.warning(f"USB device found: {self.dev}")

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.

        :returns
            None
        """
        cycle_time: cython.double = 1.0 / 5.0  # 5 frames per second
        start_time: cython.double = time.perf_counter()
        # trigger_cycle: cython.double = start_time


        self.pins = self.gpio.read()[0]
        print(f"===while {self.pins>>8=:08b}")
        start_cycle: cython.double = start_time

        while (self.pins & self.EOF) == self.EOF and (self.count < self.__max_count):
            # if time.perf_counter() > trigger_cycle:
            #    end_wait = time.perf_counter()
            #    trigger_cycle = start_time + (self.count + 2) * cycle_time
            #    self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED | self.OK1)
            #    self.gpio.write(self.LED)
            #    self.pins = self.gpio.read()[0]
            #    print(f"time_perf {self.pins>>8=:08b}")
            #    self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED)

            if (self.pins & self.OK1) != self.OK1:
                stop_cycle: cython.double = time.perf_counter()
                delta: cython.double = stop_cycle - start_cycle
                start_cycle: cython.double = stop_cycle

                elapsed_time: cython.double = start_cycle - start_time

                self.count += 1

                fps: cython.double = 1.0 / delta # (self.count + 1) / elapsed_time
                cycle_time = 1.0 / fps
                # cycle_time = 1.0 / 10.0

                # turn on led to show processing of frame has started - reset OK1
                self.gpio.write(0x0000)

                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                work_time_start: cython.double = time.perf_counter()
                self.signal_subject.on_next((self.count, start_cycle))
                work_time: cython.double = time.perf_counter() - work_time_start

                # latency
                latency_time: cython.double  = time.perf_counter()

                # reset OK1 - might be redundant - remove after thorough testing
                # self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED | self.OK1)
                # self.gpio.write(self.OK1)
                # self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED)

                self.pins = self.gpio.read()[0]

                while (self.pins & self.OK1) != self.OK1:
                    time.sleep(self.CYCLE_SLEEP)
                    self.pins = self.gpio.read()[0]
                    print(f"Latency LOOP OK1 expected to be 0 {self.pins & self.OK1=:01b}")

                # turn off led to show processing of frame has been delegated to another thread or has been finished
                self.gpio.write(self.LED)
                latency_time: cython.double = time.perf_counter() - latency_time

                if latency_time > self.LATENCY_THRESHOLD:
                    self.logger.warning(f"Suspicious high latency {latency_time} for frame {self.count} !")

                end_cycle: cython.double = time.perf_counter()

                # wait_time: cython.double = cycle_time - ((work_time_start - start_cycle) + work_time + latency_time)
                wait_time: cython.double =  cycle_time - (end_cycle - start_cycle)
                if wait_time <= 0.0:
                    self.logger.warning(f"Negative wait time {wait_time} s for frame {self.count} !")

                timing.append({
                    "count": self.count,
                    "cycle": end_cycle - start_cycle,
                    "work": work_time,
                    "read": -1.0,
                    "latency": latency_time,
                    "wait_time": wait_time,
                    "total_work": delta, # (work_time_start - start_cycle) + work_time + latency_time
                })

                if end_cycle - start_cycle >= cycle_time:
                    logging.warning(
                        f"Frame {self.count} exceeds cycle time of {cycle_time} with {end_cycle - start_cycle} at fps={fps}."
                        f" Next {int(((end_cycle - start_cycle) / cycle_time) + 0.5)} frame(s) might be skipped"
                    )

            # Retrieve pins
            self.pins = self.gpio.read()[0]

        # Signal the completion of frame processing and EoF detection
        self.signal_subject.on_completed()
