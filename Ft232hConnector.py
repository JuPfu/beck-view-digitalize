import logging
import time

import usb
from pyftdi.ftdi import Ftdi
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

    def __init__(self, ftdi: Ftdi, signal_subject: Subject, max_count: int) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
            ftdi: Ftdi -- Ftdi device driver
            signal_subject: Subject -- A subject that emits signals triggered by opto-coupler OK1.
            max_count: int -- Emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
        """

        self._initialize_device()  # Initialize USB device

        self.MSB = 8
        # Set up the LED to indicate frame processing
        # switch LED direction to output and set initial led value
        self.LED = ((1 << 1) << self.MSB)  # Pin 1 of MSB aka AC1
        # Set up opto-coupler OK1 to trigger frame processing
        # switch to output and set initial trigger value to false
        self.OK1 = ((1 << 2) << self.MSB)  # Pin 2 of MSB aka AC2
        # Set up opto-coupler OK2 to trigger End Of Film (EoF)
        # switch to output and set initial eof value
        self.EOF = ((1 << 3) << self.MSB)  # Pin 3 of MSB aka AC3

        self.gpio = GpioMpsseController()

        self.gpio.configure('ftdi:///1',
                            direction=self.LED | self.OK1 | self.EOF,
                            frequency=ftdi.frequency_max,
                            initial=self.LED | self.EOF)

        # high latency improves performance - may be due to more work getting done asynchronously
        ftdi.set_latency_timer(128)

        # temporarily print the (TX, RX) tuple of hardware FIFO sizes
        # print(f"{ftdi.fifo_sizes=}")
        #
        # # temporarily print mpsse support
        # print(f"{ftdi.is_mpsse=}")
        # print(f"{ftdi.mpsse_bit_delay=}")  # Minimum delay between execution of two MPSSE SET_BITS commands in seconds
        # # temporarily print latency timer
        # print(f"{ftdi.get_latency_timer()=}")
        # # temporarily print available pins
        # print(f"{self.gpio.all_pins=:016b}")
        #
        # print(f"{ftdi.has_drivezero=}")
        ftdi.enable_drivezero_mode(self.EOF)

        # Set direction to input for OK1 and EOF and lED to output
        self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.OK1 | self.EOF)

        ftdi.set_frequency(
            ftdi.frequency_max)  # Set the frequency at which sequence of GPIO samples are read and written.

        # initialize pins with current values
        self.pins = self.gpio.read()[0]
        # temporarily print available start value of pins
        print(f"<<<{self.pins=:016b}")

        self.signal_subject = signal_subject
        self.__max_count = max_count + 50  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2

        self.count = -1  # Initialize frame count

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
        logging.info(f"USB device found: {self.dev}")

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.

        :returns
            None
        """
        cycle_time: float = 1.0 / 5.0  # 5 frames per second
        start_time: float = time.perf_counter()
        start_wait: float = start_time
        end_wait: float = start_time
        # trigger_cycle: float = start_time
        # save_time: float = start_time

        #  while (self.pins & self.EOF) and (self.count < self.__max_count):
        while (self.pins & self.EOF) != self.EOF and (self.count < self.__max_count):
            # if time.perf_counter() > trigger_cycle: #  and (self.pins & self.OK1) != self.OK1:
            #     end_wait = time.perf_counter()
            #     trigger_cycle = start_time + (self.count + 2) * cycle_time
            #     self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED | self.OK1)
            #     self.gpio.write(self.OK1)
            #     self.pins = self.gpio.read()[0]
            #     self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED)

            if (self.pins & self.OK1) == self.OK1:
                start_cycle: float = time.perf_counter()
                elapsed_time = start_cycle - start_time

                self.count += 1

                fps: float = (self.count + 1) / elapsed_time
                cycle_time = 1.0 / fps
                # cycle_time = 1.0 / 10.0

                # turn on led to show processing of frame has started - reset OK1
                self.gpio.write(0x00)

                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                work_time_start: float = time.perf_counter()
                self.signal_subject.on_next((self.count, start_cycle))
                work_time = time.perf_counter() - work_time_start

                # latency
                latency_time: float = time.perf_counter()

                # reset OK1
                self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED | self.OK1)
                self.gpio.write(0x00)
                self.pins = self.gpio.read()[0]
                self.gpio.set_direction(pins=self.EOF | self.OK1 | self.LED, direction=self.LED)
                # self.pins = self.gpio.read()[0]

                while self.pins & self.OK1:
                    time.sleep(0.001)
                    self.pins = self.gpio.read()[0]
                    print(f"Latency LOOP OK1 expected to be 0 {self.pins & self.OK1=:01b}")

                # turn off led to show processing of frame has been delegated to another thread or has been finished
                self.gpio.write(self.LED)
                latency_time = time.perf_counter() - latency_time

                if latency_time > 0.01:
                    print(f"LATENCY {latency_time=}for {self.count=} to  large!!!")

                end_cycle = time.perf_counter()

                # save_time = time.perf_counter()
                wait_time = cycle_time - ((work_time_start - start_cycle) + work_time + latency_time)

                if wait_time <= 0.0:
                    print(f"WAIT TIME {wait_time=} for {self.count=} to small!!!")

                timing.append({
                    "count": self.count,
                    "cycle": end_cycle - start_cycle,
                    "intro": work_time_start - start_cycle,
                    "work": work_time,
                    "read": -1.0,
                    "latency": latency_time,
                    "wait_time": wait_time,
                    "total_work": (work_time_start - start_cycle) + work_time + latency_time,
                    "total_calc": (work_time_start - start_cycle) + work_time + latency_time + wait_time
                })

                if end_cycle - start_cycle >= cycle_time:
                    logging.warning(
                        f"Frame {self.count} exceeds cycle time of {cycle_time} with {end_cycle - start_cycle} at fps={fps}."
                        f" Next {int(((end_cycle - start_cycle) / cycle_time) + 0.5)} frame(s) might be skipped"
                    )

                start_wait = time.perf_counter()

            # Retrieve pins
            self.pins = self.gpio.read()[0]

        # Signal the completion of frame processing and EoF detection
        self.signal_subject.on_completed()
