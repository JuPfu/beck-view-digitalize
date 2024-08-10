import logging
import time

import board
import digitalio
import usb
from pyftdi.ftdi import Ftdi
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

        self.signal_subject = signal_subject
        self.__max_count = max_count + 50  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2

        self.count = -1  # Initialize frame count

        # Set up the LED to indicate frame processing
        self.__led = digitalio.DigitalInOut(board.C1)
        # switch direction to output and set initial led value
        self.__led.switch_to_output(value=True)

        # Set up opto-coupler OK1 to trigger frame processing
        self.__opto_coupler_ok1 = digitalio.DigitalInOut(board.C2)
        # switch to output and set initial trigger value to false
        self.__opto_coupler_ok1.switch_to_output(value=False)
        # switch to INPUT mode
        self.__opto_coupler_ok1.switch_to_input()  # pull is set to None

        # Set up opto-coupler OK2 to trigger End Of Film (EoF)
        self.__eof = digitalio.DigitalInOut(board.C3)
        # switch to output and set initial eof value
        self.__eof.switch_to_output(value=False)
        # switch to INPUT mode
        self.__eof.switch_to_input()  # pull is set to None

        # high latency improves performance - may be due to more work getting done asynchronously on the host
        ftdi.set_latency_timer(128)
        # set maximum frequency for MPSSE clock
        ftdi.set_frequency(ftdi.frequency_max)

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

        # :returns
            None
        """

        cycle_time = 1.0 / 10.0  # 10 frames per second
        start_time = time.perf_counter()
        loop_time = start_time
        start_cycle = start_time
        end_cycle = start_time
        delta = start_time

        while not self.__eof.value and (self.count < self.__max_count):
            if self.__opto_coupler_ok1.value:
                start_cycle = time.perf_counter()
                delta = start_cycle - end_cycle
                elapsed_time = start_cycle - start_time

                self.count += 1
                fps = (self.count + 1) / elapsed_time
                cycle_time = 1.0 / fps

                self.__led.value = False  # Turn on led to show processing of frame has started

                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                work_time_start = time.perf_counter()
                self.signal_subject.on_next((self.count, start_cycle))
                work_time = time.perf_counter() - work_time_start

                while self.__opto_coupler_ok1.value:
                    pass

                self.__led.value = True  # Turn off LED

                end_cycle = time.perf_counter()

                timing.append({
                    "count": self.count,
                    "cycle": end_cycle - start_cycle,
                    "work": work_time,
                    "delta": delta,
                    "wait_time": cycle_time - (end_cycle - start_cycle) - 0.0001
                })

                if end_cycle - start_cycle - 0.0001 < cycle_time:
                    start_cycle = end_cycle + cycle_time - (end_cycle - start_cycle)
                else:
                    logging.warning(
                        f"Maximum cycle time {end_cycle - start_cycle} exceeded {cycle_time} for fps={fps} at frame {self.count}. "
                        f"Next {int(((end_cycle - start_cycle) / cycle_time) + 0.5)} frame(s) might be skipped"
                    )
                    start_cycle += cycle_time

        self.signal_subject.on_completed()
