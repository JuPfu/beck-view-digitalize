import asyncio
import logging
import time

import board
import digitalio
import usb
from reactivex import Subject


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

    def __init__(self, signal_subject: Subject, max_count: int) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
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
        self.__led.switch_to_output(value=False)

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

    async def send_signal(self, count: int, perf_counter: float) -> None:
        self.signal_subject.on_next((count, perf_counter()))

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.

        :returns
            None
        """

        while not self.__eof.value and self.count < self.__max_count:
            if self.__opto_coupler_ok1.value:
                self.count += 1

                # turn on led to show processing of frame has started
                self.__led.value = True
                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                asyncio.run(self.send_signal(self.count, time.perf_counter()))

                # Wait for self.__opto_coupler_ok1 (ok1) to change to false
                # Latency of ok1 is about one millisecond
                while self.__opto_coupler_ok1.value:
                    time.sleep(0.0005)

                # turn off led to show processing of frame has been delegated to another thread or has been finished
                self.__led.value = False

        # Signal the completion of frame processing and EoF detection
        self.signal_subject.on_completed()
