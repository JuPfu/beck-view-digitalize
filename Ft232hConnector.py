import asyncio
import logging
import time

import usb
from pyftdi.ftdi import Ftdi
from pyftdi.gpio import GpioMpsseController
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

    def __init__(self, ftdi: Ftdi, signal_subject: Subject, max_count: int) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
            signal_subject: Subject -- A subject that emits signals triggered by opto-coupler OK1.

            max_count: int -- Emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
        """

        self.ftdi = ftdi

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

        # Set direction to output and switch to initial value of false for the specified pins
        self.gpio.configure('ftdi:///1',
                            direction=self.LED | self.OK1 | self.EOF,
                            frequency=6000000.0,
                            initial=0x0200)

        # Set direction to input for OK1 and  OK2
        self.gpio.set_direction(pins=self.OK1 | self.EOF, direction=0x0200)

        # Set  latency to 1ms
        ftdi.set_latency_timer(6)

        # initialize pins with current values
        self.pins = self.gpio.read()[0]

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

    async def send_signal(self, count: int, perf_counter: float) -> None:
        self.signal_subject.on_next((count, perf_counter))

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.

        :returns
            None
        """

        while not (self.pins & self.EOF) and (self.count < self.__max_count):
            if (self.pins & self.OK1):
                self.count += 1

                # turn on led to show processing of frame has started
                self.gpio.write(0x0000)

                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                asyncio.run(self.send_signal(self.count, time.perf_counter()))

                # turn off led to show processing of frame has been delegated to another thread or has been finished
                self.gpio.write(self.LED)

            self.pins = self.gpio.read()[0]

        # Signal the completion of frame processing and EoF detection
        self.signal_subject.on_completed()
