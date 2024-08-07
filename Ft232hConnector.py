import logging
import time
from struct import unpack
from typing import Tuple, Union, Iterable

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

    def __init__(self, signal_subject: Subject, max_count: int) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
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
        self.ftdi = self.gpio.ftdi

        # Set direction to output and switch to initial value of false for the specified pins
        self.direction = self.LED | self.OK1 | self.EOF
        self.gpio.configure('ftdi:///1',
                            direction=self.direction,
                            frequency=30000000,
                            initial=0x0200)

        # Set direction to input for OK1 and OK2
        self.direction = 0x0200
        self.gpio.set_direction(pins=self.OK1 | self.EOF | self.LED, direction=self.direction)

        # high latency improves performance - may be due to more work getting done asynchronously on the host
        self.ftdi.set_latency_timer(128)

        # initialize pins with current values
        self.pins = self.gpio.read()[0]

        self.signal_subject = signal_subject
        self.__max_count = max_count + 50  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
        self.__max_count = 1000
        self.count = -1  # Initialize frame count

        self.cmd = bytearray([Ftdi.GET_BITS_LOW, Ftdi.GET_BITS_HIGH, Ftdi.SEND_IMMEDIATE])

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

    def read_mpsse(self) -> Tuple[int]:
        """
        Optimized read for gpio values
        """
        self.ftdi.write_data(self.cmd)  # write command and ...
        data = self.ftdi.read_data_bytes(2, 4)  # receive data
        return unpack('<1H', data)  # format little endian ('<') one ('1') unsigned short ('H')

    def write_mpsse(self, out: Union[bytes, bytearray, Iterable[int], int]) -> None:
        """
        Optimized write for gpio values
        """
        low_dir = self.direction & 0xFF
        high_dir = (self.direction >> 8) & 0xFF
        low_data = out & 0xFF
        high_data = (out >> 8) & 0xFF
        cmd = bytearray([Ftdi.SET_BITS_LOW, low_data, low_dir, Ftdi.SET_BITS_HIGH, high_data, high_dir])
        self.ftdi.write_data(cmd)

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

        while not (self.pins & self.EOF) and (self.count < self.__max_count):
            loop_time = time.perf_counter()
            if loop_time > start_cycle or self.pins & self.OK1:
                start_cycle = time.perf_counter()
                delta = start_cycle - end_cycle
                elapsed_time = start_cycle - start_time

                self.count += 1
                fps = (self.count + 1) / elapsed_time
                cycle_time = 1.0 / fps

                self.write_mpsse(0)  # Turn on led to show processing of frame has started

                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                work_time_start = time.perf_counter()
                self.signal_subject.on_next((self.count, time.perf_counter()))
                work_time = time.perf_counter() - work_time_start

                while self.pins & self.OK1:
                    self.pins = self.read_mpsse()[0]

                self.write_mpsse(self.LED)  # Turn off LED
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

            self.pins = self.read_mpsse()[0]

        self.signal_subject.on_completed()

        if self.gpio.is_connected:
            self.gpio.close()
