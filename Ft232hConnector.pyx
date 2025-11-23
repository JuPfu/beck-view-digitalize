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
    CYCLE_SLEEP = 0.001  # Sleep time in seconds
    LATENCY_THRESHOLD = 0.01  # Suspicious latency threshold in seconds
    INITIAL_COUNT = -1

    def __init__(self, ftdi: Ftdi, signal_subject: Subject, max_count: cython.int, gui: cython.bint) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
            ftdi: Ftdi -- Ftdi device driver
            signal_subject: Subject -- A subject that emits signals triggered by opto-coupler OK1.
            max_count: int -- Emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
        """

        self.gui = gui

        self._initialize_logging()
        self._initialize_device()  # Initialize USB device

        cdef unsigned int MSB, OK1, EOF

        self.MSB = 8
        # Set up opto-coupler OK1 to trigger frame processing
        self.OK1 = ((1 << 2) << self.MSB)  # Pin 2 of MSB aka AC2
        # Set up opto-coupler OK2 to trigger End Of Film (EoF)
        self.EOF = ((1 << 3) << self.MSB)  # Pin 3 of MSB aka AC3

        self.gpio: GpioMpsseController = GpioMpsseController()

        try:
            ftdi.validate_mpsse()
        except FtdiError as err:
            self.logger.error(f"Ftdi MPSSE error: {err}")
            sys.exit(1)

        self.gpio.configure('ftdi:///1',
                            direction=0x0,
                            frequency=ftdi.frequency_max,
                            initial=self.OK1 | self.EOF)

        # set direction to output for EOF and OK1 and set their values to low (0)
        self.gpio.set_direction(pins=self.EOF | self.OK1, direction=self.EOF | self.OK1)
        self.gpio.write(0x0)

        # high latency improves performance - may be due to more work getting done asynchronously
        ftdi.set_latency_timer(128)

        # Set the frequency at which sequence of GPIO samples are read and written.
        ftdi.set_frequency(ftdi.frequency_max)

        # Set direction to input for OK1 and EOF
        self.gpio.set_direction(pins=self.EOF | self.OK1, direction=0x0)

        self.signal_subject: Subject = signal_subject
        self.__max_count = max_count + 50  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2

    def _initialize_logging(self) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        if self.gui and not self.logger.handlers:
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
            self.logger.error(f"No USB device with 'vendor Id = 0x0403 and product id = 0x6014' found!")
            sys.exit(3)
        self.logger.info(f"USB device found: {self.dev}")

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.

        :returns
            None
        """
        cdef int count = self.INITIAL_COUNT  # Initialize frame count

        cdef double cycle_time = 1.0 / 5.0  # 5 frames per second
        cdef double start_time = time.perf_counter()
        cdef unsigned int pins = self.gpio.read_pins()
        cdef double start_cycle = start_time
        cdef double stop_cycle = 0.0
        cdef double delta = 0.0

        cdef double work_time_start = 0.0
        cdef double work_time = 0.0

        cdef double latency_start = 0.0
        cdef double latency_time = 0.0

        cdef double end_cycle = 0.0

        cdef double wait_time = 0.0

        while (pins & self.EOF) != self.EOF and count < self.__max_count:
            if (pins & self.OK1) == self.OK1:
                stop_cycle = time.perf_counter()
                delta = stop_cycle - start_cycle
                start_cycle = stop_cycle

                count += 1

                cycle_time = delta

                # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
                work_time_start = time.perf_counter()
                self.signal_subject.on_next((count, start_cycle))
                work_time = time.perf_counter() - work_time_start

                if work_time > cycle_time:
                    self.logger.warning(f"Work time took {work_time*1000:.2f} ms")

                # latency
                latency_start = time.perf_counter()

                pins = self.gpio.read_pins()
                while (pins & self.OK1) == self.OK1:
                    time.sleep(0) # yield - USB FTDI latency is around 0.125–1ms even with 128 latency timer
                    pins = self.gpio.read_pins()

                latency_time = time.perf_counter() - latency_start
                if latency_time > self.LATENCY_THRESHOLD:
                    self.logger.warning(f"Suspicious high latency {latency_time} for frame {count} !")

                end_cycle = time.perf_counter()

                wait_time = cycle_time - (end_cycle - start_cycle)

                timing.append({
                    "count": count,
                    "cycle": end_cycle - start_cycle,
                    "work": work_time,
                    "read": -1.0,
                    "latency": latency_time,
                    "wait_time": wait_time,
                    "total_work": delta,
                })

                if wait_time <= 0.0:
                    self.logger.warning(
                        f"Negative wait time {wait_time:.5f} s for frame {count} at fps={1.0 / delta}."
                        f" Next {int(((end_cycle - start_cycle) / cycle_time) + 0.5)} frame(s) might be skipped"
                    )

            # Retrieve pins
            time.sleep(0)
            pins = self.gpio.read_pins()

        # Signal the completion of frame processing and EoF detection
        self.signal_subject.on_completed()
