import time

import board
import digitalio
import usb.util
from reactivex.subject import Subject


# see circuit diagram in README.md

class Ft232hConnector:
    """
    Ft232hConnector class for interfacing with the opto-coupler signals and controlling frame processing.

    This class provides the necessary functionality to control the digital signals from opto-couplers,
    manage frame processing, and handle the End Of Film (EoF) signal.

    Args:
        opto_coupler_signal_subject (Subject): A subject that emits signals triggered by opto-coupler OK1.
    """

    # 15-m-Cassette about 3.600 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 30-m-Cassette about 7.200 frames (±50 frames due to exposure and cut tolerance at start and end)
    MAXCOUNT = 1500  # 7250  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2

    def __init__(self, opto_coupler_signal_subject: Subject) -> None:
        """
        Initialize the Ft232hConnector instance with the provided subjects and set up necessary components.

        Args:
            opto_coupler_signal_subject (Subject): A subject that emits signals triggered by opto-coupler OK1.
        """

        self.__optoCouplerSignalSubject = opto_coupler_signal_subject
        self.__count = 0

        # Find the USB device with specified Vendor and Product IDs
        self.__dev = usb.core.find(idVendor=0x0403, idProduct=0x6014)
        if self.__dev is None:
            raise ValueError("USB device not found.")
        else:
            print(self.__dev)

        # Set up the LED to indicate frame processing
        self.__led = digitalio.DigitalInOut(board.C1)
        self.__led.direction = digitalio.Direction.OUTPUT
        self.__led.value = False

        # Set up opto-coupler OK1 to trigger frame processing
        self.__opto_coupler_ok1 = digitalio.DigitalInOut(board.C2)
        self.__opto_coupler_ok1.direction = digitalio.Direction.INPUT

        # Set up opto-coupler OK2 to trigger End Of Film (EoF)
        self.__eof = digitalio.DigitalInOut(board.C3)
        self.__eof.direction = digitalio.Direction.OUTPUT
        # set initial eof value
        self.__eof.value = False

        # switch to INPUT mode
        self.__eof = digitalio.DigitalInOut(board.C3)
        self.__eof.direction = digitalio.Direction.INPUT

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.
        """
        while not self.__eof.value and self.__count < self.MAXCOUNT:
            if self.__opto_coupler_ok1.value:
                self.__count += 1

                # turn on led to show processing of frame has started
                self.__led.value = True
                # Emit the frame count through the opto_coupler_signal_subject
                self.__optoCouplerSignalSubject.on_next(self.__count)
                #
                # Wait for self.__opto_coupler_ok1 (ok1) to change to false
                # Latency of ok1 is about one millisecond
                #
                while self.__opto_coupler_ok1.value:
                    time.sleep(0.0005)

                # turn off led to show processing of frame has been delegated to another thread or has been finished
                self.__led.value = False

        # Signal the completion of frame processing and EoF detection
        self.__optoCouplerSignalSubject.on_completed()
