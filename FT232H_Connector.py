import time

import board
import digitalio
import usb.util

# see circuit diagram in README.md

class FT232H_Connector:
    # 15-m-Cassette about 3.600 frames (±50 frames due to exposure and cut tolerance at start and end)
    # 30-m-Cassette about 7.200 frames (±50 frames due to exposure and cut tolerance at start and end)
    MAXCOUNT = 7250 # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2

    def __init__(self, optoCouplerSignalSubject):
        self.__optoCouplerSignalSubject = optoCouplerSignalSubject
        self.__count = 0

        self.__dev = usb.core.find(idVendor=0x0403, idProduct=0x6014)
        print(self.__dev)

        # turn LED on while processing a frame
        self.__led = digitalio.DigitalInOut(board.C1)
        self.__led.direction = digitalio.Direction.OUTPUT
        self.__led.value = False

        # opto-coupler OK1 triggers digitalizing of current frame
        self.__optoCouplerOK1 = digitalio.DigitalInOut(board.C2)
        self.__optoCouplerOK1.direction = digitalio.Direction.INPUT

        # opto-coupler OK2 triggers EOF (End Of Film)
        self.__eof = digitalio.DigitalInOut(board.C3)
        self.__eof.direction = digitalio.Direction.OUTPUT
        # set initial eof value
        self.__eof.value = True

        # switch to INPUT mode
        self.__eof = digitalio.DigitalInOut(board.C3)
        self.__eof.direction = digitalio.Direction.INPUT

    def signal_input(self, cap):
        while self.__eof.value and self.__count < self.MAXCOUNT:
            if self.__optoCouplerOK1.value:
                # turn on led to show processing of frame has started
                self.__led.value = True
                # ...todo: explain what is going on here
                self.__optoCouplerSignalSubject.on_next(cap)
                self.__count = self.__count + 1
                #
                # Wait for self.__optoCouplerOK1 (OK1) to change to false
                # Latency of OK1 is about one millisecond
                #
                while self.__optoCouplerOK1.value:
                    time.sleep(0.0005)

                # turn off led to show processing of frame has been delegated to another thread or has been finished
                self.__led.value = False
