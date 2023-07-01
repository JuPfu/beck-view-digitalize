import time

import board
import digitalio
import usb.util


class FT232H_Connector:
    # 15-m-Kassette etwa 3.600 Bilder (±50 Bilder Belichtungs- und Schnitttoleranz an Anfang und Ende)
    # 30-m-Kassette etwa 7.200 Bilder (±50 Bilder Belichtungs- und Schnitttoleranz an Anfang und Ende).
    MAXCOUNT = 50

    def __init__(self, optoCouplerSignalSubject):
        self.__optoCouplerSignalSubject = optoCouplerSignalSubject
        self.__count = 0

        self.__dev = usb.core.find(idVendor=0x0403, idProduct=0x6014)
        print(self.__dev)

        # LED an, wenn Bild / Frame verarbeitet wird
        self.__led = digitalio.DigitalInOut(board.C1)
        self.__led.direction = digitalio.Direction.OUTPUT
        self.__led.value = False

        # Sensor triggert Digitalisierung eines Bildes / Frames
        self.__optoCoupler = digitalio.DigitalInOut(board.C2)
        self.__optoCoupler.direction = digitalio.Direction.INPUT

        # Sensor triggert EOF (End Of Film)
        self.__eof = digitalio.DigitalInOut(board.C3)
        self.__eof.direction = digitalio.Direction.INPUT
        print(f"self.__eof={self.__eof.value}")

    def signal_input(self, cap):
        while self.__eof.value and self.__count < self.MAXCOUNT:
            if self.__optoCoupler.value:
                self.__led.value = True
                self.__optoCouplerSignalSubject.on_next(cap)
                self.__count = self.__count + 1
                # Wechsel Status Opto-Koppler abwarten
                # while self.photocell.value:
                #     time.sleep(0.002)
                self.__led.value = False
