import asyncio
import time

from reactivex import Subject


class Ft232hEmulator:
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

        self.signal_subject = signal_subject
        self.__max_count = max_count + 50  # emergency break if EoF (End of Film) is not recognized by opto-coupler OK2
        self.__max_count = 1000
        self.count = -1  # Initialize frame count

    def signal_input(self) -> None:
        """
        Process the input signals and trigger frame processing when opto-coupler OK1 is triggered.

        :returns
            None
        """

        if self.count < self.__max_count:
            self.count += 1

            # Emit the tuple of frame count and time stamp through the opto_coupler_signal_subject
            self.signal_subject.on_next((self.count, time.perf_counter()))

    async def trigger_ok1(self):
        ts = time.perf_counter()
        te = ts
        while self.count < self.__max_count:
            round  =  ts
            ts  = time.perf_counter()
            print(f">>> trigger_ok1")
            self.signal_input()
            te  = time.perf_counter()
            print(f"<<<trigger_ok1 {(te-ts)}  {self.count=}   {0.05 - (te-ts)}  {0.05 - (ts - round)}")
            await asyncio.sleep(0.05  - (te - ts))
            print(f"===>{time.perf_counter() -  round}")

        self.signal_subject.on_completed()
