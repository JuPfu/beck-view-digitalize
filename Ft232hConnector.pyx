# Ft232hConnector.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
Modernized Option B FT232H connector for beck-view-digitize.

- Connector owns the FTDI device internally (no Ftdi instance passed).
- Polls GPIO pins on a dedicated thread and publishes events via a reactivex Subject.
- Thread-safe timing collection in TimingResult singleton.
- Provides close() method for clean shutdown.
"""

import cython
from cython.view cimport array

import signal
import logging
import sys
import time
import threading

import usb

from pyftdi.ftdi import Ftdi
from pyftdi.gpio import GpioAsyncController

from reactivex import Subject

from TimingResult cimport TimingResult as CTimingResult
from TimingResult import TimingResult as PyTimingResult

cdef object timing = None  # python singleton (holds PyTimingResult instance)
# C-level module global (must match .pxd)
cdef CTimingResult timing_view  # initially NULL


cdef object sigsub = None

def sigint_handler(signum, frame):
    print("SIGINT captured, graceful exit.")
    sigsub.on_completed()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

cdef class Ft232hConnector:
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


    def __init__(self, object ftdi, object signal_subject, int max_count, bint gui):
        """
        - ftdi
        - signal_subject: reactivex Subject for on_next/on_completed
        - max_count: maximum expected frames
        - gui: log to stdout
        """
        self.LATENCY_THRESHOLD = 0.01
        self.INITIAL_COUNT = -1
        self.gui = gui

        self._initialize_logging()

        self.signal_subject = signal_subject
        sigsub = signal_subject
        self.max_count = max_count + 100
        self._timing_lock = threading.Lock()

        # open FTDI device internally
        self._ftdi = ftdi
        try:
            self._ftdi.open_from_url("ftdi:///1")
        except Exception as e:
            self.logger.error(f"Could not open FTDI device: {e}")
            raise

        try:
            self._ftdi.set_latency_timer(16)
        except Exception:
            pass

        self._OK1_mask = 1 << 6         # ADBUS 6 (D6)
        self._END_OF_FILM_mask = 1 << 7 # ADBUS 7 (D7)

        # configure GPIO
        self._gpio = GpioAsyncController()

        try:
            self._gpio.configure(
                'ftdi:///1',
                direction=0x0,
                frequency=1000000, # 1 MHz
                initial=self._OK1_mask | self._END_OF_FILM_mask
            )
            self._gpio.set_frequency(1000000)
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask,
                                     direction=self._END_OF_FILM_mask | self._OK1_mask)
            self._gpio.write(0x0)
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask, direction=0x0)
        except Exception as e:
            self.logger.error(f"[Ft232hConnector] gpio configure/setup failed: {e}")
            raise

        global timing, timing_view

        # timing singleton
        if timing is None:
            timing = PyTimingResult(self.max_count + 10)

        # store python-level and c-level handles
        self.timing_view = <CTimingResult> timing
        timing_view = self.timing_view   # assigns the C-level global

        # thread control
        self._stop_event = threading.Event()
        self._thread = None
        self.running = False

    def _initialize_logging(self) -> None:
        """
        Configure logging for the application.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        if self.gui:
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

    def start(self) -> None:
        """Start the polling thread."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="ft232h-poller", daemon=True)
        self.running = True
        self._thread.start()
        self.logger.info("[Ft232hConnector] Ready to receive signals from projector")

    def stop(self, object timeout=None) -> None:
        """Stop polling thread safely."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        self.running = False
        self.logger.info("[Ft232hConnector] Stopped receiving signals from projector")

    def close(self) -> None:
        """Stop poller and release FTDI/GPIO safely."""
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._gpio.close()
        except Exception:
            pass
        try:
            self._ftdi.close()
        except Exception:
            pass

    def signal_input(self):
        self.start()

    def _poll_loop(self) -> None:
        """Poll GPIO and publish events to the Subject."""
        cdef int pins, last_pins, count
        _ok = self._OK1_mask
        _eof = self._END_OF_FILM_mask
        subj = self.signal_subject
        stop_event = self._stop_event
        LATENCY_THRESHOLD = self.LATENCY_THRESHOLD
        timing_view = self.timing_view

        count = self.INITIAL_COUNT
        start_cycle = time.perf_counter()
        wait_time = start_cycle
        wait_time_start = start_cycle

        last_pins = 0
        pins = self._gpio.read(1, True)

        while (pins & _eof) != _eof and count < self.max_count:
            # rising edge detection for OK1
            if ((last_pins & _ok) == 0) and ((pins & _ok) == _ok):
                stop_cycle = time.perf_counter()
                delta = stop_cycle - start_cycle
                start_cycle = stop_cycle
                wait_time = stop_cycle - wait_time_start

                count += 1

                subj.on_next((count, start_cycle))

                work_time = time.perf_counter() - start_cycle

                # busy-wait for OK1 to drop
                latency_start = time.perf_counter()
                while ((pins & _ok) == _ok):
                    time.sleep(0.0001)
                    pins = self._gpio.read(1, True)
                latency = time.perf_counter() - latency_start

                with self._timing_lock:
                    try:
                        timing_view.append(
                            count,
                            float(time.perf_counter() - start_cycle),
                            float(work_time),
                            0.0,
                            float(latency),
                            float(wait_time),
                            float(delta)
                        )
                    except Exception:
                        self.logger.warning(f"[Ft232hConnector] Could not add data to timing_view {count=}")
                        pass

                if latency > LATENCY_THRESHOLD:
                    self.logger.warning(f"[Ft232hConnector] High latency {latency:.6f}s for frame {count}")

                wait_time_start = time.perf_counter()

            last_pins = pins
            pins = self._gpio.read(1, True)
            time.sleep(0.0001)

        # --- END OF FILM ---
        if (pins & _eof) == _eof:
            self.logger.info("[Ft232hConnector] EOF encountered!")
        else:
            self.logger.info(f"[Ft232hConnector] count of frames reached or superseeded maximum count = {self.max_count}!")

        # Signal the completion of frame processing and EoF detection
        try:
            subj.on_completed()
        except Exception as e:
            self.logger.error(f"[Ft232hConnector] on_completed failed: {e}")


    def log_timing_results(self):
        """
        Log all collected timing_view entries.
        Safe for Cython memoryview layout used in TimingResult.
        """
        cdef double[:, :] buf

        with self._timing_lock:
            tv = self.timing_view
            total = tv.size

            self.logger.info(f"[Ft232hConnector] Logging timing_view with {total} entries")

            buf = tv.buf

            for i in range(total):
                try:
                    self.logger.info(
                        f"timing[{i}]: "
                        f"count={buf[i,0]:.0f}, "
                        f"cycle={buf[i,1]:.6f}, "
                        f"work={buf[i,2]:.6f}, "
                        f"read={buf[i,3]:.6f}, "
                        f"latency={buf[i,4]:.6f}, "
                        f"wait_time={buf[i,5]:.6f}, "
                        f"total_work={buf[i,6]:.6f}"
                    )
                except Exception as e:
                    self.logger.warning(f"[Ft232hConnector] Failed to read timing_view[{i}]: {e}")


cpdef CTimingResult get_timing_view():
    """
    Return the C-level TimingResult pointer (may be NULL if not initialized).
    Caller must not dereference without checking.
    """
    return timing_view

cpdef object get_timing():
    """Return the Python-level TimingResult singleton (or None)."""
    return timing




