# Ft232hConnector.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
Modernised FT232H connector for beck-view-digitize with thread safety and resilience.

- Dedicated polling thread reads GPIO pins and publishes events via a reactivex Subject.
- Timing is recorded into a module-level TimingResult singleton called `timing`.
- Thread-safe access ensures no segfaults from multithreaded NumPy memoryview usage.
"""

import cython
from cython.view cimport array

# Python imports (Python objects)
import logging
import sys
import time
import threading

import usb
from pyftdi.ftdi import Ftdi, FtdiError
from pyftdi.gpio import GpioMpsseController
from reactivex import Subject

# TimingResult
from TimingResult cimport TimingResult as CTimingResult
from TimingResult import TimingResult as PyTimingResult

# Module-level singleton for timing storage
from Ft232hConnector cimport get_timing
timing = None  # will be set at runtime

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ft232hConnector")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


cdef class Ft232hConnector:
    """
    FT232H connector.

    Usage:
        conn = Ft232hConnector(ftdi, signal_subject, max_count, gui)
        conn.start()         # starts internal poller thread
        ...
        conn.stop()          # stops thread cleanly
    """

    # constants
    cdef double LATENCY_THRESHOLD
    cdef int INITIAL_COUNT

    # user-requested flag
    cdef bint gui

    # typed reference to TimingResult
    cdef CTimingResult timing_view

    # Python-level references (ftdi, gpio)
    cdef object _ftdi
    cdef object _gpio

    cdef object signal_subject

    # masks
    cdef int _OK1_mask
    cdef int _END_OF_FILM_mask

    # thread state
    cdef object _stop_event
    cdef object _thread
    cdef bint running

    # max count
    cdef int max_count

    # lock for thread-safe TimingResult access
    cdef object _timing_lock

    def __init__(self, object ftdi, object signal_subject, int max_count, bint gui):
        """
        Initialize connector.

        - ftdi: pyftdi Ftdi instance (Python object)
        - signal_subject: reactivex Subject for .on_next/.on_completed
        - max_count: expected max frames (used to size timing buffer)
        - gui: whether GUI logging to stdout is desired
        """

        self.LATENCY_THRESHOLD = 0.01
        self.INITIAL_COUNT = -1
        self.gui = gui
        self.signal_subject = signal_subject
        self.max_count = max_count + 100
        self._ftdi = ftdi
        self._gpio = GpioMpsseController()
        self._timing_lock = threading.Lock()

        # bit masks
        MSB = 8
        self._OK1_mask = ((1 << 2) << MSB)
        self._END_OF_FILM_mask = ((1 << 3) << MSB)

        # initialize USB device check
        self._init_device()

        # configure GPIO & FTDI
        try:
            self._gpio.configure(
                'ftdi:///1',
                direction=0x0,
                frequency=self._ftdi.frequency_max,
                initial=self._OK1_mask | self._END_OF_FILM_mask
            )
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask,
                                     direction=self._END_OF_FILM_mask | self._OK1_mask)
            self._gpio.write(0x0)
            try:
                self._ftdi.set_latency_timer(128)
            except Exception:
                pass
            try:
                self._ftdi.set_frequency(self._ftdi.frequency_max)
            except Exception:
                pass
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask, direction=0x0)
        except Exception as e:
            logger.error(f"[Ft232hConnector] gpio configure/setup failed: {e}")
            raise

        # initialize global TimingResult singleton
        global timing
        if timing is None:
            timing = PyTimingResult(self.max_count)
        self.timing_view = <CTimingResult> timing

        # thread control
        self._stop_event = threading.Event()
        self._thread = None
        self.running = False

    def _init_device(self):
        """Check USB device presence; exit on failure."""
        try:
            dev = usb.core.find(idVendor=0x0403, idProduct=0x6014)
            if dev is None:
                logger.error("No USB device with vendor 0x0403 / product 0x6014 found!")
                raise RuntimeError("FT232H device not found")
            logger.info(f"FT232H USB device found: {dev}")
        except Exception as e:
            logger.error(f"[Ft232hConnector] USB init failed: {e}")
            raise

    def start(self) -> None:
        """Start polling thread safely."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="ft232h-poller", daemon=True)
        self._thread.start()
        self.running = True
        logger.info("[Ft232hConnector] Poller started")

    def stop(self, object timeout=None) -> None:
        """Stop polling thread and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.running = False
        logger.info("[Ft232hConnector] Poller stopped")

    def _read_pins_safe(self):
        """
        Read GPIO pins robustly.
        Returns an int representing pin state.
        """
        try:
            rv = self._gpio.read(1)
        except Exception:
            return 0
        if isinstance(rv, (tuple, list)):
            return int(rv[0]) if len(rv) > 0 else 0
        try:
            return int(rv)
        except Exception:
            return 0

    def _poll_loop(self) -> None:
        """
        Poll GPIO pins and publish events safely.
        Thread-safe TimingResult access and robust GPIO read.
        """
        cdef int pins, last_pins, count
        _ok = self._OK1_mask
        _eof = self._END_OF_FILM_mask
        subj = self.signal_subject
        stop_event = self._stop_event
        LATENCY_THRESHOLD = self.LATENCY_THRESHOLD
        timing_view = self.timing_view

        count = self.INITIAL_COUNT
        start_cycle = time.perf_counter()
        pins = self._read_pins_safe()
        last_pins = pins

        while not stop_event.is_set() and (pins & _eof) != _eof and count < (self.max_count - 1):
            pins = self._read_pins_safe()

            # EOF detection
            if (pins & _eof) == _eof:
                try:
                    if not stop_event.is_set():
                        subj.on_next((count, start_cycle))
                except Exception:
                    pass
                break

            # rising edge detection for OK1
            if ((last_pins & _ok) == 0) and ((pins & _ok) == _ok):
                stop_cycle = time.perf_counter()
                delta = stop_cycle - start_cycle
                start_cycle = stop_cycle

                count += 1
                try:
                    if not stop_event.is_set():
                        subj.on_next((count, start_cycle))
                except Exception:
                    pass

                work_time = 0.0

                # busy-wait for OK1 to drop
                latency_start = time.perf_counter()
                while ((pins & _ok) == _ok) and not stop_event.is_set():
                    time.sleep(0)
                    pins = self._read_pins_safe()
                latency = time.perf_counter() - latency_start

                end_cycle = time.perf_counter()
                wait_time = delta - (end_cycle - start_cycle)

                # thread-safe append
                with self._timing_lock:
                    try:
                        timing_view.append(
                            count,
                            float(end_cycle - start_cycle),
                            float(work_time),
                            float(-1.0),
                            float(latency),
                            float(wait_time),
                            float(delta)
                        )
                    except Exception:
                        pass

                if latency > LATENCY_THRESHOLD:
                    logger.warning(f"[Ft232hConnector] High latency {latency:.6f}s for frame {count}")

            last_pins = pins
            time.sleep(0)

        # Ensure completion published
        try:
            if not stop_event.is_set():
                subj.on_completed()
        except Exception:
            pass

    def close(self):
        """
        Cleanly stop the polling thread and close FTDI/GPIO resources.
        Safe to call multiple times.
        """
        # Ensure poller stops before closing hardware handles
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

        logger.info("[Ft232hConnector] Closed FTDI/GPIO resources")



cpdef object get_timing():
    global timing
    return timing
