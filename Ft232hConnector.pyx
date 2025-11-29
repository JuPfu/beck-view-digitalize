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
import logging
import sys
import time
import threading

import usb
from pyftdi.ftdi import Ftdi
from pyftdi.gpio import GpioMpsseController
from reactivex import Subject

from TimingResult cimport TimingResult as CTimingResult
from TimingResult import TimingResult as PyTimingResult

# module-level singleton for timing
from Ft232hConnector cimport get_timing
timing = None

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ft232hConnector")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


cdef class Ft232hConnector:
    """
    FT232H connector (Option B) — owns the FTDI device internally.
    """

    cdef double LATENCY_THRESHOLD
    cdef int INITIAL_COUNT
    cdef bint gui
    cdef CTimingResult timing_view

    cdef object _ftdi
    cdef object _gpio
    cdef object signal_subject

    cdef int _OK1_mask
    cdef int _END_OF_FILM_mask

    cdef object _stop_event
    cdef object _thread
    cdef bint running

    cdef int max_count
    cdef object _timing_lock

    def __init__(self, object signal_subject, int max_count, bint gui):
        """
        - signal_subject: reactivex Subject for on_next/on_completed
        - max_count: maximum expected frames
        - gui: log to stdout
        """
        self.LATENCY_THRESHOLD = 0.01
        self.INITIAL_COUNT = -1
        self.gui = gui
        self.signal_subject = signal_subject
        self.max_count = max_count + 100
        self._timing_lock = threading.Lock()

        # open FTDI device internally
        self._ftdi = Ftdi()
        try:
            self._ftdi.open_mpsse_from_url("ftdi:///1")
        except Exception as e:
            logger.error(f"Could not open FTDI device: {e}")
            raise

        # configure GPIO
        self._gpio = GpioMpsseController()
        MSB = 8
        self._OK1_mask = ((1 << 2) << MSB)
        self._END_OF_FILM_mask = ((1 << 3) << MSB)

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

        # timing singleton
        global timing
        if timing is None:
            timing = PyTimingResult(self.max_count)
        self.timing_view = <CTimingResult> timing

        # thread control
        self._stop_event = threading.Event()
        self._thread = None
        self.running = False

        # start poller automatically
        self.start()

    def start(self) -> None:
        """Start the polling thread."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="ft232h-poller", daemon=True)
        self._thread.start()
        self.running = True
        logger.info("[Ft232hConnector] Poller started")

    def stop(self, object timeout=None) -> None:
        """Stop polling thread safely."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.running = False
        logger.info("[Ft232hConnector] Poller stopped")

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

    def _read_pins_safe(self):
        """Robust GPIO read."""
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
        pins = self._read_pins_safe()
        last_pins = pins

        while not stop_event.is_set() and (pins & _eof) != _eof and count < self.max_count:
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

                # busy-wait for OK1 to drop
                latency_start = time.perf_counter()
                while ((pins & _ok) == _ok) and not stop_event.is_set():
                    time.sleep(0)
                    pins = self._read_pins_safe()
                latency = time.perf_counter() - latency_start

                with self._timing_lock:
                    try:
                        timing_view.append(
                            count,
                            float(time.perf_counter() - start_cycle),
                            0.0,
                            -1.0,
                            float(latency),
                            0.0,
                            float(delta)
                        )
                    except Exception:
                        pass

                if latency > LATENCY_THRESHOLD:
                    logger.warning(f"[Ft232hConnector] High latency {latency:.6f}s for frame {count}")

            last_pins = pins
            time.sleep(0)

        # ensure completion
        try:
            if not stop_event.is_set():
                subj.on_completed()
        except Exception:
            pass


cpdef object get_timing():
    """Return the TimingResult singleton."""
    global timing
    return timing
