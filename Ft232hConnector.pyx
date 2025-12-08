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
from pyftdi.gpio import GpioMpsseController, GpioAsyncController
from reactivex import Subject

from TimingResult cimport TimingResult as CTimingResult
from TimingResult import TimingResult as PyTimingResult

cdef object timing = None  # python singleton (holds PyTimingResult instance)
# C-level module global (must match .pxd)
cdef CTimingResult timing_view  # initially NULL

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ft232hConnector")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


cdef class Ft232hConnector:
    """
    FT232H connector — owns the FTDI device internally.
    """

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

        self.signal_subject = signal_subject
        self.max_count = 300 # max_count + 100
        self._timing_lock = threading.Lock()

        # open FTDI device internally
        self._ftdi = ftdi
        try:
            self._ftdi.open_from_url("ftdi:///1")
        except Exception as e:
            logger.error(f"Could not open FTDI device: {e}")
            raise

        try:
            self._ftdi.set_latency_timer(16)
        except Exception:
            pass

        # self._gpio.set_baudrate(6000000)
        #try:
        #   self._ftdi.set_frequency(self._ftdi.frequency_max)
        #except Exception:
        #   pass

        # configure GPIO
        self._gpio = GpioAsyncController()
        MSB = 0
        self._OK1_mask = ((1 << 6) << MSB)
        self._END_OF_FILM_mask = ((1 << 7) << MSB)

        try:
            self._gpio.configure(
                'ftdi:///1',
                direction=0x0,
                frequency=6000000, # self._ftdi.frequency_max,
                initial=self._OK1_mask | self._END_OF_FILM_mask
            )
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask,
                                     direction=self._END_OF_FILM_mask | self._OK1_mask)
            self._gpio.write(0x0)
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask, direction=0x0)
        except Exception as e:
            logger.error(f"[Ft232hConnector] gpio configure/setup failed: {e}")
            raise

        global timing, timing_view

        # timing singleton
        if timing is None:
            timing = PyTimingResult(self.max_count)

        # store python-level and c-level handles
        self.timing_view = <CTimingResult> timing
        timing_view = self.timing_view   # assigns the C-level global

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

        lc = 0.0

        while not stop_event.is_set() and (pins & _eof) != _eof and count < self.max_count:
            lc = lc + 1.0
            # rising edge detection for OK1
            if ((last_pins & _ok) == 0) and ((pins & _ok) == _ok):
                stop_cycle = time.perf_counter()
                delta = stop_cycle - start_cycle
                start_cycle = stop_cycle
                wait_time = stop_cycle - wait_time_start

                count += 1
                # temporarily commented out to test wait-time
                try:
                    if not stop_event.is_set():
                        subj.on_next((count, start_cycle))
                except Exception:
                    pass

                work_time = time.perf_counter() - start_cycle

                # busy-wait for OK1 to drop
                latency_start = time.perf_counter()
                while ((pins & _ok) == _ok):
                    time.sleep(0.0005)
                    pins = self._gpio.read(1, True)
                latency = time.perf_counter() - latency_start

                with self._timing_lock:
                    try:
                        timing_view.append(
                            count,
                            float(time.perf_counter() - start_cycle),
                            float(work_time),
                            float(lc),
                            float(latency),
                            float(wait_time),
                            float(delta)
                        )
                    except Exception:
                        logger.warning(f"[Ft232hConnector] Could not add data to timing_view {count=}")
                        pass

                lc = 0
                if latency > LATENCY_THRESHOLD:
                    logger.warning(f"[Ft232hConnector] High latency {latency:.6f}s for frame {count}")

                wait_time_start = time.perf_counter()

            last_pins = pins
            pins = self._gpio.read(1, True)
            time.sleep(0.0005)

        try:
            self.log_timing_results()
        except Exception:
            logger.error("[Ft232hConnector] Failed to log timing_view after EOF")

        # ensure completion
        try:
            if not stop_event.is_set():
                subj.on_completed()
        except Exception:
            pass


    def log_timing_results(self):
        """
        Log all collected timing_view entries.
        Safe for Cython memoryview layout used in TimingResult.
        """
        cdef double[:, :] buf

        with self._timing_lock:
            tv = self.timing_view
            total = tv.size

            logger.info(f"[Ft232hConnector] Logging timing_view with {total} entries")

            buf = tv.buf

            for i in range(total):
                try:
                    logger.info(
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
                    logger.warning(f"[Ft232hConnector] Failed to read timing_view[{i}]: {e}")


cpdef CTimingResult get_timing_view():
    """
    Return the C-level TimingResult pointer (may be NULL if not initialized).
    Caller must not dereference without checking.
    """
    return timing_view

cpdef object get_timing():
    """Return the Python-level TimingResult singleton (or None)."""
    return timing




