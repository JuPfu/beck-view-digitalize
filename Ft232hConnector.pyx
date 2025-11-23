# Ft232hConnector.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
Modernised FT232H connector for beck-view-digitize.

- Dedicated polling thread reads the GPIO pins and publishes events via a
  reactivex Subject provided by the caller.
- Timing is recorded into a module-level TimingResult singleton called `timing`.
"""

import cython
from cython.view cimport array

# Python imports (keep FTDI/pyftdi objects as Python-level)
import logging
import sys
import time
import threading

import usb
from pyftdi.ftdi import Ftdi, FtdiError
from pyftdi.gpio import GpioMpsseController
from reactivex import Subject

# TimingResult: we cimport the Cython class for fast typed access,
# and import the Python constructor to create the module-level singleton.
from TimingResult cimport TimingResult as CTimingResult
from TimingResult import TimingResult as PyTimingResult

# module-level timing singleton (constructed lazily in __init__ below if needed)
timing = None  # will be set to TimingResult(max_frames) at runtime

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

    # typed reference to TimingResult for speed inside Cython
    cdef CTimingResult timing_view

    def __init__(self, object ftdi, object signal_subject, int max_count, bint gui):
        """
        Create connector.

        - ftdi: pyftdi Ftdi instance (Python object)
        - signal_subject: reactivex Subject for .on_next/.on_completed
        - max_count: expected max frames (used to size timing buffer)
        - gui: whether GUI logging to stdout is desired
        """
        self.LATENCY_THRESHOLD = 0.01
        self.INITIAL_COUNT = -1

        self.gui = gui
        self.signal_subject = signal_subject

        # keep raw Python references (we don't cimport Ftdi)
        self._ftdi = ftdi
        self._gpio = GpioMpsseController()

        # masks: high byte MSB shift matches earlier wiring (MSB=8)
        MSB = 8
        self._OK1_mask = ((1 << 2) << MSB)    # AC2
        # avoid name EOF due to stdio macro; use END_OF_FILM instead
        self._END_OF_FILM_mask = ((1 << 3) << MSB)  # AC3

        # initialize USB device check (raise/exit on error)
        self._init_device()

        # configure gpio & ftdi performance tuning
        try:
            # configure controller (device selector matches previous usage)
            self._gpio.configure('ftdi:///1',
                                 direction=0x0,
                                 frequency=self._ftdi.frequency_max,
                                 initial=self._OK1_mask | self._END_OF_FILM_mask)
            # set output-then-low then input as in previous code path
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask,
                                     direction=self._END_OF_FILM_mask | self._OK1_mask)
            self._gpio.write(0x0)
            # latency / frequency
            try:
                self._ftdi.set_latency_timer(128)
            except Exception:
                # best-effort: ignore if attribute not present
                pass
            try:
                self._ftdi.set_frequency(self._ftdi.frequency_max)
            except Exception:
                pass
            # finally set direction to inputs for the pins we read
            self._gpio.set_direction(pins=self._END_OF_FILM_mask | self._OK1_mask, direction=0x0)
        except Exception as e:
            logger.error(f"[Ft232hConnector] gpio configure/setup failed: {e}")
            raise

        # allocate/reset global timing singleton
        global timing
        if timing is None:
            # create Python TimingResult singleton sized to max_count + safety margin
            timing = PyTimingResult(max_count + 100)
        # also keep a typed view for faster .append calls
        self.timing_view = <CTimingResult> timing

        # internal thread control
        self._stop_event = threading.Event()
        self._thread = None
        self.running = False

    def _init_device(self):
        """Check USB device presence; exit on failure to match previous behaviour."""
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
        """Start the polling thread. Safe to call multiple times (idempotent)."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="ft232h-poller", daemon=True)
        self._thread.start()
        self.running = True
        logger.info("[Ft232hConnector] Poller started")

    def stop(self, object timeout=None) -> None:
        """Stop the polling thread and wait for it to finish."""
        # signal and join
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.running = False
        logger.info("[Ft232hConnector] Poller stopped")

    cpdef object get_timing(self):
        """Return the TimingResult singleton (Python object)."""
        global timing
        return timing

    # ------------------------------------------------------------------
    # Internal poll loop: small, hot, and keeps attribute lookups minimal.
    # All IO (gpio.read_pins) remains a Python call; we micro-optimize by
    # binding frequently used names to locals.
    # ------------------------------------------------------------------
    cdef void _record_timing(self,
                             int count,
                             double cycle_span,
                             double work_time,
                             double read_time,
                             double latency_time,
                             double wait_time,
                             double total_work) nogil:
        """
        This helper records timing with nogil -> but actually we must acquire GIL
        because TimingResult.append likely needs the GIL. So we keep a tiny nogil wrapper
        only to show intent; we call append under the GIL below in the poll loop.
        """
        # not used: kept as placeholder if we later implement a nogil C append.
        pass

    def _poll_loop(self) -> None:
        """
        Poll GPIO pins in a loop and publish events.

        Strategy / optimizations:
        - bind masks and methods to local variables to reduce attribute lookups
        - use time.perf_counter() for timestamps
        - use self._stop_event.wait(0) to yield / check stop condition if available
        - sleep with 0 to yield timeslice (minimal delay)
        """
        # localize frequently used attributes for speed
        _gpio = self._gpio
        _ok = self._OK1_mask
        _eof = self._END_OF_FILM_mask
        subj = self.signal_subject
        timing_py = timing  # Python TimingResult object (module singleton)
        append_fn = timing_py.append  # bind method once
        stop_event = self._stop_event
        read_pins = _gpio.read_pins  # bind method
        LATENCY_THRESHOLD = self.LATENCY_THRESHOLD

        count = self.INITIAL_COUNT
        start_cycle = time.perf_counter()

        # initial read
        try:
            pins = read_pins()
        except Exception:
            # fallback: try older API
            try:
                pins = _gpio.read(1)[0]
            except Exception:
                pins = 0

        last_pins = pins

        while not stop_event.is_set() and (pins & _eof) != _eof and count < (timing_py.max_frames - 1):
            # fast path: read pins
            try:
                pins = read_pins()
            except Exception:
                try:
                    pins = _gpio.read(1)[0]
                except Exception:
                    pins = last_pins  # keep previous if we can't read

            # EOF check: if EOF bit set -> publish completion event (edge 3) and break
            if (pins & _eof) == _eof:
                # publish EOF via Subject.on_next or directly on_completed
                try:
                    subj.on_completed()
                except Exception:
                    pass
                break

            # rising edge detection for OK1: last had 0 and now 1
            if ((last_pins & _ok) == 0) and ((pins & _ok) == _ok):
                # timestamping and publishing
                stop_cycle = time.perf_counter()
                delta = stop_cycle - start_cycle
                start_cycle = stop_cycle

                count += 1

                # call subscriber (fast)
                try:
                    subj.on_next((count, start_cycle))
                except Exception:
                    # subscriber may raise; keep polling
                    pass

                # measure work time (time taken by subscriber call)
                # (We cannot measure internal work accurately here; keep zero or small sample)
                # For compatibility with previous timing fields:
                work_time = 0.0

                # now busy-wait for signal to drop (OK1 goes low)
                latency_start = time.perf_counter()
                # spin/yield loop - minimal delay to reduce CPU burn
                while ((pins & _ok) == _ok) and not stop_event.is_set():
                    # yield timeslice - using sleep(0) is typically the lightest
                    time.sleep(0)
                    try:
                        pins = read_pins()
                    except Exception:
                        try:
                            pins = _gpio.read(1)[0]
                        except Exception:
                            break
                latency = time.perf_counter() - latency_start

                # compute end cycle / wait_time
                end_cycle = time.perf_counter()
                wait_time = delta - (end_cycle - start_cycle)  # conservative estimate

                # append to timing buffer (Python-level append)
                try:
                    append_fn(
                        float(count),
                        float(end_cycle - start_cycle),  # cycle
                        float(work_time),
                        float(-1.0),                     # read time not measured here
                        float(latency),
                        float(wait_time),
                        float(delta)                     # total_work
                    )
                except Exception:
                    # ignore timing storage failures to not break capture
                    pass

                # log suspicious latency
                if latency > LATENCY_THRESHOLD:
                    logger.warning(f"[Ft232hConnector] Suspicious high latency {latency:.6f}s for frame {count}")

            # update last_pins and continue
            last_pins = pins

            # small yield - keeps loop responsive but allows other threads to run
            time.sleep(0)

        # Ensure completion published if not already
        try:
            subj.on_completed()
        except Exception:
            pass

    # convenience destructor
    def __dealloc__(self):
        try:
            self.stop(timeout=0.5)
        except Exception:
            pass
