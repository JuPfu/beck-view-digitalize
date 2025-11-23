# Ft232hConnector.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
Drop-in FT232H connector with dedicated poller thread + single-slot mailbox.
Designed to be robust and easy to compile in Cython.
"""

import cython
import time
import logging
import sys
from threading import Thread, Event as ThreadEvent

import usb
from pyftdi.ftdi import FtdiError
from pyftdi.gpio import GpioMpsseController
from reactivex import Subject

# TimingResult: your Cython TimingResult class providing append(...)
from TimingResult cimport TimingResult

# global timing buffer (module-singleton)
timing = TimingResult()

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

cdef class EventSlot:
    """
    Single-slot mailbox for the latest event.
    Fields are typed (C-level) but methods run with GIL.
    edge: 1=rising OK1, 3=END_OF_FILM
    """
    cdef public unsigned long ts_us   # microseconds (perf_counter*1e6)
    cdef public unsigned char edge
    cdef public unsigned char pending

    def __cinit__(self):
        self.ts_us = 0
        self.edge = 0
        self.pending = 0

    cpdef void store(self, unsigned long ts_us, unsigned char edge):
        """
        Store an event (overwrites previous). Called by poller thread (GIL held).
        """
        # store values, mark pending last for visibility
        self.ts_us = ts_us
        self.edge = edge
        self.pending = 1

    cpdef tuple consume(self):
        """
        Consume and return (ts_seconds: float, edge:int) or (None, 0) if no pending event.
        """
        if self.pending:
            ts_us = self.ts_us
            ed = self.edge
            self.pending = 0
            return (ts_us / 1e6, int(ed))
        return (None, 0)


cdef class Poller:
    """
    Poller thread: reads GPIO and publishes events to EventSlot.
    Pure Python thread; uses simple logic and minimal overhead.
    """
    cdef EventSlot _slot
    cdef object _gpio
    cdef unsigned int _ok_mask
    cdef unsigned int _eof_mask
    cdef object _stop_event  # Threading Event
    cdef object _thread
    cdef public int running

    def __cinit__(self, EventSlot slot, object gpio, unsigned int ok_mask, unsigned int eof_mask):
        self._slot = slot
        self._gpio = gpio
        self._ok_mask = ok_mask
        self._eof_mask = eof_mask
        self._stop_event = ThreadEvent()
        self._thread = None
        self.running = 0

    cpdef void start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        self.running = 1

    cpdef void stop(self, object timeout=None):
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            if timeout is None:
                self._thread.join()
            else:
                # timeout may be float or int
                self._thread.join(float(timeout))
        except Exception:
            pass
        self._thread = None
        self.running = 0

    cdef unsigned int _read_pins(self):
        """
        Try to use read_pins() if available, else fallback to read(1)[0].
        Returns an int mask (0..255+).
        """
        try:
            fn = getattr(self._gpio, "read_pins", None)
            if fn is not None:
                return <unsigned int> fn()
            data = self._gpio.read(1)
            return <unsigned int> data[0]
        except Exception:
            # on error return 0 and continue
            return 0

    def _run(self):
        """
        Poll loop. Publish rising OK1 edges and END_OF_FILM (edge=3).
        Uses a tiny yield to avoid 100% CPU while keeping latency low.
        """
        cdef unsigned int last = 0
        cdef unsigned int pins = 0
        stop_ev = self._stop_event
        try:
            last = self._read_pins()
        except Exception:
            last = 0

        while not stop_ev.is_set():
            pins = self._read_pins()

            # END_OF_FILM detection
            if (pins & self._eof_mask) == self._eof_mask:
                try:
                    ts_us = <unsigned long> int(time.perf_counter() * 1e6)
                    self._slot.store(ts_us, <unsigned char>3)
                except Exception:
                    pass
                break

            # rising edge detection (0 -> 1)
            if ((last & self._ok_mask) == 0) and ((pins & self._ok_mask) == self._ok_mask):
                try:
                    ts_us = <unsigned long> int(time.perf_counter() * 1e6)
                    self._slot.store(ts_us, <unsigned char>1)
                except Exception:
                    pass

            last = pins
            # yield to scheduler: keeps latency small but avoids pure busy-spin
            time.sleep(0)

        # stop
        self.running = 0


cdef class Ft232hConnector:
    """
    Main connector class. API is compatible with previous version:
      - __init__(ftdi_obj, signal_subject, max_count, gui)
      - signal_input()
      - stop()
      - close()
    """

    cdef public object gpio
    cdef public object dev
    cdef public EventSlot _event_slot
    cdef public Poller _poller
    cdef public TimingResult timing
    cdef public unsigned int OK1
    cdef public unsigned int END_OF_FILM
    cdef public int __max_count
    cdef public int gui
    cdef public object signal_subject
    cdef double LATENCY_THRESHOLD

    def __init__(self, object ftdi, object signal_subject, int max_count, bint gui):
        self.gui = gui

        self._initialize_logging()
        self._initialize_device()

        cdef unsigned int MSB = 8
        self.OK1 = ((1 << 2) << MSB)
        self.END_OF_FILM = ((1 << 3) << MSB)

        self.gpio = GpioMpsseController()

        try:
            ftdi.validate_mpsse()
        except Exception as err:
            self.logger.error(f"Ftdi MPSSE error: {err}")
            raise

        self.gpio.configure('ftdi:///1',
                            direction=0x0,
                            frequency=ftdi.frequency_max,
                            initial=self.OK1 | self.END_OF_FILM)

        # same init sequence you used previously
        self.gpio.set_direction(pins=self.END_OF_FILM | self.OK1, direction=self.END_OF_FILM | self.OK1)
        self.gpio.write(0x0)

        ftdi.set_latency_timer(128)
        ftdi.set_frequency(ftdi.frequency_max)
        self.gpio.set_direction(pins=self.END_OF_FILM | self.OK1, direction=0x0)

        self.signal_subject = signal_subject
        self.__max_count = max_count + 50
        self.LATENCY_THRESHOLD = 0.01

        # reset timing buffer (C-level)
        global timing
        timing.reset(max_count)   # allocate buffer

        # event slot + poller
        self._event_slot = EventSlot()
        self._poller = Poller(self._event_slot, self.gpio, <unsigned int> self.OK1, <unsigned int> self.END_OF_FILM)
        self._poller.start()

    def _initialize_logging(self) -> None:
        self.logger = logging.getLogger(__name__)
        if self.gui and not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)

    def _initialize_device(self) -> None:
        self.dev = usb.core.find(idVendor=0x0403, idProduct=0x6014)
        if self.dev is None:
            self.logger.error("No USB device with vendorId=0x0403 productId=0x6014 found!")
            raise RuntimeError("FTDI device not found")
        self.logger.info(f"USB device found: {self.dev}")

    def signal_input(self) -> None:
        """
        Consumer loop. Consume events from the single-slot mailbox and forward via signal_subject.
        """
        cdef int count = -1
        cdef double start_cycle = time.perf_counter()
        cdef double cycle_time = 1.0 / 5.0
        cdef double delta = 0.0
        cdef double work_time = 0.0
        cdef double latency_time = 0.0
        cdef double end_cycle = 0.0
        cdef double wait_time = 0.0

        cdef EventSlot slot = self._event_slot
        cdef object subj = self.signal_subject
        cdef TimingResult tbuf = self.timing
        cdef unsigned int maxc = self.__max_count

        while True:
            ts_edge = slot.consume()
            ts, edge = ts_edge

            # END_OF_FILM or stop
            if edge == 3:
                break

            if ts is None:
                if count >= maxc:
                    break
                time.sleep(0)  # yield
                continue

            now = ts
            delta = now - start_cycle
            start_cycle = now

            count += 1
            cycle_time = delta

            work_time_start = time.perf_counter()
            try:
                subj.on_next((count, start_cycle))
            except Exception as e:
                self.logger.error(f"signal_subject.on_next error: {e}")
            work_time = time.perf_counter() - work_time_start

            if work_time > cycle_time:
                self.logger.warning(f"Work time took {work_time*1000:.2f} ms")

            # wait for OK1 release; poll underlying gpio briefly
            latency_start = time.perf_counter()
            try:
                pins = self.gpio.read_pins() if hasattr(self.gpio, "read_pins") else self.gpio.read(1)[0]
            except Exception:
                pins = 0
            while (pins & self.OK1) == self.OK1:
                time.sleep(0)
                try:
                    pins = self.gpio.read_pins() if hasattr(self.gpio, "read_pins") else self.gpio.read(1)[0]
                except Exception:
                    pins = 0

            latency_time = time.perf_counter() - latency_start
            if latency_time > self.LATENCY_THRESHOLD:
                self.logger.warning(f"Suspicious high latency {latency_time} for frame {count} !")

            end_cycle = time.perf_counter()
            wait_time = cycle_time - (end_cycle - start_cycle)

            # append to TimingResult (C-level buffer)
            try:
                tbuf.append(
                    float(count),
                    float(end_cycle - start_cycle),
                    float(work_time),
                    float(-1.0),
                    float(latency_time),
                    float(wait_time),
                    float(delta)
                )
            except Exception:
                pass

            if wait_time <= 0.0:
                self.logger.warning(
                    f"Negative wait time {wait_time:.5f} s for frame {count} at fps={1.0 / delta}."
                )

            if count >= maxc:
                break

        # completed
        try:
            subj.on_completed()
        except Exception:
            pass

    def stop(self) -> None:
        """
        Stop the poller thread.
        """
        try:
            if self._poller is not None:
                self._poller.stop(timeout=1.0)
        except Exception:
            pass

    def close(self) -> None:
        """
        Convenience cleanup wrapper.
        """
        self.stop()
