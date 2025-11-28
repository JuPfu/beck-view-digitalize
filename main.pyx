# main.pyx
# Entry point for beck-view-digitize.
# Signal handlers registered here in main thread; orchestrates clean shutdown.

# cython: language_level=3
import sys
import threading
from argparse import Namespace
from multiprocessing import freeze_support

from pyftdi.ftdi import Ftdi
from reactivex import Subject

from CommandLineParser import CommandLineParser

# Import our worker classes
from DigitizeVideo import DigitizeVideo
from Ft232hConnector import Ft232hConnector

# Small module-level globals to allow signal handler to request shutdown.
# They are assigned in main() below.
_dv = None        # DigitizeVideo instance
_ft232h = None    # Ft232hConnector instance
_shutdown_event = None  # threading.Event used to coordinate shutdown


def _signal_handler(signum, frame):
    """
    Minimal signal handler executed in main thread:
    - sets the shutdown_event to request cooperative termination
    - does NOT block or perform long-running actions
    """
    global _shutdown_event, _dv
    try:
        if _shutdown_event is not None:
            _shutdown_event.set()
    except Exception:
        pass
    # Avoid calling heavy logic here; main() will perform final stop/cleanup.


def main():
    global _dv, _ft232h, _shutdown_event

    freeze_support()

    # parse arguments
    args: Namespace = CommandLineParser().parse_args()

    # create shutdown event in main thread and register signal handlers
    _shutdown_event = threading.Event()
    import signal
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # initialize FTDI device driver
    ftdi = Ftdi()

    # list available ftdi devices
    try:
        print(f"List attached FT232H devices: {ftdi.list_devices()}")
    except Exception as e:
        print(f"Error listing FT232H devices: {e}")
        sys.exit(1)

    try:
        # open a dedicated ftdi device contained in the list of ftdi devices
        ftdi.open_mpsse_from_url("ftdi:///1")
    except Exception as e:
        print(f"Error accessing FT232H chip: {e}")
        sys.exit(1)

    # create reactive subject used to receive signals from FT232H connector
    optocoupler_signal_subject = Subject()

    # create DigitizeVideo and pass shutdown_event so it can respond to signals
    _dv = DigitizeVideo(args, optocoupler_signal_subject, shutdown_event=_shutdown_event)

    # create and start FT232H poller AFTER DigitizeVideo is ready to receive signals
    _ft232h = Ft232hConnector(ftdi, optocoupler_signal_subject, args.maxcount, args.gui)
    _ft232h.start()

    # Block here until shutdown_event is set (clean, cooperative shutdown requested)
    try:
        _dv.run_main_loop()   # this returns after final_write_to_disk() and cleanup()
    except Exception as e:
        try:
            print(f"Unexpected error while running capture loop: {e}", file=sys.stderr)
        except Exception:
            pass

    # Shutdown sequence (main thread, performed in a deterministic order)
    # 1) Stop FT232H poller (join background thread)
    try:
        if _ft232h is not None:
            _ft232h.stop(timeout=1.0)
    except Exception:
        pass

    # 2) Close FTDI device (safe now that poller stopped)
    try:
        ftdi.close()
    except Exception:
        pass

    # 3) Ensure DigitizeVideo final cleanup was done (run_main_loop already called final_write_to_disk & cleanup).
    #    But in case run_main_loop returned early for some reason, be defensive:
    try:
        if _dv is not None:
            _dv.final_write_to_disk()
            _dv.cleanup()
    except Exception:
        pass

    # Final exit
    return 0


if __name__ == '__main__':
    sys.exit(main())
