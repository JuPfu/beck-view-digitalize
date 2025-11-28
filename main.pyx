# cython: language_level=3
# main.pyx
"""
Main entrypoint for beck-view-digitize.

Responsibilities:
- Parse command-line args (CommandLineParser)
- Create FTDI device and open it
- Create shared Subject for optocoupler events
- Instantiate DigitizeVideo and Ft232hConnector (defensive to constructor variants)
- Start poller and wait for EOF or Ctrl-C
- Perform coordinated shutdown and cleanup
"""

from multiprocessing import freeze_support
import faulthandler
faulthandler.enable(all_threads=True)

import sys
import threading
import time
import signal

from argparse import Namespace

from pyftdi.ftdi import Ftdi
from reactivex import Subject

from CommandLineParser import CommandLineParser

# import Cython-extension classes (these must be available)
from DigitizeVideo import DigitizeVideo
from Ft232hConnector import Ft232hConnector

def main():
    freeze_support()

    # parse args (CommandLineParser should return argparse.Namespace)
    args: Namespace = CommandLineParser().parse_args()
    print(f"Starting with args={args}")

    # create the Subject that carries optocoupler events to subscribers
    optocoupler_signal_subject = Subject()

    # make a completion event so the main thread can wait until EOF
    completion_event = threading.Event()

    # subscribe a tiny handler to set the event when EOF/completion occurs
    # subscribe supports keyword callbacks: on_next, on_error, on_completed
    def _on_completed():
        print("[main] Received on_completed from opto subject — signalling completion_event")
        completion_event.set()

    # keep subscription minimal (DigitizeVideo also subscribes)
    optocoupler_signal_subject.subscribe(on_completed=_on_completed,
                           on_error=lambda e: print(f"[main] Subject error: {e}"))

    # create and open FTDI device
    ftdi = None
    try:
        ftdi = Ftdi()
        print("[main] Listing attached FT232H devices:", ftdi.list_devices())
        # open MPSSE device — conservative approach: fail fast on error
        ftdi.open_mpsse_from_url("ftdi:///1")
        print("[main] FTDI opened successfully")
    except Exception as e:
        print(f"[main] Warning: could not open FTDI device: {e}", file=sys.stderr)
        # We continue: some connector variants open FTDI themselves.
        # Keep ftdi as None and let Ft232hConnector handle it if it wants to.

    # instantiate DigitizeVideo first (it subscribes to the subject and prepares resources)
    try:
        digitizer = DigitizeVideo(args, optocoupler_signal_subject)
    except Exception as e:
        print(f"[main] Failed to create DigitizeVideo: {e}", file=sys.stderr)
        # If DigitizeVideo creation fails, ensure FTDI closed then exit
        if ftdi is not None:
            try:
                ftdi.close()
            except Exception:
                pass
        raise

    # instantiate Ft232hConnector — be defensive to support both constructor signatures:
    #  - old: Ft232hConnector(ftdi, subject, max_count, gui)
    #  - new: Ft232hConnector(subject, max_count, gui)   (connector opens FTDI itself)
    ft_conn = None
    try:
        if ftdi is not None:
            # try old-style constructor first (explicit ftdi)
            try:
                ft_conn = Ft232hConnector(ftdi, optocoupler_signal_subject, args.maxcount, args.gui)
            except TypeError:
                # fallback to connector-owned-ftdi constructor
                ft_conn = Ft232hConnector(optocoupler_signal_subject, args.maxcount, args.gui)  # type: ignore
        else:
            # no explicit ftdi: expect connector to open FTDI itself
            ft_conn = Ft232hConnector(optocoupler_signal_subject, args.maxcount, args.gui)  # type: ignore
    except NameError:
        # In case of weird name encoding editor paste: try straightforward attempts
        try:
            ft_conn = Ft232hConnector(optocoupler_signal_subject, args.maxcount, args.gui)  # type: ignore
        except Exception as e:
            print(f"[main] Could not instantiate Ft232hConnector: {e}", file=sys.stderr)
            # make sure to cleanup digitizer and exit
            try:
                digitizer.final_write_to_disk()
            except Exception:
                pass
            if ftdi is not None:
                try:
                    ftdi.close()
                except Exception:
                    pass
            raise
    except Exception as e:
        # Any other error — cleanup and re-raise
        print(f"[main] Could not instantiate Ft232hConnector: {e}", file=sys.stderr)
        try:
            digitizer.final_write_to_disk()
        except Exception:
            pass
        if ftdi is not None:
            try:
                ftdi.close()
            except Exception:
                pass
        raise

    # Start the poller/thread in the connector
    try:
        ft_conn.start()
    except Exception as e:
        print(f"[main] Failed to start FT232H poller: {e}", file=sys.stderr)
        # cleanup then exit
        try:
            digitizer.final_write_to_disk()
        except Exception:
            pass
        try:
            ft_conn.stop()
        except Exception:
            pass
        if ftdi is not None:
            try:
                ftdi.close()
            except Exception:
                pass
        raise

    print("[main] Poller started — waiting for EOF or Ctrl-C")

    # Wait for completion_event (set by subject.on_completed) or KeyboardInterrupt
    try:
        while not completion_event.wait(timeout=1.0):
            # simply loop waiting; this allows Ctrl+C to be handled by the signal handlers
            pass
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt received — shutting down")

    # Coordinated shutdown
    print("[main] Initiating coordinated shutdown...")

    try:
        ft_conn.stop()
    except Exception as e:
        print(f"[main] Error stopping ft_conn: {e}", file=sys.stderr)

    # ensure digitizer writes remaining frames and cleans up
    try:
        digitizer.final_write_to_disk()
    except Exception as e:
        print(f"[main] Error during digitizer final write: {e}", file=sys.stderr)

    # close FTDI handle only if we still own it (we opened it above)
    if ftdi is not None:
        try:
            ftdi.close()
            print("[main] FTDI closed.")
        except Exception as e:
            print(f"[main] Error closing FTDI: {e}", file=sys.stderr)

    print("[main] Shutdown complete.")
