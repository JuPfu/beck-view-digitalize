# cython: language_level=3
# main.pyx

"""
Main entrypoint for beck-view-digitize.

Responsibilities:
- Parse command-line args (via CommandLineParser)
- Create FTDI device and open it
- Create shared Subject for optocoupler events
- Instantiate DigitizeVideo (subscriber)
- Instantiate Ft232hConnector(ftdi, subject, max_count, gui)
- Start poller, wait for EOF or Ctrl-C
- Perform coordinated shutdown and cleanup
"""

from multiprocessing import freeze_support
import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)

import os, subprocess

import sys
import threading
import signal

from argparse import Namespace

from pyftdi.ftdi import Ftdi
from reactivex import Subject

from CommandLineParser import CommandLineParser
from DigitizeVideo import DigitizeVideo
from Ft232hConnector import Ft232hConnector


def main():
    freeze_support()

    print(os.environ.get("PKG_CONFIG_PATH"))
    subprocess.run(["pkg-config", "--cflags", "--libs", "spng"], check=True)

    #
    # Parse CLI arguments
    #
    args: Namespace = CommandLineParser().parse_args()

    #
    # Create optocoupler event Subject
    #
    optocoupler_signal_subject = Subject()

    #
    # Event that will be set when on_completed() fires (EOF)
    #
    completion_event = threading.Event()

    # subscribe a tiny handler to set the event when EOF/completion occurs
    # subscribe supports keyword callbacks: on_next, on_error, on_completed
    def _on_completed():
        print("[main] Received on_completed — signalling completion_event")
        completion_event.set()

    optocoupler_signal_subject.subscribe(
        on_completed=_on_completed,
        on_error=lambda e: print(f"[main] Subject error: {e}")
    )

    #
    # Open FTDI device
    #
    try:
        ftdi = Ftdi()
        print("[main] Listing attached FT232H devices:", ftdi.list_devices())
        # open MPSSE device — conservative approach: fail fast on error
        ftdi.open_mpsse_from_url("ftdi:///1")
        print("[main] FTDI opened successfully")
    except Exception as e:
        print(f"[main] ERROR: cannot open FTDI device: {e}", file=sys.stderr)
        sys.exit(1)

    #
    # Instantiate DigitizeVideo first so it subscribes immediately
    #
    try:
        digitizer = DigitizeVideo(args, optocoupler_signal_subject)
    except Exception as e:
        print(f"[main] Failed to create DigitizeVideo: {e}", file=sys.stderr)
        try:
            ftdi.close()
        except Exception:
            pass
        sys.exit(1)

    #
    # Instantiate Ft232hConnector using *strict signature*
    # Ft232hConnector(ftdi, signal_subject, max_count, gui)
    #
    try:
        ft_conn = Ft232hConnector(
            ftdi,
            optocoupler_signal_subject,
            args.maxcount,
            args.gui
        )
    except Exception as e:
        print(f"[main] Failed to create Ft232hConnector: {e}", file=sys.stderr)
        try:
            ftdi.close()
        except Exception:
            pass
        sys.exit(1)

    digitizer.connect(ft_conn)

    #
    # Start poller thread
    #
    try:
        ft_conn.start()
    except Exception as e:
        print(f"[main] Failed to start poller: {e}", file=sys.stderr)
        try:
            digitizer.final_write_to_disk()
        except Exception:
            pass
        try:
            ft_conn.stop()
        except Exception:
            pass
        try:
            ftdi.close()
        except Exception:
            pass
        sys.exit(1)

    print("[main] Poller started — waiting for EOF or Ctrl-C")

    #
    # Main wait loop
    #
    try:
        while not completion_event.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — initiating shutdown...")

    #
    # Coordinated shutdown
    #
    print("[main] Coordinated shutdown starting...")

    try:
        ft_conn.stop()
    except Exception as e:
        print(f"[main] Error stopping Ft232hConnector: {e}", file=sys.stderr)

    try:
        digitizer.final_write_to_disk()
    except Exception as e:
        print(f"[main] Error in final_write_to_disk: {e}", file=sys.stderr)

    try:
        ftdi.close()
        print("[main] FTDI closed.")
    except Exception as e:
        print(f"[main] Error closing FTDI: {e}", file=sys.stderr)

    print("[main] Shutdown complete.")
