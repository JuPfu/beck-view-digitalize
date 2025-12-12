# cython: language_level=3
# cython.infer_types(True)
import sys
import threading
from argparse import Namespace
from multiprocessing import freeze_support

from pyftdi.ftdi import Ftdi
from reactivex import Subject

from CommandLineParser import CommandLineParser

"""
Technologies used in this project

ReactiveX for Signal Handling:
    Reactivex is being used for handling photo cell signals. This allows for asynchronous and non-blocking operation
    improving responsiveness.

Spawning Processes:
    Circumvent the Global Interpreter Lock (GIL) by using separate processes for writing images to persistent
    storage.

Chunk Processing for Efficiency:
    Usage of a chunk of images for writing to persistent storage is an efficient strategy that reduces the number of
    context switches and system calls.

Shared Memory for Fast Data Transfer:
    Employing shared memory for transferring image data to a separate process (inter-process communication).

Monitoring:
    The inclusion of a monitoring window provides valuable insights into program execution.

Logging:
    Logging capabilities to help analyse potential problems.
"""

def main():
    freeze_support()

    # retrieve command line arguments
    args: Namespace = CommandLineParser().parse_args()

    # initialize FTDI device driver
    ftdi = Ftdi()

    # list available ftdi devices
    # on macOS do a `ls -lta /dev/cu*` when the ftdi microcontroller is connected
    try:
        print(f"List attached FT232H devices: {ftdi.list_devices()}")
    except Exception as e:
        print(f"Error listing FT232H devices: {e}")
        sys.exit(1)

    try:
        # open a dedicated ftdi device contained in the list of ftdi devices
        # URL Scheme
        # ftdi://[vendor][:[product][:serial|:bus:address|:index]]/interface
        ftdi.open_from_url("ftdi:///1")
    except Exception as e:
        print(f"Error accessing FT232H chip: {e}")
        sys.exit(1)

    from DigitizeVideo import DigitizeVideo
    from Ft232hConnector import Ft232hConnector

    optocoupler_signal_subject = Subject()

    wait_subject = Subject()

    #
    # Event that will be set when on_completed() fires (EOF)
    #
    completion_event = threading.Event()

    # subscribe a tiny handler to set the event when EOF/completion occurs
    # subscribe supports keyword callbacks: on_next, on_error, on_completed
    def _on_completed():
        completion_event.set()

    wait_subject.subscribe(
        on_completed=_on_completed,
        on_error=lambda e: print(f"[main] Subject error: {e}")
    )

    #
    # Instantiate DigitizeVideo first so it subscribes immediately
    #
    try:
        digitizer = DigitizeVideo(args, optocoupler_signal_subject, wait_subject)
    except Exception as e:
        print(f"[main] Failed to create DigitizeVideo: {e}", file=sys.stderr)
        try:
            ftdi.close()
        except Exception:
            pass
        sys.exit(1)

    ft232h = Ft232hConnector(ftdi, optocoupler_signal_subject, args.maxcount, args.gui)

    digitizer.connect(ft232h)

    # start recording - wait for signal(s) to take picture(s)
    ft232h.signal_input()

    # Main wait loop
    try:
        while not completion_event.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — initiating shutdown...")

    # Shutdown
    print("[main] Shutdown starting...")

    try:
        ftdi.close()
        print("[main] FTDI closed.")
    except Exception as e:
        print(f"[main] Error closing FTDI: {e}", file=sys.stderr)

    print("[main] Shutdown complete.")
