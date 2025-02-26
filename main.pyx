# cython: language_level=3
# cython.infer_types(True)
import cython
import signal
import sys
from argparse import Namespace
from multiprocessing import freeze_support

from pyftdi.ftdi import Ftdi
from reactivex import Subject

from CommandLineParser import CommandLineParser
from SignalHandler import signal_handler

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

# Signal handler is called on interrupt (ctrl-c) and terminate
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
        print(f"Error listing FT232H chip: {e}")
        sys.exit(1)

    try:
        # open a dedicated ftdi device contained in the list of ftdi devices
        # URL Scheme
        # ftdi://[vendor][:[product][:serial|:bus:address|:index]]/interface
        ftdi.open_mpsse_from_url("ftdi:///1")
    except Exception as e:
        print(f"Error accessing FT232H chip: {e}")
        sys.exit(1)

    from DigitizeVideo import DigitizeVideo
    from Ft232hConnector import Ft232hConnector

    optocoupler_signal_subject = Subject()

    # create class instances
    DigitizeVideo(args, optocoupler_signal_subject)

    ft232h = Ft232hConnector(ftdi, optocoupler_signal_subject, args.maxcount)

    # start recording - wait for signal(s) to take picture(s)
    ft232h.signal_input()

    # disconnect FT232H
    ftdi.close()


if __name__ == '__main__':
    main()
