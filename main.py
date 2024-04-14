from argparse import Namespace

from pyftdi.ftdi import Ftdi
from reactivex.subject import Subject

from CommandLineParser import CommandLineParser

"""
Technologies used in this project

ReactiveX for Signal Handling: 
    Reactivex is being used for handling photo cell signals. This allows for asynchronous and non-blocking operation 
    improving responsiveness.

Monitoring: 
    The inclusion of a monitoring window provides valuable insights into program execution.

Logging:
    Logging capabilities to help analyse potential problems.
"""


def main():
    # retrieve command line arguments
    args: Namespace = CommandLineParser().parse_args()

    # initialize the Future Technology Devices International chip
    ftdi = Ftdi()
    # list available ftdi devices
    # on macOS do a `ls -lta /dev/cu*` when the ftdi microcontroller is connected
    print(f"List Devices: {ftdi.list_devices()}")
    # open a dedicated ftdi device contained in the list of ftdi devices
    # URL Scheme
    # ftdi://[vendor][:[product][:serial|:bus:address|:index]]/interface
    ftdi.open_from_url("ftdi:///1")

    from DigitalizeVideo import DigitalizeVideo
    from Ft232hConnector import Ft232hConnector

    optocoupler_signal_subject: Subject = Subject()

    # create class instances
    DigitalizeVideo(args.device, args.output_path, optocoupler_signal_subject)

    ft232h = Ft232hConnector(optocoupler_signal_subject, args.maxcount)

    # start recording - wait for signal(s) to take picture(s)
    ft232h.signal_input()

    # disconnect FT232H
    ftdi.close()


if __name__ == '__main__':
    main()
