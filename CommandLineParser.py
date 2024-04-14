import argparse
import pathlib


class CommandLineParser:
    def __init__(self) -> None:
        # Initialize the argument parser with description
        self.parser = argparse.ArgumentParser(
            description='Digitise analog super 8 film.'
        )
        # Add argument for version
        self.parser.add_argument('--version', action='version', version='1.0.0')
        self.parser.add_argument(
            '-d', '--device',
            dest="device",
            type=int,
            nargs='?',
            choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            default=0,
            help='Device number of attached camera.'
        )
        # Add arguments for output path
        self.parser.add_argument(
            '-o', '--output-path',
            dest="opath",
            type=pathlib.Path,
            nargs='?',
            default=pathlib.Path(".").resolve(),
            help='Output directory for digitised png frames'
        )
        self.parser.add_argument(
            '-mc', '--max-count',
            dest="maxcount",
            type=int,
            nargs='?',
            choices=[500, 950, 1850, 3650, 7250, 14450],
            default=7250,
            help='The End Of Film (EOF) is signalled by optocoupler 2. In case this signal is not emitted stop '
                 'digitising when specified number of images is reached.'
        )

    def parse_args(self) -> argparse.Namespace:
        # Parse arguments and return the namespace
        return self.parser.parse_args()
