import argparse
import pathlib


class CommandLineParser:
    def __init__(self) -> None:
        # Initialize the argument parser with description
        self.parser = argparse.ArgumentParser(
            description='Digitize analog super 8 film.'
        )
        # Add argument for version
        self.parser.add_argument('--version', action='version', version='1.0.0')
        # Add argument for device selection
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
            dest="output_path",
            type=pathlib.Path,
            nargs='?',
            default=pathlib.Path(".").resolve(),
            help='Output directory for digitized png frames.'
        )
        # Add arguments for emergency brake
        self.parser.add_argument(
            '-m', '--max-count',
            dest="maxcount",
            type=int,
            nargs='?',
            choices=range(1, 50000),
            default=7250,
            help='The End Of Film (EOF) is signalled by optocoupler 2. In case the optocoupler 2 signal is not '
                 'emitted stop digitizing when specified number of images is reached.'
        )
        # Add arguments for monitoring frames
        self.parser.add_argument(
            '-d', '--display_monitor',
            dest="monitor",
            action="store_true",
            default=False,
            help='Display monitoring window.'
        )
        # Add arguments for chunk size
        self.parser.add_argument(
            '-c', '--chunk-size',
            dest="chunk_size",
            type=int,
            nargs='?',
            default=12,
            help='Chunk size (number of frames) passed to each process.'
        )

    def parse_args(self) -> argparse.Namespace:
        # Parse arguments and return the namespace
        return self.parser.parse_args()
