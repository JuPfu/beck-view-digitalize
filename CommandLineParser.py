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
            choices=range(0,10),
            default=0,
            help='Device number of attached camera - default is device 0.'
        )
        # Add arguments for output path
        self.parser.add_argument(
            '-o', '--output-path',
            dest="output_path",
            type=pathlib.Path,
            nargs='?',
            default=pathlib.Path(".").resolve(),
            help='Output directory for digitized png frames - default is current directory.'
        )
        # Add arguments for emergency brake
        self.parser.add_argument(
            '-m', '--max-count',
            dest="maxcount",
            type=int,
            nargs='?',
            choices=[300, 500, 3650, 7250],
            default=7250,
            help='The End Of Film (EOF) is signalled by optocoupler 2. In case the optocoupler 2 signal is not '
                 'emitted stop digitizing when specified number of images is reached - default is 7250 frames.'
        )
        # Add arguments for monitoring frames
        self.parser.add_argument(
            '-s', '--show_monitor',
            dest="monitor",
            action="store_true",
            default=False,
            help='Show monitoring window'
        )
        # Add arguments for chunk size
        self.parser.add_argument(
            '-c', '--chunk-size',
            dest="chunk_size",
            type=int,
            nargs='?',
            default=8,
            help='Chunk size (number of frames) passed to each process - default is 8  frames'
        )

    def parse_args(self) -> argparse.Namespace:
        # Parse arguments and return the namespace
        return self.parser.parse_args()
