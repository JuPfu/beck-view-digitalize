import argparse
import pathlib


class CommandLineParser:
    def __init__(self) -> None:
        # Initialize the argument parser with description
        self.parser = argparse.ArgumentParser(
            description='Digitize analog 16mm films.'
        )
        # Add argument for version
        self.parser.add_argument('--version', action='version', version='1.0.2')
        # Add argument for device selection
        self.parser.add_argument(
            '-d', '--device',
            dest="device",
            type=int,
            nargs='?',
            choices=range(0, 10),
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
        self.parser.add_argument(
            '-wh', '--width_height',
            dest="width_height",
            type=int,
            nargs=2,
            metavar=("width", "height"),
            default=[1920, 1080],
            help='Width and height of image frames - default is (1920, 1080)'
        )
        # Add arguments for emergency brake
        self.parser.add_argument(
            '-m', '--max-count',
            dest="maxcount",
            type=int,
            nargs='?',
            choices=[3600, 7200, 14400, 21800, 43600, 60000],
            default=21800,
            help='The End Of Film (EOF) is signalled by optocoupler 2. In case the optocoupler 2 signal is not '
                 'emitted stop digitizing when specified number of images is reached - default is 21800 frames.'
        )
        # Add arguments for chunk size
        self.parser.add_argument(
            '-c', '--chunk-size',
            dest="chunk_size",
            type=int,
            nargs='?',
            default=12,
            help='Chunk size (number of frames) passed to each process - default is 12 frames'
        )
        # Add arguments for direct show settings menu
        self.parser.add_argument(
            '--show-menu',
            dest="settings",
            action="store_true",
            default=True,
            help='Display direct show settings menu - default is True'
        )
        # Add arguments for exposure bracketing
        self.parser.add_argument(
            '-b', '--bracketing',
            dest="bracketing",
            action="store_true",
            default=False,
            help='Take multiple exposures of one frame with varying exposure time - default is no bracketing, which means just one exposure per frame'
        )

    def parse_args(self) -> argparse.Namespace:
        # Parse arguments and return the namespace
        return self.parser.parse_args()
