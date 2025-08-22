# cython: language_level=3
# cython.infer_types(True)
import signal
import sys
from types import FrameType

def signal_handler(self, signum: int, frame: FrameType | None) -> None:
    """
    Handle interrupt signals.
    """
    name = signal.Signals(signum).name
    self.logger.warning(f"Program terminated by signal '{name}' at {frame}")
    sys.exit(1)