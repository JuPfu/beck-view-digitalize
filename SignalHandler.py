import signal
from types import FrameType


def signal_handler(self, signum: int, frame: FrameType | None) -> None:
    """
    Handle interrupt signals.
    """
    signame = signal.Signals(signum).name
    self.logger.warning(f"Program terminated by signal '{signame}' at {frame}")
    exit(1)