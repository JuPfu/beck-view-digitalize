# main.pyx
# cython: language_level=3
import faulthandler, sys
faulthandler.enable(all_threads=True)

from argparse import Namespace
from multiprocessing import freeze_support
from reactivex import Subject

from CommandLineParser import CommandLineParser
from DigitizeVideo import DigitizeVideo

def main():
    freeze_support()

    args: Namespace = CommandLineParser().parse_args()

    # Subject used for optocoupler events; connector will publish into it.
    optocoupler_signal_subject = Subject()

    # Create DigitizeVideo (this will create & start the FT232H connector in Option B).
    dv = DigitizeVideo(args, optocoupler_signal_subject)

    # Wait until the connector signals EOF (subject.on_completed()) or user interrupts.
    try:
        dv.logger.info("Waiting for completion (EOF) — press Ctrl-C to interrupt.")
        completed = dv.wait_for_completion(None)  # block indefinitely
        if completed:
            dv.logger.info("Completion event received.")
        else:
            dv.logger.warning("Wait for completion timed out.")
    except KeyboardInterrupt:
        dv.logger.warning("Keyboard interrupt received; shutting down.")
    finally:
        try:
            dv.final_write_to_disk()
        except Exception:
            pass
        try:
            dv.cleanup()
        except Exception:
            pass

if __name__ == '__main__':
    main()
