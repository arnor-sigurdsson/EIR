import copy
import logging

from termcolor import colored
from tqdm import tqdm


class TQDMLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        record_copy = copy.copy(record)

        if record_copy.levelno == logging.DEBUG:
            record_copy.levelname = colored(record_copy.levelname, "blue")
        elif record_copy.levelno == logging.INFO:
            record_copy.levelname = colored(record_copy.levelname, "green")
        elif record_copy.levelno == logging.WARNING:
            record_copy.levelname = colored(record_copy.levelname, "yellow")
        elif record_copy.levelno == logging.ERROR:
            record_copy.levelname = colored(record_copy.levelname, "red")

        return super().format(record_copy)


def get_logger(name: str, tqdm_compatible: bool = False) -> logging.Logger:
    """
    Creates a logger with a debug level and a custom format.

    tqdm_compatible: Overwrite default stream.write in favor of tqdm.write
    to avoid breaking progress bar.
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.DEBUG)

    handler: logging.Handler | TQDMLoggingHandler
    if tqdm_compatible:
        handler = TQDMLoggingHandler()
    else:
        handler = logging.StreamHandler()

    formatter = ColoredFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger_.addHandler(handler)
    return logger_
