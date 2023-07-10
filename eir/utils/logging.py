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
        levelname = record.levelname
        if record.levelno == logging.DEBUG:
            levelname = colored(levelname, "blue")
        elif record.levelno == logging.INFO:
            levelname = colored(levelname, "green")
        elif record.levelno == logging.WARNING:
            levelname = colored(levelname, "yellow")
        elif record.levelno == logging.ERROR:
            levelname = colored(levelname, "red")

        record.levelname = levelname
        return super().format(record)


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
