import logging
from abc import abstractmethod, ABC
from typing import Union, Optional, Type
from rich.logging import RichHandler
import inspect
import sys

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

_level_to_sign = {
    "info": "ℹ️",
    "debug": "🔎",
    "warning": "⚠️",
    "error": "❌",
    "critical": "💣",
    "stdout": "🔵",
    "stderr": "🟠",
}

not_found_sign = "❗️❓"

# used for redirecting stdout and stderr to the logger
class StreamToLogger():
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.add_to_log(line, self.log_level)
    
    def flush(self):
        pass

class BaseLogger(ABC):
    def __init__(self, filename: Union[str,None] = None):
        if filename is None:
            self._start_stream_logger()
        else:
            self._start_file_logger(filename)

        # redirect stdout and stderr to logger
        sys.stdout = StreamToLogger(self, "stdout")
        sys.stderr = StreamToLogger(self, "stderr")

    @abstractmethod
    def _start_file_logger(self, filename):
        pass

    @abstractmethod
    def _start_stream_logger(self):
        pass

    def add_to_log(self, s, level="info", cf: Optional[inspect.Traceback] = None):
        if level not in _level_to_sign:
            raise ValueError("Unrecognized level value: {}. Must be one of: {}".format(level, _level_to_sign))
        # append the file name and line number of the caller
        caller_frame = cf if cf is not None else inspect.getframeinfo(inspect.currentframe().f_back)
        caller_filename = caller_frame.filename.split("/")[-1]
        message = _level_to_sign[level] + "  " + s + " ({}:{})".format(caller_filename, caller_frame.lineno)
        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "critical":
            self.logger.critical(message)
        elif level == "stdout":
            self.logger.info(message)
        elif level == "stderr":
            self.logger.info(message)

class SimpleLogger(BaseLogger):
    def __init__(self, filename: Union[str,None] = None):
        super().__init__(filename)

    def _start_file_logger(self, filename):
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level = logging.NOTSET,
            format = "%(asctime)s %(levelname)-8s %(message)s",
            datefmt = "[%Y-%m-%d %H:%M:%S]",
            filename = filename,
        )
        self.logger = logging.getLogger()

    def _start_stream_logger(self):
        raise NotImplementedError

class RichLogger(BaseLogger):
    def __init__(self, filename: Union[str,None] = None):
        super().__init__(filename)

    def _start_file_logger(self, filename):
        raise NotImplementedError

    def _start_stream_logger(self):
        logging.basicConfig(
            level=logging.NOTSET,
            format="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            handlers=[RichHandler(omit_repeated_times=False)],
        )
        self.logger = logging.getLogger("rich")

def cprint(msg: str, logger: Optional[Type[BaseLogger]] = None, log_level: str = "info"):
    if logger is not None:
        logger.add_to_log(msg, log_level, cf = inspect.getframeinfo(inspect.currentframe().f_back))
    else:
        print(msg)