from source import mute_constants
from logging import handlers
import logging

# log format
log_formatter = logging.Formatter('%(asctime)s,%(message)s')


# handler settings
log_handler = handlers.TimedRotatingFileHandler(filename=mute_constants.LOG_PATH + 'mute.log',
                                                when='midnight',
                                                interval=1,
                                                encoding='utf-8')
log_handler.setFormatter(log_formatter)
log_handler.suffix = "%Y-%m-%d"


# logger set
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)


def info(msg):
    logger.info(msg)


def debug(msg):
    logger.debug(msg)


def error(msg):
    logger.error(msg)
