import logging

NULL_LOGGER = logging.getLogger('null')
NULL_LOGGER.handlers = logging.NullHandler()


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name)
