import logging

NULL_LOGGER = logging.getLogger('null')
NULL_LOGGER.handlers = [logging.NullHandler()]
NULL_LOGGER.propagate = False
