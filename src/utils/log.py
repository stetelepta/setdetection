import logging


def setup_logger(level=logging.DEBUG, logfile=None):
    assert logfile is not None, "specify a logile"

	# setup formatter
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # create a logger
    logger = logging.getLogger()

    # set the loglevel
    logger.setLevel(level)

    # remove previous handlers if there are any to prevent duplicate log messages
    logger.handlers = []
    
    # setup handler and add to logger
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
