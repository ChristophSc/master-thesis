import logging
import os
import datetime
from config import config, dump_config


def logger_init(started_file):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if config().log.to_file:
        log_filename = os.path.join(config().log.dir,
                                    started_file + "_" + config().log.prefix + datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S") + ".log")
        logging.getLogger().addHandler(logging.FileHandler(log_filename))
    if config().log.dump_config:
        dump_config()