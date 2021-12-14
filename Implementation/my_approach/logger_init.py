import logging
import os
import datetime
from config import config, dump_config, get_config_path
from shutil import copyfile

def logger_init(started_file):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    log_dir = None
    if config().log.log_train:
        # create directory where training log, graphs and config is stored
        if started_file == "pretrain":
            log_dir  = os.path.join(config().log.dir,datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S") + "_" + started_file + "_" + config().pretrain_config.lower() + "_" + config().task.dir)
        else:
            log_dir  = os.path.join(config().log.dir,datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S") + "_" + started_file + "_"  + config().g_config.lower() + "_"  + config().d_config.lower() + "_"  + config().task.dir)
        os.mkdir(log_dir) 
        # create training log file
        log_filename = os.path.join(log_dir, "training.log")
        logging.getLogger().addHandler(logging.FileHandler(log_filename))
    if config().log.log_config:
        # copy config file
        copyfile(get_config_path(), os.path.join(log_dir, "config.yaml"))         
    if config().log.dump_config:
        dump_config()
    return log_dir