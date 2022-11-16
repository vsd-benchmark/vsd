import logging
import os
import time
import numpy as np
import torch
import random
import json
from datetime import timedelta

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(log_dir, dump=True):
    filepath = os.path.join(log_dir, 'net_launcher_log.log')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # Safety check
    # if os.path.exists(filepath) and opt.checkpoint == "":
    #     logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    if dump:
        # create file handler and set level to info
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to info
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if dump:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info('Created main log at ' + str(filepath))
    return logger


def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

class ParamsBase():

    def str(self):
        attrs = [item for item in self.__dir__() if not item.startswith('_')]
        str = 'Params:: ' + ''.join(['{} : {}\n'.format(at, getattr(self, at)) for at in attrs])
        return str

    def todict(self):
        attrs = [item for item in self.__dir__() if not item.startswith('_')]
        d = {at: getattr(self, at) for at in attrs}
        d.pop('str')
        d.pop('todict')
        d.pop('save')
        return d

    def save(self):
        r = [item.split(' : ') for item in self.str().split('\n')]
        d = {i[0]: i[1] for i in r[:-1]}
        with open(self.log_dir + '/params.json', 'w') as f:
            json.dump(d, f)
