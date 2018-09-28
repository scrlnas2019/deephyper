import os
import signal
import sys
from pprint import pformat
from random import random
from importlib import import_module
import numpy as np
import tensorflow as tf

import time

import deephyper.model.arch as a
from deephyper.search import util
from deephyper.model.utilities.nas_cmdline import create_parser

from nas.model.trainer import BasicTrainer

logger = util.conf_logger('deephyper.search.nas')

def run(param_dict):
    logger.debug('Starting...')
    config = param_dict

    load_data = util.load_attr_from(config['load_data']['func'])

    config['create_structure']['func'] = util.load_attr_from(
        config['create_structure']['func'])

    config['create_cell']['func'] = util.load_attr_from(config['create_cell']['func'])

    logger.debug('[PARAM] Loading data')
    # Loading data
    (t_X, t_y), (v_X, v_y) = load_data(dest='DATA')
    logger.debug('[PARAM] Data loaded')

    config['input_shape'] = list(np.shape(t_X))[1:]
    config['output_shape'] = list(np.shape(v_X))[1:]

    config[a.data] = { a.train_X: t_X,
                       a.train_Y: t_y,
                       a.valid_X: v_X,
                       a.valid_Y: v_y }

    architecture = config['arch_seq']

    # For all the Net generated by the CONTROLLER
    trainer = BasicTrainer(config)

    # Run the trainer and get the rewards
    result = trainer.get_rewards(architecture)

    logger.debug(f'[REWARD/RESULT] = {result}')
    return result

if __name__ == '__main__':
    parser = create_parser()
    cmdline_args = parser.parse_args()
    param_dict = cmdline_args.config
    run(param_dict)
