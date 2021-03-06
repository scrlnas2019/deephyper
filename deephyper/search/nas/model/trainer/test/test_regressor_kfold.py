from random import random

import numpy as np
from tensorflow import keras

from deephyper.search.nas.model.keras.trainers.regressor_kfold import KerasTrainerRegressorKfold
from deephyper.search import util
from deephyper.benchmark.nas.toy.pb_keras import Problem

print(Problem)
config = dict(Problem.space)

load_data = util.load_attr_from(config['load_data']['func'])
config['load_data']['func'] = load_data

# config['create_structure']['func'] = util.load_attr_from(
#     config['create_structure']['func'])

print('[PARAM] Loading data')
# Loading data
kwargs = config['load_data'].get('kwargs')
(t_X, t_y), (v_X, v_y) = load_data() if kwargs is None else load_data(**kwargs)
print('[PARAM] Data loaded')

# Set data shape
input_shape = list(np.shape(t_X))[1:]
output_shape = list(np.shape(t_y))[1:]

config['data'] = {
    'train_X': t_X,
    'train_Y': t_y,
    'valid_X': v_X,
    'valid_Y': v_y
}

structure = config['create_structure']['func'](input_shape, output_shape, **config['create_structure']['kwargs'])
arch_seq = [random() for _ in range(structure.num_nodes)]
structure.set_ops(arch_seq)
model = structure.create_model(train=True)


trainer = KerasTrainerRegressorKfold(config=config, model=model)
trainer.train()
