import os

import numpy as np

from deephyper.benchmark import HpProblem
from deephyper.benchmark.benchmark_functions_wrappers import polynome_2

# Problem definition
Problem = HpProblem()

num_dim = 10
for i in range(num_dim):
    Problem.add_dim(f'e{i}', (-10, 10), i)


# Definition of the function which runs the model
def run(param_dict):
    f, (a, b), _ = polynome_2()

    num_dim = 10
    x = np.array([param_dict[f'e{i}'] for i in range(num_dim)])

    return f(x) # the objective

if __name__ == '__main__':
    print(Problem)
