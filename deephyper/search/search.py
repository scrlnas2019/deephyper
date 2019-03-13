import argparse
from pprint import pformat
import logging
from deephyper.search import util
from deephyper.evaluator import Evaluator

logger = logging.getLogger(__name__)

class Namespace:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__dict__[k] = v

class Search:
    """Abstract representation of a black box optimization search.

    A search comprises 3 main objects: a problem, a run function and an evaluator:
        The `problem` class defines the optimization problem, providing details like the search domain.  (You can find many kind of problems in `deephyper.benchmark`)
        The `run` function executes the black box function/model and returns the objective value which is to be optimized.
        The `evaluator` abstracts the run time environment (local, supercomputer...etc) in which run functions are executed.

    Args:
        problem (str):
        run (str):
        evaluator (str): in ['balsam', 'subprocess', 'processPool', 'threadPool']
    """
    def __init__(self, problem, run, evaluator, **kwargs):
        _args = {}
        _args.update(kwargs)
        _args['problem'] = problem
        _args['run'] = evaluator
        _args['evaluator'] = evaluator
        self.args = Namespace(**_args)

        self.problem = util.generic_loader(problem, 'Problem')

        # --run is passed, which means load the run function
        if run is not None:
            self.run_func = util.generic_loader(run, 'run')
            self.run_cmd = None
            assert self.args.run_cmd is None
            assert evaluator != 'balsam-direct'
            logger.info('Evaluator will execute the callable: '+run)
            self.evaluator = Evaluator.create(self.run_func, method=evaluator)

        # --run-cmd is passed, which means balsam-direct will launch the command-line
        else:
            self.run_func = None
            self.run_cmd = self.args.run_cmd
            assert self.run_cmd is not None
            assert evaluator == 'balsam-direct', f'{evaluator} Evaluator cannot use run-cmd'
            logger.info('Evaluator will execute the command line: '+self.run_cmd)
            self.evaluator = Evaluator.create(self.run_cmd, method='balsam-direct') 

        self.num_workers = self.evaluator.num_workers
        logger.info(f'Options: '+pformat(self.args.__dict__, indent=4))
        logger.info('Hyperparameter space definition: '+pformat(self.problem.space, indent=4))
        logger.info(f'Created {self.args.evaluator} evaluator')
        logger.info(f'Evaluator: num_workers is {self.num_workers}')

    def main(self):
        raise NotImplementedError

    @classmethod
    def parse_args(cls, arg_str=None):
        base_parser = cls._base_parser()
        parser = cls._extend_parser(base_parser)
        if arg_str is not None:
            return parser.parse_args(arg_str)
        else:
            return parser.parse_args()

    @staticmethod
    def _extend_parser(base_parser):
        raise NotImplementedError

    @staticmethod
    def _base_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--problem",
            required=True,
            help="DeepHyper Problem instance that defines the search space",
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--run',
            help='''Dotted-Python path to callable that runs the
            model. For example, "MyModels.mnist.cnn_model.run" will cause the function run() to
            be imported from the `cnn_model` module inside the package `MyModels.mnist`.'''
        )
        group.add_argument('--run-cmd',
            help='''Command line arguments to run the model executable. 
            Only compatible with the balsam-direct Evaluator.
            The hyperparameters will be appended to the command line as a JSON-formatted string.
            It is the model code's responsibility to parse this JSON and pass it on to the
            constructed model.'''
        )
        parser.add_argument('--evaluator',
            default='subprocess',
            choices=Evaluator.EVALUATOR_CHOICES,
            help="Choose the type of evaluator to execute model evaluation tasks"
        )
        return parser
