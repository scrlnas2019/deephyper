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
        _args = vars(self.parse_args(''))
        kwargs['problem'] = problem
        kwargs['run'] = run
        kwargs['evaluator'] = evaluator
        _args.update(kwargs)
        self.args = Namespace(**_args)
        self.problem = util.generic_loader(problem, 'Problem')
        self.run_func = util.generic_loader(run, 'run')
        logger.info('Evaluator will execute the function: '+run)
        if kwargs.get('cache_key') is None:
            self.evaluator = Evaluator.create(self.run_func, method=evaluator)
        else:
            self.evaluator = Evaluator.create(self.run_func, method=evaluator, cache_key=kwargs['cache_key'])
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
            default="deephyper.benchmark.hps.polynome2.Problem"
        )
        parser.add_argument("--run",
            default="deephyper.benchmark.hps.polynome2.run"
        )
        parser.add_argument("--backend",
            default='tensorflow',
            help="Keras backend module name"
        )
        parser.add_argument('--max-evals',
            type=int, default=100,
            help='maximum number of evaluations'
        )
        parser.add_argument('--eval-timeout-minutes',
            type=int,
            default=4096,
            help="Kill evals that take longer than this"
        )
        parser.add_argument('--evaluator',
            default='subprocess',
            choices=['balsam', 'subprocess', 'processPool', 'threadPool'],
            help="The evaluator is an object used to run the model."
        )
        return parser
