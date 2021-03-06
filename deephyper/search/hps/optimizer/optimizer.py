from sys import float_info
from skopt import Optimizer as SkOptimizer
from numpy import inf
import logging

logger = logging.getLogger(__name__)

class Optimizer:
    SEED = 12345
    KAPPA = 1.96

    def __init__(self, problem, num_workers, args):
        assert args.learner in ["RF", "ET", "GBRT", "GP", "DUMMY"], f"Unknown scikit-optimize base_estimator: {args.learner}"

        self.space = problem.space
        n_init = inf if args.learner=='DUMMY' else num_workers
        self._optimizer = SkOptimizer(
            self.space.values(),
            base_estimator=args.learner,
            acq_optimizer='sampling',
            acq_func=args.acq_func,
            acq_func_kwargs={'kappa':self.KAPPA},
            random_state=self.SEED,
            n_initial_points=n_init
        )

        assert args.liar_strategy in "cl_min cl_mean cl_max".split()
        self.strategy = args.liar_strategy
        self.evals = {}
        self.counter = 0
        logger.info("Using skopt.Optimizer with %s base_estimator" % args.learner)

    def _get_lie(self):
        if self.strategy == "cl_min":
            return min(self._optimizer.yi) if self._optimizer.yi else 0.0
        elif self.strategy == "cl_mean":
            return self._optimizer.yi.mean() if self._optimizer.yi else 0.0
        else:
            return  max(self._optimizer.yi) if self._optimizer.yi else 0.0

    def _xy_from_dict(self):
        XX = list(self.evals.keys())
        YY = [self.evals[x] for x in XX]
        return XX, YY

    def to_dict(self, x):
        return {k:v for k,v in zip(self.space, x)}

    def _ask(self):
        x = self._optimizer.ask()
        y = self._get_lie()
        self._optimizer.tell(x,y)
        self.evals[tuple(x)] = y
        logger.debug(f'_ask: {x} lie: {y}')
        return self.to_dict(x)

    def ask(self, n_points=None, batch_size=20):
        if n_points is None:
            self.counter += 1
            return self._ask()
        else:
            self.counter += n_points
            batch = []
            for i in range(n_points):
                batch.append(self._ask())
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def ask_initial(self, n_points):
        XX = self._optimizer.ask(n_points=n_points)
        for x in XX:
            self.evals[tuple(x)] = 0.0
        self.counter += n_points
        return [self.to_dict(x) for x in XX]

    def tell(self, xy_data):
        assert isinstance(xy_data, list), f"where type(xy_data)=={type(xy_data)}"
        maxval = max(self._optimizer.yi) if self._optimizer.yi else 0.0
        for x,y in xy_data:
            key = tuple(x[k] for k in self.space)
            assert key in self.evals, f"where key=={key} and self.evals=={self.evals}"
            logger.debug(f'tell: {x} --> {key}: evaluated objective: {y}')
            self.evals[key] = (y if y < float_info.max else maxval)

        self._optimizer.Xi = []
        self._optimizer.yi = []
        XX, YY = self._xy_from_dict()
        assert len(XX) == len(YY) == self.counter, (
            f"where len(XX)=={len(XX)},"
            f"len(YY)=={len(YY)}, self.counter=={self.counter}")
        self._optimizer.tell(XX, YY)
        assert len(self._optimizer.Xi) == len(self._optimizer.yi) == self.counter, (
            f"where len(self._optimizer.Xi)=={len(self._optimizer.Xi)}, "
            f"len(self._optimizer.yi)=={len(self._optimizer.yi)},"
            f"self.counter=={self.counter}")
