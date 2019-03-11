import csv
from collections import OrderedDict
from contextlib import suppress as dummy_context
from math import isnan
from numpy import integer, floating, ndarray
import json
import uuid
import logging
import os
import sys
import time
import types

from deephyper.evaluator import runner
from deephyper.search.util import generic_loader
logger = logging.getLogger(__name__)

class Encoder(json.JSONEncoder):
    """
    Enables JSON dump of numpy data
    """
    def default(self, obj):
        if isinstance(obj, uuid.UUID): return obj.hex
        if isinstance(obj, integer): return int(obj)
        elif isinstance(obj, floating): return float(obj)
        elif isinstance(obj, ndarray): return obj.tolist()
        elif isinstance(obj, types.FunctionType):
            return f'{obj.__module__}.{obj.__name__}'
        else: return super(Encoder, self).default(obj)

class Evaluator:
    FAIL_RETURN_VALUE = sys.float_info.max
    PYTHON_EXE = os.environ.get('DEEPHYPER_PYTHON_BACKEND', sys.executable)
    WORKERS_PER_NODE = int(os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))
    KERAS_BACKEND = os.environ.get('KERAS_BACKEND', 'tensorflow')
    os.environ['KERAS_BACKEND'] = KERAS_BACKEND
    EVALUATOR_CHOICES = ['balsam', 'balsam-direct', 'subprocess', 'processPool', 'threadPool']
    assert os.path.isfile(PYTHON_EXE)

    @staticmethod
    def create(run, cache_key=None, method='balsam'):
        '''
        Args:
            run: string commandline OR Python Callable object
        '''
        assert method in EVALUATOR_CHOICES

        if method != 'balsam-direct': assert callable(run)
        else: assert isinstance(run, str)

        if method == "balsam":
            from deephyper.evaluator._balsam import BalsamEvaluator
            Eval = BalsamEvaluator
        elif method == "balsam-direct":
            from deephyper.evaluator._balsam_direct import BalsamDirectEvaluator
            Eval = BalsamDirectEvaluator
        elif method == "subprocess":
            from deephyper.evaluator._subprocess import SubprocessEvaluator
            Eval = SubprocessEvaluator
        elif method == "processPool":
            from deephyper.evaluator._processPool import ProcessPoolEvaluator
            Eval = ProcessPoolEvaluator
        else:
            from deephyper.evaluator._threadPool import ThreadPoolEvaluator
            Eval = ThreadPoolEvaluator

        return Eval(run, cache_key=cache_key)

    def __init__(self, run, cache_key=None):
        self.pending_evals = {} # uid --> Future
        self.finished_evals = OrderedDict() # uid --> scalar
        self.requested_evals = [] # keys
        self.key_uid_map = {} # map keys to uids

        self.transaction_context = dummy_context
        self._start_sec = time.time()
        self.elapsed_times = {}

        self.num_workers = 0

        if cache_key is not None:
            assert callable(cache_key)
            self._gen_uid = cache_key
        else:
            self._gen_uid = lambda d: self.encode(d)

        if callable(run):
            self._run_function = run
            self._run_cmd = None
            module = run.__module__
            if moduleName == '__main__':
                raise RuntimeError(f'Evaluator will not execute function "{run_function.__name__}" '
                "because it is in the __main__ module.  Please provide a function "
                "imported from an external module!")
        else:
            self._run_function = None
            self._run_cmd = run

    def encode(self, x):
        if not isinstance(x, dict):
            raise ValueError(f'Expected dict, but got {type(x)}')
        return json.dumps(x, cls=Encoder)

    def _elapsed_sec(self):
        return time.time() - self._start_sec

    def decode(self, key):
        '''from JSON string to x (list)'''
        x = json.loads(key)
        if not isinstance(x, dict):
            raise ValueError(f'Expected dict, but got {type(x)}')
        return x

    def add_eval(self, x):
        key = self.encode(x)
        self.requested_evals.append(key)
        uid = self._gen_uid(x)
        if uid in self.key_uid_map.values():
            logger.info(f"UID: {uid} already evaluated; skipping execution")
        else:
            future = self._eval_exec(x)
            logger.info(f"Submitted new eval of {x}")
            future.uid = uid
            self.pending_evals[uid] = future
        self.key_uid_map[key] = uid

    def add_eval_batch(self, XX):
        with self.transaction_context():
            for x in XX: self.add_eval(x)

    def _eval_exec(self, x):
        raise NotImplementedError

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        raise NotImplementedError

    @staticmethod
    def _parse(run_stdout):
        y = Evaluator.FAIL_RETURN_VALUE
        for line in run_stdout.split('\n'):
            if "DH-OUTPUT:" in line.upper():
                try:
                    y = float(line.split()[-1])
                except ValueError as e:
                    logger.exception("Could not parse DH-OUTPUT line:\n"+line)
                    y = Evaluator.FAIL_RETURN_VALUE
                break
        if isnan(y): y = Evaluator.FAIL_RETURN_VALUE
        return y

    @property
    def _runner_executable(self):
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        assert moduleName != '__main__'
        module = sys.modules[moduleName]
        modulePath = os.path.dirname(os.path.abspath(module.__file__))
        runnerPath = os.path.abspath(runner.__file__)
        runner_exec = ' '.join((self.PYTHON_EXE, runnerPath, modulePath, moduleName,
                            funcName))
        return runner_exec

    def await_evals(self, to_read, timeout=None):
        keys = list(map(self.encode, to_read))
        uids = [self._gen_uid(x) for x in to_read]
        futures = {uid : self.pending_evals[uid]
                   for uid in set(uids) if uid in self.pending_evals}
        logger.info(f"Waiting on {len(futures)} evals to finish...")

        logger.info(f'Blocking on completion of {len(futures)} pending evals')
        self.wait(futures.values(), timeout=timeout, return_when='ALL_COMPLETED')
        # TODO: on TimeoutError, kill the evals that did not finish; return infinity
        for uid in futures:
            y = futures[uid].result()
            self.elapsed_times[uid] = self._elapsed_sec()
            del self.pending_evals[uid]
            self.finished_evals[uid] = y
        for (key, uid, x) in zip(keys, uids, to_read):
            y = self.finished_evals[uid]
            logger.info(f"x: {x} y: {y}")
            try: self.requested_evals.remove(key)
            except ValueError: pass
            yield (x,y)

    def get_finished_evals(self):
        futures = self.pending_evals.values()
        try:
            waitRes = self.wait(futures, timeout=0.5, return_when='ANY_COMPLETED')
        except TimeoutError:
            pass
        else:
            for future in (waitRes.done + waitRes.failed):
                uid = future.uid
                y = future.result()
                logger.info(f'New eval finished: {uid} --> {y}')
                self.elapsed_times[uid] = self._elapsed_sec()
                del self.pending_evals[uid]
                self.finished_evals[uid] = y

        for key in self.requested_evals[:]:
            uid = self.key_uid_map[key]
            if uid in self.finished_evals:
                self.requested_evals.remove(key)
                x = self.decode(key)
                y = self.finished_evals[uid]
                logger.debug(f"Requested eval x: {x} y: {y}")
                yield (x,y)

    @property
    def counter(self):
        return len(self.finished_evals) + len(self.pending_evals)

    def dump_evals(self):
        if not self.finished_evals: return

        with open('results.json', 'w') as fp:
            json.dump(self.finished_evals, fp, indent=4, sort_keys=True, cls=Encoder)

        resultsList = []

        for key,uid in self.key_uid_map.items():
            if uid not in self.finished_evals: continue
            result = self.decode(key)
            result['objective'] = self.finished_evals[uid]
            result['elapsed_sec'] = self.elapsed_times[uid]
            resultsList.append(result)

        with open('results.csv', 'w') as fp:
            columns = resultsList[0].keys()
            writer = csv.DictWriter(fp, columns)
            writer.writeheader()
            writer.writerows(resultsList)
