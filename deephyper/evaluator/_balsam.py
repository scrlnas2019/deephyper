import logging
import os
import json
from io import StringIO

from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist
from balsam.launcher import dag
from balsam.launcher.async import FutureTask
from balsam.launcher.async import wait as balsam_wait
from balsam.core.models import ApplicationDefinition as AppDef

from deephyper.evaluator import Evaluator
logger = logging.getLogger(__name__)

LAUNCHER_NODES = int(os.environ.get('BALSAM_LAUNCHER_NODES', 1))
JOB_MODE = os.environ.get('BALSAM_JOB_MODE', None)
if JOB_MODE is None:
    logger.warning("Could not detect job mode. Assuming serial")
    JOB_MODE = "serial"
JOB_MODE = JOB_MODE.lower().strip()
assert JOB_MODE in ['serial', 'mpi']

logger.debug(f"Detected LAUNCHER_NODES = {LAUNCHER_NODES}")
logger.debug(f"Detected JOB_MODE = {JOB_MODE}")

class BalsamEvaluator(Evaluator):
    """Evaluator using balsam software.

    Documentation to balsam : https://balsam.readthedocs.io
    This class helps us to run task on HPC systems with more flexibility and ease of use.

    Args:
        run_function (func): takes one parameter of type dict and returns a scalar value.
        cache_key (func): takes one parameter of type dict and returns a hashable type, used as the key for caching evaluations. Multiple inputs that map to the same hashable key will only be evaluated once. If ``None``, then cache_key defaults to a lossless (identity) encoding of the input dict.
    """
    def __init__(self, run_function, cache_key=None):
        super().__init__(run_function, cache_key)
        if dag.current_job is None:
            raise RuntimeError("Balsam evaluator must be running inside context of a Balsam launcher")
        self.set_run_resources()
        if JOB_MODE == 'serial':
            self.num_workers = max(1, LAUNCHER_NODES*self.WORKERS_PER_NODE - 2)
            logger.debug(f"Serial mode: models per node = {self.WORKERS_PER_NODE}")
        else:
            run_nodes = self.run_resources['num_nodes']
            self.num_workers = max(
                1, 
                (LAUNCHER_NODES - dag.current_job.num_nodes) // run_nodes
            )
            logger.debug(f"MPI mode: models occupy whole number of nodes")

        logger.debug(f"Total number of workers: {self.num_workers}")
        logger.info(f"Backend runs will use Python: {self.PYTHON_EXE}")
        self._init_app()
        logger.info(f"Backend runs will execute function: {self.appName}")
        self.transaction_context = transaction.atomic


    def set_run_resources(self):
        resource_defaults = {
            'num_nodes': 1,
            'ranks_per_node': 1,
            'threads_per_rank': 64,
            'node_packing_count': self.WORKERS_PER_NODE,
        }
        resources = dag.current_job.data.get('run_resources', {})
        for key,val in resource_defaults.items():
            if key not in resources: resources[key] = val

        self.workflow = dag.current_job.workflow
        self.run_resources = resources

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        return balsam_wait(futures, timeout=timeout, return_when=return_when)

    def _init_app(self):
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        self.appName = '.'.join((moduleName, funcName))
        try:
            app = AppDef.objects.get(name=self.appName)
        except ObjectDoesNotExist:
            logger.info(f"ApplicationDefinition did not exist for {self.appName}; creating new app in BalsamDB")
            app = AppDef(name=self.appName, executable=self._runner_executable)
            app.save()
        else:
            logger.info(f"BalsamEvaluator will use existing app {self.appName}: {app.executable}")

    def _eval_exec(self, x):
        jobname = f"task{self.counter}"
        args = f"'{self.encode(x)}'"
        task = dag.add_job(
                    name = jobname,
                    workflow = self.workflow,
                    application = self.appName,
                    args = args,
                    environ_vars = os.environ.copy() # propagates to model
                    **self.run_resources
                   )
        logger.debug(f"Created job {jobname}")
        logger.debug(f"Args: {args}")

        future = FutureTask(task, self._on_done, fail_callback=self._on_fail)
        future.task_args = args
        return future

    @staticmethod
    def _on_done(job): #def _on_done(job, process_data):
        # Objective in job.data?
        try: objective = float(job.data['dh-objective'])
        except KeyError,ValueError: pass
        else: return objective

        # Objective in postprocess.log ?
        try:
            post_out = job.read_file_in_workdir(f'postprocess.log')
            objective = Evaluator._parse(post_out)
        except ValueError as e:
            pass
        else:
            if objective != Evaluator.FAIL_RETURN_VALUE: return objective

        # Otherwise, take what we get in {job.name}.out
        stdout = job.read_file_in_workdir(f'{job.name}.out')
        objective = Evaluator._parse(stdout)
        return objective

    @staticmethod
    def _on_fail(job):
        logger.info(f'Task {job.cute_id} failed; setting objective as float_max')
        return Evaluator.FAIL_RETURN_VALUE
