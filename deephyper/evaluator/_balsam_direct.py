import logging
import shlex
import os
from deephyper.evaluator._balsam import BalsamEvaluator
from balsam.core.models import ApplicationDefinition

logger = logging.getLogger(__name__)

class BalsamDirectEvaluator(BalsamEvaluator):

    def _init_app(self):
        run_cmd = self._run_cmd
        self.appName = run_cmd
        app, created = ApplicationDefinition.objects.get_or_create(
            name = self.appName,
            defaults = {'executable': run_cmd},
        )
        if created:
            logger.info(f"Created app {self.appName} in Balsam DB to run the model")
