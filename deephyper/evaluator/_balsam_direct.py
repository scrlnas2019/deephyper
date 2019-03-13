import shlex
import os
from deephyper.evaluator._balsam import BalsamEvaluator

class BalsamDirectEvaluator(BalsamEvaluator):

    def _init_app(self):
        run_cmd = self._run_cmd
        self.appName = ' '.join(
            os.path.basename(s for s in shlex.split(run_cmd))
        )
        app, created = ApplicationDefinition.objects.get_or_create(
            name = self.app_name,
            executable = run_cmd
        )
        if created:
            logger.info(f"Created app {name} in Balsam DB to run the model")