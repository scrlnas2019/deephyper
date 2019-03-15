import argparse
from importlib import import_module
import os
import shutil
import sys
from pprint import pprint
import configparser
from balsam.core.models import ApplicationDefinition, BalsamJob, QueuedLaunch
from balsam.service import service

SEARCH_APPS = {
    'ambs' : ('deephyper', 'search', 'hps', 'ambs', 'AMBS'),
}

def get_config(ini_path):
    here = os.path.dirname(__file__)
    default_ini_list = glob.glob(os.path.join(here, '*.ini'))
    user_ini = os.path.expanduser('~/.deephyper/run.ini')
    if ini_path is None:
        if not os.path.exists(user_ini):
            print("***************")
            print("  Creating default runtime configurations in ~/.deephyper:")
            dirname = os.path.dirname(user_ini)
            os.makedirs(dirname, exist_ok=True)
            for default_ini in default_ini_list: shutil.copy(src=default_ini, dst=dirname)
            print("  Please edit this file to define a global runtime configuration, then re-run this command.")
            print("  If you want to use a problem-specific configuration, create a copy of this file and pass it with the -c flag")
            print("***************")
            sys.exit(1)
        return read_ini(user_ini)
    else:
        if not os.path.isfile(ini_path):
            raise FileNotFoundError(f"No such config file: {ini_path}")
        return read_ini(ini_path)

def read_ini(ini_path):
    fname = os.path.abspath(os.path.expanduser(ini_path))
    assert os.path.isfile(fname), f'No such file: {fname}'
    _config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    _config.optionxform = str
    _config.read(fname)
    config = {}
    sections = ('environ', 'run_resources', 'search_resources')
    for section in sections:
        config[section] = dict(_config[section])
    return config

def create_job(problem, run, run_cmd, workflow, **kwargs):
    script_path = kwargs['script_path']
    app_name = kwargs['app_name']
    app, created = ApplicationDefinition.objects.get_or_create(
        name = app_name,
        executable = f'{sys.executable} {script_path}'
    )
    if created:
        print(f"Created app {app_name} in the Balsam DB (ID {app.pk})")

    args = [f'--problem {problem}']
    if run is not None:
        assert run_cmd is None
        args.append(f'--run {run}')
        args.append('--evaluator balsam')
    else:
        assert run_cmd is not None
        args.append(f'--run-cmd "{run_cmd}"')
        args.append('--evaluator balsam-direct')

    for arg in kwargs['search_defaults']:
        if arg in kwargs:
            args.append(f'--{arg.replace("_","-")} {kwargs[arg]}')

    run_resources = kwargs['run_resources']
    search_resources = kwargs['search_resources']
    for key in 'num_nodes ranks_per_node threads_per_rank node_packing_count'.split():
        run_resources[key] = int(run_resources[key])
        search_resources[key] = int(search_resources[key])
        

    envs = kwargs['environ'].copy()
    envs['DEEPHYPER_WORKERS_PER_NODE'] = run_resources['node_packing_count']
    environ_vars = ':'.join(f'{key}={val}' for key,val in envs.items())

    search_job = BalsamJob(
        name = app_name,
        workflow = workflow,
        application = app_name,
        environ_vars = environ_vars,
        **search_resources,
        args = ' '.join(args),
        data = dict(run_resources=run_resources),
    )

    search_job.save()
    print('\n*****')
    print(f"Created {app_name} search task {search_job.cute_id} with workflow tag {workflow}")
    print(f"Submit a Balsam job to run this search:")
    print(f'balsam submit-launch --wf-filter {workflow} -n <nodes> -t <minutes> -q <queue> -A <project> --job-mode <mpi or serial>')
    print('******\n')

def add_base_args(subparser):
    p = subparser
    p.add_argument('workflow', help="Unique workflow tag for this search run")
    p.add_argument('-c', '--config-file',
        help='''Take search parameters from a config file. Any parameters passed as
                       command line arguments will override the parameters set in
                       the config file.  This is a good way to avoid repeatedly
                       typing lengthy command lines that are 90% similar. Use 
                       'config-demo' to print a demo config file from which you can
                       create your own.''',
    )
    p.add_argument('-p', '--problem', 
        help='''DeepHyper Problem instance
        (myPackage.mySubPackage.problem.Problem) *or* /path/to/problem.py
        containing the Problem instance''',
        required=True
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',
        help='''Dotted-Python path to callable that runs the
        model. For example, "MyModels.mnist.cnn_model.run" will cause the function run() to
        be imported from the `cnn_model` module inside the package
        `MyModels.mnist`. Alternatively, provide /path/to/model.py
        Python module file that will be imported (run() must be
        accessible from the outermost global scope)'''
    )
    group.add_argument('--run-cmd',
        help='''Command line arguments to run the model executable directly.
        Use this if the model run() cannot be loaded into the DeepHyper Python
        environment (e.g. model code is not Python3.6 compatible, or you need to run model in
        a Singularity container).  The hyperparameters will be appended to the command line as a JSON-formatted string.
        It is the model's responsibility to parse this JSON and pass it on to the
        constructed model.'''
    )

def build_parser():
    root_parser = argparse.ArgumentParser(description="Create DeepHyper search jobs in Balsam")
    subparsers = root_parser.add_subparsers(title="DeepHyper Search Menu")

    for app_name, path in SEARCH_APPS.items():
        mod_path, class_name = '.'.join(path[:-1]), path[-1]
        search_module = import_module(mod_path)
        search_class = getattr(search_module, class_name)

        # help text is first line of module docstring
        script_path = os.path.abspath(search_module.__file__)
        doc = search_module.__doc__
        doc = doc.split('\n')[0] if isinstance(doc, str) else ''

        # add subparser for search
        p = subparsers.add_parser(app_name, help=doc)
        p.set_defaults(app_name=app_name)
        add_base_args(p)
        search_class._extend_parser(p)
        p.set_defaults(script_path=script_path)
        
        _dummy = argparse.ArgumentParser()
        search_class._extend_parser(_dummy)
        defaults = vars(_dummy.parse_args(''))
        p.set_defaults(search_defaults=defaults)

    return root_parser

def main():
    config = get_config(None)
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.config_file is not None:
        config = get_config(args.config_file)
    config.update(vars(args))
    create_job(**config)

if __name__ == "__main__":
    main()
