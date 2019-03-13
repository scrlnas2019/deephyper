"""
Command line script to run Python function in an external process

Usage: python runner.py <modulePath> <moduleName> <funcName> <json-params>

Loads Python module <moduleName> located in the <modulePath> directory.
The function <funcName> must be in module global scope (i.e. not nested
inside a class), accept one dictionary argument, and return a scalar objective
value. The passed dictionary is given by <json-params>, a JSON-formatted dictionary 
escaped by single quotes.
"""
from __future__ import print_function
import importlib
import sys
import json
if sys.version_info.major == 2:
    ModuleNotFoundError = ImportError

def load_module(name, path):
    try:
        mod = importlib.import_module(name)
    except (ModuleNotFoundError, ImportError):
        sys.path.insert(0, path)
        mod = importlib.import_module(name)
    return mod

def check_rank():
    rank = 0
    if 'mpi4py' in sys.modules:
        mpi4py = sys.modules['mpi4py']
        if hasattr(mpi4py, 'MPI'):
            MPI = mpi4py.MPI
            if MPI.Is_initialized() and not MPI.Is_finalized():
                rank = MPI.COMM_WORLD.Get_rank()
    elif 'horovod.tensorflow' in sys.modules:
        hvd = sys.modules['horovod.tensorflow']
        try: rank = hvd.rank()
        except ValueError: rank = 0
    return rank

if __name__ == "__main__":
    modulePath = sys.argv[1]
    moduleName = sys.argv[2]
    module = load_module(moduleName, modulePath)

    funcName = sys.argv[3]
    json_params = sys.argv[4]
    param_dict = json.loads(json_params)
    func = getattr(module, funcName)

    retval = func(param_dict)

    if check_rank() == 0:
        print("DH-OUTPUT:", retval)
