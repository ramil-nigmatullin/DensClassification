"""TeNPy - a Python library for Tensor Network Algorithms

TeNPy is a library for algorithms working with tensor networks,
e.g., matrix product states and -operators,
designed to study the physics of strongly correlated quantum systems.
The code is intended to be accessible for newcommers
and yet powerful enough for day-to-day research.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
# This file marks this directory as a python package.

import logging
logger = logging.getLogger(__name__)  # main logger for tenpy

# load and provide subpackages on first input
# note that the order matters!
from . import tools
from . import linalg
from . import algorithms
from . import networks
from . import models
from . import simulations
from . import version  # needs to be after linalg!
from .simulations.simulation import run_simulation, resume_from_checkpoint

#: hard-coded version string
__version__ = version.version

#: full version from git description, and numpy/scipy/python versions
__full_version__ = version.full_version

__all__ = [
    "algorithms", "linalg", "models", "networks", "simulations", "tools", "version", "show_config",
    "run_simulation", "resume_from_checkpoint", "console_main"
]


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`tenpy.version.version_summary`.
    """
    print(version.version_summary)


def console_main():
    """Command line interface.


    See also :func:`run_simulation` for the python interface running a simulation.

    When tenpy is installed correctly via pip/conda, a command line script called ``tenpy-run``
    is set up, which calls this function, i.e., you can do the following in the terminal::

        tenpy-run --help

    Equivalently, you can also invoke the tenpy module from your python interpreter like this::

        python -m tenpy --help

    ..
        Sphinx includes the output of ``tenpy-run --help`` here.
    """
    import numpy as np
    import scipy
    import sys
    import importlib
    parser = _setup_arg_parser()

    args = parser.parse_args()
    # import extra modules
    context = {'tenpy': globals(), 'np': np, 'scipy': scipy}
    if args.import_module:
        sys.path.insert(0, '.')
        for module_name in args.import_module:
            module = importlib.import_module(module_name)
            context[module_name] = module
    # load parameters_file
    options = {}
    if args.parameters_file:
        import yaml
        with open(args.parameters_file, 'r') as stream:
            options = yaml.safe_load(stream)
    # update extra options
    if args.option:
        for key, val_string in args.option:
            val = eval(val_string, context)
            set_recursive(options, key, val, insert_dicts=True)
    if len(options) == 0:
        raise ValueError("No options supplied! Check your command line arguments!")
    if 'output_filename' not in options:
        raise ValueError("No output filename specified - refuse to run without saving anything!")
    run_simulation(args.sim_class, **options)


def _setup_arg_parser(width=None):
    import argparse
    import textwrap

    desc = "Command line interface to run a TeNPy simulation."
    epilog = textwrap.dedent("""\
    Examples
    --------

    In the simplest case, you just give a single yaml file with all the parameters as argument:

        tenpy-run my_params.yml

    If you implemented a custom simulation class called ``MyGreatSimulation`` in a
    file ``my_simulations.py``, you can use it like this:

        tenpy-run -i my_simulations -c MyGreatSimulation my_params.yml

    Further, you can overwrite one or multiple options of the parameters file:

        tenpy-run my_params.yml -o output_filename '"output.h5"' -o model_params/Jz 2.

    Note that string values for the options require double quotes on the command line.
    """)

    def formatter(prog):
        return argparse.RawDescriptionHelpFormatter(prog,
                                                    indent_increment=4,
                                                    max_help_position=8,
                                                    width=width)

    parser = argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=formatter)
    parser.add_argument('--import-module',
                        '-i',
                        metavar='MODULE',
                        action='append',
                        help="Import the given python MODULE before setting up the simulation. "
                        "This is useful if the module contains user-defined subclasses. "
                        "Use python-style names like `numpy` without the .py ending.")
    parser.add_argument('--sim-class',
                        '-c',
                        default='GroundStateSearch',
                        help="selects the Simulation (sub)class, "
                        "e.g. 'GroundStateSearch' (default) or 'RealTimeEvolution'.")
    parser.add_argument('parameters_file',
                        nargs='?',
                        help="A yaml (*.yml) file with the simulation parameters/options.")
    opt_help = textwrap.dedent("""\
        Allows overwriting some options from the yaml files.
        KEY can be recursive separated by ``/``, e.g. ``algorithm_params/trunc_params/chi``.
        VALUE is initialized by python's ``eval(VALUE)`` with `np`, `scipy` and `tenpy` defined.
        Thus ``'1.2'`` and ``'2.*np.pi*1.j/6'`` or ``'np.linspace(0., 1., 6)'`` will work if you
        include the quotes on the command line to ensure that the VALUE is passed as a single
        argument.""")
    parser.add_argument('--option',
                        '-o',
                        nargs=2,
                        action='append',
                        metavar=('KEY', 'VALUE'),
                        help=opt_help)
    parser.add_argument('--version', '-v', action='version', version=__full_version__)
    return parser
