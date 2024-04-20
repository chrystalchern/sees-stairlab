"""
# Synopsis

>`render.py [<options>] <model-file>`

>**Chrystal Chern**, and **Claudio Perez**


This script plots the geometry of a structural
model given a SAM JSON file.


## Matlab
In order to install the Matlab bindings, open Matlab in a
directory containing the files `render.py` and `render.m`,
and run the following command in the Matlab interpreter:

    render --install

Once this process is complete, the command `render` can be
called from Matlab, just as described below for the command
line.

# Usage
This script can be used either as a module, or as a command
line utility. When invoked from the command line on
**Windows**, {NAME} should be `python -m render`. For example:

    python -m render model.json --axes 2 --view elev

"""
import sys

import yaml
import numpy as np
import sees
from sees import RenderError
from sees.cli import parse_args
from sees.views import VIEWS

NAME="sees"
EXAMPLES="""
Examples:
    Plot the structural model defined in the file `sam.json`:
        $ {NAME} sam.json

    Plot displaced structure with unit translation at nodes
    5, 3 and 2 in direction 2 at scale of 100:

        $ {NAME} -d 5:2,3:2,2:2 -s100 --vert 2 sam.json
"""

# Script functions
#----------------------------------------------------

# Argument parsing is implemented manually because in
# the past I have found the standard library module
# `argparse` to be slow.

AXES = dict(zip(("long","tran","vert","sect","elev", "plan"), range(6)))

def dof_index(dof: str):
    try: return int(dof)
    except: return AXES[dof]


def install_me(install_opt=None):
    import os
    import subprocess
    import textwrap
    if install_opt == "dependencies":
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *REQUIREMENTS.strip().split("\n")
        ])
        sys.exit()
    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup
    name = sys.argv[0]

    sys.argv = sys.argv[:1] + ["develop", "--user"]
    package = name[:-3].replace(".", "").replace("/","").replace("\\","")
    # if True:
    #     print(package)
    #     print(name[:-3])
    #     print(sys.argv)
    #     sys.exit()

    setup(name=package,
          version=__version__,
          description="",
          long_description=textwrap.indent(HELP, ">\t\t"),
          author="",
          author_email="",
          url="",
          py_modules=[package],
          scripts=[name],
          license="",
          install_requires=[*REQUIREMENTS.strip().split("\n")],
    )

TESTS = [
    (False,"{NAME} sam.json -d 2:plan -s"),
    (True, "{NAME} sam.json -d 2:plan -s50"),
    (True, "{NAME} sam.json -d 2:3    -s50"),
    (True, "{NAME} sam.json -d 5:2,3:2,2:2 -s100 --vert 2 sam.json")
]

def main():
    config = parse_args(sys.argv)

    if config is None:
        sys.exit()

    try:
        sees.render(**config)

    except (FileNotFoundError, RenderError) as e:
        # Catch expected errors to avoid printing an ugly/unnecessary stack trace.
        print(e, file=sys.stderr)
        print("         Run '{NAME} --help' for more information".format(NAME=NAME), file=sys.stderr)
        sys.exit()

if __name__ == "__main__":
    main()

