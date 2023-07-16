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
import os
import sys

import yaml
import numpy as np
import sees
from sees import config, RenderError
from sees.views import VIEWS

__version__ = "0.0.4"

NAME = "sees"

HELP = """
usage: {NAME} <sam-file>
       {NAME} --setup ...
       {NAME} [options] <sam-file>
       {NAME} [options] <sam-file> <res-file>
       {NAME} --section <py-file>#<py-object>

Generate a plot of a structural model.

Positional Arguments:
  <sam-file>                     JSON file defining the structural model.
  <res-file>                     JSON or YAML file defining a structural
                                 response.

Options:
  DISPLACEMENTS
  -s, --scale  <scale>           Set displacement scale factor.
  -d, --disp   <node>:<dof>...   Apply a unit displacement at node with tag
                                 <node> in direction <dof>.
  VIEWING
  -V, --view   {{elev|plan|sect}}  Set camera view.
      --vert   <int>             Specify index of model's vertical coordinate
      --hide   <object>          Hide <object>; see '--show'.
      --show   <object>          Show <object>; accepts any of:
                                    {{origin|frames|frames.displ|nodes|nodes.displ|extrude}}

  MISC.
  -o, --save   <out-file>        Save plot to <out-file>.
  -C, --conf   <conf-file>
  -c

  BACKEND
  --canvas <canvas>              trimesh, gnu, plotly, matplotlib

      --install                  Install script dependencies.
      --setup                    Run setup operations.
      --script {{sam|res}}
      --version                  Print version and exit.
  -h, --help                     Print this message and exit.



  <dof>        {{long | tran | vert | sect | elev | plan}}
               {{  0  |   1  |   2  |   3  |   4  |   5 }}
  <object>     {{origin|frames|frames.displ|nodes|nodes.displ}}
    origin
    frames
    nodes
    legend
    extrude                      extrude cross-sections
    outline                      outline extrusion
    triads
    x,y,z

    fibers
"""

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


def parse_args(argv)->dict:
    opts = config.Config()
    if os.path.exists(".render.yaml"):
        with open(".render.yaml", "r") as f:
            presets = yaml.load(f, Loader=yaml.Loader)

        config.apply_config(presets,opts)

    args = iter(argv[1:])
    for arg in args:
        try:
            if arg == "--help" or arg == "-h":
                print(HELP.format(NAME=NAME))
                sys.exit()


            elif arg == "--gnu":
                opts["plotter"] = "gnu"
            elif arg == "--plotly":
                opts["plotter"] = "plotly"
            elif arg == "--canvas":
                opts["plotter"] = next(args)

            elif arg == "--install":
                try: install_me(next(args))
                # if no directory is provided, use default
                except StopIteration: install_me()
                sys.exit()

            elif arg == "--version":
                print(__version__)
                sys.exit()

            elif arg[:2] == "-d":
                node_dof = arg[2:] if len(arg) > 2 else next(args)
                for nd in node_dof.split(","):
                    node, dof = nd.split(":")
                    opts["displ"][int(node)].append(dof_index(dof))

            elif arg[:6] == "--disp":
                node_dof = next(args)
                for nd in node_dof.split(","):
                    node, dof = nd.split(":")
                    opts["displ"][int(node)].append(dof_index(dof))

            elif arg == "--conf":
                with open(next(args), "r") as f:
                    presets = yaml.load(f, Loader=yaml.Loader)
                config.apply_config(presets,opts)


            elif arg[:2] == "-s":
                opts["scale"] = float(arg[2:]) if len(arg) > 2 else float(next(args))
            elif arg == "--scale":
                scale = next(args)
                if "=" in scale:
                    # looks like --scale <object>=<scale>
                    k,v = scale.split("=")
                    opts["objects"][k]["scale"] = float(v)
                else:
                    opts["scale"] = float(scale)

            elif arg == "--vert":
                opts["vert"] = int(next(args))

            elif arg == "--show":
                opts["show_objects"].extend(next(args).split(","))

            elif arg == "--hide":
                opts["show_objects"].pop(opts["show_objects"].index(next(args)))

            elif arg[:2] == "-V":
                opts["view"] = arg[2:] if len(arg) > 2 else next(args)
            elif arg == "--view":
                opts["view"] = next(args)

            elif arg == "--default-section":
                opts["default_section"] = np.loadtxt(next(args))

            elif arg[:2] == "-m":
                opts["mode_num"] = int(arg[2]) if len(arg) > 2 else int(next(args))

            elif arg == "--time":
                opts["time"] = next(args)

            elif arg[:2] == "-o":
                filename = arg[2:] if len(arg) > 2 else next(args)
                opts["write_file"] = filename
                if "html" in filename or "json" in filename:
                    opts["plotter"] = "plotly"

            # Final check on options
            elif arg[0] == "-" and len(arg) > 1:
                raise RenderError(f"ERROR - unknown option '{arg}'")

            elif not opts["sam_file"]:
                if arg == "-": arg = sys.stdin
                opts["sam_file"] = arg

            else:
                if arg == "-": arg = sys.stdin
                opts["res_file"] = arg

        except StopIteration:
            # `next(args)` was called in parse loop without successive arg
            raise RenderError(f"ERROR -- Argument '{arg}' expected value")

    return opts

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

if __name__ == "__main__":
    config = parse_args(sys.argv)

    try:
        sees.render(**config)

    except (FileNotFoundError, RenderError) as e:
        # Catch expected errors to avoid printing an ugly/unnecessary stack trace.
        print(e, file=sys.stderr)
        print("         Run '{NAME} --help' for more information".format(NAME=NAME), file=sys.stderr)
        sys.exit()

