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

#import sees
from sees import render
from sees.parser import parse_args
from sees.errors import RenderError


NAME="sees"

def main(argv):

    try:
        config = parse_args(argv)

        if config is None:
            sys.exit()

        artist = render(**config)

        # write plot to file if output file name provided
        if config["write_file"]:
            artist.save(config["write_file"])
            return

        # Otherwise either create popup, or start server
        elif hasattr(artist.canvas, "popup"):
            artist.canvas.popup()

        elif hasattr(artist.canvas, "to_glb"):
            import sees.server
            viewer = config["viewer_config"].get("name", None)
            port = config["server_config"].get("port", None)
            server = sees.server.Server(glb=artist.canvas.to_glb(),
                                        viewer=viewer)
            server.run(port=port)

        elif hasattr(artist.canvas, "to_html"):
            import sees.server
            port = config["server_config"].get("port", None)
            server = sees.server.Server(html=artist.canvas.to_html())
            server.run(port=port)

    except (FileNotFoundError, RenderError) as e:
        # Catch expected errors to avoid printing an ugly/unnecessary stack trace.
        print(f"ERROR - {e}", file=sys.stderr)
        print("         Run '{NAME} --help' for more information".format(NAME=NAME), file=sys.stderr)
        sys.exit(-1)


if __name__ == "__main__":
    main(sys.argv)

