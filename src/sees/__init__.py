import os
import sys
from pathlib import Path
try:
    import orjson as json
except ImportError:
    import json

import numpy as np
Array = np.ndarray
FLOAT = np.float32

from .config import Config, apply_config
from .frame import FrameArtist


Frame, Patch, Plane, Solid = range(4)

# Data shaping / Misc.
#----------------------------------------------------

# The following functions are used for reshaping data
# and carrying out other miscellaneous operations.

class RenderError(Exception): pass

def Canvas(subplots=None, backend=None):
    pass

def read_model(filename:str, shift=None)->dict:

    if isinstance(filename, str) and filename.endswith(".tcl"):
        import opensees.tcl
        with open(filename, "r") as f:
            interp = opensees.tcl.exec(f.read(), silent=True, analysis=False)
        return interp.serialize()

    try:
        with open(filename,"r") as f:
            sam = json.loads(f.read())

    except TypeError:
        sam = json.loads(filename.read())

    return sam


def render(sam_file, res_file=None, noshow=False, ndf=6,
           artist = None, #: str|"Artist" = None,
           **opts):

    # Configuration is determined by successively layering
    # from sources with the following priorities:
    #      defaults < file configs < kwds 

    config = Config()

    if sam_file is None:
        raise RenderError("ERROR -- expected required argument <sam-file>")

    # Read model data
    if isinstance(sam_file, (str, Path)):
        model_data = read_model(sam_file)

    elif hasattr(sam_file, "asdict"):
        # opensees.openseespy.Model
        model_data = sam_file.asdict()

    elif hasattr(sam_file, "read"):
        model_data = read_model(sam_file)

    elif not isinstance(sam_file, dict):
        model_data = read_model(sam_file)

    elif isinstance(sam_file, tuple):
        # (nodes, cells)
        pass
    else:
        model_data = sam_file

    if "RendererConfiguration" in model_data:
        apply_config(model_data["RendererConfiguration"], config)

    apply_config(opts, config)

    #
    # Create Artist
    #
    # A Model is created from model_data by the artist
    # so that the artist can inform it how to transform
    # things if neccessary.
    if artist is None:
        artist = "frame"

    if artist == "frame":
        artist = FrameArtist(model_data, ndf=ndf, **config)


    #
    # Read and clean displacements 
    # TODO: 
    # - remove `cases` var, 
    # - change add_state from being a generator
    # rename `name` parameter
    if res_file is not None:
        artist.add_state(res_file,
                           scale=config["scale"],
                           only=config["mode_num"])

    elif config["displ"] is not None:
        pass
        # TODO: reimplement point displacements
        # cases = [artist.add_point_displacements(config["displ"], scale=config["scale"])]

    if "Displacements" in model_data:
        cases.extend(artist.add_state(model_data["Displacements"],
                                        scale=config["scale"],
                                        only=config["mode_num"]))


    return artist

