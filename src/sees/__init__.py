import os
import sys
from pathlib import Path

from .config import Config, apply_config
from .frame import FrameArtist

class RenderError(Exception): pass

def Canvas(subplots=None, backend=None):
    pass

assets = Path(__file__).parents[0]/"assets/"

def render(sam_file, res_file=None, noshow=False, ndf=6,
           artist = None, #: str|"Artist" = None,
           **opts):

    import sees.model

    # Configuration is determined by successively layering
    # from sources with the following priorities:
    #      defaults < file configs < kwds 

    config = Config()

    if sam_file is None:
        raise RenderError("ERROR -- expected required argument <sam-file>")

    # Read model data
    if isinstance(sam_file, (str, Path)):
        model_data = sees.model.read_model(sam_file)

    elif hasattr(sam_file, "asdict"):
        # Assuming an opensees.openseespy.Model
        model_data = sam_file.asdict()

    elif hasattr(sam_file, "read"):
        model_data = sees.model.read_model(sam_file)

    elif isinstance(sam_file, tuple):
        # TODO: (nodes, cells)
        pass

    elif not isinstance(sam_file, dict):
        model_data = sees.model.read_model(sam_file)

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
    # Read and process displacements 
    #
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

