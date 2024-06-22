# Claudio Perez
import numpy as np
from dataclasses import dataclass
import warnings

@dataclass 
class DrawStyle:
    color: str

@dataclass
class LineStyle:
    color: str   = "black"
    alpha: float = 1.0
    width: float = 1

@dataclass
class NodeStyle:
    color: str   = "black"
    scale: float = 1.0

@dataclass
class MeshStyle:
    color: str   = "gray"
    alpha: float = 1.0
    edges: LineStyle = None



class Canvas:
    def build(self): ...

    def write(self, filename=None):
        raise NotImplementedError

    def annotate(self, *args, **kwds): ...

    def plot_nodes(self, vertices, indices=None, label=None, style: NodeStyle=None, data=None):
        warnings.warn("plot_lines not implemented for chosen canvas")

    def plot_lines(self, vertices, indices=None, label=None, style: LineStyle=None):
        warnings.warn("plot_lines not implemented for chosen canvas")

    def plot_mesh(self,  vertices, indices     , label=None, style: MeshStyle=None, local_coords=None):
        warnings.warn("plot_mesh not implemented for chosen canvas")

    def plot_vectors(self, locs, vecs, label=None, **kwds):
        ne = vecs.shape[0]
        for j in range(3):
            X = np.zeros((ne*3, 3))*np.nan
            for i in range(j,ne,3):
                X[i*3,:] = locs[i]
                X[i*3+1,:] = locs[i] + vecs[i]
            self.plot_lines(X, color=("red", "blue", "green")[j], label=label)


