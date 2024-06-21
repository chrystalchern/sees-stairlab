# Claudio Perez
import numpy as np
import warnings

class Canvas:
    def build(self): ...

    def write(self, filename=None):
        raise NotImplementedError

    def annotate(self, *args, **kwds): ...

    def plot_nodes(self, coords, label = None, props=None, data=None):
        warnings.warn("plot_lines not implemented for chosen canvas")

    def plot_lines(self, coords, label=None, props=None, color=None, width=None):
        warnings.warn("plot_lines not implemented for chosen canvas")

    def plot_mesh(self,vertices, triangles, **kwds):
        warnings.warn("plot_mesh not implemented for chosen canvas")

    def plot_vectors(self, locs, vecs, label=None, **kwds):
        ne = vecs.shape[0]
        for j in range(3):
            X = np.zeros((ne*3, 3))*np.nan
            for i in range(j,ne,3):
                X[i*3,:] = locs[i]
                X[i*3+1,:] = locs[i] + vecs[i]
            self.plot_lines(X, color=("red", "blue", "green")[j], label=label)


