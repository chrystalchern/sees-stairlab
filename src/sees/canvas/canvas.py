# Claudio Perez
import numpy as np
class Canvas:
    def build(self): ...
    def show(self): ...
    def write(self, filename=None): ...
    def plot_nodes(self, coords, label = None, props=None, data=None): ...

    def plot_lines(self, coords, label=None, props=None, color=None, width=None): ...

    def plot_vectors(self, locs, vecs, label=None, **kwds):
        ne = vecs.shape[0]
        for j in range(3):
            X = np.zeros((ne*3, 3))*np.nan
            for i in range(j,ne,3):
                X[i*3,:] = locs[i]
                X[i*3+1,:] = locs[i] + vecs[i]
            self.plot_lines(X, color=("red", "blue", "green")[j], label=label)

    def plot_mesh(self,vertices, triangles, **kwds): ...

