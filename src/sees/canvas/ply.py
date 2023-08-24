# Claudio Perez
from .canvas import Canvas
import numpy as np

class PlotlyCanvas(Canvas):
    def __init__(self, config=None):
        self.data = []
        self.config = config
        self.annotations = []

    def show(self):
        self.fig.show(renderer="browser")

    def build(self):
        opts = self.config
        show_axis = "axis" in opts["show_objects"]
        show_grid = "grid" in opts["show_objects"]
        import plotly.graph_objects as go
        fig = go.Figure(dict(
                data=self.data,
                layout=go.Layout(
                  scene=dict(aspectmode="data",
                     xaxis = {"showspikes": "tick" in opts["show_objects"],
                              "showbackground": show_grid, "gridcolor": "gray" if show_grid else None},
                     yaxis = {"showspikes": "tick" in opts["show_objects"],
                              "showbackground": show_grid, "gridcolor": "gray" if show_grid else None},
                     zaxis = {"showspikes": "tick" in opts["show_objects"],
                              "showbackground": show_grid, "gridcolor": "gray" if show_grid else None},
                     xaxis_title = r"$\mathbf{E}_1$",
                     xaxis_visible = "x" in opts["show_objects"],#show_axis,
                     yaxis_visible = "y" in opts["show_objects"],#show_axis,
                     zaxis_visible = "z" in opts["show_objects"],#show_axis,
                     camera=dict(
                         projection={"type": opts["camera"]["projection"]}
                     ),
                     annotations=self.annotations
                  ),
#                 showlegend=("legend" in opts["show_objects"])
                )
            ))
        fig.update(layout_coloraxis_showscale=False)
        self.fig = fig
        return self

    def write(self, filename=None):
        opts = self.config
        if "html" in filename:
            import plotly
            fig = self.fig
            html = plotly.io.to_html(fig, div_id=str(id(self)), **opts["save_options"]["html"])
            with open(opts["write_file"],"w+") as f:
                f.write(html)
        elif "json" in opts["write_file"]:
            with open(opts["write_file"],"w+") as f:
                self.fig.write_json(f)

    def make_hover_data(self, data, ln=None):
        if ln is None:
            items = np.array([d.values for d in data])
            keys = data[0].keys()
        else:
            items = np.array([list(data.values())]*ln)
            keys = data.keys()
        return {
            "hovertemplate": "<br>".join(f"{k}: %{{customdata[{v}]}}" for v,k in enumerate(keys)),
            "customdata": list(items),
        }


    def plot_nodes(self, coords, label = None, props=None, data=None):
        name = label or "nodes"
        x,y,z = coords.T
        keys  = ["tag","crd"]

        data = {
                "name": name,
                "x": x, "y": y, "z": z,
                "type": "scatter3d","mode": "markers",
                "hovertemplate": "<br>".join(f"{k}: %{{customdata[{v}]}}" for v,k in enumerate(keys)),
                "customdata": data,
                "marker": {
                    "symbol": "square",
                    **self.config["objects"]["nodes"]["default"]
                },
                "showlegend": False
        }
        self.data.append(data)

    def plot_lines(self, coords, label=None, props=None, color=None, width=None):
        x,y,z = coords.T
        props = {"color": color or "#808080", "alpha": 0.6}
        data = {
            "name": label if label is not None else "",
            "type": "scatter3d",
            "mode": "lines",
            "x": x, "y": y, "z": z,
            "line": {"color": props["color"] if color is None else color, "width": width},
            "hoverinfo":"skip"
        }
        self.data.append(data)

    def annotate(self, text, coords, **kwds):
        self.annotations.append({
            "text": text,
            "showarrow": False,
            "xanchor": "left",
            "yshift": -10,
            "xshift":   5,
            "font": {"color": "black", "size": 15},
            "x": coords[0], "y": coords[1], "z": coords[2],
            **kwds
        })

    def plot_vectors(self, locs, vecs, **kwds):

        ne = vecs.shape[0]
        for j in range(3):
            X = np.zeros((ne*3, 3))*np.nan
            for i in range(j,ne,3):
                X[i*3,:] = locs[i]
                X[i*3+1,:] = locs[i] + vecs[i]

            color = kwds.get("color", ("red", "blue", "green")[j])

            # _label = label if label is not None else ""
            label = kwds.get("label", "")
            if isinstance(label, list):
                label = label[j]
            self.data.append({
                "name": label,
                "type": "scatter3d",
                "mode": "lines",
                "x": X.T[0], "y": X.T[1], "z": X.T[2],
                "line": {"color": color, "width": 4},
                "hoverinfo":"skip",
                "showlegend": False
            })

    def plot_mesh(self, vertices, triangles):
        x,y,z = zip(*vertices)
        i,j,k = zip(*triangles)
        self.data.append({
            #"name": label if label is not None else "",
            "type": "mesh3d",
            "x": x, "y": y, "z": z, "i": i, "j": j, "k": k,
            "hoverinfo":"skip",
            "lighting": dict(ambient=0.9), #roughness=0.5, specular=0.2),
            "opacity": 1.0,
            "color": "gray",
            # "color": "white",
            # "opacity": 0.2
        })
