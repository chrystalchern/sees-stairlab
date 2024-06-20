import os
import sys
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

import numpy as np
Array = np.ndarray
from scipy.linalg import block_diag

from sees.state import read_state, State
from sees.config import Config, apply_config

from sees.model import FrameModel

# Data shaping / Misc.
#----------------------------------------------------

# The following functions are used for reshaping data
# and carrying out other miscellaneous operations.


def _is_basic_frame(el):
    return     "beam" in el["type"].lower() \
            or "dfrm" in el["type"].lower()

#   return "zero" not in el["type"].lower()



def get_section_geometries(model, config=None):
    if config is not None and "standard_section" in config:
        outline = {
            "square":  np.array([[-1., -1.],
                                 [ 1., -1.],
                                 [ 1.,  1.],
                                 [-1.,  1.]]),

            "tee":     np.array([[ 6.0,  0.0],
                                 [ 6.0,  4.0],
                                 [-6.0,  4.0],
                                 [-6.0,  0.0],
                                 [-2.0,  0.0],
                                 [-2.0, -8.0],
                                 [ 2.0, -8.0],
                                 [ 2.0,  0.0]])/10

        }[config["standard_section"]]

        config["default_section"] = outline
        return {
            elem["name"]: outline
            for elem in model["assembly"].values() if "sections" in elem
        }

    outlines = {}
    for name,section in model["sections"].items():
        get_section_shape(section, model["sections"], outlines)

    return {
        # TODO: For now, only using the first of the element's cross
        #       sections
        elem["name"]: np.array([outlines[s] for s in elem["sections"]][0])

        for elem in model["assembly"].values() if "sections" in elem
    }

# Kinematics
#----------------------------------------------------

# The following functions implement various kinematic
# relations for standard frame models.

def elastic_curve(x: Array, v: Array, L:float)->Array:
    "compute points along Euler's elastica"
    if len(v) == 2:
        ui, uj, (vi, vj) = 0.0, 0.0, v
    else:
        ui, vi, uj, vj = v
    xi = x/L                        # local coordinates
    N1 = 1.-3.*xi**2+2.*xi**3
    N2 = L*(xi-2.*xi**2+xi**3)
    N3 = 3.*xi**2-2*xi**3
    N4 = L*(xi**3-xi**2)
    y = ui*N1 + vi*N2 + uj*N3 + vj*N4
    return y.flatten()

def elastic_tangent(x: Array, v: Array, L: float)->Array:
    if len(v) == 2:
        ui, uj, (vi, vj) = 0.0, 0.0, v
    else:
        ui, vi, uj, vj = v
    xi = x/L
    M3 = 1 - xi
    M4 = 6/L*(xi-xi**2)
    M5 = 1 - 4*xi+3*xi**2
    M6 = -2*xi + 3*xi**2
    return (ui*M3 + vi*M5 + uj*M4 + vj*M6).flatten()

def orientation(xyz, yvec=None):
    """Create a rotation matrix between local E and global e basis
    """
    dx = xyz[-1] - xyz[0]
    L = np.linalg.norm(dx)
    e1 = dx/L
    e2 = np.array(yvec)
    v3 = np.cross(e1,e2)
    norm_v3 = np.linalg.norm(v3)

    e3 = v3 / norm_v3
    R = np.stack([e1,e2,e3])
    return R


def rotation(xyz: Array, vert=None)->Array:
    """Create a rotation matrix between local e and global E
    """
    if vert is None: vert = (0,0,1)
    dx = xyz[-1] - xyz[0]
    L = np.linalg.norm(dx)
    e1 = dx/L
    v13 = np.atleast_1d(vert)
    v2 = -np.cross(e1,v13)
    norm_v2 = np.linalg.norm(v2)
    if norm_v2 < 1e-8:
        v2 = -np.cross(e1,np.array([*reversed(vert)]))
        norm_v2 = np.linalg.norm(v2)
    e2 = v2 / norm_v2
    v3 =  np.cross(e1,e2)
    e3 = v3 / np.linalg.norm(v3)
    R = np.stack([e1,e2,e3])
    return R

def displaced_profile(
        coord: Array,
        displ: Array,        #: Displacements
        vect : Array = None, #: Element orientation vector
        npoints: int = 10,
        tangent: bool = False
    )->Array:
    n = npoints
    #           (------ndm------)
    reps = 4 if len(coord[0])==3 else 2

    # 3x3 rotation into local system
    Q = rotation(coord, vect)
    # Local displacements
    u_local = block_diag(*[Q]*reps)@displ
    # Element length
    L = np.linalg.norm(coord[-1] - coord[0])

    # longitudinal, transverse, vertical, section, elevation, plan
    li, ti, vi, si, ei, pi = u_local[:6]
    lj, tj, vj, sj, ej, pj = u_local[6:]

    Lnew = L + lj - li
    xaxis = np.linspace(0.0, Lnew, n)

    plan_curve = elastic_curve(xaxis, [ti, pi, tj, pj], Lnew)
    elev_curve = elastic_curve(xaxis, [vi,-ei, vj,-ej], Lnew)

    local_curve = np.stack([xaxis + li, plan_curve, elev_curve])

    if tangent:
        plan_tang = elastic_tangent(xaxis, [ti, pi, tj, pj], Lnew)
        elev_tang = elastic_tangent(xaxis, [vi,-ei, vj,-ej], Lnew)

        local_tang = np.stack([np.linspace(0,0,n), plan_tang, elev_tang])
        return (
            Q.T@local_curve + coord[0][None,:].T,
            Q.T@local_tang
        )

    return Q.T@local_curve + coord[0][None,:].T


class FrameArtist:
    ndm: int
    ndf: int
    model: "FrameModel"
    canvas: "Canvas"

    def __init__(self, model_data, response=None, ndf=None, loc=None, vert=2, **kwds):
        config = Config()
        if "config" in kwds:
            apply_config(kwds.pop("config"), config)

        apply_config(kwds, config)
        self.config = config

        canvas_type = config.get("canvas", "matplotlib")
        if not isinstance(canvas_type, str):
            self.canvas = canvas_type
        elif canvas_type == "matplotlib":
            import sees.canvas.mpl
            self.canvas = sees.canvas.mpl.MatplotlibCanvas(**config["canvas_config"])
        elif canvas_type == "femgl":
            import sees.canvas.femgl
            self.canvas = sees.canvas.femgl.FemGlCanvas(self.model, **config["canvas_config"])
        elif canvas_type == "plotly":
            import sees.canvas.ply
            self.canvas = sees.canvas.ply.PlotlyCanvas(**config["canvas_config"])
        elif canvas_type == "gltf":
            import sees.canvas.gltf
            self.canvas = sees.canvas.gltf.GltfLibCanvas(**config["canvas_config"])
        elif canvas_type == "trimesh":
            import sees.canvas.tri
            self.canvas = sees.canvas.tri.TrimeshCanvas(**config["canvas_config"])
        else:
            raise ValueError("Unknown canvas type " + str(canvas_type))

        self.canvas.config = config


        self.ndm = 3

        if ndf is None:
            ndf = 6

        elif ndf == 3:
            self.ndm = 2

        if vert == 3:
            R = np.eye(3)
        else:
            R = np.array(((1,0, 0),
                          (0,0,-1),
                          (0,1, 0)))

        self._plot_rotation = R

        self.model = FrameModel(model_data, shift=loc, rot=R, **kwds.get("model_config", {}))

        # Create permutation matrix
        if ndf == 3:
            P = np.array(((1,0, 0),
                          (0,1, 0),
                          (0,0, 0),

                          (0,0, 0),
                          (0,0, 0),
                          (0,0, 1)))
        else:
            P = np.eye(6)


        self.dofs2plot = block_diag(*[R]*2)@P

        self.displ_states = {} # defaultdict(lambda : np.zeros((len(self.model["nodes"]), 6)))


    def add_point_displacements(self, displ, scale=1.0, name=None):
        displ_array = self.displ_states[name]
        for i,n in enumerate(self.model["nodes"]):
            for dof in displ[n]:
                displ_array[i, dof] = 1.0

        displ_array[:,3:] *= scale/100
        displ_array[:,:3] *= scale
        return name

    def _add_displ_case(self, state, name=None, scale=1.0):

        if name in self.displ_states:
            self.displ_states[name].update(state)
        else:
            self.displ_states[name] = state
        return name


    def add_state(self, res_file, scale=1.0, only=None, type=None):

        if not isinstance(res_file, (dict, Array, State)):
            state = read_state(res_file, only=only,
                               model=self.model,
                               scale=scale,
                               transform=self.dofs2plot)
        else:
            state = res_file

        # Test type of first item in dict; if dict of dicts,
        # its a collection of responses, otherwise, just a
        # single response
        if isinstance(state, dict):
            for k, v in state.items():
                self._add_displ_case(v, name=k, scale=scale)

        else:
            self._add_displ_case(state, scale=scale)


    def plot_origin(self, **kwds):
        xyz = np.zeros((3,3))
        uvw = self._plot_rotation.T*kwds.get("scale", 1.0)
        off = [[0, -kwds.get("scale", 1.0)/2, 0],
               [0]*3,
               [0]*3] #kwds.get("scale", 1.0)/10
#       uvw = np.eye(3)*kwds.get("scale", 1.0)
        self.canvas.plot_vectors(xyz, uvw, **kwds)
        if hasattr(self.canvas, "annotate"):
            for i,label in enumerate(kwds.get("label", [])):
                self.canvas.annotate(label, (xyz+uvw)[i]+off[i])

    def add_elem_data(self):
        N = 3
        exclude_keys = {"type", "instances", "nodes", "crd", "crdTransformation"}

        if "prototypes" not in self.model:
            elem_types = defaultdict(lambda: defaultdict(list))
            for elem in self.model["assembly"].values():
                elem_types[elem["type"]]["elems"].append(elem["name"])
                elem_types[elem["type"]]["coords"].append(elem["crd"])
                elem_types[elem["type"]]["data"].append([
                    str(v) for k,v in elem.items() if k not in exclude_keys
                ])
                if "keys" not in elem_types[elem["type"]]:
                    elem_types[elem["type"]]["keys"] = [
                        k for k in elem.keys() if k not in exclude_keys
                    ]
        else:
            elem_types = {
                f"{elem['type']}<{elem['name']}>": {
                    "elems": [self.model["assembly"][i]["name"] for i in elem["instances"]],
                    "data":  [
                        [str(v) for k,v in elem.items() if k not in exclude_keys]
                        #for _ in range(len(elem["instances"]))
                    ]*(len(elem["instances"])),
                    "coords": [self.model["assembly"][i]["crd"] for i in elem["instances"]],
                    "keys":   [k for k in elem.keys() if k not in exclude_keys]

                } for elem in self.model["prototypes"]["elements"]
            }

        for name, elem in elem_types.items():
            coords = np.zeros((len(elem["elems"])*(N+1),self.ndm))
            coords.fill(np.nan)
            for i,crd in enumerate(elem["coords"]):
                coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(*crd, N)

            # coords = coords.reshape(-1,4,N)[:,-N]
            coords = coords.reshape(-1,4,3)[:,-3]

            x,y,z = coords.T
            keys  = elem["keys"]
            data  = np.array(elem["data"])

            # TODO: Make this nicer
            self.canvas.data.append({
                "name": name,
                "x": x, "y": y, "z": z,
                "type": "scatter3d", "mode": "markers", # "lines", #
                "hovertemplate": "<br>".join(f"{k}: %{{customdata[{v}]}}" for v,k in enumerate(keys)),
                "customdata": data,
                "opacity": 0.6 if "zerolength" in name.lower() and "zerolength" in self.config["show_objects"] else 0
                #"marker": {"opacity": 0.0,"size": 0.0, "line": {"width": 0.0}}
            })


    def plot_chords(self, assembly, state=None, layer=None):
        frame = self.model
        nodes = self.model["nodes"]
        ndm   = self.model["ndm"]

        N = 2
        coords = np.zeros((len(frame["assembly"])*(N+1),3))
        coords.fill(np.nan)

        for i,el in enumerate(frame["assembly"].values()):
            coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(el["crd"][0], el["crd"][-1], N)

            # exclude zero-length elements
            if "zero" not in el["type"].lower() and state is not None:
                coords[(N+1)*i:(N+1)*i+N,:] += [
                    state.node_array(n)[:ndm] for n in (el["nodes"][0], el["nodes"][-1])
                ]

        self.canvas.plot_lines(coords[:,:self.ndm])


    def plot_extruded_frames(self, state=None, rotations=None):
        from scipy.spatial.transform import Rotation

        sections = get_section_geometries(self.model, self.config)

        nodes = self.model["nodes"]
        # N = 2
        # N = 20 if displ is not None else 2
        N = 2 if state is not None else 2

        coords = []
        triang = []
        I = 0
        for i,el in enumerate(self.model["assembly"].values()):
            # if int(el["name"]) < 30: continue
            try:
                sect = sections[el["name"]]
            except:
                if int(el["name"]) < 1e3:
                    sect = self.config["default_section"]
                else:
                    sect = np.array([[-48, -48],
                                     [ 48, -48],
                                     [ 48,  48],
                                     [-48,  48]])

            ne = len(sect)
            R  = rotation(el["crd"], None)
            if state is not None:
                glob_displ = state.cell_array(tag).flatten()
                vect = None #np.array(el["trsfm"]["vecInLocXZPlane"])[axes]
                X,r = displaced_profile(el["crd"], glob_displ, vect=vect, npoints=N, tangent=True)
                X = X.T

                R = [Rotation.from_euler("xzy",v).as_matrix()@R for v in r.T]

            else:
                X  = np.linspace(el["crd"][0], el["crd"][-1], N)
                R = [R]*N

            for j in range(N):
                for k,edge in enumerate(sect):
                    coords.append(X[j  , :] + R[j].T@[0, *edge])
                    if j == 0:
                        continue

                    elif k < ne-1:
                        triang.extend([
                            [I+    ne*j + k, I+    ne*j + k + 1, I+ne*(j-1) + k],
                            [I+ne*j + k + 1, I+ne*(j-1) + k + 1, I+ne*(j-1) + k]
                        ])
                    else:
                        # elif j < N-1:
                        triang.extend([
                            [I+    ne*j + k,    I + ne*j , I+ne*(j-1) + k],
                            [      I + ne*j, I + ne*(j-1), I+ne*(j-1) + k]
                        ])

            I += N*ne

        x,y,z = zip(*coords)
        i,j,k = zip(*triang)
        self.canvas.data.append({
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

        show_edges = True
        if show_edges:
            IDX = np.array((
                (0, 2),
                (0, 1)
            ))
            nan = np.zeros(3)*np.nan
            coords = np.array(coords)
            tri_points = np.array([
                coords[idx]  if (j+1)%3 else nan for j,idx in enumerate(np.array(triang).reshape(-1))
                # coords[i]  if j%2 else nan for j,idx in enumerate(np.array(triang)) for i in idx[IDX[j%2]]
            ])
            Xe, Ye, Ze = tri_points.T
            self.canvas.data.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": Xe, "y": Ye, "z": Ze,
                "hoverinfo":"skip",
                "opacity": 1.0,
                "line": { "color": "black", "width": 5}
            })



    def plot_displaced_assembly(self, assembly, state=None, label=None):
        frame = self.model
        nodes = self.model["nodes"]
        N = 10 if state is not None else 2
        coords = np.zeros((len(frame["assembly"])*(N+1), 3))
        coords.fill(np.nan)

        for i,(tag,el) in enumerate(frame["assembly"].items()):
            # exclude zero-length elements
            if _is_basic_frame(el) and state is not None:
                glob_displ = state.cell_array(tag).flatten()
                vect = None #np.array(el["trsfm"]["vecInLocXZPlane"])[axes]
                coords[(N+1)*i:(N+1)*i+N,:] = displaced_profile(el["crd"], glob_displ, vect=vect, npoints=N).T
            elif len(el["crd"]) == 2:
                coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(*el["crd"], N)

        self.canvas.plot_lines(coords[:, :self.ndm], color="red", label=label)

    def plot_nodes(self, state=None, data=None):
        coord = self.model["coord"]
        if state is not None:
            coord = coord + state.node_array()[:, :3] #self.ndm]
        self.canvas.plot_nodes(coord[:,:self.ndm], data=data)

    def add_triads(self):
        ne = len(self.model["assembly"])
        xyz, uvw = np.nan*np.zeros((2, ne, 3, 3))

        for i,el in enumerate(self.model["assembly"].values()):
            axes = self.model.frame_orientation(el["name"]) #el["crd"], el["trsfm"]["yvec"])
            if axes is None:
                continue

            scale = np.linalg.norm(el["crd"][-1] - el["crd"][0])/10
            coord = sum(i for i in el["crd"])/len(el["nodes"])
            xyz[i,:,:] = np.array([coord]*3)
            uvw[i,:,:] = scale*axes

#           self.canvas.plot_vectors(np.array([coord]*3), scale*rotation(el["crd"], el["trsfm"]["yvec"]))
        self.canvas.plot_vectors(xyz.reshape(ne*3,3), uvw.reshape(ne*3,3))

#   def plot_frame_axes(self):
#       for elem in self.model["assembly"].values():
#           xyz = (elem["crd"][-1] + elem["crd"][0])/2
#           uvw = np.eye(3)/np.linalg.norm(elem["crd"][-1] - elem["crd"][0])
#           self.canvas.plot_vectors(xyz, uvw)

#   def add_elem_data(self):
#       N = 3
#       coords = np.zeros((len(self.model["assembly"])*(N+1),self.ndm))
#       coords.fill(np.nan)
#       for i,el in enumerate(self.model["assembly"].values()):
#           coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(*el["crd"], N)

#       coords = coords.reshape(-1,4,3)[:,-3]

#       x,y,z = coords.T
#       keys  = ["tag",]
#       frames = np.array(list(self.model["assembly"].keys()),dtype=FLOAT)[:,None]
#       try:
#           # TODO: Make this nicer
#           self.canvas.data.append({
#                   "name": "frames",
#                   "x": x, "y": y, "z": z,
#                   "type": "scatter3d","mode": "markers",
#                   "hovertemplate": "<br>".join(f"{k}: %{{customdata[{v}]}}" for v,k in enumerate(keys)),
#                   "customdata": frames,
#                   "opacity": 0
#                   #"marker": {"opacity": 0.0,"size": 0.0, "line": {"width": 0.0}}
#           })
#       except:
#           pass


    def draw(self):
        if "frames" in self.config["show_objects"]:
            self.plot_chords(self.model["assembly"])
            try:
                self.add_elem_data()
            except:
                pass

        if "nodes" in self.config["show_objects"]:
#           self.plot_nodes(data=list(np.array(list(self.model["nodes"].keys()),dtype=str)[:,None]))
            self.plot_nodes(data=[[str(k), list(map(str, n["crd"]))] for k,n in self.model["nodes"].items()])

        if "origin" in self.config["show_objects"]:
            self.plot_origin(**self.config["objects"]["origin"])

        if "triads" in self.config["show_objects"]:
            self.add_triads()

        for layer, state in self.displ_states.items():
            if "chords" in self.config["show_objects"]:
                self.plot_chords(self.model["assembly"], state=state, layer=layer)
            self.plot_displaced_assembly(self.model["assembly"], state=state, label=layer)

        if "extrude" in self.config["show_objects"]:
            displ = None
            if len(self.displ_states) == 1:
                displ = next(iter(self.displ_states.values()))
            try:
                self.plot_extruded_frames(displ)
            except Exception as e:
                raise e
                print("Warning -- ", e, file=sys.stderr)

        self.canvas.build()
        return self

    def write(self, filename):
        self.canvas.write(filename)

    def repl(self):
        from opensees.repl.__main__ import OpenSeesREPL
        self.canvas.plt.ion()

        try:
            from IPython import get_ipython
            get_ipython().run_magic_line('matplotlib')
        except:
            pass

        repl = OpenSeesREPL()

        def plot(*args):
            if len(args) == 0:
                return self.plot()

            elif hasattr(self, "plot_"+args[0]):
                return getattr(self, "plot_"+args[0])(*args[1:])

            elif hasattr(self, args[0]):
                return getattr(self, args[0])(*args[1:])

        repl.interp._interp.createcommand("plot", plot)
        # repl.interp._interp.createcommand("show", lambda *args: self.canvas.show())
        repl.repl()


