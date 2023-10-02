import os
import sys
from collections import defaultdict
try:
    import orjson as json
except ImportError:
    import json

import yaml
import numpy as np
Array = np.ndarray
FLOAT = np.float32
from scipy.linalg import block_diag

from .config import Config, apply_config


# Data shaping / Misc.
#----------------------------------------------------

# The following functions are used for reshaping data
# and carrying out other miscellaneous operations.

class RenderError(Exception): pass

def clean_model(sam:dict, shift: Array = None, rot=None)->dict:
    """
    Process OpenSees JSON output and return dict with the form:

        {<elem tag>: {"crd": [<coordinates>], ...}}
    """
    try:
        sam = sam["StructuralAnalysisModel"]
    except KeyError:
        pass

    ndm = 3
    R = np.eye(ndm) if rot is None else rot

    geom = sam.get("geometry", sam.get("assembly"))
    if shift is None:
        shift = np.zeros(ndm)
    else:
        shift = np.asarray(shift)
    try:
        #coord = np.array([R@n.pop("crd") for n in geom["nodes"]], dtype=FLOAT) + shift
        coord = np.array([R@n["crd"] for n in geom["nodes"]], dtype=FLOAT) + shift
    except:
        coord = np.array([R@[*n.pop("crd"), 0.0] for n in geom["nodes"]], dtype=FLOAT) + shift

    nodes = {
            n["name"]: {**n, "crd": coord[i], "idx": i}
                for i,n in enumerate(geom["nodes"])
    }

    ndm = len(next(iter(nodes.values()))["crd"])


    try:
        trsfm = {t["name"]: t for t in sam["properties"]["crdTransformations"]}
    except KeyError:
        trsfm = {}

    elems =  {
      e["name"]: dict(
        **e,
        crd=np.array([nodes[n]["crd"] for n in e["nodes"]], dtype=FLOAT),
        trsfm=trsfm[e["crdTransformation"]]
            if "crdTransformation" in e and e["crdTransformation"] in trsfm
            else dict(yvec=R@e["yvec"] if "yvec" in e else None)
      ) for e in geom["elements"]
    }

    try:
        sections = {s["name"]: s for s in sam["properties"]["sections"]}
    except:
        sections = {}

    output = dict(nodes=nodes, assembly=elems, coord=coord, sam=sam, sections=sections, ndm=ndm)

    if "prototypes" in sam:
        output.update({"prototypes": sam["prototypes"]})

    return output

def get_section_shape(section, sections=None, outlines=None):
    from scipy.spatial import ConvexHull

    # Rotation to change coordinates from x-y to z-y
    R = np.array(((0,-1),
                  (1, 0))).T

    # Treat aggregated sections
    if "section" in section:
        if section["section"] not in outlines:
            outlines[section["name"]] = get_section_shape(sections[section["section"]], sections, outlines)
        else:
            outlines[section["name"]] = outlines[section["section"]]

    elif "bounding_polygon" in section:
        outlines[section["name"]] = [R@s for s in section["bounding_polygon"]]

    elif "fibers" in section:
        #outlines[section["name"]] = _alpha_shape(np.array([f["coord"] for f in section["fibers"]]))
        points = np.array([f["coord"] for f in section["fibers"]])
        outlines[section["name"]] = points[ConvexHull(points).vertices]

    return outlines[section["name"]]

def get_section_geometries(model, renderer=None):

    sections, outlines = {}, {}
    for name,section in model["sections"].items():
        get_section_shape(section, model["sections"], outlines)

    return {
        # TODO: For now, only using the first of the element's cross
        #       sections
        elem["name"]: np.array([outlines[s] for s in elem["sections"]][0])

        for elem in model["assembly"].values() if "sections" in elem
    }

def read_displacements(res_file):

    if not isinstance(res_file, str):
        return yaml.load(res_file, Loader=yaml.Loader)

    from urllib.parse import urlparse
    res_path = urlparse(res_file)
    if "json" in res_path[2]:
        with open(res_path[2], "r") as f:
            res = json.loads(f.read())
    else:
        with open(res_path[2], "r") as f:
            res = yaml.load(f, Loader=yaml.Loader)
    if res_path[4]: # query parameters passed
        res = res[int(res_path[4].split("=")[-1])]

    return res




def read_model(filename:str, shift=None)->dict:

    if isinstance(filename, str) and filename.endswith(".tcl"):
        import opensees.tcl
        with open(filename, "r") as f:
            interp = opensees.tcl.eval(f.read(), silent=True, analysis=False)
        return interp.serialize()

    try:
        with open(filename,"r") as f:
            sam = json.loads(f.read())
    except TypeError:
        sam = json.loads(filename.read())
    return sam


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
    return (ui*M3+vi*M5+uj*M4+vj*M6).flatten()

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


class SkeletalRenderer:
    def __init__(self, model, response=None, ndf=None, loc=None, vert=2, **kwds):
        self.ndm = 3

        if ndf is None: ndf = 6

        if vert == 3:
            R = np.eye(3)
        else:
            R = np.array(((1,0, 0),
                          (0,0,-1),
                          (0,1, 0)))

        self.plot_rotation = R

        self.model = clean_model(model, shift=loc, rot=R)

        # Create permutation matrix
        if self.model["ndm"] == 2:
            P = np.array(((1,0, 0),
                          (0,1, 0),
                          (0,0, 0),

                          (0,0, 0),
                          (0,0, 0),
                          (0,0, 1)))
        else:
            P = np.eye(6)

        self.dofs2plot = block_diag(*[R]*2)@P


        self.response_layers = defaultdict(lambda : np.zeros((len(self.model["nodes"]), ndf)))


        config = Config()
        if "config" in kwds:
            apply_config(kwds.pop("config"), config)
        apply_config(kwds, config)
        self.config = config


        plotter = config.get("plotter", "matplotlib")
        if plotter == "matplotlib":
            import sees.canvas.mpl
            self.canvas = sees.canvas.mpl.MatplotlibCanvas()
        elif plotter == "plotly":
            import sees.canvas.ply
            self.canvas = sees.canvas.ply.PlotlyCanvas()
        elif plotter == "gltf":
            import sees.canvas.gltflib
            self.canvas = sees.canvas.gltflib.GltfLibCanvas()
        elif plotter == "trimesh":
            import sees.canvas.tri
            self.canvas = sees.canvas.tri.TrimeshCanvas()
        else:
            raise ValueError("Unknown plotter " + str(plotter))

        self.canvas.config = config

    def add_point_displacements(self, displ, scale=1.0, name=None):
        displ_array = self.response_layers[name]
        for i,n in enumerate(self.model["nodes"]):
            for dof in displ[n]:
                displ_array[i, dof] = 1.0

        displ_array[:,3:] *= scale/100
        displ_array[:,:3] *= scale
        return name

    def add_displacement_case(self, displ, name=None, scale=1.0):
        tol = 1e-14
        displ_array = self.response_layers[name]

        for i,n in enumerate(self.model["nodes"]):
            try:
                displ_array[i,:] = self.dofs2plot@displ[n]
            except KeyError:
                pass

        # apply cutoff
        displ_array[np.abs(displ_array) < tol] = 0.0

        # apply scale
        displ_array *= scale
        return name


    def add_displacements(self, res_file, scale=1.0, name=None):
        model = self.model

        if not isinstance(res_file, (dict, Array)):
            displ = read_displacements(res_file)
        else:
            displ = res_file

        # Test type of first item in dict; if dict of dicts,
        # its a collection of responses, otherwise, just a
        # single response
        if isinstance(next(iter(displ.values())), dict):
            if name is not None:
                yield self.add_displacement_case(displ[name], name=name, scale=scale)
            else:
                for k, v in displ.items():
                    yield self.add_displacement_case(v, name=k, scale=scale)
        else:
            yield self.add_displacement_case(displ, scale=scale)


    def label_nodes(self): ...


    def plot_origin(self, **kwds):
        xyz = np.zeros((3,3))
        uvw = np.eye(3)*kwds.get("scale", 1.0)
        self.canvas.plot_vectors(xyz, uvw, **kwds)
        if hasattr(self.canvas, "annotate"):
            for i,label in enumerate(kwds.get("label", [])):
                self.canvas.annotate(label, (xyz+uvw)[i])

#   def plot_frame_axes(self):
#       for elem in self.model["assembly"].values():
#           xyz = (elem["crd"][-1] + elem["crd"][0])/2
#           uvw = np.eye(3)/np.linalg.norm(elem["crd"][-1] - elem["crd"][0])
#           self.canvas.plot_vectors(xyz, uvw)

    def add_elem_data(self):
        N = 3
        coords = np.zeros((len(self.model["assembly"])*(N+1),self.ndm))
        coords.fill(np.nan)
        for i,el in enumerate(self.model["assembly"].values()):
            coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(*el["crd"], N)

        coords = coords.reshape(-1,4,3)[:,-3]

        x,y,z = coords.T
        keys  = ["tag",]
        frames = np.array(list(self.model["assembly"].keys()),dtype=FLOAT)[:,None]
        try:
            # TODO: Make this nicer
            self.canvas.data.append({
                    "name": "frames",
                    "x": x, "y": y, "z": z,
                    "type": "scatter3d","mode": "markers",
                    "hovertemplate": "<br>".join(f"{k}: %{{customdata[{v}]}}" for v,k in enumerate(keys)),
                    "customdata": frames,
                    "opacity": 0
                    #"marker": {"opacity": 0.0,"size": 0.0, "line": {"width": 0.0}}
            })
        except:
            pass

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

    def plot_chords(self, assembly, displ=None, layer=None):
        frame = self.model
        nodes = self.model["nodes"]
        ndm   = self.model["ndm"]

        N = 2
        coords = np.zeros((len(frame["assembly"])*(N+1),self.ndm))
        coords.fill(np.nan)

        for i,el in enumerate(frame["assembly"].values()):
            coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(el["crd"][0], el["crd"][-1], N)

            # exclude zero-length elements
            if "zero" not in el["type"].lower() and displ is not None:
                coords[(N+1)*i:(N+1)*i+N,:] += [
                    displ[nodes[n]["name"]][:ndm] for n in (el["nodes"][0], el["nodes"][-1])
                ]

        self.canvas.plot_lines(coords)


    def plot_extruded_frames(self, displ=None, rotations=None):
        from scipy.spatial.transform import Rotation

        sections = get_section_geometries(self.model, self)

        nodes = self.model["nodes"]
        # N = 2
        # N = 20 if displ is not None else 2
        N = 2 if displ is not None else 2

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
            if displ is not None:
                glob_displ = [
                    u for n in el["nodes"]
                        for u in displ[nodes[n]["idx"]]
                ]
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
            nan = np.zeros(self.ndm)*np.nan
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



    def plot_displaced_assembly(self, assembly, displ=None, label=None):
        frame = self.model
        nodes = self.model["nodes"]
        N = 10 if displ is not None else 2
        coords = np.zeros((len(frame["assembly"])*(N+1),self.ndm))
        coords.fill(np.nan)

        for i,el in enumerate(frame["assembly"].values()):
            # exclude zero-length elements
            if "zero" not in el["type"].lower() and displ is not None:
                glob_displ = [
                    u for n in el["nodes"]
                        for u in displ[nodes[n]["idx"]]
                ]
                vect = None #np.array(el["trsfm"]["vecInLocXZPlane"])[axes]
                coords[(N+1)*i:(N+1)*i+N,:] = displaced_profile(el["crd"], glob_displ, vect=vect, npoints=N).T
            else:
                coords[(N+1)*i:(N+1)*i+N,:] = np.linspace(*el["crd"], N)

        self.canvas.plot_lines(coords, color="red", label=label)

    def plot_nodes(self, displ=None, data=None):
        coord = self.model["coord"]
        if displ is not None:
            coord = coord + displ[:, :self.ndm]
        self.canvas.plot_nodes(coord, data=data)

    def add_triads(self, displ=None):
        ne = len(self.model["assembly"])
        xyz, uvw = np.nan*np.zeros((2, ne, 3, 3))

        for i,el in enumerate(self.model["assembly"].values()):
            scale = np.linalg.norm(el["crd"][-1] - el["crd"][0])/10
            coord = sum(i for i in el["crd"])/len(el["nodes"])
            xyz[i,:,:] = np.array([coord]*3)
            uvw[i,:,:] = scale*orientation(el["crd"], el["trsfm"]["yvec"])
#           self.canvas.plot_vectors(np.array([coord]*3), scale*rotation(el["crd"], el["trsfm"]["yvec"]))
        self.canvas.plot_vectors(xyz.reshape(ne*3,3), uvw.reshape(ne*3,3))


    def plot(self):
        if "frames" in self.config["show_objects"]:
            self.plot_chords(self.model["assembly"])
            try:
                self.add_elem_data()
            except:
                pass

        if "nodes" in self.config["show_objects"]:
            self.plot_nodes(data=list(np.array(list(self.model["nodes"].keys()),dtype=FLOAT)[:,None]))

        if "origin" in self.config["show_objects"]:
            self.plot_origin(**self.config["objects"]["origin"])

        if "triads" in self.config["show_objects"]:
            self.add_triads()

        for layer, displ in self.response_layers.items():
            if "chords" in self.config["show_objects"]:
                self.plot_chords(self.model["assembly"], displ=displ, layer=layer)
            self.plot_displaced_assembly(self.model["assembly"], displ=displ, label=layer)

        if "extrude" in self.config["show_objects"]:
            displ = None
            if len(self.response_layers) == 1:
                displ = next(iter(self.response_layers.values()))
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



def render(sam_file, res_file=None, noshow=False, **opts):
    # Configuration is determined by successively layering
    # from sources with the following priorities:
    #      defaults < file configs < kwds 

    config = Config()

    if sam_file is None:
        raise RenderError("ERROR -- expected required argument <sam-file>")

    # Read and clean model
    if not isinstance(sam_file, dict):
        model = read_model(sam_file)
    else:
        model = sam_file

    if "RendererConfiguration" in model:
        apply_config(model["RendererConfiguration"], config)

    apply_config(opts, config)

    renderer = SkeletalRenderer(model, **config)

    # Read and clean displacements 
    # TODO: 
    # - remove `cases` var, 
    # - change add_displacements from being a generator
    # rename `name` parameter
    if res_file is not None:
        cases = renderer.add_displacements(res_file, scale=config["scale"],
                                           name=config["mode_num"])
        list(cases)

    elif config["displ"] is not None:
        cases = [renderer.add_point_displacements(config["displ"], scale=config["scale"])]

    if "Displacements" in model:
        cases.extend(renderer.add_displacements(model["Displacements"],scale=config["scale"],
                                                name=config["mode_num"]))


    # write plot to file if file name provided
    if config["write_file"]:
        renderer.plot()
        renderer.write(config["write_file"])

    else:
        renderer.plot()
        if not noshow:
            renderer.canvas.show()
        # renderer.repl()

    return renderer

