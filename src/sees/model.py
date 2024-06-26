# Claudio Perez
import warnings
from collections import defaultdict

import numpy as np

try:
    import orjson as json
except ImportError:
    import json

_EYE3 = np.eye(3)

_OUTLINES = {
    None:      None,
    "square":  np.array([[-1., -1.],
                         [ 1., -1.],
                         [ 1.,  1.],
                         [-1.,  1.]])/10,

    "tee":     np.array([[ 6.0,  0.0],
                         [ 6.0,  4.0],
                         [-6.0,  4.0],
                         [-6.0,  0.0],
                         [-2.0,  0.0],
                         [-2.0, -8.0],
                         [ 2.0, -8.0],
                         [ 2.0,  0.0]])/10
}



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


class Model:
    def __iter__(self):
        # this method allows: nodes, cells = Model(mesh)
        return iter((self.nodes, self.cells))

    @property
    def nodes(self)->dict:
        pass

    @property
    def cells(self)->dict:
        pass

    def iter_nodes(self): ...

    def node_location(self, tag): ...

    def node_information(self, tag): ...

    def iter_cells(self, filt=None): ...

    def cell_type(self, tag):       ... # line triangle quadrilateral 

    def cell_exterior(self, tag):   ...

    def cell_interior(self, tag):   ...

    def cell_rotation(self, tag, state): ...

    def cell_position(self, tag, state):   ...

    def cell_outline(self,  tag):   ...

    def cell_information(self, tag): ...



class FrameModel:
    def __init__(self, sam:dict, shift = None, rot=None, **kwds):

        self._frame_outlines = None
        self._extrude_default = _OUTLINES[kwds.get("extrude_default", "square")]
        self._extrude_outline = _OUTLINES[kwds.get("extrude_outline", None)]

        ndm = 3
        R = np.eye(ndm) if rot is None else rot

        if shift is None:
            shift = np.zeros(ndm)
        else:
            shift = np.asarray(shift)

        #
        self._data = _from_opensees(sam, shift, R)# output

        self.ndm = self._data["ndm"]
        self.ndf = self._data["ndf"]

    def __getitem__(self, key):
        return self._data[key]

    def cell_nodes(self, tag=None):
        if tag is None:
            if not hasattr(self, "_cell_nodes"):
                self._cell_nodes = {k: e["nodes"] for k, e in self["assembly"].items()}
            return self._cell_nodes
        else:
            return self["assembly"][tag]["nodes"]

    def cell_indices(self, tag=None):
        if not hasattr(self, "_cell_indices"):
            self._cell_indices = {
                elem["name"]: tuple(self.node_indices(n) for n in elem["nodes"])
                for elem in self["assembly"].values()
            }

        if tag is not None:
            return self._cell_indices[tag]
        else:
            return self._cell_indices

    def cell_properties(self, tag=None):
        if tag is not None:
            return self["assembly"][tag]

    def cell_prototypes(self)->"iter":
        exclude_keys = {"type", "instances", "nodes",
                        "crd", "crdTransformation"}

        if not self["prototypes"]:
            elem_types = defaultdict(dict)

            for elem in self["assembly"].values():
                type = elem["type"]
                if type not in elem_types:
                    elem_types[type]["name"] = type
                    elem_types[type]["variants"] = []
                    elem_types[type]["instances"] = [elem["name"]]
                    elem_types[type]["properties"] = {
                        k: v for k,v in elem.items() if k not in exclude_keys
                    }
                else:
                    elem_types[type]["instances"].append(elem["name"])


            elem_types = list(elem_types.values())
        else:
            elem_types = [
                {
                    "name": f"{elem['type']}<{elem['name']}>",
                    "instances": [self["assembly"][i]["name"] for i in elem["instances"]],
                    "properties":  [
                        [str(v) for k,v in elem.items() if k not in exclude_keys]
                        #for _ in range(len(elem["instances"]))
                    ]*(len(elem["instances"])),
                    "coords": [self["assembly"][i]["crd"] for i in elem["instances"]],
                    "keys":   [k for k in elem.keys() if k not in exclude_keys]

                } for elem in self["prototypes"].get("elements", [])
            ]
        return elem_types

    def iter_node_tags(self):
        for tag in self["nodes"]:
            yield tag

    def iter_cell_tags(self):
        for tag in self["assembly"]:
            yield tag


    def iter_nodes(self, keys=None):
        pass

    def iter_cells(self, keys=None):
        pass

    def node_properties(self, tag=None)->dict:
        return self["nodes"][tag]

    def node_indices(self, tag=None):
        if not hasattr(self, "_node_indices"):
            self._node_indices = {
                tag: i for i, tag in enumerate(self["nodes"])
            }
        return self._node_indices[tag]

    def node_rotation(self, tag=None, state=None):
        if state is None:
            eye = np.eye(3)
            if tag is None:
                return [eye for i in self.iter_node_tags()]
            else:
                return eye
        else:
            return state.node_array(tag, dof=state.rotation)

    def node_position(self, tag=None, state=None):

        if tag is None:
            pos = np.array([n["crd"] for n in self["nodes"].values()])
        else:
            pos = self["nodes"][tag]["crd"]

        if state is not None:
            pos = pos + state.node_array(tag, dof=state.position)

        return pos

    def cell_position(self, tag, state=None):
        if isinstance(tag, list):
            return np.array([[
                    self.node_position(node, state) for node in self["assembly"][t]["nodes"]
                ] for t in tag
            ])
        else:
            return np.array([ self.node_position(node, state)
                              for node in self["assembly"][tag]["nodes"] ])

    def frame_orientation(self, tag, state=None):
        el  = self["assembly"][tag]
        xyz = el["crd"]

        v1  = xyz[-1] - xyz[0]
        L   = np.linalg.norm(v1)
        e1  = v1/L

        if self.ndm == 2:
            v2 = np.array([0, 1, 0])

        if "yvec" in el["trsfm"] and el["trsfm"]["yvec"] is not None:
            v2  = np.array(el["trsfm"]["yvec"])

        elif "vecInLocXZPlane" in el["trsfm"]:
            v13 =  np.atleast_1d(el["trsfm"]["vecInLocXZPlane"])
            v2  = -np.cross(e1,v13)

        else:
            return _EYE3

        e2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(e1,e2)
        e3 = v3 / np.linalg.norm(v3)
        return np.stack([e1,e2,e3])


    def cell_exterior(self, tag):
        type = self["assembly"][tag]["type"].lower()

        if "frm" in type or "beamcol" in type:
            return self.cell_indices(tag)

        elif ("quad" in type or \
              "shell" in type and ("q" in type) or ("mitc" in type)):
            return self.cell_indices(tag)

        return []

    def cell_interpolation(self, tag):
        pass

    def cell_triangles(self, tag):
        type = self["assembly"][tag]["type"].lower()

        if "frm" in type or "beamcol" in type:
            return []

        elif ("quad" in type or \
              "shell" in type and ("q" in type) or ("mitc" in type)):
            nodes = self.cell_indices(tag)

            if len(nodes) == 4:
                return [[nodes[0], nodes[1], nodes[2]],
                        [nodes[2], nodes[3], nodes[0]]]
        return []

    def add_hook():
        pass

    def cell_surface(self, tag):
        if self._frame_outlines is None:
            self._frame_outlines = _get_frame_outlines(self)

        if self._extrude_outline is not None:
            return self._extrude_outline
        elif tag in self._frame_outlines:
            return self._frame_outlines[tag]
        else:
            return self._extrude_default


def _from_opensees(sam, shift, R):
    # Process OpenSees JSON format
    try:
        sam = sam["StructuralAnalysisModel"]
    except KeyError:
        pass

    geom = sam.get("geometry", sam.get("assembly"))

    try:
        #coord = np.array([R@n.pop("crd") for n in geom["nodes"]], dtype=float) + shift
        coord = np.array([R@n["crd"] for n in geom["nodes"]], dtype=float) + shift
        ndm = 3
    except:
        coord = np.array([R@[*n["crd"], 0.0] for n in geom["nodes"]], dtype=float) + shift
        ndm = 2

    nodes = {
        n["name"]: {**n, "crd": coord[i], "idx": i}
            for i,n in enumerate(geom["nodes"])
    }

#   ndm = len(next(iter(nodes.values()))["crd"])
    ndf = next(iter(nodes.values())).get("ndf", None)

    trsfm = {}
    for t in sam.get("properties", {}).get("crdTransformations", []):
        trsfm[int(t["name"])] = {
                k: val for k,val in t.items() if k != "vecInLocXZPlane"
        }
        if ndm == 3:
            trsfm[int(t["name"])]["vecInLocXZPlane"] = R@t["vecInLocXZPlane"]


    elems =  {
        e["name"]: dict(
        **e,
        crd=np.array([nodes[n]["crd"] for n in e["nodes"]], dtype=float),
        trsfm=trsfm[int(e["crdTransformation"])]
            if "crdTransformation" in e and int(e["crdTransformation"]) in trsfm
            else dict(yvec=R@e["yvec"] if "yvec" in e else None)
        ) for e in geom["elements"]
    }

    try:
        sections = {s["name"]: s for s in sam["properties"]["sections"]}
    except:
        sections = {}

    output = dict(nodes=nodes,
                  assembly=elems,
                  sam=sam,
                  sections=sections,
                  prototypes=sam.get("prototypes", {}),
                  ndm=ndm,
                  ndf=ndf
    )

    return output


def _add_section_shape(section, sections=None, outlines=None):
    import scipy.spatial

    # Rotation to change coordinates from x-y to z-y
    R = np.array(((0,-1),
                  (1, 0))).T

    if "section" in section:
        # Treat aggregated sections
        if section["section"] not in outlines:
            outlines[section["name"]] = _add_section_shape(sections[section["section"]], sections, outlines)
        else:
            outlines[section["name"]] = outlines[section["section"]]

    elif "bounding_polygon" in section:
        outlines[section["name"]] = [R@s for s in section["bounding_polygon"]]

    elif "fibers" in section:
        points = np.array([f["coord"] for f in section["fibers"]])
        try:
            outlines[section["name"]] = points[scipy.spatial.ConvexHull(points).vertices]
        except scipy.spatial._qhull.QhullError as e:
            from sees.utility.alpha_shape import alpha_shape
            outlines[section["name"]] = alpha_shape(points, 1)
           #warnings.warn(str(e))


def _get_frame_outlines(model):
    sections = {}
    for name,section in model["sections"].items():
        _add_section_shape(section, model["sections"], sections)

    outlines = {
        # TODO: For now, only using the first of the element's cross
        #       sections
        elem["name"]: np.array(sections[elem["sections"][0]])

        for elem in model["assembly"].values()
            if "sections" in elem and elem["sections"][0] in sections
    }

    return outlines

