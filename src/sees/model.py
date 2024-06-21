# Claudio Perez
import numpy as np

try:
    import orjson as json
except ImportError:
    import json



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

    def cell_orientation(self, tag): ...

    def cell_position(self, tag, state):   ...

    def cell_outline(self,  tag):   ...


    def cell_information(self, tag): ...

class SolidModel:
    pass

OUTLINES = {
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

def _add_section_shape(section, sections=None, outlines=None):
    from scipy.spatial import ConvexHull

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
        #outlines[section["name"]] = _alpha_shape(np.array([f["coord"] for f in section["fibers"]]))
        points = np.array([f["coord"] for f in section["fibers"]])
        outlines[section["name"]] = points[ConvexHull(points).vertices]


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

class FrameModel:
#   nodes
#   cells
#   elements
#   sections
#   prototypes

    def __getitem__(self, key):
        return self._data[key]

#   @property
#   def nodes(self):
#       return {k: n["crd"] for k, n in self["nodes"].items()}
#   @property
#   def cells(self):
#       return {k: e["nodes"] for k, e in self["assembly"].items()}

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

    def iter_node_tags(self):
        for tag in self["nodes"]:
            yield tag

    def node_indices(self, tag=None):
        if not hasattr(self, "_node_indices"):
            self._node_indices = {
                tag: i for i, tag in enumerate(self["nodes"])
            }
        return self._node_indices[tag]

    def node_position(self, tag=None, state=None):

        if tag is None:
            pos = np.array([n["crd"] for n in self["nodes"].values()])
        else:
            pos = self["nodes"][tag]["crd"]

        if state is not None:
            pos += state.node_array(tag)

        return pos

    def cell_position(self, tag, state=None):
        return np.array([ self.node_position(node, state)
                          for node in self["assembly"][tag]["nodes"] ])

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

    def frame_orientation(self, tag):
        import sees.frame
        el = self["assembly"][tag]

        xyz = el["crd"]
        dx = xyz[-1] - xyz[0]
        L = np.linalg.norm(dx)
        e1 = dx/L

        if "yvec" in el["trsfm"] and el["trsfm"]["yvec"] is not None:
            e2 = np.array(el["trsfm"]["yvec"])
            v3 = np.cross(e1,e2)
            norm_v3 = np.linalg.norm(v3)
            e3 = v3 / norm_v3
            return np.stack([e1,e2,e3])

        elif "vecInLocXZPlane" in el["trsfm"]:
            v13 = np.atleast_1d(el["trsfm"]["vecInLocXZPlane"])
            v2 = -np.cross(e1,v13)
            norm_v2 = np.linalg.norm(v2)
            if norm_v2 < 1e-8:
                v2 = -np.cross(e1,np.array([*reversed(vert)]))
                norm_v2 = np.linalg.norm(v2)
            e2 = v2 / norm_v2
            v3 =  np.cross(e1,e2)
            e3 = v3 / np.linalg.norm(v3)
            return np.stack([e1,e2,e3])
        else:
            print(el)


    def frame_outline(self, tag):
        if self._frame_outlines is None:
            self._frame_outlines = _get_frame_outlines(self)

        if self._extrude_outline is not None:
            return self._extrude_outline
        elif tag in self._frame_outlines:
            return self._frame_outlines[tag]
        else:
            return self._extrude_default


    def __init__(self, sam:dict, shift = None, rot=None, **kwds):

        """
        Process OpenSees JSON output and return dict with the form:

            {<elem tag>: {"crd": [<coordinates>], ...}}
        """
        try:
            sam = sam["StructuralAnalysisModel"]
        except KeyError:
            pass

        self._frame_outlines = None
        self._extrude_default = OUTLINES[kwds.get("extrude_default", "square")]
        self._extrude_outline = OUTLINES[kwds.get("extrude_outline", None)]

        ndm = 3
        R = np.eye(ndm) if rot is None else rot

        geom = sam.get("geometry", sam.get("assembly"))

        if shift is None:
            shift = np.zeros(ndm)
        else:
            shift = np.asarray(shift)

        try:
            #coord = np.array([R@n.pop("crd") for n in geom["nodes"]], dtype=float) + shift
            coord = np.array([R@n["crd"] for n in geom["nodes"]], dtype=float) + shift
        except:
            coord = np.array([R@[*n["crd"], 0.0] for n in geom["nodes"]], dtype=float) + shift

        nodes = {
                n["name"]: {**n, "crd": coord[i], "idx": i}
                    for i,n in enumerate(geom["nodes"])
        }

        ndm = len(next(iter(nodes.values()))["crd"])


        try:
            trsfm = {int(t["name"]): t for t in sam["properties"]["crdTransformations"]}
        except KeyError:
            trsfm = {}

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
#                     coord=coord,
                      sam=sam,
                      sections=sections,
                      ndm=ndm)

        if "prototypes" in sam:
            output.update({"prototypes": sam["prototypes"]})

        self._data = output

