import numpy as np

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

    def node_iter(self): ...

    def node_location(self, tag): ...

    def node_information(self, tag): ...

    def cell_iter(self, filt=None): ...

    def cell_type(self, tag):       ... # line triangle quadrilateral 

    def cell_exterior(self, tag):   ...

    def cell_interior(self, tag):   ...

    def cell_location(self, tag):   ...

    def cell_outline(self,  tag):   ...

    def cell_orientation(self, tag): ...

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

        for elem in model["assembly"].values() if "sections" in elem and elem["sections"][0] in sections
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

    @property
    def nodes(self):
        return {k: n["crd"] for k, n in self["nodes"].items()}

    @property
    def cells(self):
        return {k: e["nodes"] for k, e in self["assembly"].items()}

    def frame_orientation(self, el):
        import sees.frame
        if "yvec" in el:
            return sees.frame.orientation(el["crd"], el["trsfm"]["yvec"])

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
            trsfm = {t["name"]: t for t in sam["properties"]["crdTransformations"]}
        except KeyError:
            trsfm = {}

        elems =  {
          e["name"]: dict(
            **e,
            crd=np.array([nodes[n]["crd"] for n in e["nodes"]], dtype=float),
            trsfm=trsfm[e["crdTransformation"]]
                if "crdTransformation" in e and e["crdTransformation"] in trsfm
                else dict(yvec=R@e["yvec"] if "yvec" in e else None)
          ) for e in geom["elements"]
        }

        try:
            sections = {s["name"]: s for s in sam["properties"]["sections"]}
        except:
            sections = {}

        output = dict(nodes=nodes,
                      assembly=elems,
                      coord=coord,
                      sam=sam,
                      sections=sections,
                      ndm=ndm)

        if "prototypes" in sam:
            output.update({"prototypes": sam["prototypes"]})

        self._data = output

