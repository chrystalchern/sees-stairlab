# Claudio Perez
"""
- glTF uses a right-handed coordinate system, that is, the cross product of +X and +Y yields +Z.
- glTF defines +Y as up.
- The front of a glTF asset faces +Z.
- All angles are in radians.
- Positive rotation is counterclockwise.
- Rotations are given as quaternions stored as a tuple (x,y,z,w),
  where the w-component is the cosine of half of the rotation angle.
  For example, the quaternion [ 0.259, 0.0, 0.0, 0.966 ] describes a rotation
  about 30 degrees, around the x-axis.
"""
from pathlib import Path
import numpy as np
import pygltflib
from .canvas import Canvas

GLTF_T = {
    "float32": pygltflib.FLOAT,
    "uint8":   pygltflib.UNSIGNED_BYTE,
    "uint16":  pygltflib.UNSIGNED_SHORT,
}

def _split(a, value, source=None):
    # like str.split, but for arrays
    if source is None:
        source = a
    if np.isnan(value):
        idx = np.where(~np.isnan(source))[0]
    else:
        idx = np.where(source != value)[0]
    return np.split(a[idx],np.where(np.diff(idx)!=1)[0]+1)

class GltfLibCanvas(Canvas):
    vertical = 2

    def __init__(self, config=None):
        self.config = config

        # Quaternion, equivalent to rotation matrix:
        #  1  0  0
        #  0  0  1
        #  0 -1  0
        #                          x, y, z, scalar
        self._rotation = [-0.7071068, 0, 0, 0.7071068]

        self.index_t = "uint16"
        self.float_t = "float32"

        self.gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[])],
            nodes=[],
            meshes=[],
            accessors=[],
            materials=[
                pygltflib.Material(
                    name="black",
                    doubleSided=True,
                    alphaMode=pygltflib.MASK,
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[0, 0, 0, 1]
                    )
                ),
                pygltflib.Material(
                    name="white",
                    doubleSided=True,
                    alphaMode=pygltflib.MASK,
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[1, 1, 1, 1]
                    )
                ),
                pygltflib.Material(
                    name="red",
                    doubleSided=True,
                    alphaMode=pygltflib.MASK,
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[1, 0, 0, 1]
                    )
                ),
                pygltflib.Material(
                    name="green",
                    doubleSided=True,
                    alphaMode=pygltflib.MASK,
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[0, 1, 0, 1]
                    )
                ),
                pygltflib.Material(
                    name="blue",
                    doubleSided=True,
                    alphaMode=pygltflib.MASK,
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[0, 0, 1, 1]
                    )
                ),
                pygltflib.Material(
                    name="gray",
                    doubleSided=True,
                    alphaMode=pygltflib.MASK,
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[0.9, 0.9, 0.9, 1]
                    )
                ),
            ],
            bufferViews=[],
            buffers=[pygltflib.Buffer(byteLength=0)],
        )
        self.gltf._glb_data = bytes()

        self._init_nodes()

        # Map pairs of (color, alpha) to material's index in material list
        self._color = {(m.name,m.pbrMetallicRoughness.baseColorFactor[3]): i
                       for i,m in enumerate(self.gltf.materials)}

    def _init_nodes(self):
        #
        #
        #
        index_t = "uint8"
        points = np.array(
            [
                [-1.0, -1.0,  1.0],
                [ 1.0, -1.0,  1.0],
                [-1.0,  1.0,  1.0],
                [ 1.0,  1.0,  1.0],
                [ 1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0],
                [-1.0,  1.0, -1.0],
            ],
            dtype="float32",
        )/100

        triangles = np.array(
            [
                [0, 1, 2],
                [3, 2, 1],
                [1, 0, 4],
                [5, 4, 0],
                [3, 1, 6],
                [4, 6, 1],
                [2, 3, 7],
                [6, 7, 3],
                [0, 2, 5],
                [7, 5, 2],
                [5, 7, 4],
                [6, 4, 7],
            ],
            dtype=index_t,
        )
        triangles_binary_blob = triangles.flatten().tobytes()
        points_binary_blob = points.tobytes()

        self.gltf.accessors.extend([
            pygltflib.Accessor(
                bufferView=self._push_data(triangles_binary_blob,
                                           pygltflib.ELEMENT_ARRAY_BUFFER),
                componentType=GLTF_T[index_t],
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=self._push_data(points_binary_blob,
                                           pygltflib.ARRAY_BUFFER),
                componentType=GLTF_T[self.float_t],
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            )
        ])


        indices_access = len(self.gltf.accessors)-2 # indices
        points_access  = len(self.gltf.accessors)-1 # points
        self.gltf.meshes.append(
               pygltflib.Mesh(
                 primitives=[
                     pygltflib.Primitive(
                         mode=pygltflib.TRIANGLES,
                         attributes=pygltflib.Attributes(POSITION=points_access),
                         material=0,#material,
                         indices=indices_access
                     )
                 ]
               )
        )

        self._node_mesh = len(self.gltf.meshes) - 1

    def _use_asset(self, name, scale, rotation, material):
        pass

#   def plot_vectors(self, locs, vecs, **kwds):

#       ne = vecs.shape[0]
#       for j in range(3):
#           X = np.zeros((ne*3, 3))*np.nan
#           for i in range(j,ne,3):
#               X[i*3,:] = locs[i]
#               X[i*3+1,:] = locs[i] + vecs[i]

#           color = kwds.get("color", ("red", "blue", "green")[j])

#           # _label = label if label is not None else ""
#           label = kwds.get("label", "")
#           if isinstance(label, list):
#               label = label[j]
#           self.data.append({
#               "name": label,
#               "type": "scatter3d",
#               "mode": "lines",
#               "x": X.T[0], "y": X.T[1], "z": X.T[2],
#               "line": {"color": color, "width": 4},
#               "hoverinfo":"skip",
#               "showlegend": False
#           })

    def plot_nodes(self, coords, label = None, props=None, data=None, **kwds):
        name = label or "nodes"
        x,y,z = coords.T

#       indices_access, points_access = self._node_access

        material = self._get_material(kwds.get("color", "black"),
                                      kwds.get("alpha",      1))


        for coord in coords:
            self.gltf.nodes.append(pygltflib.Node(
                    mesh=self._node_mesh,
                    #rotation=self._rotation,
                    translation=coord.tolist(),
                )
            )
            self.gltf.scenes[0].nodes.append(len(self.gltf.nodes)-1)



    def _get_material(self, color, alpha=1):
        if (color, alpha) in self._color:
            return self._color[(color,alpha)]

        elif isinstance(color, str) and color[0] == "#":
            # Remove leading hash in hex
            hx  = color.lstrip("#")
            # Convert hex to RGB
            rgb = [int(hx[i:i+2], 16)/255 for i in (0, 2, 4)]

        elif isinstance(color, tuple):
            rgb = color

        else:
            raise TypeError("Unexpected type for color")

        # Store index for new material
        self._color[(color, alpha)] = len(self.gltf.materials)
        # Create and add new material
        self.gltf.materials.append(
            pygltflib.Material(
                name=str(color),
                doubleSided=True,
                alphaMode=pygltflib.MASK,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=[*rgb, alpha]
                )
            ),
        )
        return self._color[(color, alpha)]


    def _push_data(self, data, target, byteStride=None):
        self.gltf.bufferViews.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteStride=byteStride,
                    byteOffset=self.gltf.buffers[0].byteLength,
                    byteLength=len(data),
                    target=target,
                )
        )

        self.gltf._glb_data += data
        self.gltf.buffers[0].byteLength += len(data)
        return len(self.gltf.bufferViews)-1


    def plot_lines(self, vertices, **kwds):
        material = self._get_material(kwds.get("color", "gray"),
                                      kwds.get("alpha",      1))


        # vertices is given with nans separating line groups, but
        # GLTF does not accept nans so we have to filter these
        # out, and add distinct meshes for each line group
        assert np.all(np.isnan(vertices[np.isnan(vertices[:,0]), :]))
        points  = np.array(vertices[~np.isnan(vertices[:,0]),:], dtype="float32")
        points_buffer = self._push_data(points.tobytes(), pygltflib.ARRAY_BUFFER)

        self.gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=points_buffer,
                componentType=GLTF_T[self.float_t],
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            )
        )
        points_access = len(self.gltf.accessors) - 1

        for indices in _split(np.arange(len(vertices), dtype=self.index_t), np.nan, vertices[:,0]):
            # here, n adjusts indices by the number of nan rows that were removed so far
            n  = sum(np.isnan(vertices[:indices[0],0]))
            indices_array = indices - n
            indices_binary_blob = indices_array.tobytes()

            if len(indices_array) <= 1:
                print(indices_array)
                continue

            self.gltf.accessors.extend([
                pygltflib.Accessor(
                    bufferView=self._push_data(indices_binary_blob,
                                               pygltflib.ELEMENT_ARRAY_BUFFER),
                    componentType=GLTF_T[self.index_t],
                    count=indices_array.size,
                    type=pygltflib.SCALAR,
                    max=[int(indices_array.max())],
                    min=[int(indices_array.min())],
                )
            ])
            self.gltf.meshes.append(
                   pygltflib.Mesh(
                     primitives=[
                         pygltflib.Primitive(
                             mode=pygltflib.LINES,
                             attributes=pygltflib.Attributes(POSITION=points_access),
                             material=material,
                             # most recently added accessor
                             indices=len(self.gltf.accessors)-1,
                             # TODO: use this mechanism to add annotation data
                             extras={},
                         )
                     ]
                   )
            )

            self.gltf.nodes.append(pygltflib.Node(
                    mesh=len(self.gltf.meshes)-1,
                    rotation=self._rotation
                )
            )
            self.gltf.scenes[0].nodes.append(len(self.gltf.nodes)-1)


    def plot_mesh(self, vertices, triangles, lines=None, **kwds):
        points    = np.array(vertices, dtype=self.float_t)
        triangles = np.array(triangles,dtype=self.index_t)

        triangles_binary_blob = triangles.flatten().tobytes()
        points_binary_blob = points.tobytes()

        self.gltf.accessors.extend([
            pygltflib.Accessor(
                bufferView=self._push_data(triangles_binary_blob, pygltflib.ELEMENT_ARRAY_BUFFER),
                componentType=GLTF_T[self.index_t],
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=self._push_data(points_binary_blob, pygltflib.ARRAY_BUFFER),
                componentType=GLTF_T[self.float_t],
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            )
        ])

        material = self._get_material(kwds.get("color", "gray"),
                                      kwds.get("alpha",      1))

        self.gltf.meshes.append(
               pygltflib.Mesh(
                 primitives=[
                     pygltflib.Primitive(
                         mode=pygltflib.TRIANGLES,
                         attributes=pygltflib.Attributes(POSITION=len(self.gltf.accessors)-1),
                         material=material,
                         indices=len(self.gltf.accessors)-2
                     )
                 ]
               )
        )
        self.gltf.nodes.append(pygltflib.Node(
                mesh=len(self.gltf.meshes)-1,
                rotation=self._rotation
            )
        )
        self.gltf.scenes[0].nodes.append(len(self.gltf.nodes)-1)


    def show(self):
        import bottle
        page = self._form_html("./model.glb")

        glb = b"".join(self.gltf.save_to_bytes())
        app = bottle.Bottle()
        app.route("/")(lambda : page )
        app.route("/model.glb")(lambda : glb)
        try:
            bottle.run(app, host="localhost", port=9090)
        except KeyboardInterrupt:
            pass


    def _form_html_(self, src):
#       with open(Path(__file__).parents[0]/"three"/"webgl_instancing_scatter.html", "r") as f:
#       with open(Path(__file__).parents[0]/"three"/"index.html", "r") as f:
        with open(Path(__file__).parents[0]/"three"/"gltf.html", "r") as f:
            return f.read()

    def _form_html(self, src):
        import textwrap

        return textwrap.dedent(f"""
          <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
            "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
          <html xmlns="http://www.w3.org/1999/xhtml" lang="en">
          <head>
            <meta charset="utf-8">
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js">
            </script>
          </head>
          <body>
            <model-viewer alt="rendering"
                          src="{src}"
                          ar
                          style="width: 100%; height: 1000px;"
                          shadow-intensity="1" camera-controls touch-action="pan-y">
            </model-viewer>
          </body>
          </html>
        """)

    def write(self, filename=None):
        opts = self.config

        self.gltf.save(filename)

#       if "glb" in opts["write_file"][-3:]:
#           glb = b"".join(self.gltf.save_to_bytes())
#           with open(opts["write_file"],"wb+") as f:
#               f.write(glb)
