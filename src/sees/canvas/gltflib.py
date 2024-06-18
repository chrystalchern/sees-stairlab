# Claudio Perez
from pathlib import Path
import numpy as np
import pygltflib
from .canvas import Canvas

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
    def __init__(self, config=None):
        self.config = config

        self.gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[])],
            nodes=[],
            meshes=[],
            accessors=[],
            materials=[
                pygltflib.Material(
                    name="black",
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[0, 0, 0, 1]
                    )
                ),
                pygltflib.Material(
                    name="white",
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[1, 1, 1, 1]
                    )
                ),
                pygltflib.Material(
                    name="gray",
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorFactor=[0.8, 0.8, 0.8, 1]
                    )
                ),
            ],
            bufferViews=[],
            buffers=[pygltflib.Buffer(byteLength=0)],
        )
        self.gltf._glb_data = bytes()

        self.colors = {m.name: m for m in self.gltf.materials}


    def _push_data(self, data):
        self.gltf.bufferViews.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(self.gltf._glb_data),
                    byteLength=len(data),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,
                )
        )

        self.gltf._glb_data += data
        self.gltf.buffers[0].byteLength += len(data)
        return len(self.gltf.bufferViews)-1

    def plot_lines(self, vertices, **kwds):
        # vertices is given with nans separating line groups, but
        # GLTF does not accept nans so we have to filter these
        # out, and add distinct meshes for each line group
        points  = np.array(vertices[~np.isnan(vertices[:,0]),:], dtype="float32")
        points_binary_blob  = points.tobytes()

        self.gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=self._push_data(points_binary_blob),
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            )
        )
        point_accessor = len(self.gltf.accessors) - 1

        for indices in _split(np.arange(len(vertices), dtype="uint8"), np.nan, vertices[:,0]):
            # here, n adjusts indices by the number of nan rows that were removed so far
            n  = sum(np.isnan(vertices[:indices[0],0]))
            indices_binary_blob = (indices - n).flatten().tobytes()
            self.gltf.accessors.append(
                pygltflib.Accessor(
                    bufferView=self._push_data(indices_binary_blob),
                    componentType=pygltflib.UNSIGNED_BYTE,
                    count=indices.size,
                    type=pygltflib.SCALAR,
                    max=[int(indices.max())],
                    min=[int(indices.min())],
                )
            )
            self.gltf.meshes.append(
                   pygltflib.Mesh(
                     primitives=[
                         pygltflib.Primitive(
                             mode=pygltflib.LINES,
                             attributes=pygltflib.Attributes(POSITION=point_accessor),
                             material=0,
                             # most recently added accessor
                             indices=len(self.gltf.accessors)-1
                         )
                     ]
                   )
            )

            self.gltf.nodes.append(pygltflib.Node(mesh=len(self.gltf.meshes)-1))
            self.gltf.scenes[0].nodes.append(len(self.gltf.nodes)-1)


    def plot_mesh(self, vertices, triangles, lines=None, **kwds):
        points    = np.array(vertices, dtype="float32")
        triangles = np.array(triangles,dtype="uint8")

        triangles_binary_blob = triangles.flatten().tobytes()
        points_binary_blob = points.tobytes()

        self.gltf.accessors.extend([
            pygltflib.Accessor(
                bufferView=self._push_data(triangles_binary_blob),
                componentType=pygltflib.UNSIGNED_BYTE,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=self._push_data(points_binary_blob),
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            )
        ])
        self.gltf.meshes.append(
               pygltflib.Mesh(
                 primitives=[
                     pygltflib.Primitive(
                         mode=pygltflib.TRIANGLES,
                         attributes=pygltflib.Attributes(POSITION=len(self.gltf.accessors)-1),
                         indices=len(self.gltf.accessors)-2
                     )
                 ]
               )
        )
        self.gltf.nodes.append(pygltflib.Node(mesh=len(self.gltf.meshes)-1))
        self.gltf.scenes[0].nodes.append(len(self.gltf.nodes)-1)


    def show(self):
        import bottle
        page = self._form_html("./model.glb")

        glb = b"".join(self.gltf.save_to_bytes())
        app = bottle.Bottle()
        app.route("/")(lambda : page )
        app.route("/model.glb")(lambda : glb)
        bottle.run(app, host="localhost", port=8080)


    def _form_html(self, src):
#       with open(Path(__file__).parents[0]/"three"/"webgl_instancing_scatter.html", "r") as f:
        with open(Path(__file__).parents[0]/"three"/"index.html", "r") as f:
            return f.read()

    def _form_html_(self, src):
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

        glb = b"".join(self.gltf.save_to_bytes())

        if "glb" in opts["write_file"][-3:]:
            with open(opts["write_file"],"wb+") as f:
                f.write(glb)

