# Claudio Perez
import numpy as np
import pygltflib
from .canvas import Canvas

class GltfLibCanvas(Canvas):
    def __init__(self, config=None):
        self.config = config

    def plot_mesh(self,vertices, triangles, lines=None):
        points = np.array(vertices, dtype="float32")
        triangles = np.array(triangles,dtype="uint8")

        triangles_binary_blob = triangles.flatten().tobytes()
        points_binary_blob = points.tobytes()

        self.gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=pygltflib.Attributes(POSITION=1), indices=0
                        )
                    ]
                )
            ],
            accessors=[
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_BYTE,
                    count=triangles.size,
                    type=pygltflib.SCALAR,
                    max=[int(triangles.max())],
                    min=[int(triangles.min())],
                ),
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=len(points),
                    type=pygltflib.VEC3,
                    max=points.max(axis=0).tolist(),
                    min=points.min(axis=0).tolist(),
                ),
            ],
            bufferViews=[
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(triangles_binary_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(triangles_binary_blob),
                    byteLength=len(points_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(triangles_binary_blob) + len(points_binary_blob)
                )
            ],
        )

        self.gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)


    def write(self, filename=None):
        import trimesh
        opts = self.config

        glb = b"".join(self.gltf.save_to_bytes())

        if "glb" in opts["write_file"][-3:]:
            with open(opts["write_file"],"wb+") as f:
                f.write(glb)

