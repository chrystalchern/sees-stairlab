import sys
import numpy as np
from scipy.spatial.transform import Rotation

import sees.frame
import shps.curve


def draw_extruded_frames(artist, state=None, options=None):
    ndm = 3

    model = artist.model

    nodes = artist.model["nodes"]

    coords = []
    triang = []

    I = 0
    for i,el in enumerate(artist.model["assembly"].values()):
        # TODO: probably better to loop over sections dict rather
        #       than elements
        outline = model.frame_outline(el["name"])
        if outline is None:
            continue

        outline = outline*options["objects"]["sections"]["scale"]

        N  = len(el["nodes"])

        noe = len(outline) # number of outline edges
        if state is not None:
            glob_displ = state.cell_array(el["name"], state.position)
            X = shps.curve.displace(el["crd"], glob_displ, N).T
            R = state.cell_array(el["name"], state.rotation)
        else:
            outline = outline*0.98
            X = np.array(el["crd"])
            R = [sees.frame.orientation(el["crd"], el["trsfm"]["yvec"]).T]*N



        # Loop over sample points along element length to assemble
        # `coord` and `triang` arrays
        for j in range(N):
            # loop over section edges
            for k,edge in enumerate(outline):
                # append rotated section coordinates to list of coordinates
                coords.append(X[j, :] + R[j]@[0, *edge])

                if j == 0:
                    # skip the first section
                    continue

                elif k < noe-1:
                    triang.extend([
                        # tie two triangles to this edge
                        [I+    noe*j + k,   I+    noe*j + k + 1,    I+noe*(j-1) + k],
                        [I+noe*j + k + 1,   I+noe*(j-1) + k + 1,    I+noe*(j-1) + k]
                    ])
                else:
                    # elif j < N-1:
                    triang.extend([
                        [I+    noe*j + k,    I + noe*j , I+noe*(j-1) + k],
                        [      I + noe*j, I + noe*(j-1), I+noe*(j-1) + k]
                    ])

        I += N*noe

    artist.canvas.plot_mesh(coords, triang,
                              color   = "gray" , #if state is not None else "white",
                              opacity = None   if state is not None else 0.2
                             )

    show_edges = True

    if not show_edges:
        return

    IDX = np.array((
        (0, 2),
        (0, 1)
    ))

    nan = np.zeros(artist.ndm)*np.nan
    coords = np.array(coords)
    if "extrude.sections" in options["show_objects"]:
        tri_points = np.array([
            coords[idx]  if (j+1)%3 else nan for j,idx in enumerate(np.array(triang).reshape(-1))
        ])
    else:
        tri_points = np.array([
            coords[i]  if j%2 else nan
            for j,idx in enumerate(np.array(triang)) for i in idx[IDX[j%2]]
        ])

    artist.canvas.plot_lines(tri_points,
                               color="black" if state is not None else "#808080",
                               width=4)

class so3:
    @classmethod
    def exp(cls, vect):
        return Rotation.from_rotvec(vect).as_matrix()

def _add_moment(artist, loc=None, axis=None):
    import meshio
    loc = [1.0, 0.0, 0.0]
    axis = [0, np.pi/2, 0]
    mesh_data = meshio.read('chrystals_moment.stl')
    coords = mesh_data.points

    coords = np.einsum('ik, kj -> ij',  coords,
                       so3.exp([0, 0, -np.pi/4])@so3.exp(axis))
    coords = 1e-3*coords + loc
#   for node in coords:
#       node = so3.exp(axis)@node
    for i in mesh_data.cells:
        if i.type == "triangle":
            triangles =  i.data #mesh_data.cells['triangle']

    artist.canvas.plot_mesh(coords, triangles)


import sees
from sees import RenderError, read_model
def _render(sam_file, res_file=None, noshow=False, **opts):
    # Configuration is determined by successively layering
    # from sources with the following priorities:
    #      defaults < file configs < kwds 

    config = sees.config.Config()


    if sam_file is None:
        raise RenderError("ERROR -- expected positional argument <sam-file>")

    # Read and clean model
    if not isinstance(sam_file, dict):
        model = read_model(sam_file)
    else:
        model = sam_file

    if "RendererConfiguration" in model:
        sees.apply_config(model["RendererConfiguration"], config)

    sees.apply_config(opts, config)

    artist = sees.FrameArtist(model, **config)

    draw_extruded_frames(artist, options=opts)

    # -----------------------------------------------------------
#   if "IterationHistory" in sam_file:
    soln = sees.state.read_state(sam_file, artist.model, **opts)
    if "time" not in opts:
        soln = soln[soln.times[-1]]

    draw_extruded_frames(artist, soln, opts)
    # -----------------------------------------------------------
    _add_moment(artist)
    # -----------------------------------------------------------

    camera = dict(
      up=dict(x=0, y=0, z=1),
      center=dict(x=0, y=0, z=0),
      eye=dict(x=1.25, y=1.25, z=1.25)
    )

    #fig.update_layout(scene_camera=camera, title=name)

    # write plot to file if file name provided
    if config["write_file"]:
        artist.draw()
        artist.write(config["write_file"])

    else:
        artist.draw()
        if not noshow:
            artist.canvas.show()

    return artist


if __name__ == "__main__":
    import sees.__main__
    config = sees.__main__.parse_args(sys.argv)

    try:
        _render(**config)

    except (FileNotFoundError, RenderError) as e:
        # Catch expected errors to avoid printing an ugly/unnecessary stack trace.
        print(e, file=sys.stderr)
        print("         Run '{NAME} --help' for more information".format(NAME=sys.argv[0]), file=sys.stderr)
        sys.exit()

