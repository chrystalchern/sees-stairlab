
# Configuring
#============
# The following configuration options are available:
from collections import defaultdict

Config = lambda : {
  "model_config":  {},
  "canvas_config": {}, # kwds for canvas
  "viewer_config": {},
  "server_config": {},

  "show_objects":  ["frames", "frames.displ", "nodes", "legend", "elastica"],
  "mode_num"    :  None,
  "hide_objects":  ["origin"],
  "sam_file":      None,
  "res_file":      None,
  "write_file":    None,
  "displ":         defaultdict(list),
  "scale":         100.0,
  "vert":          2,
  "view":          "iso",

  "camera": {
      "view": "iso",               # iso | plan| elev[ation] | sect[ion]
      "projection": "orthographic" # perspective | orthographic
  },

  # Artist
  "displacements": {"scale": 100, "color": "#660505"},

  "objects": {
      "origin": {"color": "black", "scale": 1.0},
      "frames" : {
          "displaced": {"color": "red", "npoints": 20}
      },
      "nodes": {
          "scale": 1.0,
          "default": {"size": 3, "color": "#000000"},
          "displaced" : {},
          "fixed"  : {},
      },
      "sections": {"scale": 1.0}
  },

  # Canvas
  "canvas":        "matplotlib",
  "save_options": {
      # Options for when writing to an HTML file.
      "html": {
          "include_plotlyjs": True,
          "include_mathjax" : "cdn",
          "full_html"       : True
      }
  }
}

def apply_config(conf, opts):
    for k,v in conf.items():
        if isinstance(v,dict):
            apply_config(v, opts[k])
        else:
            opts[k] = v

