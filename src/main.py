from pathlib import Path
from stl_utils import load_stl, print_mesh_summary, compute_face_geometry

mesh = load_stl(Path("../data/esfera.stl"))
print_mesh_summary(mesh)

geom = compute_face_geometry(mesh)

centers = geom["centers"]
areas = geom["areas"]
normals = geom["normals"]