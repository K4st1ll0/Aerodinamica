from pathlib import Path
from stl_utils import load_stl, print_mesh_summary, compute_face_geometry
from stl_utils import plot_geom

mesh = load_stl(Path("../data/esfera.stl"))
print_mesh_summary(mesh)

geom = compute_face_geometry(mesh)

centers = geom["centers"]
areas = geom["areas"]
normals = geom["normals"]

plot_geom(
    geom,
    show_mesh=True,
    show_centers=True,
    show_normals=True,
    normal_scale=8.0,
    color_by="area",
    title="Sphere STL - face geometry",
)