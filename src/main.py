from pathlib import Path
from stl_utils import load_stl, print_mesh_summary, compute_face_geometry
from stl_utils import plot_geom
from newton_solver import solve_newton_case
import numpy as np

mesh = load_stl(Path("data/esfera.stl"))
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

# ============================================================
# PARÁMETROS DEL CASO
# ============================================================
alpha_deg = 20.0
S_ref = 1.0
L_ref = 1.0
r_ref = np.array([0.0, 0.0, 0.0])

# ============================================================
# CÁLCULO NEWTON
# ============================================================
result = solve_newton_case(
    centers=centers,
    areas=areas,
    normals=normals,
    alpha_deg=alpha_deg,
    S_ref=S_ref,
    L_ref=L_ref,
    r_ref=r_ref,
)

# ============================================================
# RESULTADOS BÁSICOS
# ============================================================
CF = result["CF_total"]
CM = result["CM_total"]

print("\n=== RESULTADOS NEWTON ===")
print(f"alpha_deg = {result['alpha_deg']}")

print(f"CF_total_x = {CF[0]}")
print(f"CF_total_y = {CF[1]}")
print(f"CF_total_z = {CF[2]}")

print(f"CM_total_x = {CM[0]}")
print(f"CM_total_y = {CM[1]}")
print(f"CM_total_z = {CM[2]}")
print("\nNúmero de caras a barlovento:", np.sum(result["mu"] > 0.0))
print("Número de caras a sotavento :", np.sum(result["mu"] <= 0.0))
