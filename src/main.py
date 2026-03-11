from pathlib import Path
import numpy as np

from stl_utils import load_stl, print_mesh_summary, compute_face_geometry
from stl_utils import plot_geom
from newton_solver import solve_newton_case

# mesh = load_stl(Path("data/esfera.stl"))              # Esfera Moodle
# mesh = load_stl(Path("data/Capsula/PruebaARD.stl"))   # Prueba STL1 SAG-50mm
# mesh = load_stl(Path("data/Capsula/PruebaARD2.stl"))  # Prueba STL2 SAG-0.01mm, muchísimos triángulos
# mesh = load_stl(Path("data/Capsula/PruebaARD3.stl"))  # Prueba STL3 SAG-15mm step-100mm
mesh = load_stl(Path("data/Capsula/PruebaARD4.stl"))    # Prueba STL4 con mesh smoothing

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

# Ejes de proyección
eD = np.array([1.0, 0.0, 0.0])  # drag
eL = np.array([0.0, 0.0, 1.0])  # lift
eM = np.array([0.0, 1.0, 0.0])  # momento alrededor de y

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
    eD=eD,
    eL=eL,
    eM=eM,
)

# ============================================================
# RESULTADOS BÁSICOS
# ============================================================
CF = result["CF_total"]
CMv = result["CM_total"]

print("\n=== RESULTADOS NEWTON ===")
print(f"alpha_deg = {result['alpha_deg']}")

print(f"CF_total_x = {CF[0]}")
print(f"CF_total_y = {CF[1]}")
print(f"CF_total_z = {CF[2]}")

print(f"CM_total_x = {CMv[0]}")
print(f"CM_total_y = {CMv[1]}")
print(f"CM_total_z = {CMv[2]}")

print(f"CD = {result['CD']}")
print(f"CL = {result['CL']}")
print(f"CM = {result['CM']}")

print("\nPrimeros 10 valores de cp:")
print(result["cp"][:10])

print("\nNúmero de caras a barlovento:", result["n_windward"])
print("Número de caras a sotavento :", result["n_leeward"])