from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stl_utils import load_stl, print_mesh_summary, compute_face_geometry
from stl_utils import plot_geom
from newton_solver import solve_newton_case


def estimate_reference_values(mesh, geom):
    """
    Estima S_ref, L_ref y r_ref a partir de la geometría.
    Convención práctica:
    - flujo principal en x
    - S_ref = área proyectada frontal aproximada en plano yz
    - L_ref = longitud en x
    - r_ref = centroide global de caras ponderado por área
    """
    bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    bounds_max = np.asarray(mesh.bounds[1], dtype=float)
    extents = np.asarray(mesh.extents, dtype=float)

    centers = geom["centers"]
    areas = geom["areas"]

    # Longitud de referencia: extensión en x
    L_ref = float(extents[0])

    # Área de referencia: caja frontal yz
    # Es una aproximación sencilla y estable para arrancar
    S_ref = float(extents[1] * extents[2])

    # Punto de referencia: centroide superficial aproximado
    r_ref = np.average(centers, axis=0, weights=areas)

    return S_ref, L_ref, r_ref, bounds_min, bounds_max, extents


def plot_cp_map(mesh, geom, cp, title="Cp map"):
    """
    Colorea la malla con el valor de cp por cara.
    """
    face_vertices = geom["face_vertices"]

    cp = np.asarray(cp, dtype=float)
    cp_min = float(cp.min())
    cp_max = float(cp.max())

    if np.isclose(cp_max, cp_min):
        cp_norm = np.zeros_like(cp)
    else:
        cp_norm = (cp - cp_min) / (cp_max - cp_min)

    face_colors = cm.viridis(cp_norm)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(
        face_vertices,
        facecolors=face_colors,
        edgecolor="k",
        linewidth=0.2,
        alpha=0.95,
    )
    ax.add_collection3d(poly)

    bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    bounds_max = np.asarray(mesh.bounds[1], dtype=float)
    center = 0.5 * (bounds_min + bounds_max)
    extents = bounds_max - bounds_min
    max_range = 0.5 * np.max(extents)

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    mappable = cm.ScalarMappable(cmap="viridis")
    mappable.set_array(cp)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("cp")

    plt.tight_layout()
    plt.show()


def print_case_results(name, result):
    CF = result["CF_total"]
    CMv = result["CM_total"]

    print(f"\n=== {name} ===")
    print(f"alpha_deg   = {result['alpha_deg']}")
    print(f"CF_total_x  = {CF[0]}")
    print(f"CF_total_y  = {CF[1]}")
    print(f"CF_total_z  = {CF[2]}")
    print(f"CM_total_x  = {CMv[0]}")
    print(f"CM_total_y  = {CMv[1]}")
    print(f"CM_total_z  = {CMv[2]}")
    print(f"CD          = {result['CD']}")
    print(f"CL          = {result['CL']}")
    print(f"CM          = {result['CM']}")
    print(f"n_windward  = {result['n_windward']}")
    print(f"n_leeward   = {result['n_leeward']}")
    print(f"cp_min      = {np.min(result['cp'])}")
    print(f"cp_max      = {np.max(result['cp'])}")


# ============================================================
# CARGA DE MALLA
# ============================================================

# mesh = load_stl(Path("data/esfera.stl"))
# mesh = load_stl(Path("data/Capsula/PruebaARD.stl"))
# mesh = load_stl(Path("data/Capsula/PruebaARD2.stl"))
# mesh = load_stl(Path("data/Capsula/PruebaARD3.stl"))
mesh = load_stl(Path("data/Capsula/PruebaARD4.stl"))

print_mesh_summary(mesh)

geom = compute_face_geometry(mesh)

centers = geom["centers"]
areas = geom["areas"]
normals = geom["normals"]

# ============================================================
# VISUALIZACIÓN GEOMÉTRICA
# ============================================================

diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
normal_scale = 0.03 * diag

plot_geom(
    geom,
    show_mesh=True,
    show_centers=True,
    show_normals=True,
    normal_scale=normal_scale,
    color_by="area",
    title="STL - face geometry",
)

# ============================================================
# REFERENCIAS GEOMÉTRICAS
# ============================================================

S_ref, L_ref, r_ref, bounds_min, bounds_max, extents = estimate_reference_values(mesh, geom)

print("\n=== REFERENCIAS USADAS ===")
print(f"S_ref = {S_ref}")
print(f"L_ref = {L_ref}")
print(f"r_ref = {r_ref}")

# ============================================================
# EJES DE PROYECCIÓN
# ============================================================
# Como el flujo base va hacia -x, el drag positivo lo tomamos en -x
eD = np.array([-1.0, 0.0, 0.0])
eL = np.array([0.0, 0.0, -1.0])   # ajustado a la convención del flujo
eM = np.array([0.0, 1.0, 0.0])    # pitching moment alrededor de y

# ============================================================
# CASO 1: alpha = 0 deg
# ============================================================

result_0 = solve_newton_case(
    centers=centers,
    areas=areas,
    normals=normals,
    alpha_deg=0.0,
    S_ref=S_ref,
    L_ref=L_ref,
    r_ref=r_ref,
    eD=eD,
    eL=eL,
    eM=eM,
)

print_case_results("RESULTADOS NEWTON - alpha 0 deg", result_0)

# ============================================================
# CASO 2: alpha = 20 deg
# ============================================================

result_20 = solve_newton_case(
    centers=centers,
    areas=areas,
    normals=normals,
    alpha_deg=20.0,
    S_ref=S_ref,
    L_ref=L_ref,
    r_ref=r_ref,
    eD=eD,
    eL=eL,
    eM=eM,
)

print_case_results("RESULTADOS NEWTON - alpha 20 deg", result_20)

print("\nPrimeros 10 valores de cp (alpha=20):")
print(result_20["cp"][:10])

# ============================================================
# MAPA DE CP
# ============================================================

plot_cp_map(mesh, geom, result_20["cp"], title="Cp map - Newton - alpha 20 deg")