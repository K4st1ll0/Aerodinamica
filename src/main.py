from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stl_utils import load_stl, print_mesh_summary, compute_face_geometry, plot_geom
from MN import solve_newton_case
from MNM import solve_modified_newton_case

### falta guardar en json
### falta comparar dos mallas

def estimate_reference_values(mesh, geom):
    """
    Estima S_ref, L_ref y r_ref a partir de la geometría.
    Convención práctica:
    - flujo principal en y
    - S_ref = área frontal aproximada en plano xz
    - L_ref = longitud en y
    - r_ref = centroide global de caras ponderado por área
    """
    bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    bounds_max = np.asarray(mesh.bounds[1], dtype=float)
    extents = np.asarray(mesh.extents, dtype=float)

    centers = geom["centers"]
    areas = geom["areas"]

    # Longitud de referencia: extensión en y
    L_ref = float(extents[1])

    # Área de referencia: caja frontal xz
    S_ref = float(extents[0] * extents[2])

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


def save_results_csv(filepath, rows):
    fieldnames = [
        "alpha_deg",
        "CD",
        "CL",
        "CM",
        "CF_total_x",
        "CF_total_y",
        "CF_total_z",
        "CM_total_x",
        "CM_total_y",
        "CM_total_z",
        "n_windward",
        "n_leeward",
        "cp_min",
        "cp_max",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados guardados en: {filepath}")


def save_cp_faces_csv(filepath, centers, cp):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_center", "y_center", "z_center", "cp"])

        for c, cp_i in zip(centers, cp):
            writer.writerow([c[0], c[1], c[2], cp_i])

    print(f"Cp por cara guardado en: {filepath}")


def run_alpha_sweep(
    alphas_deg,
    centers,
    areas,
    normals,
    S_ref,
    L_ref,
    r_ref,
    eD,
    eL,
    eM,
):
    rows = []

    for alpha_deg in alphas_deg:
        result = solve_newton_case(
            centers=centers,
            areas=areas,
            normals=normals,
            alpha_deg=float(alpha_deg),
            S_ref=S_ref,
            L_ref=L_ref,
            r_ref=r_ref,
            eD=eD,
            eL=eL,
            eM=eM,
        )

        print_case_results(f"RESULTADOS NEWTON - alpha {alpha_deg} deg", result)

        CF = result["CF_total"]
        CMv = result["CM_total"]

        rows.append({
            "alpha_deg": result["alpha_deg"],
            "CD": result["CD"],
            "CL": result["CL"],
            "CM": result["CM"],
            "CF_total_x": CF[0],
            "CF_total_y": CF[1],
            "CF_total_z": CF[2],
            "CM_total_x": CMv[0],
            "CM_total_y": CMv[1],
            "CM_total_z": CMv[2],
            "n_windward": result["n_windward"],
            "n_leeward": result["n_leeward"],
            "cp_min": float(np.min(result["cp"])),
            "cp_max": float(np.max(result["cp"])),
        })

    return rows

def run_mnm_alpha_sweep(
    alphas_deg,
    Mach,
    centers,
    areas,
    normals,
    S_ref,
    L_ref,
    r_ref,
    eD,
    eL,
    eM,
    gamma=1.4,
):
    rows = []

    for alpha_deg in alphas_deg:
        result = solve_modified_newton_case(
            centers=centers,
            areas=areas,
            normals=normals,
            alpha_deg=float(alpha_deg),
            Mach=float(Mach),
            S_ref=S_ref,
            L_ref=L_ref,
            r_ref=r_ref,
            eD=eD,
            eL=eL,
            eM=eM,
            gamma=gamma,
        )

        print(f"\n=== RESULTADOS MNM - alpha {alpha_deg} deg - M {Mach} ===")
        print(f"cp_max      = {result['cp_max']}")
        print(f"CD          = {result['CD']}")
        print(f"CL          = {result['CL']}")
        print(f"CM          = {result['CM']}")
        print(f"n_windward  = {result['n_windward']}")
        print(f"n_leeward   = {result['n_leeward']}")
        print(f"cp_min      = {np.min(result['cp'])}")
        print(f"cp_max_face = {np.max(result['cp'])}")

        CF = result["CF_total"]
        CMv = result["CM_total"]

        rows.append({
            "alpha_deg": result["alpha_deg"],
            "Mach": result["Mach"],
            "gamma": result["gamma"],
            "cp_max_stagnation": result["cp_max"],
            "CD": result["CD"],
            "CL": result["CL"],
            "CM": result["CM"],
            "CF_total_x": CF[0],
            "CF_total_y": CF[1],
            "CF_total_z": CF[2],
            "CM_total_x": CMv[0],
            "CM_total_y": CMv[1],
            "CM_total_z": CMv[2],
            "n_windward": result["n_windward"],
            "n_leeward": result["n_leeward"],
            "cp_min": float(np.min(result["cp"])),
            "cp_max": float(np.max(result["cp"])),
        })

    return rows

def save_results_csv_mnm(filepath, rows):
    fieldnames = [
        "alpha_deg",
        "Mach",
        "gamma",
        "cp_max_stagnation",
        "CD",
        "CL",
        "CM",
        "CF_total_x",
        "CF_total_y",
        "CF_total_z",
        "CM_total_x",
        "CM_total_y",
        "CM_total_z",
        "n_windward",
        "n_leeward",
        "cp_min",
        "cp_max",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados MNM guardados en: {filepath}")

def run_mnm_mach_sweep(
    Mach_list,
    centers,
    areas,
    normals,
    alpha_deg,
    S_ref,
    L_ref,
    r_ref,
    eD,
    eL,
    eM,
    gamma=1.4,
):
    rows = []

    for Mach in Mach_list:
        result = solve_modified_newton_case(
            centers=centers,
            areas=areas,
            normals=normals,
            alpha_deg=float(alpha_deg),
            Mach=float(Mach),
            S_ref=S_ref,
            L_ref=L_ref,
            r_ref=r_ref,
            eD=eD,
            eL=eL,
            eM=eM,
            gamma=gamma,
        )

        CF = result["CF_total"]
        CMv = result["CM_total"]

        print(f"\n=== RESULTADOS MNM - alpha {alpha_deg} deg - M {Mach} ===")
        print(f"cp_max_stagnation = {result['cp_max']}")
        print(f"CF_total_x        = {CF[0]}")
        print(f"CF_total_y        = {CF[1]}")
        print(f"CF_total_z        = {CF[2]}")
        print(f"CM_total_x        = {CMv[0]}")
        print(f"CM_total_y        = {CMv[1]}")
        print(f"CM_total_z        = {CMv[2]}")
        print(f"CD                = {result['CD']}")
        print(f"CL                = {result['CL']}")
        print(f"CM                = {result['CM']}")
        print(f"n_windward        = {result['n_windward']}")
        print(f"n_leeward         = {result['n_leeward']}")
        print(f"cp_min            = {np.min(result['cp'])}")
        print(f"cp_max_face       = {np.max(result['cp'])}")

        rows.append({
            "Mach": result["Mach"],
            "alpha_deg": result["alpha_deg"],
            "gamma": result["gamma"],
            "cp_max_stagnation": result["cp_max"],
            "CD": result["CD"],
            "CL": result["CL"],
            "CM": result["CM"],
            "CF_total_x": CF[0],
            "CF_total_y": CF[1],
            "CF_total_z": CF[2],
            "CM_total_x": CMv[0],
            "CM_total_y": CMv[1],
            "CM_total_z": CMv[2],
            "n_windward": result["n_windward"],
            "n_leeward": result["n_leeward"],
            "cp_min": float(np.min(result["cp"])),
            "cp_max": float(np.max(result["cp"])),
        })

    return rows



def main():
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

    eD = np.array([0.0, -1.0, 0.0])  # drag positivo en -y
    eL = np.array([0.0, 0.0, -1.0])  # lift en -z
    eM = np.array([1.0, 0.0, 0.0])   # momento alrededor de x

    # ============================================================
    # BARRIDO DE ÁNGULOS
    # ============================================================

    alphas_deg = [0.0, 10.0, 20.0, 30.0]

    '''
    rows = run_alpha_sweep(
        alphas_deg=alphas_deg,
        centers=centers,
        areas=areas,
        normals=normals,
        S_ref=S_ref,
        L_ref=L_ref,
        r_ref=r_ref,
        eD=eD,
        eL=eL,
        eM=eM,
    )

    save_results_csv("results_newton.csv", rows)
    '''

    # ============================================================
    # BARRIDO DE ÁNGULOS - MNM
    # ============================================================

    Mach = 8.0
    gamma = 1.4

    rows_mnm = run_mnm_alpha_sweep(
        alphas_deg=alphas_deg,
        Mach=Mach,
        centers=centers,
        areas=areas,
        normals=normals,
        S_ref=S_ref,
        L_ref=L_ref,
        r_ref=r_ref,
        eD=eD,
        eL=eL,
        eM=eM,
        gamma=gamma,
    )

    save_results_csv_mnm("results_mnm_M8.csv", rows_mnm)
  
      # ============================================================
    # BARRIDO EN MACH PARA SENSIBILIDAD
    # ============================================================

    Mach_list = [2.0, 4.0, 8.0, 12.0]

    rows_mach = run_mnm_mach_sweep(
        Mach_list=Mach_list,
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
        gamma=gamma,
    )

    save_results_csv_mnm("results_mnm_mach_sweep.csv", rows_mach)
    
    result_mnm_20 = solve_modified_newton_case(
        centers=centers,
        areas=areas,
        normals=normals,
        alpha_deg=20.0,
        Mach=8.0,
        S_ref=S_ref,
        L_ref=L_ref,
        r_ref=r_ref,
        eD=eD,
        eL=eL,
        eM=eM,
        gamma=1.4,
    )

    print("\nPrimeros 10 valores de cp MNM (alpha=20, M=8):")
    print(result_mnm_20["cp"][:10])
    print(f"cp_max de remanso = {result_mnm_20['cp_max']}")

    save_cp_faces_csv("cp_faces_mnm_alpha20_M8.csv", centers, result_mnm_20["cp"])
    plot_cp_map(mesh, geom, result_mnm_20["cp"], title="Cp map - MNM - alpha 20 deg - M8")
    # ============================================================
    # MAPA DE CP PARA alpha = 20 deg
    # ============================================================

    '''
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

    print("\nPrimeros 10 valores de cp (alpha=20):")
    print(result_20["cp"][:10])

    save_cp_faces_csv("cp_faces_alpha20.csv", centers, result_20["cp"])
    plot_cp_map(mesh, geom, result_20["cp"], title="Cp map - Newton - alpha 20 deg")
    '''

if __name__ == "__main__":
    main()