"""
runner.py — Implementación del pipeline de cálculo.

Contiene toda la lógica de ejecución: geometría, sweeps, plots, CSV,
ensamblado de casos y exportación.

main.py solo define configuración y llama a run_pipeline(config).
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stl_utils import load_stl, print_mesh_summary, compute_face_geometry, plot_geom
from MN  import solve_newton_case
from MNM import solve_modified_newton_case
from export import (
    build_case_dict, build_reference_block, build_results_json,
    save_json, generate_html, _check,
)
from generate_3d_report import generate_3d_html


# ══════════════════════════════════════════════════════════════════════════════
# Geometría y referencias
# ══════════════════════════════════════════════════════════════════════════════

def get_capsule_refs(mesh, geom):
    """S_ref = caja frontal xz, L_ref = extensión y, r_ref = centroide ponderado."""
    extents = np.asarray(mesh.extents, dtype=float)
    centers = geom["centers"]
    areas   = geom["areas"]
    S_ref = float(extents[0] * extents[2])
    L_ref = float(extents[1])
    r_ref = np.average(centers, axis=0, weights=areas)
    return S_ref, L_ref, r_ref


def get_sphere_refs(geom):
    """S_ref = pi*R^2, L_ref = 2R, r_ref = origen."""
    areas = geom["areas"]
    R     = np.sqrt(areas.sum() / (4 * np.pi))
    return float(np.pi * R**2), float(2 * R), np.zeros(3), R


def wind_axes(alpha_deg):
    """Devuelve (eD, eL, eM) en ejes viento para un alpha dado."""
    a    = np.deg2rad(alpha_deg)
    eD   = np.array([0., -np.cos(a), -np.sin(a)])
    eD  /= np.linalg.norm(eD)
    eM   = np.array([1., 0., 0.])
    eL   = np.cross(eM, eD); eL /= np.linalg.norm(eL)
    return eD, eL, eM


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_cp_map(mesh, geom, cp, title="Cp map", plots_dir=None):
    """Mapa de Cp coloreado sobre la malla 3D."""
    face_vertices = geom["face_vertices"]
    cp = np.asarray(cp, dtype=float)
    cp_min, cp_max = float(cp.min()), float(cp.max())
    cp_norm = np.zeros_like(cp) if np.isclose(cp_max, cp_min) \
              else (cp - cp_min) / (cp_max - cp_min)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        face_vertices,
        facecolors=cm.viridis(cp_norm), edgecolor="k",
        linewidth=0.2, alpha=0.95,
    ))
    bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    bounds_max = np.asarray(mesh.bounds[1], dtype=float)
    center    = 0.5 * (bounds_min + bounds_max)
    max_range = 0.5 * np.max(bounds_max - bounds_min)
    ax.set_xlim(center[0]-max_range, center[0]+max_range)
    ax.set_ylim(center[1]-max_range, center[1]+max_range)
    ax.set_zlim(center[2]-max_range, center[2]+max_range)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    mappable = cm.ScalarMappable(cmap="viridis")
    mappable.set_array(cp)
    plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1).set_label("cp")
    plt.tight_layout()
    if plots_dir is not None:
        fname = title.replace(" ", "_").replace("/", "_").replace("°", "deg") + ".png"
        plt.savefig(Path(plots_dir) / fname, dpi=150)
        print(f"  Figura guardada: {Path(plots_dir) / fname}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Sweeps
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS_MN = [
    "alpha_deg", "CD", "CL", "CM",
    "CF_total_x", "CF_total_y", "CF_total_z",
    "CM_total_x", "CM_total_y", "CM_total_z",
    "n_windward", "n_leeward", "cp_min", "cp_max",
]
CSV_FIELDS_MNM = [
    "alpha_deg", "Mach", "gamma", "cp_max_stagnation",
    "CD", "CL", "CM",
    "CF_total_x", "CF_total_y", "CF_total_z",
    "CM_total_x", "CM_total_y", "CM_total_z",
    "n_windward", "n_leeward", "cp_min", "cp_max",
]


def run_mn_sweep(alphas_deg, centers, areas, normals, S_ref, L_ref, r_ref, eD, eL, eM):
    rows = []
    for alpha in alphas_deg:
        r = solve_newton_case(
            centers=centers, areas=areas, normals=normals, alpha_deg=float(alpha),
            S_ref=S_ref, L_ref=L_ref, r_ref=r_ref, eD=eD, eL=eL, eM=eM,
        )
        CF = r["CF_total"]; CMv = r["CM_total"]
        print(f"  MN  α={alpha:5.1f}°  CD={r['CD']:.5f}  CL={r['CL']:.5f}  CM={r['CM']:.5f}"
              f"  wind={r['n_windward']}")
        rows.append({
            "alpha_deg": r["alpha_deg"], "CD": r["CD"], "CL": r["CL"], "CM": r["CM"],
            "CF_total_x": CF[0], "CF_total_y": CF[1], "CF_total_z": CF[2],
            "CM_total_x": CMv[0], "CM_total_y": CMv[1], "CM_total_z": CMv[2],
            "n_windward": r["n_windward"], "n_leeward": r["n_leeward"],
            "cp_min": float(np.min(r["cp"])), "cp_max": float(np.max(r["cp"])),
        })
    return rows


def run_mnm_sweep(alphas_deg, Mach, centers, areas, normals, S_ref, L_ref, r_ref, eD, eL, eM, gamma=1.4):
    rows = []
    for alpha in alphas_deg:
        r = solve_modified_newton_case(
            centers=centers, areas=areas, normals=normals,
            alpha_deg=float(alpha), Mach=float(Mach),
            S_ref=S_ref, L_ref=L_ref, r_ref=r_ref, eD=eD, eL=eL, eM=eM, gamma=gamma,
        )
        CF = r["CF_total"]; CMv = r["CM_total"]
        print(f"  MNM α={alpha:5.1f}°  M={Mach}  CD={r['CD']:.5f}  CL={r['CL']:.5f}"
              f"  CM={r['CM']:.5f}  cp_max={r['cp_max']:.5f}")
        rows.append({
            "alpha_deg": r["alpha_deg"], "Mach": r["Mach"], "gamma": r["gamma"],
            "cp_max_stagnation": r["cp_max"],
            "CD": r["CD"], "CL": r["CL"], "CM": r["CM"],
            "CF_total_x": CF[0], "CF_total_y": CF[1], "CF_total_z": CF[2],
            "CM_total_x": CMv[0], "CM_total_y": CMv[1], "CM_total_z": CMv[2],
            "n_windward": r["n_windward"], "n_leeward": r["n_leeward"],
            "cp_min": float(np.min(r["cp"])), "cp_max": float(np.max(r["cp"])),
        })
    return rows


def run_mach_sweep(Mach_list, alpha_deg, centers, areas, normals, S_ref, L_ref, r_ref, eD, eL, eM, gamma=1.4):
    rows = []
    for Mach in Mach_list:
        r = solve_modified_newton_case(
            centers=centers, areas=areas, normals=normals,
            alpha_deg=float(alpha_deg), Mach=float(Mach),
            S_ref=S_ref, L_ref=L_ref, r_ref=r_ref, eD=eD, eL=eL, eM=eM, gamma=gamma,
        )
        CF = r["CF_total"]; CMv = r["CM_total"]
        print(f"  MNM M={Mach:5.1f}  α={alpha_deg}°  CD={r['CD']:.5f}"
              f"  cp_max={r['cp_max']:.5f}")
        rows.append({
            "Mach": r["Mach"], "alpha_deg": r["alpha_deg"], "gamma": r["gamma"],
            "cp_max_stagnation": r["cp_max"],
            "CD": r["CD"], "CL": r["CL"], "CM": r["CM"],
            "CF_total_x": CF[0], "CF_total_y": CF[1], "CF_total_z": CF[2],
            "CM_total_x": CMv[0], "CM_total_y": CMv[1], "CM_total_z": CMv[2],
            "n_windward": r["n_windward"], "n_leeward": r["n_leeward"],
            "cp_min": float(np.min(r["cp"])), "cp_max": float(np.max(r["cp"])),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(filepath, rows, fieldnames):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {filepath}")


def save_cp_csv(filepath, centers, cp):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_center", "y_center", "z_center", "cp"])
        for c, cp_i in zip(centers, cp):
            w.writerow([c[0], c[1], c[2], cp_i])
    print(f"  Cp CSV → {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(config: dict, root: Path, csv_dir: Path, plots_dir: Path, results_dir: Path):
    """Ejecuta el pipeline completo a partir del dict de configuración."""

    STL_CAPSULE = config["STL_CAPSULE"]
    STL_SPHERE  = config["STL_SPHERE"]
    STL_COARSE  = config.get("STL_COARSE")
    STL_FINE    = config.get("STL_FINE")

    ALPHAS_DEG  = config["ALPHAS_DEG"]
    MACH        = config["MACH"]
    MACH_SWEEP  = config["MACH_SWEEP"]
    GAMMA       = config.get("GAMMA", 1.4)

    CP_MAP_ALPHA  = config.get("CP_MAP_ALPHA")
    CP_MAP_METHOD = config.get("CP_MAP_METHOD", "MNM")

    RUN_MN_SWEEP         = config.get("RUN_MN_SWEEP", True)
    RUN_MNM_SWEEP        = config.get("RUN_MNM_SWEEP", True)
    RUN_MACH_SWEEP       = config.get("RUN_MACH_SWEEP", True)
    RUN_MESH_SENSITIVITY = config.get("RUN_MESH_SENSITIVITY", True)

    MESH_ALPHA = config.get("MESH_ALPHA", 10.0)
    MESH_MACH  = config.get("MESH_MACH", 8.0)
    MESH_STLS  = config.get("MESH_STLS", [])

    SHOW_GEOMETRY     = config.get("SHOW_GEOMETRY", False)
    ALPHA_TRIM_DEG    = config.get("ALPHA_TRIM_DEG")
    ALPHA_TRIM_METHOD = config.get("ALPHA_TRIM_METHOD", "MNM")

    GROUP_ID = config["GROUP_ID"]
    MEMBERS  = config["MEMBERS"]

    eD0 = np.array([0., -1.,  0.])
    eL0 = np.array([0.,  0., -1.])
    eM0 = np.array([1.,  0.,  0.])

    # ══════════════════════════════════════════════════════════════════════
    # CÁPSULA
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print(f"Cargando cápsula: {STL_CAPSULE.name}")
    print("═"*60)
    mesh_cap = load_stl(STL_CAPSULE)
    print_mesh_summary(mesh_cap)
    geom_cap = compute_face_geometry(mesh_cap)
    c_cap, a_cap, n_cap = geom_cap["centers"], geom_cap["areas"], geom_cap["normals"]
    S_cap, L_cap, r_cap = get_capsule_refs(mesh_cap, geom_cap)
    n_tri_cap = len(a_cap)
    print(f"\n  S_ref={S_cap:.2f} mm²  L_ref={L_cap:.2f} mm  r_ref_y={r_cap[1]:.2f} mm")

    if SHOW_GEOMETRY:
        diag = np.linalg.norm(mesh_cap.bounds[1] - mesh_cap.bounds[0])
        plot_geom(geom_cap, show_mesh=True, show_centers=True, show_normals=True,
                  normal_scale=0.03*diag, color_by="area", title="Cápsula — geometría")

    if RUN_MN_SWEEP:
        print(f"\n── Barrido MN  | α={ALPHAS_DEG}°")
        rows_mn = run_mn_sweep(ALPHAS_DEG, c_cap, a_cap, n_cap,
                               S_cap, L_cap, r_cap, eD0, eL0, eM0)
        save_csv(csv_dir / "results_mn.csv", rows_mn, CSV_FIELDS_MN)

    if RUN_MNM_SWEEP:
        print(f"\n── Barrido MNM | α={ALPHAS_DEG}°  M={MACH}")
        rows_mnm = run_mnm_sweep(ALPHAS_DEG, MACH, c_cap, a_cap, n_cap,
                                 S_cap, L_cap, r_cap, eD0, eL0, eM0, GAMMA)
        save_csv(csv_dir / f"results_mnm_M{int(MACH)}.csv", rows_mnm, CSV_FIELDS_MNM)

    if RUN_MACH_SWEEP:
        print(f"\n── Barrido Mach MNM | M={MACH_SWEEP}  α=0°")
        rows_mach = run_mach_sweep(MACH_SWEEP, 0.0, c_cap, a_cap, n_cap,
                                   S_cap, L_cap, r_cap, eD0, eL0, eM0, GAMMA)
        save_csv(csv_dir / "results_mnm_mach_sweep.csv", rows_mach, CSV_FIELDS_MNM)

    if ALPHA_TRIM_DEG is not None:
        print(f"\n── Coeficientes en α_trim = {ALPHA_TRIM_DEG}°  |  {ALPHA_TRIM_METHOD}  M={MACH}")
        eD_t, eL_t, eM_t = wind_axes(ALPHA_TRIM_DEG)
        if ALPHA_TRIM_METHOD == "MN":
            r_t = solve_newton_case(
                centers=c_cap, areas=a_cap, normals=n_cap, alpha_deg=ALPHA_TRIM_DEG,
                S_ref=S_cap, L_ref=L_cap, r_ref=r_cap, eD=eD_t, eL=eL_t, eM=eM_t,
            )
        else:
            r_t = solve_modified_newton_case(
                centers=c_cap, areas=a_cap, normals=n_cap,
                alpha_deg=ALPHA_TRIM_DEG, Mach=MACH,
                S_ref=S_cap, L_ref=L_cap, r_ref=r_cap, eD=eD_t, eL=eL_t, eM=eM_t, gamma=GAMMA,
            )
        CD_t = float(r_t["CD"]); CL_t = float(r_t["CL"]); CM_t = float(r_t["CM"])
        LD_t = CL_t / CD_t if CD_t != 0 else float("inf")
        print(f"  CD={CD_t:.5f}  CL={CL_t:.5f}  CM={CM_t:.5f}  L/D={LD_t:.4f}")

    if CP_MAP_ALPHA is not None:
        print(f"\n── Mapa Cp | {CP_MAP_METHOD}  α={CP_MAP_ALPHA}°  M={MACH}")
        if CP_MAP_METHOD == "MN":
            r_cp = solve_newton_case(
                centers=c_cap, areas=a_cap, normals=n_cap, alpha_deg=CP_MAP_ALPHA,
                S_ref=S_cap, L_ref=L_cap, r_ref=r_cap, eD=eD0, eL=eL0, eM=eM0,
            )
        else:
            r_cp = solve_modified_newton_case(
                centers=c_cap, areas=a_cap, normals=n_cap,
                alpha_deg=CP_MAP_ALPHA, Mach=MACH,
                S_ref=S_cap, L_ref=L_cap, r_ref=r_cap, eD=eD0, eL=eL0, eM=eM0, gamma=GAMMA,
            )
        title_cp = f"Cp map - {CP_MAP_METHOD} - alpha {CP_MAP_ALPHA} deg - M{int(MACH)}"
        save_cp_csv(
            csv_dir / f"cp_faces_{CP_MAP_METHOD.lower()}_a{int(CP_MAP_ALPHA)}_M{int(MACH)}.csv",
            c_cap, r_cp["cp"],
        )
        plot_cp_map(mesh_cap, geom_cap, r_cp["cp"], title=title_cp, plots_dir=plots_dir)

    # ══════════════════════════════════════════════════════════════════════
    # ESFERA
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print(f"Cargando esfera: {STL_SPHERE.name}")
    print("═"*60)
    mesh_sp = load_stl(STL_SPHERE)
    geom_sp = compute_face_geometry(mesh_sp)
    c_sp, a_sp, n_sp = geom_sp["centers"], geom_sp["areas"], geom_sp["normals"]
    S_sp, L_sp, r_sp, R_sp = get_sphere_refs(geom_sp)
    n_tri_sp = len(a_sp)
    print(f"  R={R_sp:.2f} mm  S_ref={S_sp:.2f} mm²  L_ref={L_sp:.2f} mm  triángulos={n_tri_sp}")

    if STL_COARSE is not None and STL_COARSE.exists():
        print(f"  Cargando malla gruesa: {STL_COARSE.name}")
        mesh_co = load_stl(STL_COARSE)
        geom_co = compute_face_geometry(mesh_co)
        c_co, a_co, n_co = geom_co["centers"], geom_co["areas"], geom_co["normals"]
        n_tri_co = len(a_co)
    else:
        c_co, a_co, n_co, n_tri_co = c_cap, a_cap, n_cap, n_tri_cap
        STL_COARSE = STL_CAPSULE

    if STL_FINE is not None and STL_FINE.exists():
        print(f"  Cargando malla fina:   {STL_FINE.name}")
        mesh_fi = load_stl(STL_FINE)
        geom_fi = compute_face_geometry(mesh_fi)
        c_fi, a_fi, n_fi = geom_fi["centers"], geom_fi["areas"], geom_fi["normals"]
        n_tri_fi = len(a_fi)
    else:
        c_fi, a_fi, n_fi, n_tri_fi = c_cap, a_cap, n_cap, n_tri_cap
        STL_FINE = STL_CAPSULE

    # ══════════════════════════════════════════════════════════════════════
    # 9 CASOS OBLIGATORIOS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("Calculando los 9 casos obligatorios para results.json …")
    print("═"*60)

    def _mn(centers, areas, normals, alpha, S, L, r, eD, eL, eM):
        return solve_newton_case(
            centers=centers, areas=areas, normals=normals, alpha_deg=float(alpha),
            S_ref=S, L_ref=L, r_ref=r, eD=eD, eL=eL, eM=eM,
        )

    def _mnm(centers, areas, normals, alpha, M, S, L, r, eD, eL, eM, g=1.4):
        return solve_modified_newton_case(
            centers=centers, areas=areas, normals=normals,
            alpha_deg=float(alpha), Mach=float(M),
            S_ref=S, L_ref=L, r_ref=r, eD=eD, eL=eL, eM=eM, gamma=g,
        )

    def _case(case_id, geo_name, stl_path, tri, model, M, alpha, res):
        eD_w, eL_w, eM_w = wind_axes(alpha)
        CF = res["CF_total"]; CMv = res["CM_total"]
        CD = float(np.dot(CF, eD_w))
        CL = float(np.dot(CF, eL_w))
        CM = float(np.dot(CMv, eM_w))
        print(f"  {case_id:45s}  CD={CD:.5f}  CL={CL:.5f}  CM={CM:.5f}")
        return build_case_dict(
            case_id=case_id, geometry_name=geo_name,
            stl_file=str(stl_path.relative_to(root)).replace("\\", "/"),
            triangles=tri, model=model, Mach=float(M), alpha_deg=float(alpha),
            CF_total=CF, CM_total=CMv, CD=CD, CL=CL, CM=CM,
        )

    eD0w, eL0w, eM0w = wind_axes(0.0)
    eD10, eL10, eM10 = wind_axes(10.0)
    eD20, eL20, eM20 = wind_axes(20.0)

    r1  = _mn (c_sp,  a_sp,  n_sp,  0,  S_sp,  L_sp,  r_sp,  eD0w,  eL0w,  eM0w)
    r2  = _mnm(c_sp,  a_sp,  n_sp,  0,  2,     S_sp,  L_sp,  r_sp,  eD0w,  eL0w,  eM0w)
    r3  = _mnm(c_sp,  a_sp,  n_sp,  0,  8,     S_sp,  L_sp,  r_sp,  eD0w,  eL0w,  eM0w)
    r4  = _mn (c_cap, a_cap, n_cap, 0,  S_cap, L_cap, r_cap, eD0w,  eL0w,  eM0w)
    r5  = _mnm(c_cap, a_cap, n_cap, 0,  8,     S_cap, L_cap, r_cap, eD0w,  eL0w,  eM0w)
    r6  = _mnm(c_cap, a_cap, n_cap, 10, 8,     S_cap, L_cap, r_cap, eD10,  eL10,  eM10)
    r7  = _mnm(c_cap, a_cap, n_cap, 20, 8,     S_cap, L_cap, r_cap, eD20,  eL20,  eM20)
    r8  = _mn (c_co,  a_co,  n_co,  10, S_cap, L_cap, r_cap, eD10,  eL10,  eM10)
    r9  = _mn (c_fi,  a_fi,  n_fi,  10, S_cap, L_cap, r_cap, eD10,  eL10,  eM10)

    cases = [
        _case("sphere_MN_a0_M8",               "sphere",  STL_SPHERE,  n_tri_sp,  "MN",  8,  0,  r1),
        _case("sphere_MNM_a0_M2",              "sphere",  STL_SPHERE,  n_tri_sp,  "MNM", 2,  0,  r2),
        _case("sphere_MNM_a0_M8",              "sphere",  STL_SPHERE,  n_tri_sp,  "MNM", 8,  0,  r3),
        _case("capsule_MN_a0_M8",              "capsule", STL_CAPSULE, n_tri_cap, "MN",  8,  0,  r4),
        _case("capsule_MNM_a0_M8",             "capsule", STL_CAPSULE, n_tri_cap, "MNM", 8,  0,  r5),
        _case("capsule_MNM_a10_M8",            "capsule", STL_CAPSULE, n_tri_cap, "MNM", 8,  10, r6),
        _case("capsule_MNM_a20_M8",            "capsule", STL_CAPSULE, n_tri_cap, "MNM", 8,  20, r7),
        _case("capsule_MN_mesh_coarse_a10_M8", "capsule", STL_COARSE,  n_tri_co,  "MN",  8,  10, r8),
        _case("capsule_MN_mesh_fine_a10_M8",   "capsule", STL_FINE,    n_tri_fi,  "MN",  8,  10, r9),
    ]

    # ══════════════════════════════════════════════════════════════════════
    # SENSIBILIDAD DE MALLA
    # ══════════════════════════════════════════════════════════════════════
    if RUN_MESH_SENSITIVITY and MESH_STLS:
        print("\n" + "═"*60)
        print(f"Sensibilidad de malla | alpha={MESH_ALPHA}° | M={MESH_MACH}")
        print("═"*60)
        eD_ms, eL_ms, eM_ms = wind_axes(MESH_ALPHA)
        rows_mesh = []
        for stl_p in MESH_STLS:
            if not stl_p.exists():
                print(f"  [SKIP] {stl_p.name} — no encontrado")
                continue
            m_ms  = load_stl(stl_p)
            g_ms  = compute_face_geometry(m_ms)
            c_ms, a_ms, n_ms = g_ms["centers"], g_ms["areas"], g_ms["normals"]
            S_ms, L_ms, r_ms = get_capsule_refs(m_ms, g_ms)
            n_f = len(a_ms)
            print(f"  {stl_p.name:25s}  ({n_f} caras)")
            for model in ("MN", "MNM"):
                if model == "MN":
                    r_ms_res = solve_newton_case(
                        centers=c_ms, areas=a_ms, normals=n_ms,
                        alpha_deg=MESH_ALPHA, S_ref=S_ms, L_ref=L_ms, r_ref=r_ms,
                        eD=eD_ms, eL=eL_ms, eM=eM_ms,
                    )
                else:
                    r_ms_res = solve_modified_newton_case(
                        centers=c_ms, areas=a_ms, normals=n_ms,
                        alpha_deg=MESH_ALPHA, Mach=MESH_MACH,
                        S_ref=S_ms, L_ref=L_ms, r_ref=r_ms,
                        eD=eD_ms, eL=eL_ms, eM=eM_ms, gamma=GAMMA,
                    )
                rows_mesh.append({
                    "stl":       stl_p.name,
                    "n_faces":   n_f,
                    "model":     model,
                    "alpha_deg": MESH_ALPHA,
                    "Mach":      MESH_MACH,
                    "CD":        r_ms_res["CD"],
                    "CL":        r_ms_res["CL"],
                    "CM":        r_ms_res["CM"],
                    "n_windward": r_ms_res["n_windward"],
                })
                print(f"    {model}: CD={r_ms_res['CD']:.5f}  CL={r_ms_res['CL']:.5f}"
                      f"  CM={r_ms_res['CM']:.5f}")
        save_csv(
            csv_dir / "results_mesh_sensitivity.csv",
            rows_mesh,
            ["stl", "n_faces", "model", "alpha_deg", "Mach", "CD", "CL", "CM", "n_windward"],
        )

    # ══════════════════════════════════════════════════════════════════════
    # EXPORTAR results.json + reports
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("Exportando results.json y reportes …")
    print("═"*60)

    eD0w, eL0w, _ = wind_axes(0.0)
    reference = build_reference_block(
        Sref_m2=S_cap * 1e-6,
        Lref_m=L_cap * 1e-3,
        moment_reference_point_m=(r_cap * 1e-3).tolist(),
        vinf_direction_body=eD0w.tolist(),
        CD_axis_body=eD0w.tolist(),
        CL_axis_body=eL0w.tolist(),
        CM_axis_body=[1., 0., 0.],
    )
    team    = {"group_id": GROUP_ID, "members": MEMBERS}
    results = build_results_json(cases=cases, team=team, reference=reference)

    save_json(results, results_dir / "results.json")
    generate_html(results, results_dir / "report.html",
                  sphere_sref=S_sp * 1e-6, sphere_lref=L_sp * 1e-3)
    _check(results)

    cp_csv = (
        csv_dir / f"cp_faces_{CP_MAP_METHOD.lower()}_a{int(abs(CP_MAP_ALPHA))}_M{int(MACH)}.csv"
        if CP_MAP_ALPHA is not None else None
    )
    mesh_csv = csv_dir / "results_mesh_sensitivity.csv" if RUN_MESH_SENSITIVITY else None

    generate_3d_html(
        results_json  = results_dir / "results.json",
        cp_csv        = cp_csv,
        stl_capsule   = STL_CAPSULE,
        stl_sphere    = STL_SPHERE,
        stl_coarse    = MESH_STLS[0] if RUN_MESH_SENSITIVITY and MESH_STLS else None,
        mesh_sens_csv = mesh_csv,
        out_path      = results_dir / "report_3d.html",
    )
