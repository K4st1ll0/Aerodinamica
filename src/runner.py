"""
runner.py — Implementación del pipeline de cálculo.

Contiene toda la lógica de ejecución: geometría, sweeps, plots, CSV,
ensamblado de casos y exportación.

main.py solo define configuración y llama a run_pipeline(config).
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Estilo formal para todas las gráficas ──────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "lines.linewidth":   1.8,
    "savefig.facecolor": "white",
    "savefig.dpi":       150,
})

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
    """Devuelve (eD, eL, eM) en ejes viento para un alpha dado.
    flujo en +y a alpha=0 (escudo térmico en -y enfrenta el flujo)."""
    a    = np.deg2rad(alpha_deg)
    eD   = np.array([0., +np.cos(a), +np.sin(a)])
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
# Plots analíticos
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "MN":      "#2196F3",
    "MNM":     "#FF5722",
    "coarse":  "#E91E63",
    "fine":    "#4CAF50",
    "medium":  "#FF9800",
    "sphere":  "#00BCD4",
    "capsule": "#9C27B0",
}

def _setup_ax(ax, xlabel="", ylabel="", title="", grid=True):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=10)


def generate_plots(
    plots_dir,
    rows_mn=None,
    rows_mnm=None,
    rows_mach=None,
    sphere_mach_rows=None,
    sphere_mn_cases=None,
    mesh_sweeps_mn=None,
    mesh_sweeps_mnm=None,
    mesh_mach_sweeps_mnm=None,
    mesh_tri_counts=None,
):
    """Genera todas las gráficas analíticas y las guarda en plots_dir."""
    pd_dir   = Path(plots_dir)
    mesh_dir = pd_dir / "AnalisisMallado"
    pd_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    def _col(rows, key):
        return [r[key] for r in rows] if rows else []

    # ── 1. CD / CL / CM vs alpha — Cápsula (MN vs MNM) ─────────────────────
    if rows_mn and rows_mnm:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Cápsula — Coeficientes vs α  (MN vs MNM)", fontsize=14, fontweight="bold")
        pairs = [("CD", "CD"), ("CL", "CL"), ("CM", "CM")]
        for ax, (key, label) in zip(axes, pairs):
            a_mn  = _col(rows_mn,  "alpha_deg")
            v_mn  = _col(rows_mn,  key)
            a_mnm = _col(rows_mnm, "alpha_deg")
            v_mnm = _col(rows_mnm, key)
            ax.plot(a_mn,  v_mn,  "o-", color=_COLORS["MN"],  label="MN",  lw=2, ms=6)
            ax.plot(a_mnm, v_mnm, "s-", color=_COLORS["MNM"], label="MNM", lw=2, ms=6)
            _setup_ax(ax, "α (°)", label, f"{label} vs α")
            ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(pd_dir / "capsule_coefficients_vs_alpha.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ capsule_coefficients_vs_alpha.png")

    # ── 2. CD / CL / CM vs alpha — Cápsula (MN, por separado) ──────────────
    if rows_mn:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Cápsula — Coeficientes vs α  (MN)", fontsize=14, fontweight="bold")
        for ax, key in zip(axes, ["CD", "CL", "CM"]):
            ax.plot(_col(rows_mn, "alpha_deg"), _col(rows_mn, key),
                    "o-", color=_COLORS["MN"], lw=2.5, ms=7)
            _setup_ax(ax, "α (°)", key, f"{key} vs α (MN)")
        plt.tight_layout()
        fig.savefig(pd_dir / "capsule_MN_vs_alpha.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ capsule_MN_vs_alpha.png")

    # ── 3. CD / CL / CM vs alpha — Cápsula (MNM, por separado) ─────────────
    if rows_mnm:
        M_label = rows_mnm[0].get("Mach", "?") if rows_mnm else "?"
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Cápsula — Coeficientes vs α  (MNM, M∞={M_label})", fontsize=14, fontweight="bold")
        for ax, key in zip(axes, ["CD", "CL", "CM"]):
            ax.plot(_col(rows_mnm, "alpha_deg"), _col(rows_mnm, key),
                    "s-", color=_COLORS["MNM"], lw=2.5, ms=7)
            _setup_ax(ax, "α (°)", key, f"{key} vs α (MNM)")
        plt.tight_layout()
        fig.savefig(pd_dir / "capsule_MNM_vs_alpha.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ capsule_MNM_vs_alpha.png")

    # ── 4. CD / CL / CM y Cp_max vs Mach — Cápsula (MNM, α=0) ─────────────
    if rows_mach:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Cápsula — Coeficientes vs M∞  (MNM, α=0°)", fontsize=14, fontweight="bold")
        M = _col(rows_mach, "Mach")
        for ax, key, color in zip(
            axes.flat[:3], ["CD", "CL", "CM"],
            [_COLORS["MNM"], "#4CAF50", "#9C27B0"],
        ):
            ax.plot(M, _col(rows_mach, key), "D-", color=color, lw=2, ms=6)
            _setup_ax(ax, "M∞", key, f"{key} vs M∞ (MNM)")
        ax_cp = axes.flat[3]
        ax_cp.plot(M, _col(rows_mach, "cp_max_stagnation"), "^-",
                   color=_COLORS["sphere"], lw=2, ms=6)
        _setup_ax(ax_cp, "M∞", "Cp,max", "Cp,max de estancamiento vs M∞")
        plt.tight_layout()
        fig.savefig(pd_dir / "capsule_MNM_vs_mach.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ capsule_MNM_vs_mach.png")

    # ── 5. Esfera: CD vs Mach (MNM sweep) ───────────────────────────────────
    if sphere_mach_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Esfera — Coeficientes vs M∞  (MNM, α=0°)", fontsize=14, fontweight="bold")
        M  = _col(sphere_mach_rows, "Mach")
        CD = _col(sphere_mach_rows, "CD")
        CP = _col(sphere_mach_rows, "cp_max_stagnation")
        ax0, ax1 = axes
        ax0.plot(M, CD, "o-", color=_COLORS["sphere"], lw=2.5, ms=7, label="MNM")
        ax0.axhline(y=2.0, color=_COLORS["MN"], lw=1.5, ls="--", label="MN (CD=2)")
        if sphere_mn_cases:
            ax0.scatter([8], [sphere_mn_cases.get("sphere_MN_a0_M8_CD", None)],
                        color=_COLORS["MN"], zorder=5, s=60, marker="^", label="MN M=8")
        _setup_ax(ax0, "M∞", "CD", "CD vs M∞ — Esfera")
        ax0.legend(fontsize=10)
        ax1.plot(M, CP, "^-", color="#FF9800", lw=2.5, ms=7)
        _setup_ax(ax1, "M∞", "Cp,max", "Cp,max vs M∞ — Esfera (Rayleigh-Pitot)")
        plt.tight_layout()
        fig.savefig(pd_dir / "sphere_vs_mach.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ sphere_vs_mach.png")

    # ── helpers de malla (usados en secciones 6 y 7) ─────────────────────────
    _tri = mesh_tri_counts or {}
    _mesh_colors  = plt.cm.plasma(np.linspace(0.1, 0.85, max(len(mesh_sweeps_mn or {}), 1)))
    _mesh_markers = ["o", "s", "^", "D", "v", "P", "X"]

    def _mesh_lbl(name):
        n = _tri.get(name)
        return f"ARD_{n:,}" if n else name

    # ── 6. Error relativo — referencia = malla más fina ─────────────────────
    def _mesh_error_plots(sweeps, method_label, fname_suffix, x_key, x_label, coefs=None):
        """Plots error (%) de cada malla frente a la ultrafina.
        x_key / x_label: 'alpha_deg'/'α (°)'  o  'Mach'/'M∞'.
        """
        if not sweeps or len(sweeps) < 2:
            return
        if coefs is None:
            coefs = ["CD", "CL", "CM"]
        _tri_local = mesh_tri_counts or {}
        ref_name   = max(sweeps.keys(), key=lambda n: _tri_local.get(n, 0))
        ref_rows   = sweeps[ref_name]

        colors_err  = plt.cm.viridis(np.linspace(0.1, 0.85, len(sweeps) - 1))
        markers_err = ["o", "s", "^", "D", "v", "P"]

        for coef in coefs:
            ref_vals = {r[x_key]: r[coef] for r in ref_rows}
            fig, ax  = plt.subplots(figsize=(9, 5))
            ci = 0
            for name, rows in sweeps.items():
                if name == ref_name:
                    continue
                xs   = [r[x_key] for r in rows]
                errs = []
                for r in rows:
                    ref_v = ref_vals.get(r[x_key])
                    if ref_v is None or abs(ref_v) < 1e-12:
                        errs.append(0.0)
                    else:
                        errs.append(abs(r[coef] - ref_v) / abs(ref_v) * 100.0)
                ax.plot(xs, errs,
                        marker=markers_err[ci % len(markers_err)],
                        color=colors_err[ci], lw=2, ms=6, label=_mesh_lbl(name))
                ci += 1
            ax.axhline(y=5, color="red",    ls="--", lw=1.5, label="5%")
            ax.axhline(y=1, color="orange", ls=":",  lw=1.2, label="1%")
            ref_lbl = _mesh_lbl(ref_name)
            _setup_ax(ax, x_label, f"Δ{coef} / {coef}_ref  (%)",
                      f"Error {coef} vs {x_label} — ref. {ref_lbl}  ({method_label})")
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(mesh_dir / f"mesh_error_{coef}_{fname_suffix}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ mesh_error_{coef}_{fname_suffix}.png")

    # vs alpha
    _mesh_error_plots(mesh_sweeps_mn,  "MN",  "MN_alpha",  "alpha_deg", "α (°)")
    _mesh_error_plots(mesh_sweeps_mnm, "MNM", "MNM_alpha", "alpha_deg", "α (°)")
    # vs Mach  (MNM; MN no depende de Mach)
    _mesh_error_plots(mesh_mach_sweeps_mnm, "MNM", "MNM_mach", "Mach", "M∞",
                      coefs=["CD", "CL", "CM", "cp_max_stagnation"])

    # ── 7. Todas las mallas: coeficientes vs alpha ───────────────────────────
    if mesh_sweeps_mn:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Sensibilidad de malla — Coeficientes vs α  (MN)", fontsize=14, fontweight="bold")
        for ax, key in zip(axes, ["CD", "CL", "CM"]):
            for idx, (name, rows) in enumerate(mesh_sweeps_mn.items()):
                ax.plot(_col(rows, "alpha_deg"), _col(rows, key),
                        f"{_mesh_markers[idx % len(_mesh_markers)]}-",
                        color=_mesh_colors[idx], label=_mesh_lbl(name), lw=2, ms=6)
            _setup_ax(ax, "α (°)", key, f"{key} vs α — todas las mallas (MN)")
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(mesh_dir / "mesh_all_alpha_MN.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ mesh_all_alpha_MN.png")

    if mesh_sweeps_mnm:
        mach_label = next(iter(mesh_sweeps_mnm.values()))[0].get("Mach", "?") if mesh_sweeps_mnm else "?"
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sensibilidad de malla — Coeficientes vs α  (MNM, M∞={mach_label})", fontsize=14, fontweight="bold")
        for ax, key in zip(axes, ["CD", "CL", "CM"]):
            for idx, (name, rows) in enumerate(mesh_sweeps_mnm.items()):
                ax.plot(_col(rows, "alpha_deg"), _col(rows, key),
                        f"{_mesh_markers[idx % len(_mesh_markers)]}-",
                        color=_mesh_colors[idx], label=_mesh_lbl(name), lw=2, ms=6)
            _setup_ax(ax, "α (°)", key, f"{key} vs α — todas las mallas (MNM)")
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(mesh_dir / "mesh_all_alpha_MNM.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ mesh_all_alpha_MNM.png")

    # ── 8. Todas las mallas: CD / CL / CM / Cp_max vs Mach (MNM) ────────────
    if mesh_mach_sweeps_mnm and len(mesh_mach_sweeps_mnm) >= 1:
        alpha_lbl = next(iter(mesh_mach_sweeps_mnm.values()))[0].get("alpha_deg", "?")
        for coef, ylabel, fname in [
            ("CD",                 "CD",     "mesh_mach_CD_MNM.png"),
            ("CL",                 "CL",     "mesh_mach_CL_MNM.png"),
            ("CM",                 "CM",     "mesh_mach_CM_MNM.png"),
            ("cp_max_stagnation",  "Cp,max", "mesh_mach_Cpmax_MNM.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for idx, (name, rows) in enumerate(mesh_mach_sweeps_mnm.items()):
                M_vals = _col(rows, "Mach")
                c_vals = _col(rows, coef)
                ax.plot(M_vals, c_vals,
                        marker=_mesh_markers[idx % len(_mesh_markers)],
                        color=_mesh_colors[idx], lw=2, ms=6, label=_mesh_lbl(name))
            _setup_ax(ax, "M∞", ylabel,
                      f"{ylabel} vs M∞ — todas las mallas (MNM, α={alpha_lbl}°)")
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(mesh_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ {fname}")

    # ── 9. Cp_max vs alpha (MN vs MNM) ──────────────────────────────────────
    if rows_mn and rows_mnm:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(_col(rows_mn,  "alpha_deg"), _col(rows_mn,  "cp_max"),
                "o-", color=_COLORS["MN"],  lw=2.5, ms=7, label="MN  Cp,max")
        ax.plot(_col(rows_mn,  "alpha_deg"), _col(rows_mn,  "cp_min"),
                "o--", color=_COLORS["MN"], lw=1.5, ms=5, alpha=0.6, label="MN  Cp,min")
        ax.plot(_col(rows_mnm, "alpha_deg"), _col(rows_mnm, "cp_max_stagnation"),
                "s-", color=_COLORS["MNM"], lw=2.5, ms=7, label="MNM Cp,max")
        ax.plot(_col(rows_mnm, "alpha_deg"), _col(rows_mnm, "cp_min"),
                "s--", color=_COLORS["MNM"], lw=1.5, ms=5, alpha=0.6, label="MNM Cp,min")
        _setup_ax(ax, "α (°)", "Cp", "Cp máximo y mínimo vs α — Cápsula (MN vs MNM)")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(pd_dir / "capsule_Cp_vs_alpha.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ capsule_Cp_vs_alpha.png")

    print(f"\n  Gráficas guardadas en: {pd_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Barlovento / Sotavento + CD vs α multi-Mach
# ══════════════════════════════════════════════════════════════════════════════

def generate_windward_plots(
    centers, areas, normals, S_ref, L_ref, r_ref,
    alphas_deg: list, mach_sweep: list,
    plots_dir: Path, alpha_ref: float = 20.0, gamma: float = 1.4,
):
    """
    Guarda en plots_dir/MNM_y_Barlovento/:
      1. cp_barlovento_sotavento.png — Cp de cada cara (barlovento/sotavento) para MN y MNM
      2. CD_vs_alpha_multiMach.png   — CD vs α: curva MN + curvas MNM a varios Mach
    """
    out_dir = Path(plots_dir) / "MNM_y_Barlovento"
    out_dir.mkdir(parents=True, exist_ok=True)

    eD, eL, eM = wind_axes(alpha_ref)

    # ── Resolver MN ─────────────────────────────────────────────────────────
    r_mn  = solve_newton_case(
        centers=centers, areas=areas, normals=normals,
        alpha_deg=alpha_ref, S_ref=S_ref, L_ref=L_ref, r_ref=r_ref,
        eD=eD, eL=eL, eM=eM,
    )
    mu_mn  = np.asarray(r_mn["mu"])
    cp_mn  = np.asarray(r_mn["cp"])

    # ── Resolver MNM (Mach = mach_sweep[len//2] para la referencia) ─────────
    mach_ref = mach_sweep[len(mach_sweep) // 2] if mach_sweep else 8.0
    r_mnm = solve_modified_newton_case(
        centers=centers, areas=areas, normals=normals,
        alpha_deg=alpha_ref, Mach=mach_ref, S_ref=S_ref, L_ref=L_ref, r_ref=r_ref,
        eD=eD, eL=eL, eM=eM, gamma=gamma,
    )
    mu_mnm = np.asarray(r_mnm["mu"])
    cp_mnm = np.asarray(r_mnm["cp"])

    # ── Plot 1: Cp por cara — barlovento y sotavento ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Distribución Cp por cara — cápsula  (α={alpha_ref:.0f}°)",
                 fontsize=13, fontweight="bold")

    for ax, mu, cp, label, mach_lbl in [
        (axes[0], mu_mn,  cp_mn,  "MN",  ""),
        (axes[1], mu_mnm, cp_mnm, "MNM", f"  M={mach_ref:.0f}"),
    ]:
        phi = np.degrees(np.arccos(np.clip(mu, -1.0, 1.0)))   # 0° = estancamiento
        wind = mu > 0
        lee  = ~wind

        ax.scatter(phi[wind], cp[wind],  s=6, alpha=0.5, color="#E53935",
                   label=f"Barlovento  ({wind.sum():,} caras)")
        ax.scatter(phi[lee],  cp[lee],   s=6, alpha=0.4, color="#1E88E5",
                   label=f"Sotavento   ({lee.sum():,} caras)")
        ax.axvline(x=90, color="gray", ls="--", lw=1, label="φ = 90°")
        _setup_ax(ax, "φ (°) desde estancamiento", "Cp",
                  f"{label}{mach_lbl}")
        ax.legend(fontsize=9, markerscale=2)

    plt.tight_layout()
    fig.savefig(out_dir / "cp_barlovento_sotavento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ cp_barlovento_sotavento.png")

    # ── Plot 2: CD vs α — MN + MNM a varios Mach ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("CD vs α — MN y MNM (varios Mach)", fontsize=13, fontweight="bold")

    # Ejes cuerpo (mismos que el pipeline principal)
    _eD = np.array([0., -1., 0.])
    _eL = np.array([0.,  0., -1.])
    _eM = np.array([1.,  0., 0.])

    # Curva MN
    rows_mn = run_mn_sweep(alphas_deg, centers, areas, normals,
                           S_ref, L_ref, r_ref, _eD, _eL, _eM)
    ax.plot([r["alpha_deg"] for r in rows_mn], [r["CD"] for r in rows_mn],
            "o-", color=_COLORS["MN"], lw=2.5, ms=7, label="MN")

    # Curvas MNM a cada Mach
    mach_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(mach_sweep)))
    for idx, M in enumerate(mach_sweep):
        rows_m = run_mnm_sweep(alphas_deg, M, centers, areas, normals,
                               S_ref, L_ref, r_ref, _eD, _eL, _eM, gamma)
        ax.plot([r["alpha_deg"] for r in rows_m], [r["CD"] for r in rows_m],
                "s--", color=mach_colors[idx], lw=1.8, ms=5, label=f"MNM  M={M:.0f}")

    _setup_ax(ax, "α (°)", "CD", "CD vs α — MN y MNM a varios Mach  (Cápsula)")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    fig.savefig(out_dir / "CD_vs_alpha_multiMach.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ CD_vs_alpha_multiMach.png")


# ══════════════════════════════════════════════════════════════════════════════
# Validación: esfera vs datos MATLAB
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(config: dict, plots_dir: Path, gamma: float = 1.4):
    """Compara el modelo MN/MNM con los datos de la esfera exportados desde MATLAB.

    Genera plots en plots_dir/Validacion/:
      - validacion_Cp_vs_cosphi.png
      - validacion_Cp_vs_angulo.png
      - validacion_CD_vs_Mach.png
    """
    import trimesh

    STL_SPHERE  = config.get("STL_VALIDATION_SPHERE",
                             config.get("STL_SPHERE"))          # esfera1 o esfera
    MATLAB_CSV  = config.get("VALIDATION_CSV")                  # obligatorio
    MACH_LIST   = config.get("VALIDATION_MACH_LIST",
                             [2, 4, 6, 8, 10, 15, 20])

    if not MATLAB_CSV or not Path(MATLAB_CSV).exists():
        print(f"  [SKIP] Validación: CSV no encontrado ({MATLAB_CSV})")
        return
    if not STL_SPHERE or not Path(STL_SPHERE).exists():
        print(f"  [SKIP] Validación: STL esfera no encontrado ({STL_SPHERE})")
        return

    val_dir = Path(plots_dir) / "Validacion"
    val_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar STL ──────────────────────────────────────────────────────────
    mesh   = trimesh.load(str(STL_SPHERE), force="mesh")
    verts  = np.asarray(mesh.vertices, dtype=float)
    faces  = np.asarray(mesh.faces,    dtype=int)
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    cross  = np.cross(v1-v0, v2-v0)
    norms_ = np.linalg.norm(cross, axis=1, keepdims=True)
    tnorm  = cross / (norms_ + 1e-30)          # normales unitarias outward
    areas  = np.linalg.norm(cross, axis=1) / 2.0
    centers = (v0 + v1 + v2) / 3.0

    # ── Cargar CSV MATLAB ───────────────────────────────────────────────────
    cp_matlab = np.genfromtxt(MATLAB_CSV, delimiter=",", skip_header=1,
                               usecols=[0])
    n_stl = len(faces)
    n_csv = len(cp_matlab)
    if n_stl != n_csv:
        print(f"  [WARN] Validación: STL tiene {n_stl} caras, CSV tiene {n_csv} filas. "
              f"Usando las primeras {min(n_stl, n_csv)}.")
        n = min(n_stl, n_csv)
        cp_matlab = cp_matlab[:n]
        tnorm = tnorm[:n]
        areas = areas[:n]
        centers = centers[:n]

    # ── Ángulos ─────────────────────────────────────────────────────────────
    U_DIR   = np.array([0., 1., 0.])           # dirección flujo en MATLAB
    Stheta  = tnorm @ U_DIR                     # dot(U, outward_normal)

    # ── MN numérico sobre el STL ─────────────────────────────────────────────
    # Nota: nuestro solver usa flujo en -y (alpha=0), MATLAB usa flujo en +y.
    # Para una esfera simétrica ambas mitades dan la misma distribución Cp vs mu.
    # Usamos mu del solver como eje x para el modelo, y -Stheta para MATLAB.
    ext   = verts.max(0) - verts.min(0)
    S_ref = float(ext[0] * ext[2])
    L_ref = float(ext[1])
    r_ref = np.average(centers, axis=0, weights=areas)
    # eD en la dirección del flujo del solver ([0,-1,0])
    flow_solver = np.array([0., -1., 0.])
    eM = np.array([1., 0., 0.])
    eL = np.cross(eM, flow_solver); eL /= np.linalg.norm(eL)

    r_mn     = solve_newton_case(
        centers=centers, areas=areas, normals=tnorm,
        alpha_deg=0.0, S_ref=S_ref, L_ref=L_ref, r_ref=r_ref,
        eD=flow_solver, eL=eL, eM=eM,
    )
    cp_model = np.array(r_mn["cp"])          # Cp numérico por cara
    mu       = np.array(r_mn["mu"])          # cos(incidencia) en conv. solver
    CD_mn    = float(r_mn["CD"])
    print(f"  MN numérico:  CD={CD_mn:.5f}  n_windward={r_mn['n_windward']:,}")

    # Índices barlovento en cada convenio
    rng              = np.random.default_rng(42)
    idx_model_wind   = np.where(mu > 0)[0]           # barlovento solver
    idx_matlab_wind  = np.where(Stheta < 0)[0]       # barlovento MATLAB (-Stheta > 0)
    n_sample         = min(3000, len(idx_model_wind), len(idx_matlab_wind))
    s_model = rng.choice(idx_model_wind,  size=n_sample, replace=False)
    s_matlab = rng.choice(idx_matlab_wind, size=n_sample, replace=False)

    # x-axis: mu para el modelo, -Stheta para MATLAB (ambos = cos(incidencia local))
    x_model  = mu[s_model]
    x_matlab = -Stheta[s_matlab]

    # ── Plot 1: Cp vs cos(φ) ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_model,  cp_model[s_model],    s=20, alpha=0.6,
               color="crimson",   zorder=3, label="MN (modelo)")
    ax.scatter(x_matlab, cp_matlab[s_matlab],  s=8,  alpha=0.5,
               color="steelblue", zorder=5, label="Dato MATLAB")
    ax.set_xlabel("cos φ  (ángulo de incidencia local)", fontsize=12)
    ax.set_ylabel("Cp", fontsize=12)
    ax.set_title("Validación esfera — Cp vs cos φ\nDato MATLAB vs MN (modelo numérico)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 2.1)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    fig.savefig(val_dir / "validacion_Cp_vs_cosphi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ validacion_Cp_vs_cosphi.png")

    # ── Plot 2: Cp vs ángulo (°) ────────────────────────────────────────────
    phi_model  = np.degrees(np.arccos(np.clip(x_model,  0, 1)))
    phi_matlab = np.degrees(np.arccos(np.clip(x_matlab, 0, 1)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(phi_model,  cp_model[s_model],   s=20, alpha=0.6,
               color="crimson",   zorder=3, label="MN (modelo)")
    ax.scatter(phi_matlab, cp_matlab[s_matlab], s=8,  alpha=0.5,
               color="steelblue", zorder=5, label="Dato MATLAB")
    ax.set_xlabel("φ (°) desde estancamiento", fontsize=12)
    ax.set_ylabel("Cp", fontsize=12)
    ax.set_title("Validación esfera — Cp vs ángulo de incidencia\nDato MATLAB vs MN (modelo numérico)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-1, 91); ax.set_ylim(-0.05, 2.1)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    fig.savefig(val_dir / "validacion_Cp_vs_angulo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ validacion_Cp_vs_angulo.png")

    # ── Plot 3: CD esfera vs Mach (MNM) — sin datos MATLAB, sólo modelo ────
    CDs_mnm = []
    for M in MACH_LIST:
        r = solve_modified_newton_case(
            centers=centers, areas=areas, normals=tnorm,
            alpha_deg=0.0, Mach=float(M),
            S_ref=S_ref, L_ref=L_ref, r_ref=r_ref,
            eD=flow_solver, eL=eL, eM=eM, gamma=gamma,
        )
        CDs_mnm.append(float(np.dot(r["CF_total"], flow_solver)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(MACH_LIST, CDs_mnm, "o-", color="#E53935", lw=2.5, ms=7, label="MNM (modelo)")
    ax.axhline(CD_mn, color="#1565C0", ls="--", lw=2, label=f"MN (modelo)  CD={CD_mn:.3f}")
    ax.set_xlabel("M∞", fontsize=12)
    ax.set_ylabel("CD", fontsize=12)
    ax.set_title("CD esfera vs M∞ — modelo MN / MNM", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(val_dir / "validacion_CD_vs_Mach.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ validacion_CD_vs_Mach.png")

    print(f"  → Plots guardados en {val_dir}")


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
    RUN_SPHERE_MACH_SWEEP = config.get("RUN_SPHERE_MACH_SWEEP", True)
    RUN_MESH_ALPHA_SWEEP = config.get("RUN_MESH_ALPHA_SWEEP", True)

    MESH_ALPHA = config.get("MESH_ALPHA", 10.0)
    MESH_MACH  = config.get("MESH_MACH", 8.0)
    MESH_STLS  = config.get("MESH_STLS", [])

    SHOW_GEOMETRY     = config.get("SHOW_GEOMETRY", False)
    ALPHA_TRIM_DEG    = config.get("ALPHA_TRIM_DEG")
    ALPHA_TRIM_METHOD = config.get("ALPHA_TRIM_METHOD", "MNM")

    GROUP_ID = config["GROUP_ID"]
    MEMBERS  = config["MEMBERS"]

    # Inicializar resultados de sweeps como None (se asignan si el sweep se ejecuta)
    rows_mn = rows_mnm = rows_mach = rows_mesh = None
    sphere_mach_rows = None

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

    if RUN_SPHERE_MACH_SWEEP:
        SPHERE_MACH_SWEEP = config.get("SPHERE_MACH_SWEEP", [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
        print(f"\n── Barrido Mach MNM esfera | M={SPHERE_MACH_SWEEP}  α=0°")
        sphere_mach_rows = run_mach_sweep(
            SPHERE_MACH_SWEEP, 0.0, c_sp, a_sp, n_sp,
            S_sp, L_sp, r_sp, eD0, eL0, eM0, GAMMA,
        )
        save_csv(csv_dir / "results_sphere_mnm_mach_sweep.csv", sphere_mach_rows, CSV_FIELDS_MNM)

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

    # dict {stem: rows_mn} y {stem: rows_mnm} para todas las mallas de sensibilidad
    mesh_sweeps_mn      = {}
    mesh_sweeps_mnm     = {}
    mesh_mach_sweeps_mnm = {}   # {stem: rows_mach}  — barrido Mach por malla
    mesh_tri_counts     = {}    # {stem: n_tri}

    if RUN_MESH_ALPHA_SWEEP and MESH_STLS:
        print(f"\n── Barrido alpha (MN + MNM) para todas las mallas de sensibilidad")
        for stl_p in MESH_STLS:
            if not stl_p.exists():
                print(f"  [SKIP] {stl_p.name}"); continue
            m_sw  = load_stl(stl_p)
            g_sw  = compute_face_geometry(m_sw)
            c_sw  = g_sw["centers"]; a_sw = g_sw["areas"]; n_sw = g_sw["normals"]
            name  = stl_p.stem
            print(f"  {name}")
            rows_mn_sw = run_mn_sweep(ALPHAS_DEG, c_sw, a_sw, n_sw,
                                      S_cap, L_cap, r_cap, eD0, eL0, eM0)
            save_csv(csv_dir / f"results_mn_{name}.csv", rows_mn_sw, CSV_FIELDS_MN)
            rows_mnm_sw = run_mnm_sweep(ALPHAS_DEG, MACH, c_sw, a_sw, n_sw,
                                        S_cap, L_cap, r_cap, eD0, eL0, eM0, GAMMA)
            save_csv(csv_dir / f"results_mnm_{name}_M{int(MACH)}.csv", rows_mnm_sw, CSV_FIELDS_MNM)
            mesh_sweeps_mn[name]  = rows_mn_sw
            mesh_sweeps_mnm[name] = rows_mnm_sw
            mesh_tri_counts[name] = len(a_sw)

    if RUN_MESH_ALPHA_SWEEP and MESH_STLS and MACH_SWEEP:
        alpha_mesh = config.get("MESH_ALPHA", 10.0)
        eD_mm, eL_mm, eM_mm = wind_axes(alpha_mesh)
        print(f"\n── Barrido Mach MNM para todas las mallas (α={alpha_mesh}°)")
        for stl_p in MESH_STLS:
            if not stl_p.exists():
                print(f"  [SKIP] {stl_p.name}"); continue
            name = stl_p.stem
            if name not in mesh_tri_counts:   # aún no cargada (alpha sweep desactivado)
                m_mm = load_stl(stl_p)
                g_mm = compute_face_geometry(m_mm)
                c_mm = g_mm["centers"]; a_mm = g_mm["areas"]; n_mm = g_mm["normals"]
                mesh_tri_counts[name] = len(a_mm)
            else:
                m_mm = load_stl(stl_p)
                g_mm = compute_face_geometry(m_mm)
                c_mm = g_mm["centers"]; a_mm = g_mm["areas"]; n_mm = g_mm["normals"]
            print(f"  {name}")
            rows_mach_mm = run_mach_sweep(
                MACH_SWEEP, alpha_mesh, c_mm, a_mm, n_mm,
                S_cap, L_cap, r_cap, eD_mm, eL_mm, eM_mm, GAMMA,
            )
            mesh_mach_sweeps_mnm[name] = rows_mach_mm


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

    # ── Gráficas analíticas ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("Generando gráficas analíticas …")
    print("═"*60)
    sphere_mn_cases = {
        "sphere_MN_a0_M8_CD": float(np.dot(r1["CF_total"], wind_axes(0.0)[0])),
    }
    generate_plots(
        plots_dir=plots_dir,
        rows_mn=rows_mn if RUN_MN_SWEEP else None,
        rows_mnm=rows_mnm if RUN_MNM_SWEEP else None,
        rows_mach=rows_mach if RUN_MACH_SWEEP else None,
        sphere_mach_rows=sphere_mach_rows,
        sphere_mn_cases=sphere_mn_cases,
        mesh_sweeps_mn=mesh_sweeps_mn or None,
        mesh_sweeps_mnm=mesh_sweeps_mnm or None,
        mesh_mach_sweeps_mnm=mesh_mach_sweeps_mnm or None,
        mesh_tri_counts=mesh_tri_counts or None,
    )

    # ── Validación esfera ────────────────────────────────────────────────────
    if config.get("RUN_WINDWARD_PLOTS", True):
        print("\n" + "═"*60)
        print("Generando gráficas barlovento/sotavento + CD vs α multi-Mach …")
        print("═"*60)
        generate_windward_plots(
            centers=c_cap, areas=a_cap, normals=n_cap,
            S_ref=S_cap, L_ref=L_cap, r_ref=r_cap,
            alphas_deg=ALPHAS_DEG,
            mach_sweep=MACH_SWEEP,
            plots_dir=plots_dir,
            alpha_ref=CP_MAP_ALPHA if CP_MAP_ALPHA is not None else 20.0,
            gamma=GAMMA,
        )

    if config.get("RUN_VALIDATION", False):
        print("\n" + "═"*60)
        print("Validación: esfera vs datos MATLAB …")
        print("═"*60)
        run_validation(config, plots_dir=plots_dir, gamma=GAMMA)
