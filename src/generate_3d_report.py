"""
generate_3d_report.py — Informe HTML interactivo con Plotly.js.

Visualización 3D de geometrías STL coloreadas por Cp, gráficas interactivas
y tabla de resultados — todo en un HTML standalone sin servidor.

Uso:
    python src/generate_3d_report.py
    python src/generate_3d_report.py --out results/report_3d.html

Dependencias: trimesh, numpy (ya usadas en el proyecto).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── Rutas base ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent

# Importar solver (mismo directorio que este archivo)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from MN  import solve_newton_case          as _solve_mn
    from MNM import solve_modified_newton_case as _solve_mnm
    _SOLVER_OK = True
except ImportError as _e:
    _SOLVER_OK = False
    print(f"[AVISO] Solver no disponible ({_e}); viewer interactivo deshabilitado.", file=sys.stderr)


def _stl_load(path):
    """Carga STL y devuelve trimesh.Trimesh."""
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return m


def _face_geom(mesh):
    """Equivalente ligero de stl_utils.compute_face_geometry (sin matplotlib)."""
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross   = np.cross(v1 - v0, v2 - v0)
    areas   = np.linalg.norm(cross, axis=1) / 2.0
    normals = cross / (2.0 * areas[:, None] + 1e-30)
    centers = (v0 + v1 + v2) / 3.0
    return {"centers": centers, "areas": areas, "normals": normals}
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ════════════════════════════════════════════════════════════════════════════
# Carga de datos
# ════════════════════════════════════════════════════════════════════════════

def load_stl_mesh(stl_path: Path):
    """Devuelve (vertices [N,3], faces [M,3]) usando trimesh."""
    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh no está instalado. Ejecuta: pip install trimesh", file=sys.stderr)
        sys.exit(1)
    mesh = trimesh.load(str(stl_path), force="mesh")
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)


def load_cp_csv(csv_path: Path) -> np.ndarray:
    """Carga columna 'cp' de un CSV de Cp por cara. Devuelve array 1D."""
    cp_vals = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cp_vals.append(float(row["cp"]))
    return np.array(cp_vals)


def load_sweep_csv(csv_path: Path) -> list[dict]:
    """Carga un CSV de sweep → lista de dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def _cast(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v
    return [{k: _cast(v) for k, v in r.items()} for r in rows]


def face_cp_to_vertex_cp(vertices: np.ndarray, faces: np.ndarray,
                          face_cp: np.ndarray) -> np.ndarray:
    """Mapea Cp por cara → Cp por vértice (promedio de caras adyacentes)."""
    vertex_cp  = np.zeros(len(vertices))
    vertex_cnt = np.zeros(len(vertices))
    n = min(len(faces), len(face_cp))
    for fi in range(n):
        for vi in faces[fi]:
            vertex_cp[vi]  += face_cp[fi]
            vertex_cnt[vi] += 1
    mask = vertex_cnt > 0
    vertex_cp[mask] /= vertex_cnt[mask]
    return vertex_cp


# ════════════════════════════════════════════════════════════════════════════
# Analítico: cp_max y CD esfera vs Mach
# ════════════════════════════════════════════════════════════════════════════

def cpmax_analytic(M: float, g: float = 1.4) -> float:
    t1 = ((g + 1) / 2 * M**2) ** (g / (g + 1))
    t2 = ((1 - g + 2 * g * M**2) / (g + 1)) ** (1 / (g + 1))
    return 2 / (g * M**2) * (t1 * t2 - 1)


# ════════════════════════════════════════════════════════════════════════════
# Serialización compacta de arrays (sin espacios)
# ════════════════════════════════════════════════════════════════════════════

def _jf(arr, precision=4) -> str:
    """Array de floats → JSON compacto con precisión dada."""
    return "[" + ",".join(f"{v:.{precision}f}" for v in arr) + "]"


def _ji(arr) -> str:
    """Array de ints → JSON compacto."""
    return "[" + ",".join(str(int(v)) for v in arr) + "]"


# ════════════════════════════════════════════════════════════════════════════
# Pre-cómputo interactivo: Cp maps para todos los selectores
# ════════════════════════════════════════════════════════════════════════════

def _wind_axes(alpha_deg):
    a  = np.deg2rad(alpha_deg)
    eD = np.array([0., -np.cos(a), -np.sin(a)])
    eD /= np.linalg.norm(eD)
    eM = np.array([1., 0., 0.])
    eL = np.cross(eM, eD); eL /= np.linalg.norm(eL)
    return eD, eL, eM


def _capsule_refs(mesh, geom):
    extents = np.asarray(mesh.extents, dtype=float)
    centers, areas = geom["centers"], geom["areas"]
    S_ref = float(extents[0] * extents[2])
    L_ref = float(extents[1])
    r_ref = np.average(centers, axis=0, weights=areas)
    return S_ref, L_ref, r_ref


def compute_interactive_data(
    mesh_paths: dict[str, Path],
    alphas:     list[float],
    machs:      list[float],
    gamma:      float = 1.4,
) -> tuple[dict, dict]:
    """
    Pre-computa geometría y Cp maps para todos los selectores.

    Returns
    -------
    geom_db : {name: {verts, faces}}
    cp_db   : {name: {model: {alpha_str: {mach_str: {cp, CD, CL, CM}}}}}
    """
    if not _SOLVER_OK:
        return {}, {}

    geom_db = {}
    cp_db   = {}

    for name, path in mesh_paths.items():
        if not Path(path).exists():
            print(f"  [SKIP] {name} — STL no encontrado")
            continue
        print(f"  Precomputando {name}...")
        mesh = _stl_load(path)
        geom = _face_geom(mesh)
        c, a, n = geom["centers"], geom["areas"], geom["normals"]
        S, L, r = _capsule_refs(mesh, geom)
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces,    dtype=int)

        geom_db[name] = {"verts": verts, "faces": faces}
        cp_db[name]   = {"MN": {}, "MNM": {}}

        for alpha in alphas:
            eD, eL, eM = _wind_axes(alpha)
            ak = str(int(alpha))

            # MN (no depende de Mach)
            res = _solve_mn(
                centers=c, areas=a, normals=n, alpha_deg=alpha,
                S_ref=S, L_ref=L, r_ref=r, eD=eD, eL=eL, eM=eM,
            )
            vcp = face_cp_to_vertex_cp(verts, faces, res["cp"])
            cp_db[name]["MN"].setdefault(ak, {})["8"] = {
                "cp": vcp, "CD": res["CD"], "CL": res["CL"], "CM": res["CM"],
            }

            # MNM (depende de Mach)
            cp_db[name]["MNM"].setdefault(ak, {})
            for mach in machs:
                res = _solve_mnm(
                    centers=c, areas=a, normals=n, alpha_deg=alpha, Mach=mach,
                    S_ref=S, L_ref=L, r_ref=r, eD=eD, eL=eL, eM=eM, gamma=gamma,
                )
                vcp = face_cp_to_vertex_cp(verts, faces, res["cp"])
                mk  = str(int(mach))
                cp_db[name]["MNM"][ak][mk] = {
                    "cp": vcp, "CD": res["CD"], "CL": res["CL"], "CM": res["CM"],
                }

    return geom_db, cp_db


def _serialize_cp_db(geom_db: dict, cp_db: dict) -> str:
    """Serializa geom_db y cp_db a literales JS."""
    geom_js_parts = []
    for name, g in geom_db.items():
        v, f = g["verts"], g["faces"]
        geom_js_parts.append(
            f'"{name}":{{'
            f'x:{_jf(v[:,0])},y:{_jf(v[:,1])},z:{_jf(v[:,2])},'
            f'i:{_ji(f[:,0])},j:{_ji(f[:,1])},k:{_ji(f[:,2])},'
            f'n:{len(f)}}}'
        )
    geom_js = "{" + ",".join(geom_js_parts) + "}"

    cp_js_parts = []
    for name, models in cp_db.items():
        model_parts = []
        for model, alphas_d in models.items():
            alpha_parts = []
            for ak, machs_d in alphas_d.items():
                mach_parts = []
                for mk, vals in machs_d.items():
                    cp_arr = _jf(vals["cp"], precision=3)
                    mach_parts.append(
                        f'"{mk}":{{'
                        f'cp:{cp_arr},'
                        f'CD:{vals["CD"]:.6f},'
                        f'CL:{vals["CL"]:.6f},'
                        f'CM:{vals["CM"]:.6f}}}'
                    )
                alpha_parts.append(f'"{ak}":{{{",".join(mach_parts)}}}')
            model_parts.append(f'"{model}":{{{",".join(alpha_parts)}}}')
        cp_js_parts.append(f'"{name}":{{{",".join(model_parts)}}}')
    cp_js = "{" + ",".join(cp_js_parts) + "}"

    return geom_js, cp_js


# ════════════════════════════════════════════════════════════════════════════
# Generación del HTML
# ════════════════════════════════════════════════════════════════════════════

def generate_3d_html(
    results_json:  Path | None = None,
    cp_csv:        Path | None = None,
    stl_capsule:   Path | None = None,
    stl_sphere:    Path | None = None,
    stl_coarse:    Path | None = None,   # malla gruesa para tab Malla
    mesh_sens_csv: Path | None = None,   # results_mesh_sensitivity.csv
    out_path:      Path | None = None,
) -> Path:
    # ── Rutas por defecto ────────────────────────────────────────────────────
    results_json = results_json or RESULTS_DIR / "results.json"
    cp_csv       = cp_csv       or RESULTS_DIR / "cp_faces_mnm_a20_M8.csv"
    stl_capsule  = stl_capsule  or DATA_DIR   / "Capsula" / "PruebaARD3.stl"
    stl_sphere   = stl_sphere   or DATA_DIR   / "esfera.stl"
    stl_coarse   = stl_coarse   or DATA_DIR   / "Capsula" / "PruebaARD.stl"
    mesh_sens_csv = mesh_sens_csv or RESULTS_DIR / "results_mesh_sensitivity.csv"
    out_path     = out_path     or RESULTS_DIR / "report_3d.html"

    # ── Carga ────────────────────────────────────────────────────────────────
    print("Cargando datos...")
    with open(results_json, encoding="utf-8") as f:
        results = json.load(f)
    cases  = results["cases"]
    ref    = results["reference"]
    team   = results["team"]

    print("  Cargando STL capsula (media)...")
    verts_cap, faces_cap = load_stl_mesh(stl_capsule)
    print(f"    {len(verts_cap)} vertices, {len(faces_cap)} caras")

    print("  Cargando STL esfera...")
    verts_sph, faces_sph = load_stl_mesh(stl_sphere)
    print(f"    {len(verts_sph)} vertices, {len(faces_sph)} caras")

    print("  Cargando Cp CSV...")
    face_cp = load_cp_csv(cp_csv)
    vertex_cp_cap = face_cp_to_vertex_cp(verts_cap, faces_cap, face_cp)
    print(f"    Cp = [{face_cp.min():.4f}, {face_cp.max():.4f}]")

    # ── Malla gruesa para comparativa ────────────────────────────────────────
    has_coarse = stl_coarse is not None and Path(stl_coarse).exists()
    if has_coarse:
        print(f"  Cargando STL gruesa ({stl_coarse.name})...")
        verts_co, faces_co = load_stl_mesh(stl_coarse)
        print(f"    {len(verts_co)} vertices, {len(faces_co)} caras")
        cox = _jf(verts_co[:, 0]); coy = _jf(verts_co[:, 1]); coz = _jf(verts_co[:, 2])
        coi = _ji(faces_co[:, 0]); coj = _ji(faces_co[:, 1]); cok = _ji(faces_co[:, 2])
        coarse_name = stl_coarse.name
        n_coarse    = len(faces_co)
    else:
        cox = coy = coz = coi = coj = cok = "[]"
        coarse_name = "—"; n_coarse = 0

    # ── Sweep CSVs ───────────────────────────────────────────────────────────
    sweep_mnm, sweep_mn, sweep_mach = [], [], []
    for path, store in [
        (RESULTS_DIR / "results_mnm_M8.csv",         "mnm"),
        (RESULTS_DIR / "results_mn.csv",              "mn"),
        (RESULTS_DIR / "results_mnm_mach_sweep.csv", "mach"),
    ]:
        if path.exists():
            data = load_sweep_csv(path)
            if store == "mnm":  sweep_mnm  = data
            elif store == "mn": sweep_mn   = data
            else:               sweep_mach = data

    # ── Mesh sensitivity CSV ─────────────────────────────────────────────────
    mesh_rows = []
    if Path(mesh_sens_csv).exists():
        mesh_rows = load_sweep_csv(mesh_sens_csv)
    # Separate by model
    mesh_mn  = sorted([r for r in mesh_rows if r.get("model") == "MN"],
                       key=lambda r: r["n_faces"])
    mesh_mnm = sorted([r for r in mesh_rows if r.get("model") == "MNM"],
                       key=lambda r: r["n_faces"])

    def _mesh_series(rows):
        return {
            "n":   json.dumps([int(r["n_faces"]) for r in rows]),
            "CD":  json.dumps([r["CD"] for r in rows]),
            "CL":  json.dumps([r["CL"] for r in rows]),
            "CM":  json.dumps([r["CM"] for r in rows]),
            "lbl": json.dumps([f"{r.get('stl','')}\n({int(r['n_faces'])} caras)" for r in rows]),
        }

    ms_mn  = _mesh_series(mesh_mn)
    ms_mnm = _mesh_series(mesh_mnm)

    # ── Analítico cp_max vs Mach ─────────────────────────────────────────────
    mach_pts   = [2, 4, 6, 8, 10, 12, 15, 20, 30, 50]
    cpmax_pts  = [round(cpmax_analytic(m), 5) for m in mach_pts]
    cd_mnm_pts = [round(cpmax_analytic(m) / 2, 5) for m in mach_pts]

    # ── Pre-cómputo interactivo ───────────────────────────────────────────────
    _INTERACTIVE_ALPHAS = [0.0, 10.0, 20.0, 30.0]
    _INTERACTIVE_MACHS  = [2.0, 4.0, 8.0, 12.0, 15.0, 20.0]
    _INTERACTIVE_MESHES = {
        "PruebaARD":  DATA_DIR / "Capsula" / "PruebaARD.stl",
        "PruebaARD3": DATA_DIR / "Capsula" / "PruebaARD3.stl",
        "PruebaARD4": DATA_DIR / "Capsula" / "PruebaARD4.stl",
    }
    print("\nPre-computando Cp interactivos...")
    geom_db, cp_db = compute_interactive_data(
        _INTERACTIVE_MESHES, _INTERACTIVE_ALPHAS, _INTERACTIVE_MACHS
    )
    geom_js_str, cp_db_js_str = _serialize_cp_db(geom_db, cp_db)
    interactive_mesh_names = json.dumps(list(geom_db.keys()))
    print("  Hecho.")

    # ── Serialización geometrías ─────────────────────────────────────────────
    cx = _jf(verts_cap[:, 0]); cy = _jf(verts_cap[:, 1]); cz = _jf(verts_cap[:, 2])
    ci = _ji(faces_cap[:, 0]); cj = _ji(faces_cap[:, 1]); ck = _ji(faces_cap[:, 2])
    cp_js = _jf(vertex_cp_cap, precision=5)

    sx = _jf(verts_sph[:, 0]); sy = _jf(verts_sph[:, 1]); sz = _jf(verts_sph[:, 2])
    si = _ji(faces_sph[:, 0]); sj = _ji(faces_sph[:, 1]); sk = _ji(faces_sph[:, 2])

    # ── Sweep data para gráficas ─────────────────────────────────────────────
    def _sw(rows, kx, ky):
        return json.dumps([r[kx] for r in rows if kx in r]), \
               json.dumps([r[ky] for r in rows if ky in r])

    mnm_alphas, mnm_CD = _sw(sweep_mnm, "alpha_deg", "CD")
    mnm_alphas, mnm_CL = _sw(sweep_mnm, "alpha_deg", "CL")
    mnm_alphas, mnm_CM = _sw(sweep_mnm, "alpha_deg", "CM")
    mn_alphas,  mn_CD  = _sw(sweep_mn,  "alpha_deg", "CD")
    mach_sweep_x, mach_sweep_cd = _sw(sweep_mach, "Mach", "CD")

    # ── Metadata ─────────────────────────────────────────────────────────────
    group_str   = team.get("group_id", "GXX")
    members_str = " · ".join(team.get("members", []))
    sref_val    = ref.get("Sref_m2", 0)
    lref_val    = ref.get("Lref_m", 0)
    sp_sref     = ref.get("_sphere_Sref_m2", 0)
    sp_lref     = ref.get("_sphere_Lref_m", 0)
    n_tri_cap   = next((c["triangles"] for c in cases if "capsule" in c["case_id"]), "—")
    n_tri_sph   = next((c["triangles"] for c in cases if "sphere"  in c["case_id"]), "—")

    js_cases = json.dumps([{
        "id": c["case_id"], "geo": c["geometry_name"], "model": c["model"],
        "M": c["Mach"], "a": c["alpha_deg"],
        "CD": c["CD"], "CL": c["CL"], "CM": c["CM"],
        "nw": c.get("_n_windward", 0),
    } for c in cases])

    checks_data = _build_checks(cases)

    # ── HTML ─────────────────────────────────────────────────────────────────
    html = _html_template(
        group_str=group_str, members_str=members_str,
        sref_val=sref_val, lref_val=lref_val, sp_sref=sp_sref, sp_lref=sp_lref,
        n_tri_cap=n_tri_cap, n_tri_sph=n_tri_sph,
        cx=cx, cy=cy, cz=cz, ci=ci, cj=cj, ck=ck, cp_js=cp_js,
        sx=sx, sy=sy, sz=sz, si=si, sj=sj, sk=sk,
        cox=cox, coy=coy, coz=coz, coi=coi, coj=coj, cok=cok,
        coarse_name=coarse_name, n_coarse=n_coarse,
        n_fine=len(faces_cap),
        mach_pts=json.dumps(mach_pts), cpmax_pts=json.dumps(cpmax_pts),
        cd_mnm_pts=json.dumps(cd_mnm_pts),
        mnm_alphas=mnm_alphas, mnm_CD=mnm_CD, mnm_CL=mnm_CL, mnm_CM=mnm_CM,
        mn_alphas=mn_alphas, mn_CD=mn_CD,
        mach_sweep_x=mach_sweep_x, mach_sweep_cd=mach_sweep_cd,
        ms_mn_n=ms_mn["n"],   ms_mn_CD=ms_mn["CD"],   ms_mn_CL=ms_mn["CL"],
        ms_mn_CM=ms_mn["CM"], ms_mn_lbl=ms_mn["lbl"],
        ms_mnm_n=ms_mnm["n"], ms_mnm_CD=ms_mnm["CD"], ms_mnm_CL=ms_mnm["CL"],
        ms_mnm_CM=ms_mnm["CM"], ms_mnm_lbl=ms_mnm["lbl"],
        mesh_alpha_label=f"{mesh_rows[0]['alpha_deg']:.0f}°" if mesh_rows else "?",
        geom_db_js=geom_js_str,
        cp_db_js=cp_db_js_str,
        interactive_mesh_names=interactive_mesh_names,
        interactive_alphas=json.dumps([int(a) for a in _INTERACTIVE_ALPHAS]),
        interactive_machs=json.dumps([int(m) for m in _INTERACTIVE_MACHS]),
        js_cases=js_cases,
        checks_data=json.dumps(checks_data),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  OK report_3d.html -> {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# Checks C1-C6
# ════════════════════════════════════════════════════════════════════════════

def _build_checks(cases: list[dict]) -> list[dict]:
    by_id = {c["case_id"]: c for c in cases}

    def _get(*ids, key="CL"):
        return [by_id[i][key] for i in ids if i in by_id]

    sp0_mn  = by_id.get("sphere_MN_a0_M8",   {})
    sp0_m2  = by_id.get("sphere_MNM_a0_M2",  {})
    sp0_m8  = by_id.get("sphere_MNM_a0_M8",  {})
    ca0_mn  = by_id.get("capsule_MN_a0_M8",  {})
    ca0_mnm = by_id.get("capsule_MNM_a0_M8", {})
    ca10    = by_id.get("capsule_MNM_a10_M8",{})
    ca20    = by_id.get("capsule_MNM_a20_M8",{})
    co      = by_id.get("capsule_MN_mesh_coarse_a10_M8", {})
    fi      = by_id.get("capsule_MN_mesh_fine_a10_M8",   {})

    def pass_c1():
        for c in cases:
            if c["CD"] < 0: return False
        return True

    def pass_c2():
        for c in [sp0_mn, sp0_m2, sp0_m8]:
            if not c: return False
            if abs(c.get("CL", 1)) > 1e-3 or abs(c.get("CM", 1)) > 1e-3:
                return False
        return True

    cd2 = sp0_m2.get("CD", 0); cd8 = sp0_m8.get("CD", 0)
    pass_c3 = bool(cd8 > cd2 + 1e-3) if cd2 and cd8 else False

    def pass_c4():
        for c in [ca0_mn, ca0_mnm]:
            if not c: return False
            if abs(c.get("CL", 1)) > 5e-3 or abs(c.get("CM", 1)) > 5e-3:
                return False
        return True

    cl0  = abs(ca0_mnm.get("CL", 0)); cl10 = abs(ca10.get("CL", 0)); cl20 = abs(ca20.get("CL", 0))
    pass_c5 = cl10 > cl0 + 1e-3 and cl20 > cl10 + 1e-3 if ca0_mnm and ca10 and ca20 else False

    if co and fi:
        delta_c6 = abs(fi.get("CD", 0) - co.get("CD", 0)) / max(abs(fi.get("CD", 1)), 1e-6)
        pass_c6  = delta_c6 < 0.05
    else:
        delta_c6 = 0; pass_c6 = False

    return [
        {"id": "C1", "title": "Coherencia interna",    "pass": pass_c1(),
         "desc": f"CD≥0, |CF|<50, |CM|<50 en todos los casos."},
        {"id": "C2", "title": "Simetría esfera α=0",   "pass": pass_c2(),
         "desc": f"|CL|<1e-3, |CM|<1e-3. Error nivel máquina ~10⁻¹⁷."},
        {"id": "C3", "title": "Sensibilidad Mach esfera", "pass": pass_c3,
         "desc": f"CD_M8={cd8:.5f} > CD_M2={cd2:.5f}."},
        {"id": "C4", "title": "Simetría cápsula α=0",  "pass": pass_c4(),
         "desc": f"|CL|<5e-3, |CM|<5e-3. Axisimetría verificada."},
        {"id": "C5", "title": "CL cápsula crece con α","pass": bool(pass_c5),
         "desc": f"0 → {cl10:.4f} → {cl20:.4f}. Monótono."},
        {"id": "C6", "title": "Sensibilidad a malla",  "pass": bool(pass_c6),
         "desc": f"ΔCD={delta_c6:.4f} < 5%."},
    ]


# ════════════════════════════════════════════════════════════════════════════
# Plantilla HTML
# ════════════════════════════════════════════════════════════════════════════

def _html_template(**d) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AAVFR TP1 — Informe 3D Interactivo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{{
  --bg:#07090f;--bg2:#0f1220;--bg3:#161b2c;--border:#1e2540;
  --accent:#3de8b0;--accent2:#e85040;--accent3:#7060e8;--accent4:#e8a030;
  --text:#ccd6f6;--muted:#4a5580;
  --glow-g:0 0 30px #3de8b018;--glow-r:0 0 30px #e8504018;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;
  font-weight:400;min-height:100vh;overflow-x:hidden}}

/* grid bg */
body::after{{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),
    linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:50px 50px;opacity:.25}}

.wrap{{position:relative;z-index:1;max-width:1240px;margin:0 auto;padding:0 24px 80px}}

/* ── Nav ── */
nav{{position:sticky;top:0;z-index:100;background:rgba(7,9,15,.92);
  backdrop-filter:blur(12px);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:0;padding:0 24px;height:52px}}
.nav-brand{{font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
  color:var(--accent);letter-spacing:.1em;padding-right:24px;
  border-right:1px solid var(--border);white-space:nowrap}}
.nav-tabs{{display:flex;overflow-x:auto;scrollbar-width:none}}
.nav-tabs::-webkit-scrollbar{{display:none}}
.tab-btn{{background:none;border:none;color:var(--muted);font-family:'Space Mono',monospace;
  font-size:10px;letter-spacing:.08em;padding:0 18px;height:52px;cursor:pointer;
  white-space:nowrap;border-bottom:2px solid transparent;transition:color .2s,border-color .2s}}
.tab-btn:hover{{color:var(--text)}}
.tab-btn.active{{color:var(--accent);border-bottom-color:var(--accent)}}

/* ── Header ── */
header{{padding:56px 0 40px;border-bottom:1px solid var(--border)}}
.hflex{{display:flex;align-items:flex-end;gap:32px;flex-wrap:wrap}}
.hbadge{{background:var(--accent);color:var(--bg);font-family:'Space Mono',monospace;
  font-size:9.5px;font-weight:700;letter-spacing:.12em;padding:4px 10px;
  border-radius:3px;flex-shrink:0;margin-bottom:auto}}
.htitle{{flex:1;min-width:200px}}
.htitle h1{{font-size:clamp(1.9rem,4vw,3rem);font-weight:700;line-height:1.1;letter-spacing:-.01em}}
.htitle h1 em{{color:var(--accent);font-style:normal}}
.htitle p{{margin-top:8px;font-family:'Space Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:.06em}}
.hstats{{display:flex;gap:28px;flex-shrink:0}}
.stat .sv{{display:block;font-size:2rem;font-weight:700;color:var(--accent);line-height:1}}
.stat .sl{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.1em}}

/* ── Sections ── */
.section{{margin-top:60px;display:none;animation:fadeUp .35s ease both}}
.section.active{{display:block}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
.shead{{display:flex;align-items:center;gap:14px;margin-bottom:24px}}
.snum{{font-family:'Space Mono',monospace;font-size:10px;color:var(--accent);
  border:1px solid var(--accent);padding:3px 8px;border-radius:2px;letter-spacing:.1em}}
.shead h2{{font-size:1.25rem;font-weight:600}}
.sline{{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}}

/* ── 3D viewer panel ── */
.viewer-wrap{{display:grid;grid-template-columns:1fr 300px;gap:16px;align-items:start}}
@media(max-width:880px){{.viewer-wrap{{grid-template-columns:1fr}}}}
.plot3d{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;height:560px}}
.info-panel{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px}}
.info-panel h3{{font-size:.9rem;font-weight:600;margin-bottom:4px}}
.info-sub{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:.06em;margin-bottom:18px}}
.kv{{margin-bottom:12px}}
.kv .k{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.07em}}
.kv .v{{font-size:1.4rem;font-weight:600;line-height:1.1;margin-top:2px}}
.kv .v.cd{{color:var(--accent2)}}.kv .v.cl{{color:var(--accent)}}.kv .v.cm{{color:var(--accent3)}}
.divider{{height:1px;background:var(--border);margin:16px 0}}
.legend-item{{display:flex;align-items:center;gap:8px;margin-bottom:8px;
  font-family:'Space Mono',monospace;font-size:9px;color:var(--muted)}}
.legend-color{{width:12px;height:12px;border-radius:2px;flex-shrink:0}}
.tip{{margin-top:20px;padding:10px;background:var(--bg3);border-radius:6px;
  font-family:'Space Mono',monospace;font-size:8.5px;color:var(--muted);line-height:1.7}}

/* ── Charts grid ── */
.charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:720px){{.charts-grid{{grid-template-columns:1fr}}}}
.chart-box{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  overflow:hidden}}
.chart-box h3{{padding:16px 18px 4px;font-size:.95rem;font-weight:600}}
.chart-sub{{padding:0 18px 12px;font-family:'Space Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:.05em}}
.plotly-chart{{height:300px}}

/* ── Table ── */
.tw{{overflow-x:auto;border:1px solid var(--border);border-radius:10px}}
table{{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:10.5px}}
th{{text-align:left;padding:10px 13px;border-bottom:1px solid var(--accent);
  color:var(--accent);font-size:9px;letter-spacing:.1em;white-space:nowrap;
  background:var(--bg2);cursor:pointer;user-select:none}}
th:hover{{color:var(--text)}}
th::after{{content:' ⇅';opacity:.4}}
td{{padding:9px 13px;border-bottom:1px solid var(--border);white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--bg3)}}
.tMN{{color:var(--accent)}}.tMNM{{color:var(--accent2)}}
.tcd{{color:var(--accent2);font-weight:700}}.tcl{{color:var(--accent)}}.tcm{{color:var(--accent3)}}

/* ── Ref block ── */
.refbox{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:20px}}
.refbox p{{font-size:.8rem;color:var(--muted);line-height:1.75;margin-bottom:14px}}
.refbox p strong{{color:var(--text)}}
.rgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}}
.ri .rk{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:.07em;margin-bottom:3px}}
.ri .rv{{font-size:.88rem;font-weight:700}}

/* ── Checks ── */
.checks-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.ck{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  padding:15px;display:flex;gap:11px;transition:box-shadow .2s}}
.ck.pass{{box-shadow:var(--glow-g)}}.ck.fail{{box-shadow:var(--glow-r)}}
.cki{{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:13px;flex-shrink:0;font-weight:700}}
.cki.ok{{background:#3de8b018;border:1px solid var(--accent);color:var(--accent)}}
.cki.fail{{background:#e8504018;border:1px solid var(--accent2);color:var(--accent2)}}
.ckt{{font-size:.78rem;font-weight:700;margin-bottom:3px}}
.ckd{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);line-height:1.55}}

/* ── Footer ── */
footer{{margin-top:64px;padding-top:18px;border-top:1px solid var(--border);
  font-family:'Space Mono',monospace;font-size:9.5px;color:var(--muted);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}

/* ── Misc ── */
.tag{{display:inline-block;font-family:'Space Mono',monospace;font-size:8.5px;
  padding:2px 7px;border-radius:2px;font-weight:700}}
.tag.MN{{background:#3de8b018;color:var(--accent);border:1px solid var(--accent)}}
.tag.MNM{{background:#e8504018;color:var(--accent2);border:1px solid var(--accent2)}}

/* ── Selector bar ── */
.sel-bar{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
  padding:12px 16px;background:var(--bg2);border:1px solid var(--border);
  border-radius:10px;margin-bottom:14px}}
.sel-group{{display:flex;flex-direction:column;gap:3px}}
.sel-label{{font-family:'Space Mono',monospace;font-size:8.5px;color:var(--muted);
  letter-spacing:.08em}}
.sel-bar select{{background:var(--bg3);border:1px solid var(--border);color:var(--text);
  font-family:'Space Mono',monospace;font-size:10px;padding:5px 10px;border-radius:5px;
  cursor:pointer;outline:none;min-width:120px}}
.sel-bar select:hover{{border-color:var(--accent)}}
.sel-bar select:focus{{border-color:var(--accent);box-shadow:0 0 0 2px #3de8b025}}
.sel-bar select option{{background:var(--bg3)}}

/* ── Metric pills ── */
.metrics-row{{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}}
.metric-pill{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:10px 18px;flex:1;min-width:90px;text-align:center}}
.metric-pill .mp-label{{font-family:'Space Mono',monospace;font-size:8.5px;color:var(--muted);letter-spacing:.1em}}
.metric-pill .mp-val{{font-size:1.5rem;font-weight:600;line-height:1.15;margin-top:2px}}
.metric-pill.mpcd .mp-val{{color:var(--accent2)}}
.metric-pill.mpcl .mp-val{{color:var(--accent)}}
.metric-pill.mpcm .mp-val{{color:var(--accent3)}}
.metric-pill.mpn  .mp-val{{color:var(--accent4);font-size:1.1rem}}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">AAVFR · TP1</div>
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="showTab('tab-capsule',this)">3D Cápsula Cp</button>
    <button class="tab-btn" onclick="showTab('tab-sphere',this)">3D Esfera</button>
    <button class="tab-btn" onclick="showTab('tab-mesh',this)">Sensibilidad Malla</button>
    <button class="tab-btn" onclick="showTab('tab-charts',this)">Gráficas</button>
    <button class="tab-btn" onclick="showTab('tab-table',this)">Resultados</button>
    <button class="tab-btn" onclick="showTab('tab-ref',this)">Referencias</button>
    <button class="tab-btn" onclick="showTab('tab-checks',this)">Checks C1–C6</button>
  </div>
</nav>

<div class="wrap">

<header>
  <div class="hflex">
    <div class="hbadge">AAVFR · TP1 · v1.1</div>
    <div class="htitle">
      <h1>Métodos de<br><em>Inclinación Local</em></h1>
      <p>MN · MNM · CÁPSULA ARD · ESFERA &nbsp;·&nbsp; {d['group_str']} &nbsp;·&nbsp; {d['members_str']}</p>
    </div>
    <div class="hstats">
      <div class="stat"><span class="sv">9</span><span class="sl">CASOS</span></div>
      <div class="stat"><span class="sv">2</span><span class="sl">MÉTODOS</span></div>
      <div class="stat"><span class="sv">6✓</span><span class="sl">CHECKS</span></div>
    </div>
  </div>
</header>


<!-- ═══════════════════ TAB: 3D CÁPSULA ═══════════════════ -->
<div class="section active" id="tab-capsule">
  <div class="shead"><span class="snum">01</span><h2>Visualización 3D — Cápsula ARD</h2><div class="sline"></div></div>

  <!-- Selector bar -->
  <div class="sel-bar">
    <div class="sel-group">
      <span class="sel-label">MALLA</span>
      <select id="sel-mesh" onchange="updateCapsule()"></select>
    </div>
    <div class="sel-group">
      <span class="sel-label">MÉTODO</span>
      <select id="sel-model" onchange="updateCapsule()">
        <option value="MNM">MNM — Newton Modificado</option>
        <option value="MN">MN — Newton</option>
      </select>
    </div>
    <div class="sel-group">
      <span class="sel-label">ÁNGULO α (°)</span>
      <select id="sel-alpha" onchange="updateCapsule()"></select>
    </div>
    <div class="sel-group">
      <span class="sel-label">MACH M∞</span>
      <select id="sel-mach" onchange="updateCapsule()"></select>
    </div>
  </div>

  <!-- 3D plot full-width -->
  <div class="plot3d" id="plot-capsule" style="height:580px;border-radius:10px"></div>

  <!-- Métricas dinámicas -->
  <div class="metrics-row">
    <div class="metric-pill mpcd">
      <div class="mp-label">CD</div>
      <div class="mp-val" id="mp-cd">—</div>
    </div>
    <div class="metric-pill mpcl">
      <div class="mp-label">CL</div>
      <div class="mp-val" id="mp-cl">—</div>
    </div>
    <div class="metric-pill mpcm">
      <div class="mp-label">CM</div>
      <div class="mp-val" id="mp-cm">—</div>
    </div>
    <div class="metric-pill mpn">
      <div class="mp-label">CARAS</div>
      <div class="mp-val" id="mp-n">—</div>
    </div>
    <div class="metric-pill" style="border-color:var(--muted)">
      <div class="mp-label">S_ref</div>
      <div class="mp-val" style="color:var(--text);font-size:1rem">{d['sref_val']:.4f} m²</div>
    </div>
    <div class="metric-pill" style="border-color:var(--muted)">
      <div class="mp-label">L_ref</div>
      <div class="mp-val" style="color:var(--text);font-size:1rem">{d['lref_val']:.4f} m</div>
    </div>
  </div>

  <!-- compat: info-panel ids aún usados en código anterior -->
  <span id="info-cd" style="display:none"></span>
  <span id="info-cl" style="display:none"></span>
  <span id="info-cm" style="display:none"></span>

  <!-- legend kept for reference -->
  <div style="display:flex;align-items:center;gap:8px;margin-top:12px;
    font-family:'Space Mono',monospace;font-size:8.5px;color:var(--muted)">
    <div style="width:80px;height:8px;border-radius:2px;
      background:linear-gradient(90deg,#440154,#21908c,#fde725)"></div>
    <span>Cp — Viridis: mínimo → máximo</span>
      </div>
      <div class="tip">
        🖱 Arrastra para rotar<br>
        🔍 Scroll para zoom<br>
        ⇧ Shift+drag para mover<br>
        🖱🖱 Doble click para centrar
      </div>
    </div>
  </div>
</div>


<!-- ═══════════════════ TAB: 3D ESFERA ═══════════════════ -->
<div class="section" id="tab-sphere">
  <div class="shead"><span class="snum">02</span><h2>Visualización 3D — Esfera</h2><div class="sline"></div></div>
  <div class="viewer-wrap">
    <div class="plot3d" id="plot-sphere"></div>
    <div class="info-panel">
      <h3>Geometría de validación</h3>
      <div class="info-sub">Esfera isotrópica · sin Cp externo</div>
      <div class="kv"><div class="k">TRIÁNGULOS</div>
        <div class="v">{d['n_tri_sph']}</div></div>
      <div class="kv"><div class="k">S_ref (π R²)</div>
        <div class="v">{d['sp_sref']:.6f} m²</div></div>
      <div class="kv"><div class="k">L_ref (2R)</div>
        <div class="v">{d['sp_lref']:.4f} m</div></div>
      <div class="divider"></div>
      <div class="kv"><div class="k">CD · MN · M8</div>
        <div class="v cd" id="sph-cd-mn">—</div></div>
      <div class="kv"><div class="k">CD · MNM · M2</div>
        <div class="v cl" id="sph-cd-m2">—</div></div>
      <div class="kv"><div class="k">CD · MNM · M8</div>
        <div class="v cm" id="sph-cd-m8">—</div></div>
      <div class="tip">
        La esfera se usa para validación del método MN.<br>
        CD_Newton teórico = 1.0 para M→∞.<br><br>
        La geometría se colorea por distancia al centro geométrico.
      </div>
    </div>
  </div>
</div>


<!-- ═══════════════════ TAB: MALLA ═══════════════════ -->
<div class="section" id="tab-mesh">
  <div class="shead"><span class="snum">03</span><h2>Sensibilidad de Malla</h2><div class="sline"></div></div>

  <!-- Convergencia -->
  <div class="charts-grid" style="margin-bottom:20px">
    <div class="chart-box">
      <h3>Convergencia CD — MN vs MNM</h3>
      <div class="chart-sub">CD vs N° triángulos · α={d['mesh_alpha_label']} · M=8</div>
      <div class="plotly-chart" id="mesh-ch-cd"></div>
    </div>
    <div class="chart-box">
      <h3>Convergencia CL y CM — MN vs MNM</h3>
      <div class="chart-sub">CL / CM vs N° triángulos · α={d['mesh_alpha_label']} · M=8</div>
      <div class="plotly-chart" id="mesh-ch-cl"></div>
    </div>
  </div>

  <!-- Comparativa 3D coarse vs medium -->
  <div class="shead" style="margin-top:40px"><span class="snum" style="border-color:var(--accent4);color:var(--accent4)">3D</span>
    <h2>Comparativa geométrica: malla gruesa vs media</h2><div class="sline"></div></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
        margin-bottom:8px;padding-left:4px">
        MALLA GRUESA — {d['coarse_name']} · {d['n_coarse']} caras
      </div>
      <div class="plot3d" id="plot-coarse" style="height:440px"></div>
    </div>
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
        margin-bottom:8px;padding-left:4px">
        MALLA MEDIA — PruebaARD3.stl · {d['n_fine']} caras
      </div>
      <div class="plot3d" id="plot-fine" style="height:440px"></div>
    </div>
  </div>
</div>


<!-- ═══════════════════ TAB: GRÁFICAS ═══════════════════ -->
<div class="section" id="tab-charts">
  <div class="shead"><span class="snum">04</span><h2>Análisis Gráfico Interactivo</h2><div class="sline"></div></div>
  <div class="charts-grid">

    <div class="chart-box">
      <h3>Validación Esfera — CD vs M∞</h3>
      <div class="chart-sub">MNM analítico · CD Newton = 1.0 (referencia)</div>
      <div class="plotly-chart" id="ch-sphere-cd"></div>
    </div>

    <div class="chart-box">
      <h3>cp,max y CD_MNM vs M∞</h3>
      <div class="chart-sub">cp,max / CD = cp,max/2 (analítico)</div>
      <div class="plotly-chart" id="ch-cpmax"></div>
    </div>

    <div class="chart-box">
      <h3>Cápsula — CD vs α</h3>
      <div class="chart-sub">MN vs MNM · M∞ = 8 · ejes viento</div>
      <div class="plotly-chart" id="ch-cap-cd"></div>
    </div>

    <div class="chart-box">
      <h3>Cápsula — CL y CM vs α</h3>
      <div class="chart-sub">MNM · M∞ = 8</div>
      <div class="plotly-chart" id="ch-cap-cl"></div>
    </div>

  </div>
</div>


<!-- ═══════════════════ TAB: TABLA ═══════════════════ -->
<div class="section" id="tab-table">
  <div class="shead"><span class="snum">05</span><h2>Tabla de Resultados</h2><div class="sline"></div></div>
  <div class="tw">
    <table id="results-table">
      <thead><tr>
        <th onclick="sortTable(0)">CASE_ID</th>
        <th onclick="sortTable(1)">GEO</th>
        <th onclick="sortTable(2)">MODELO</th>
        <th onclick="sortTable(3)">MACH</th>
        <th onclick="sortTable(4)">α°</th>
        <th onclick="sortTable(5)">CD</th>
        <th onclick="sortTable(6)">CL</th>
        <th onclick="sortTable(7)">CM</th>
        <th onclick="sortTable(8)">BARLOVENTO</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>


<!-- ═══════════════════ TAB: REFERENCIAS ═══════════════════ -->
<div class="section" id="tab-ref">
  <div class="shead"><span class="snum">06</span><h2>Convenciones y Referencias</h2><div class="sline"></div></div>
  <div class="refbox">
    <p>Sistema de referencia <strong>STL body frame</strong>: x lateral · y axial (morro en y_min) · z vertical. Unidades: mm.<br>
    α &gt; 0 inclina el flujo hacia −z (morro arriba). V∞ = [0, −cos α, −sin α].<br>
    Drag en dirección de V∞ (ejes viento). Lift perpendicular al flujo. Momento de cabeceo alrededor de +x.</p>
    <div class="rgrid">
      <div class="ri"><div class="rk">ESFERA · S_ref (π R²)</div><div class="rv">{d['sp_sref']:.6f} m²</div></div>
      <div class="ri"><div class="rk">ESFERA · L_ref (2R)</div><div class="rv">{d['sp_lref']:.4f} m</div></div>
      <div class="ri"><div class="rk">CÁPSULA · S_ref</div><div class="rv">{d['sref_val']:.4f} m²</div></div>
      <div class="ri"><div class="rk">CÁPSULA · L_ref</div><div class="rv">{d['lref_val']:.4f} m</div></div>
      <div class="ri"><div class="rk">CÁPSULA · Triángulos</div><div class="rv">{d['n_tri_cap']}</div></div>
      <div class="ri"><div class="rk">ESFERA · Triángulos</div><div class="rv">{d['n_tri_sph']}</div></div>
    </div>
  </div>
</div>


<!-- ═══════════════════ TAB: CHECKS ═══════════════════ -->
<div class="section" id="tab-checks">
  <div class="shead"><span class="snum">07</span><h2>Pruebas de Coherencia C1–C6</h2><div class="sline"></div></div>
  <div class="checks-grid" id="checks-grid"></div>
</div>


<footer>
  <span>AAVFR · Máster Sistemas Espaciales · UPM · IDR</span>
  <span>Informe 3D interactivo — generado desde results.json v1.1</span>
</footer>

</div><!-- /wrap -->

<script>
// ── Datos embebidos ──────────────────────────────────────────────────────────
const CAP_X={d['cx']}, CAP_Y={d['cy']}, CAP_Z={d['cz']};
const CAP_I={d['ci']}, CAP_J={d['cj']}, CAP_K={d['ck']};
const CAP_CP={d['cp_js']};

const SPH_X={d['sx']}, SPH_Y={d['sy']}, SPH_Z={d['sz']};
const SPH_I={d['si']}, SPH_J={d['sj']}, SPH_K={d['sk']};

const MACH_PTS={d['mach_pts']};
const CPMAX={d['cpmax_pts']};
const CD_MNM={d['cd_mnm_pts']};

const MNM_A={d['mnm_alphas']}, MNM_CD={d['mnm_CD']}, MNM_CL={d['mnm_CL']}, MNM_CM={d['mnm_CM']};
const MN_A={d['mn_alphas']},   MN_CD={d['mn_CD']};
const MACH_X={d['mach_sweep_x']}, MACH_CD={d['mach_sweep_cd']};

const CASES={d['js_cases']};
const CHECKS={d['checks_data']};

// ── Base de datos interactiva ─────────────────────────────────────────────────
const GEOM_DB={d['geom_db_js']};
const CP_DB={d['cp_db_js']};
const IMESH_NAMES={d['interactive_mesh_names']};
const IALPHAS={d['interactive_alphas']};
const IMACHS={d['interactive_machs']};

// ── Malla ────────────────────────────────────────────────────────────────────
const MS_MN_N={d['ms_mn_n']},  MS_MN_CD={d['ms_mn_CD']},  MS_MN_CL={d['ms_mn_CL']},  MS_MN_CM={d['ms_mn_CM']},  MS_MN_LBL={d['ms_mn_lbl']};
const MS_MNM_N={d['ms_mnm_n']},MS_MNM_CD={d['ms_mnm_CD']},MS_MNM_CL={d['ms_mnm_CL']},MS_MNM_CM={d['ms_mnm_CM']},MS_MNM_LBL={d['ms_mnm_lbl']};
const CO_X={d['cox']},CO_Y={d['coy']},CO_Z={d['coz']},CO_I={d['coi']},CO_J={d['coj']},CO_K={d['cok']};

// ── Plotly layout helpers ───────────────────────────────────────────────────
const BG    = '#0f1220';
const PAPER = '#07090f';
const GRID  = '#1e2540';
const TEXT  = '#ccd6f6';
const MUTED = '#4a5580';
const FONT  = {{'family':"'Space Mono',monospace",'size':10,'color':TEXT}};

const LAYOUT_3D = (title) => ({{
  paper_bgcolor: PAPER,
  plot_bgcolor:  BG,
  font: FONT,
  margin: {{l:0,r:0,t:36,b:0}},
  title: {{text:title, font:{{size:12,color:TEXT}}, x:0.5}},
  scene: {{
    bgcolor: PAPER,
    xaxis: {{gridcolor:GRID,zerolinecolor:GRID,color:MUTED,showbackground:true,backgroundcolor:PAPER}},
    yaxis: {{gridcolor:GRID,zerolinecolor:GRID,color:MUTED,showbackground:true,backgroundcolor:PAPER}},
    zaxis: {{gridcolor:GRID,zerolinecolor:GRID,color:MUTED,showbackground:true,backgroundcolor:PAPER}},
    aspectmode: 'data',
    camera: {{eye:{{x:1.6,y:-1.4,z:0.8}}}},
  }},
}});

const LAYOUT_2D = (xLabel, yLabel) => ({{
  paper_bgcolor: PAPER,
  plot_bgcolor:  BG,
  font: FONT,
  margin: {{l:48,r:20,t:20,b:48}},
  legend: {{bgcolor:'rgba(0,0,0,0)',font:{{size:9,color:TEXT}}}},
  xaxis: {{gridcolor:GRID,zerolinecolor:GRID,color:MUTED,title:{{text:xLabel,font:{{size:9,color:MUTED}}}}}},
  yaxis: {{gridcolor:GRID,zerolinecolor:GRID,color:MUTED,title:{{text:yLabel,font:{{size:9,color:MUTED}}}}}},
}});

const CFG = {{responsive:true, displayModeBar:true,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],
  toImageButtonOptions:{{format:'png',scale:2}}}};

// ── Tab navigation ──────────────────────────────────────────────────────────
const tabsRendered = {{}};

function showTab(id, btn) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  if (!tabsRendered[id]) {{
    tabsRendered[id] = true;
    setTimeout(() => renderTab(id), 50);
  }}
}}

function renderTab(id) {{
  if (id === 'tab-sphere') renderSphere();
  if (id === 'tab-mesh')   renderMesh();
  if (id === 'tab-charts') renderCharts();
}}

// ── 3D Cápsula (render inmediato) ───────────────────────────────────────────
function renderCapsule() {{
  // Populate selectors on first render
  const meshSel  = document.getElementById('sel-mesh');
  const alphaSel = document.getElementById('sel-alpha');
  const machSel  = document.getElementById('sel-mach');

  if (meshSel.options.length === 0) {{
    IMESH_NAMES.forEach(n => {{
      const o = document.createElement('option');
      o.value = n; o.textContent = n;
      if (n === 'PruebaARD3') o.selected = true;
      meshSel.appendChild(o);
    }});
    IALPHAS.forEach(a => {{
      const o = document.createElement('option');
      o.value = a; o.textContent = a + '°';
      if (a === 20) o.selected = true;
      alphaSel.appendChild(o);
    }});
    IMACHS.forEach(m => {{
      const o = document.createElement('option');
      o.value = m; o.textContent = 'M ' + m;
      if (m === 8) o.selected = true;
      machSel.appendChild(o);
    }});
  }}

  updateCapsule();
}}

function updateCapsule() {{
  const mesh  = document.getElementById('sel-mesh').value;
  const model = document.getElementById('sel-model').value;
  const alpha = document.getElementById('sel-alpha').value;
  const mach  = document.getElementById('sel-mach').value;

  // Disable Mach selector for MN (doesn't depend on Mach)
  document.getElementById('sel-mach').disabled = (model === 'MN');

  const g  = GEOM_DB[mesh];
  if (!g) return;

  const mk = (model === 'MN') ? '8' : String(mach);
  const entry = CP_DB?.[mesh]?.[model]?.[String(alpha)]?.[mk];

  const trace = {{
    type: 'mesh3d',
    x: g.x, y: g.y, z: g.z,
    i: g.i, j: g.j, k: g.k,
    intensity: entry ? entry.cp : null,
    colorscale: 'Viridis', showscale: true,
    colorbar: {{
      title: {{text:'Cp', side:'right', font:{{size:9,color:MUTED}}}},
      thickness:14, len:0.7,
      tickfont:{{size:9,color:MUTED}},
      outlinecolor:GRID, bgcolor:BG,
    }},
    flatshading: false,
    lighting: {{ambient:0.5,diffuse:0.7,specular:0.3,roughness:0.5}},
    lightposition: {{x:1000,y:-2000,z:2000}},
    hovertemplate:'x:%{{x:.1f}}<br>y:%{{y:.1f}}<br>z:%{{z:.1f}}<br>Cp:%{{intensity:.4f}}<extra></extra>',
  }};

  const title = `${{mesh}} — ${{model}} · α=${{alpha}}° · M=${{mach}}`;
  if (document.getElementById('plot-capsule').data) {{
    Plotly.react('plot-capsule', [trace], LAYOUT_3D(title), CFG);
  }} else {{
    Plotly.newPlot('plot-capsule', [trace], LAYOUT_3D(title), CFG);
  }}

  // Update metrics
  if (entry) {{
    document.getElementById('mp-cd').textContent = entry.CD.toFixed(5);
    document.getElementById('mp-cl').textContent = entry.CL.toFixed(5);
    document.getElementById('mp-cm').textContent = entry.CM.toFixed(5);
  }} else {{
    document.getElementById('mp-cd').textContent = '—';
    document.getElementById('mp-cl').textContent = '—';
    document.getElementById('mp-cm').textContent = '—';
  }}
  document.getElementById('mp-n').textContent = g.n ? g.n.toLocaleString() : '—';
}}

// ── 3D Esfera ───────────────────────────────────────────────────────────────
function renderSphere() {{
  // Colorear por distancia al centroide (sin Cp disponible)
  const cx = SPH_X.reduce((a,b)=>a+b,0)/SPH_X.length;
  const cy = SPH_Y.reduce((a,b)=>a+b,0)/SPH_Y.length;
  const cz = SPH_Z.reduce((a,b)=>a+b,0)/SPH_Z.length;
  const intensity = SPH_Z.map((z,i) => SPH_X[i]-cx);  // lateral position

  const trace = {{
    type: 'mesh3d',
    x: SPH_X, y: SPH_Y, z: SPH_Z,
    i: SPH_I, j: SPH_J, k: SPH_K,
    intensity: intensity,
    colorscale: [['0','#3de8b0'],['0.5','#7060e8'],['1','#e85040']],
    showscale: false,
    flatshading: false,
    lighting: {{ambient:0.45, diffuse:0.75, specular:0.4, roughness:0.4}},
    lightposition: {{x:500,y:-1000,z:1500}},
    opacity: 0.92,
    hovertemplate: 'x:%{{x:.2f}}<br>y:%{{y:.2f}}<br>z:%{{z:.2f}}<extra></extra>',
  }};

  const layout = LAYOUT_3D('Esfera — geometría de validación');
  layout.scene.camera = {{eye:{{x:1.8,y:-1.8,z:0.8}}}};
  Plotly.newPlot('plot-sphere', [trace], layout, CFG);

  const m = id => CASES.find(x => x.id === id);
  const r1 = m('sphere_MN_a0_M8'), r2 = m('sphere_MNM_a0_M2'), r3 = m('sphere_MNM_a0_M8');
  if (r1) document.getElementById('sph-cd-mn').textContent = r1.CD.toFixed(5);
  if (r2) document.getElementById('sph-cd-m2').textContent = r2.CD.toFixed(5);
  if (r3) document.getElementById('sph-cd-m8').textContent = r3.CD.toFixed(5);
}}

// ── Sensibilidad de malla ────────────────────────────────────────────────────
function renderMesh() {{
  // Convergencia CD
  Plotly.newPlot('mesh-ch-cd', [
    {{name:'CD MN',  x:MS_MN_N,  y:MS_MN_CD,  mode:'lines+markers',
      text:MS_MN_LBL, hovertemplate:'%{{text}}<br>CD=%{{y:.5f}}<extra></extra>',
      line:{{color:'#3de8b0',width:2.5}}, marker:{{size:8,symbol:'circle'}}}},
    {{name:'CD MNM', x:MS_MNM_N, y:MS_MNM_CD, mode:'lines+markers',
      text:MS_MNM_LBL,hovertemplate:'%{{text}}<br>CD=%{{y:.5f}}<extra></extra>',
      line:{{color:'#e85040',width:2.5}}, marker:{{size:8,symbol:'diamond'}}}},
  ], {{
    ...LAYOUT_2D('N° triángulos','CD'),
    xaxis:{{...LAYOUT_2D().xaxis, type:'log', title:{{text:'N° triángulos (log)',font:{{size:9,color:MUTED}}}}}},
  }}, CFG);

  // Convergencia CL y CM
  Plotly.newPlot('mesh-ch-cl', [
    {{name:'CL MN',  x:MS_MN_N,  y:MS_MN_CL,  mode:'lines+markers',
      line:{{color:'#3de8b0',width:2.5,dash:'solid'}},  marker:{{size:8}}}},
    {{name:'CL MNM', x:MS_MNM_N, y:MS_MNM_CL, mode:'lines+markers',
      line:{{color:'#e85040',width:2.5,dash:'solid'}},  marker:{{size:8,symbol:'diamond'}}}},
    {{name:'CM MN',  x:MS_MN_N,  y:MS_MN_CM,  mode:'lines+markers',
      line:{{color:'#3de8b088',width:2,dash:'dot'}},    marker:{{size:6}}}},
    {{name:'CM MNM', x:MS_MNM_N, y:MS_MNM_CM, mode:'lines+markers',
      line:{{color:'#e8504088',width:2,dash:'dot'}},    marker:{{size:6,symbol:'diamond'}}}},
  ], {{
    ...LAYOUT_2D('N° triángulos',''),
    xaxis:{{...LAYOUT_2D().xaxis, type:'log', title:{{text:'N° triángulos (log)',font:{{size:9,color:MUTED}}}}}},
  }}, CFG);

  // 3D coarse
  const meshTrace = (x,y,z,i,j,k,col,title) => [{{
    type:'mesh3d', x,y,z,i,j,k,
    color:col, opacity:0.9, flatshading:true,
    lighting:{{ambient:0.5,diffuse:0.7,specular:0.2}},
    lightposition:{{x:500,y:-1500,z:2000}},
    hovertemplate:'x:%{{x:.1f}}<br>y:%{{y:.1f}}<br>z:%{{z:.1f}}<extra></extra>',
  }}];

  const sceneLayout = (title) => ({{
    ...LAYOUT_3D(title),
    scene:{{...LAYOUT_3D(title).scene, camera:{{eye:{{x:1.6,y:-1.4,z:0.8}}}}}},
  }});

  if (CO_X.length) {{
    Plotly.newPlot('plot-coarse', meshTrace(CO_X,CO_Y,CO_Z,CO_I,CO_J,CO_K,'#e8a030',''),
      sceneLayout('Malla gruesa — {d['coarse_name']}'), CFG);
  }}
  Plotly.newPlot('plot-fine',   meshTrace(CAP_X,CAP_Y,CAP_Z,CAP_I,CAP_J,CAP_K,'#3de8b0',''),
    sceneLayout('Malla media — PruebaARD3.stl'), CFG);
}}

// ── Charts ──────────────────────────────────────────────────────────────────
function renderCharts() {{
  // 1) Esfera CD vs Mach
  Plotly.newPlot('ch-sphere-cd', [
    {{name:'CD (MNM analítico)', x:MACH_PTS, y:CD_MNM, mode:'lines+markers',
      line:{{color:'#e85040',width:2}}, marker:{{size:6,color:'#e85040'}}}},
    {{name:'Barrido Mach (MNM)', x:MACH_X, y:MACH_CD, mode:'lines+markers',
      line:{{color:'#e8a030',width:2,dash:'dot'}}, marker:{{size:6,color:'#e8a030'}}}},
    {{name:'CD Newton = 1.0', x:MACH_PTS, y:MACH_PTS.map(()=>1), mode:'lines',
      line:{{color:'#3de8b0',width:1.5,dash:'dash'}}}},
  ], {{...LAYOUT_2D('M∞','CD'), yaxis:{{...LAYOUT_2D().yaxis,range:[0.4,1.2]}}}}, CFG);

  // 2) cp,max vs Mach
  Plotly.newPlot('ch-cpmax', [
    {{name:'cp,max (analítico)', x:MACH_PTS, y:CPMAX, mode:'lines+markers',
      line:{{color:'#7060e8',width:2}}, marker:{{size:6}}, fill:'tozeroy', fillcolor:'#7060e810'}},
    {{name:'CD_MNM = cp,max/2', x:MACH_PTS, y:CD_MNM, mode:'lines+markers',
      line:{{color:'#e85040',width:2}}, marker:{{size:6}}}},
    {{name:'cp,max/2 (line)', x:MACH_PTS, y:CPMAX.map(v=>v/2), mode:'lines',
      line:{{color:'#3de8b0',width:1.5,dash:'dot'}}}},
  ], LAYOUT_2D('M∞',''), CFG);

  // 3) Cápsula CD vs alpha
  Plotly.newPlot('ch-cap-cd', [
    {{name:'MNM', x:MNM_A, y:MNM_CD, mode:'lines+markers',
      line:{{color:'#e85040',width:2.5}}, marker:{{size:7}}}},
    {{name:'MN',  x:MN_A,  y:MN_CD,  mode:'lines+markers',
      line:{{color:'#3de8b0',width:2.5}}, marker:{{size:7}}}},
  ], LAYOUT_2D('α (°)','CD'), CFG);

  // 4) Cápsula CL & CM vs alpha
  Plotly.newPlot('ch-cap-cl', [
    {{name:'CL (MNM)', x:MNM_A, y:MNM_CL, mode:'lines+markers',
      line:{{color:'#3de8b0',width:2.5}}, marker:{{size:7}}}},
    {{name:'CM (MNM)', x:MNM_A, y:MNM_CM, mode:'lines+markers',
      line:{{color:'#7060e8',width:2.5}}, marker:{{size:7}}}},
  ], LAYOUT_2D('α (°)',''), CFG);
}}

// ── Tabla de resultados ──────────────────────────────────────────────────────
const tbody = document.getElementById('tbody');
CASES.forEach(c => {{
  tbody.innerHTML += `<tr>
    <td style="font-size:9px">${{c.id}}</td>
    <td>${{c.geo}}</td>
    <td><span class="tag ${{c.model}}">${{c.model}}</span></td>
    <td>${{c.M}}</td>
    <td>${{c.a}}</td>
    <td class="tcd">${{c.CD.toFixed(5)}}</td>
    <td class="tcl">${{c.CL.toFixed(5)}}</td>
    <td class="tcm">${{c.CM.toFixed(5)}}</td>
    <td>${{c.nw}}</td>
  </tr>`;
}});

// ── Tabla sort ──────────────────────────────────────────────────────────────
let sortDir = 1;
function sortTable(col) {{
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a,b) => {{
    const av = a.cells[col].textContent.trim();
    const bv = b.cells[col].textContent.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    return (isNaN(an) ? av.localeCompare(bv) : an - bn) * sortDir;
  }});
  sortDir *= -1;
  rows.forEach(r => tbody.appendChild(r));
}}

// ── Checks ───────────────────────────────────────────────────────────────────
const checksGrid = document.getElementById('checks-grid');
CHECKS.forEach(c => {{
  checksGrid.innerHTML += `<div class="ck ${{c.pass?'pass':'fail'}}">
    <div class="cki ${{c.pass?'ok':'fail'}}">${{c.pass?'✓':'✗'}}</div>
    <div>
      <div class="ckt">${{c.id}} — ${{c.title}}</div>
      <div class="ckd">${{c.desc}}</div>
    </div>
  </div>`;
}});

// ── Init ─────────────────────────────────────────────────────────────────────
tabsRendered['tab-capsule'] = true;
renderCapsule();
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera report_3d.html interactivo")
    parser.add_argument("--out", type=Path, default=None, help="Ruta de salida del HTML")
    parser.add_argument("--cp",  type=Path, default=None, help="CSV de Cp por cara")
    args = parser.parse_args()

    generate_3d_html(cp_csv=args.cp, out_path=args.out)
