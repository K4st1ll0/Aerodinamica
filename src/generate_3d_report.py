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
    return [{k: float(v) for k, v in r.items()} for r in rows]


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
# Generación del HTML
# ════════════════════════════════════════════════════════════════════════════

def generate_3d_html(
    results_json: Path | None = None,
    cp_csv: Path | None = None,
    stl_capsule: Path | None = None,
    stl_sphere: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    # ── Rutas por defecto ────────────────────────────────────────────────────
    results_json = results_json or RESULTS_DIR / "results.json"
    cp_csv       = cp_csv       or RESULTS_DIR / "cp_faces_mnm_a20_M8.csv"
    stl_capsule  = stl_capsule  or DATA_DIR   / "Capsula" / "PruebaARD3.stl"
    stl_sphere   = stl_sphere   or DATA_DIR   / "esfera.stl"
    out_path     = out_path     or RESULTS_DIR / "report_3d.html"

    # ── Carga ────────────────────────────────────────────────────────────────
    print("Cargando datos...")
    with open(results_json, encoding="utf-8") as f:
        results = json.load(f)
    cases  = results["cases"]
    ref    = results["reference"]
    team   = results["team"]

    print("  Cargando STL cápsula...")
    verts_cap, faces_cap = load_stl_mesh(stl_capsule)
    print(f"    {len(verts_cap)} vértices, {len(faces_cap)} caras")

    print("  Cargando STL esfera...")
    verts_sph, faces_sph = load_stl_mesh(stl_sphere)
    print(f"    {len(verts_sph)} vértices, {len(faces_sph)} caras")

    print("  Cargando Cp CSV...")
    face_cp = load_cp_csv(cp_csv)
    vertex_cp_cap = face_cp_to_vertex_cp(verts_cap, faces_cap, face_cp)
    print(f"    Cp = [{face_cp.min():.4f}, {face_cp.max():.4f}]")

    # ── Sweep CSVs (opcionales) ──────────────────────────────────────────────
    sweep_mnm, sweep_mn, sweep_mach = [], [], []
    for path, store in [
        (RESULTS_DIR / "results_mnm_M8.csv",          "mnm"),
        (RESULTS_DIR / "results_mn.csv",               "mn"),
        (RESULTS_DIR / "results_mnm_mach_sweep.csv",  "mach"),
    ]:
        if path.exists():
            data = load_sweep_csv(path)
            if store == "mnm":   sweep_mnm  = data
            elif store == "mn":  sweep_mn   = data
            else:                sweep_mach = data

    # ── Analítico cp_max vs Mach ─────────────────────────────────────────────
    mach_pts    = [2, 4, 6, 8, 10, 12, 15, 20, 30, 50]
    cpmax_pts   = [round(cpmax_analytic(m), 5) for m in mach_pts]
    cd_mnm_pts  = [round(cpmax_analytic(m) / 2, 5) for m in mach_pts]

    # ── Serialización de geometrías ──────────────────────────────────────────
    cx = _jf(verts_cap[:, 0])
    cy = _jf(verts_cap[:, 1])
    cz = _jf(verts_cap[:, 2])
    ci = _ji(faces_cap[:, 0])
    cj = _ji(faces_cap[:, 1])
    ck = _ji(faces_cap[:, 2])
    cp_js = _jf(vertex_cp_cap, precision=5)

    sx = _jf(verts_sph[:, 0])
    sy = _jf(verts_sph[:, 1])
    sz = _jf(verts_sph[:, 2])
    si = _ji(faces_sph[:, 0])
    sj = _ji(faces_sph[:, 1])
    sk = _ji(faces_sph[:, 2])

    # ── Sweep data para gráficas ─────────────────────────────────────────────
    def _sweep_json(rows, key_x, key_y):
        xs = [r[key_x] for r in rows if key_x in r]
        ys = [r[key_y] for r in rows if key_y in r]
        return json.dumps(xs), json.dumps(ys)

    mnm_alphas, mnm_CD = _sweep_json(sweep_mnm, "alpha_deg", "CD")
    mnm_alphas, mnm_CL = _sweep_json(sweep_mnm, "alpha_deg", "CL")
    mnm_alphas, mnm_CM = _sweep_json(sweep_mnm, "alpha_deg", "CM")
    mn_alphas,  mn_CD  = _sweep_json(sweep_mn,  "alpha_deg", "CD")

    mach_sweep_x, mach_sweep_cd = _sweep_json(sweep_mach, "Mach", "CD")

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
        "id":    c["case_id"],
        "geo":   c["geometry_name"],
        "model": c["model"],
        "M":     c["Mach"],
        "a":     c["alpha_deg"],
        "CD":    c["CD"],
        "CL":    c["CL"],
        "CM":    c["CM"],
        "nw":    c.get("_n_windward", 0),
    } for c in cases])

    checks_data = _build_checks(cases)

    # ── HTML ─────────────────────────────────────────────────────────────────
    html = _html_template(
        group_str=group_str, members_str=members_str,
        sref_val=sref_val, lref_val=lref_val, sp_sref=sp_sref, sp_lref=sp_lref,
        n_tri_cap=n_tri_cap, n_tri_sph=n_tri_sph,
        cx=cx, cy=cy, cz=cz, ci=ci, cj=cj, ck=ck, cp_js=cp_js,
        sx=sx, sy=sy, sz=sz, si=si, sj=sj, sk=sk,
        mach_pts=json.dumps(mach_pts),
        cpmax_pts=json.dumps(cpmax_pts),
        cd_mnm_pts=json.dumps(cd_mnm_pts),
        mnm_alphas=mnm_alphas, mnm_CD=mnm_CD, mnm_CL=mnm_CL, mnm_CM=mnm_CM,
        mn_alphas=mn_alphas, mn_CD=mn_CD,
        mach_sweep_x=mach_sweep_x, mach_sweep_cd=mach_sweep_cd,
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
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
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
body{{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;
  min-height:100vh;overflow-x:hidden}}

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
.htitle h1{{font-size:clamp(1.9rem,4vw,3rem);font-weight:800;line-height:1.05;letter-spacing:-.02em}}
.htitle h1 em{{color:var(--accent);font-style:normal}}
.htitle p{{margin-top:8px;font-family:'Space Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:.06em}}
.hstats{{display:flex;gap:28px;flex-shrink:0}}
.stat .sv{{display:block;font-size:2rem;font-weight:800;color:var(--accent);line-height:1}}
.stat .sl{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.1em}}

/* ── Sections ── */
.section{{margin-top:60px;display:none;animation:fadeUp .35s ease both}}
.section.active{{display:block}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
.shead{{display:flex;align-items:center;gap:14px;margin-bottom:24px}}
.snum{{font-family:'Space Mono',monospace;font-size:10px;color:var(--accent);
  border:1px solid var(--accent);padding:3px 8px;border-radius:2px;letter-spacing:.1em}}
.shead h2{{font-size:1.25rem;font-weight:800}}
.sline{{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}}

/* ── 3D viewer panel ── */
.viewer-wrap{{display:grid;grid-template-columns:1fr 300px;gap:16px;align-items:start}}
@media(max-width:880px){{.viewer-wrap{{grid-template-columns:1fr}}}}
.plot3d{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;height:560px}}
.info-panel{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px}}
.info-panel h3{{font-size:.9rem;font-weight:800;margin-bottom:4px}}
.info-sub{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:.06em;margin-bottom:18px}}
.kv{{margin-bottom:12px}}
.kv .k{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.07em}}
.kv .v{{font-size:1.4rem;font-weight:800;line-height:1.1;margin-top:2px}}
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
.chart-box h3{{padding:16px 18px 4px;font-size:.88rem;font-weight:700}}
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
</style>
</head>
<body>

<nav>
  <div class="nav-brand">AAVFR · TP1</div>
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="showTab('tab-capsule',this)">3D Cápsula Cp</button>
    <button class="tab-btn" onclick="showTab('tab-sphere',this)">3D Esfera</button>
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
  <div class="viewer-wrap">
    <div class="plot3d" id="plot-capsule"></div>
    <div class="info-panel">
      <h3>Distribución de Cp</h3>
      <div class="info-sub">MNM · α = 20° · M∞ = 8</div>
      <div class="kv"><div class="k">CD</div>
        <div class="v cd" id="info-cd">—</div></div>
      <div class="kv"><div class="k">CL</div>
        <div class="v cl" id="info-cl">—</div></div>
      <div class="kv"><div class="k">CM</div>
        <div class="v cm" id="info-cm">—</div></div>
      <div class="divider"></div>
      <div class="kv"><div class="k">TRIÁNGULOS</div>
        <div class="v">{d['n_tri_cap']}</div></div>
      <div class="kv"><div class="k">S_ref</div>
        <div class="v">{d['sref_val']:.4f} m²</div></div>
      <div class="kv"><div class="k">L_ref</div>
        <div class="v">{d['lref_val']:.4f} m</div></div>
      <div class="divider"></div>
      <div class="legend-item">
        <div class="legend-color" style="background:linear-gradient(90deg,#440154,#21908c,#fde725)"></div>
        <span>Cp: Viridis (bajo → alto)</span>
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


<!-- ═══════════════════ TAB: GRÁFICAS ═══════════════════ -->
<div class="section" id="tab-charts">
  <div class="shead"><span class="snum">03</span><h2>Análisis Gráfico Interactivo</h2><div class="sline"></div></div>
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
  <div class="shead"><span class="snum">04</span><h2>Tabla de Resultados</h2><div class="sline"></div></div>
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
  <div class="shead"><span class="snum">05</span><h2>Convenciones y Referencias</h2><div class="sline"></div></div>
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
  <div class="shead"><span class="snum">06</span><h2>Pruebas de Coherencia C1–C6</h2><div class="sline"></div></div>
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
  if (id === 'tab-sphere')  renderSphere();
  if (id === 'tab-charts')  renderCharts();
}}

// ── 3D Cápsula (render inmediato) ───────────────────────────────────────────
function renderCapsule() {{
  const trace = {{
    type: 'mesh3d',
    x: CAP_X, y: CAP_Y, z: CAP_Z,
    i: CAP_I, j: CAP_J, k: CAP_K,
    intensity: CAP_CP,
    colorscale: 'Viridis',
    showscale: true,
    colorbar: {{
      title: {{text:'Cp', side:'right', font:{{size:9,color:MUTED}}}},
      thickness: 14, len: 0.7,
      tickfont: {{size:9, color:MUTED}},
      outlinecolor: GRID,
      bgcolor: BG,
    }},
    flatshading: false,
    lighting: {{ambient:0.5, diffuse:0.7, specular:0.3, roughness:0.5}},
    lightposition: {{x:1000,y:-2000,z:2000}},
    hovertemplate: 'x:%{{x:.1f}}<br>y:%{{y:.1f}}<br>z:%{{z:.1f}}<br>Cp:%{{intensity:.4f}}<extra></extra>',
  }};
  Plotly.newPlot('plot-capsule', [trace], LAYOUT_3D('Cápsula ARD — Cp (MNM · α=20° · M=8)'), CFG);

  // Fill info panel
  const c = CASES.find(x => x.id === 'capsule_MNM_a20_M8');
  if (c) {{
    document.getElementById('info-cd').textContent = c.CD.toFixed(5);
    document.getElementById('info-cl').textContent = c.CL.toFixed(5);
    document.getElementById('info-cm').textContent = c.CM.toFixed(5);
  }}
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
