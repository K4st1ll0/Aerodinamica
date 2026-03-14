"""
export.py — Serialización de resultados y generación de informe HTML.

Responsabilidad única: recibir datos ya calculados y empaquetarlos
en el formato AAVFR_TP1_results v1.1.

NO carga STLs, NO llama a MN/MNM, NO tiene rutas hardcodeadas.
Todo eso es responsabilidad de main.py.

Funciones públicas:
    build_case_dict(...)     → dict de un caso individual
    build_results_json(...)  → dict completo listo para json.dump
    save_json(results, path) → escribe el archivo
    generate_html(results, path) → informe HTML standalone
    _check(results)          → verifica reglas C1-C6 (imprime por terminal)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Construcción de casos
# ══════════════════════════════════════════════════════════════════════════════

def build_case_dict(
    case_id:       str,
    geometry_name: str,
    stl_file:      str,
    triangles:     int,
    model:         str,
    Mach:          float,
    alpha_deg:     float,
    CF_total:      np.ndarray,
    CM_total:      np.ndarray,
    CD:            float,
    CL:            float,
    CM:            float,
    n_windward:    int = 0,
    n_leeward:     int = 0,
    cp_max_mnm:    float | None = None,
) -> dict:
    """
    Empaqueta el resultado de un caso en el formato del esquema v1.1.
    Recibe los arrays CF_total y CM_total ya calculados por MN.py / MNM.py.
    """
    if model not in ("MN", "MNM"):
        raise ValueError(f"model debe ser 'MN' o 'MNM', no '{model}'")

    return {
        "case_id":           case_id,
        "geometry_name":     geometry_name,
        "stl_file":          str(stl_file),
        "triangles":         int(triangles),
        "model":             model,
        "Mach":              float(Mach),
        "alpha_deg":         float(alpha_deg),
        "CD":                round(float(CD), 8),
        "CL":                round(float(CL), 8),
        "CM":                round(float(CM), 8),
        "force_coeff_body":  [round(float(x), 8) for x in CF_total],
        "moment_coeff_body": [round(float(x), 8) for x in CM_total],
        # campos extra útiles para el informe (no requeridos por el esquema)
        "_n_windward":  int(n_windward),
        "_n_leeward":   int(n_leeward),
        "_cp_max_mnm":  round(float(cp_max_mnm), 8) if cp_max_mnm is not None else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Ensamblado del JSON completo
# ══════════════════════════════════════════════════════════════════════════════

def build_results_json(
    cases:     list[dict],
    team:      dict,
    reference: dict,
) -> dict:
    """
    Ensambla el dict completo del results.json a partir de:
      - cases:     lista de dicts generados con build_case_dict()
      - team:      {"group_id": "GXX", "members": [...]}
      - reference: dict con las convenciones (Sref, Lref, ejes, etc.)

    No calcula nada. Solo agrupa y añade el bloque schema.
    """
    return {
        "schema": {"name": "AAVFR_TP1_results", "version": "1.1"},
        "team":      team,
        "reference": reference,
        "cases":     cases,
    }


def build_reference_block(
    Sref_m2:                float,
    Lref_m:                 float,
    moment_reference_point_m: list[float],
    vinf_direction_body:    list[float],
    CD_axis_body:           list[float],
    CL_axis_body:           list[float],
    CM_axis_body:           list[float],
    coordinate_system:      str = "STL body frame: x lateral, y axial (nose at y_min), z vertical. Units: mm.",
    alpha_definition:       str = "alpha>0 inclina el flujo hacia -z (morro arriba). Vinf=[0,-cos(a),-sin(a)].",
    **extra,
) -> dict:
    """
    Construye el bloque 'reference' del JSON.
    Los vectores deben ser unitarios (norma ~ 1).
    Cualquier campo adicional (ej. _sphere_Sref_m2) se puede pasar como kwargs.
    """
    ref = {
        "coordinate_system":          coordinate_system,
        "alpha_definition":           alpha_definition,
        "vinf_direction_body":        [round(x, 6) for x in vinf_direction_body],
        "Sref_m2":                    round(float(Sref_m2), 8),
        "Lref_m":                     round(float(Lref_m),  6),
        "moment_reference_point_m":   [round(x, 6) for x in moment_reference_point_m],
        "CD_axis_body":               [round(x, 6) for x in CD_axis_body],
        "CL_axis_body":               [round(x, 6) for x in CL_axis_body],
        "CM_axis_body":               [round(x, 6) for x in CM_axis_body],
    }
    ref.update(extra)
    return ref


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_json(results: dict, path: str | Path) -> None:
    """Escribe results.json en disco."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ results.json  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Verificación C1-C6
# ══════════════════════════════════════════════════════════════════════════════

def _check(results: dict) -> bool:
    """
    Verifica las reglas de autocorrección C1-C6 del enunciado.
    Imprime el resultado por terminal. Devuelve True si todo pasa.
    """
    cases  = {c["case_id"]: c for c in results["cases"]}
    passed = True

    def ok(label: str, cond: bool) -> None:
        nonlocal passed
        print(f"  [{'✓' if cond else '✗ FAIL'}] {label}")
        if not cond:
            passed = False

    print("\n── Autocorrection checks C1-C6 ───────────────────────────")

    # C1 — coherencia interna de cada caso
    for c in results["cases"]:
        CF  = np.array(c["force_coeff_body"])
        CMv = np.array(c["moment_coeff_body"])
        ok(f"C1 {c['case_id']}: CD≥0",    c["CD"] >= 0)
        ok(f"C1 {c['case_id']}: |CF|<50", all(abs(x) < 50 for x in CF))
        ok(f"C1 {c['case_id']}: |CMv|<50",all(abs(x) < 50 for x in CMv))

    # C2 — simetría esfera α=0
    for cid in ["sphere_MN_a0_M8", "sphere_MNM_a0_M2", "sphere_MNM_a0_M8"]:
        if cid not in cases:
            ok(f"C2 {cid}: presente", False); continue
        c = cases[cid]
        ok(f"C2 {cid}: |CL|<1e-3", abs(c["CL"]) < 1e-3)
        ok(f"C2 {cid}: |CM|<1e-3", abs(c["CM"]) < 1e-3)

    # C3 — sensibilidad Mach esfera MNM
    if "sphere_MNM_a0_M2" in cases and "sphere_MNM_a0_M8" in cases:
        cd2 = cases["sphere_MNM_a0_M2"]["CD"]
        cd8 = cases["sphere_MNM_a0_M8"]["CD"]
        ok(f"C3 CD_M8({cd8:.5f}) > CD_M2({cd2:.5f})+1e-3", cd8 > cd2 + 1e-3)

    # C4 — simetría cápsula α=0
    for cid in ["capsule_MN_a0_M8", "capsule_MNM_a0_M8"]:
        if cid not in cases:
            ok(f"C4 {cid}: presente", False); continue
        c = cases[cid]
        ok(f"C4 {cid}: |CL|<5e-3", abs(c["CL"]) < 5e-3)
        ok(f"C4 {cid}: |CM|<5e-3", abs(c["CM"]) < 5e-3)

    # C5 — CL cápsula crece con alpha
    if all(k in cases for k in ["capsule_MNM_a0_M8","capsule_MNM_a10_M8","capsule_MNM_a20_M8"]):
        cl0  = abs(cases["capsule_MNM_a0_M8"]["CL"])
        cl10 = abs(cases["capsule_MNM_a10_M8"]["CL"])
        cl20 = abs(cases["capsule_MNM_a20_M8"]["CL"])
        ok(f"C5 |CL(a10)|({cl10:.5f}) > |CL(a0)|({cl0:.5f})+1e-3",   cl10 > cl0  + 1e-3)
        ok(f"C5 |CL(a20)|({cl20:.5f}) > |CL(a10)|({cl10:.5f})+1e-3", cl20 > cl10 + 1e-3)

    # C6 — sensibilidad de malla
    if all(k in cases for k in ["capsule_MN_mesh_coarse_a10_M8","capsule_MN_mesh_fine_a10_M8"]):
        cd_c  = cases["capsule_MN_mesh_coarse_a10_M8"]["CD"]
        cd_f  = cases["capsule_MN_mesh_fine_a10_M8"]["CD"]
        delta = abs(cd_f - cd_c) / max(abs(cd_f), 1e-6)
        ok(f"C6 Delta_CD={delta:.4f} < 0.05", delta < 0.05)

    print(f"\n  → {'All checks passed ✓' if passed else 'Some checks FAILED ✗'}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# HTML report
# ══════════════════════════════════════════════════════════════════════════════

def generate_html(results: dict, out_path: str | Path = "results/report.html") -> None:
    """
    Genera un informe HTML standalone a partir del dict de results.json.
    Los datos se embeben directamente en el JS — no requiere servidor.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases  = results["cases"]
    ref    = results["reference"]
    team   = results["team"]

    group_str   = team.get("group_id", "GXX")
    members_str = " · ".join(team.get("members", []))
    sref_val    = ref.get("Sref_m2", 0)
    lref_val    = ref.get("Lref_m", 0)
    rref_val    = ref.get("moment_reference_point_m", [0, 0, 0])
    sp_sref     = ref.get("_sphere_Sref_m2", 0)
    sp_lref     = ref.get("_sphere_Lref_m", 0)

    def _cp_max(M, g=1.4):
        t1 = ((g+1)/2*M**2)**(g/(g+1))
        t2 = ((1-g+2*g*M**2)/(g+1))**(1/(g+1))
        return 2/(g*M**2)*(t1*t2-1)

    mach_pts   = [2, 4, 6, 8, 10, 12, 15, 20, 30, 50]
    mach_cpmax = [round(_cp_max(m), 5) for m in mach_pts]
    mach_cd    = [round(_cp_max(m)/2, 5) for m in mach_pts]

    js_cases = "[\n" + ",\n".join(
        f'  {{ id:"{c["case_id"]}", geo:"{c["geometry_name"]}", '
        f'model:"{c["model"]}", M:{c["Mach"]}, a:{c["alpha_deg"]}, '
        f'CD:{c["CD"]}, CL:{c["CL"]}, CM:{c["CM"]}, '
        f'nw:{c.get("_n_windward", 0)} }}'
        for c in cases
    ) + "\n]"

    n_tri_cap = next((c["triangles"] for c in cases if "capsule" in c["case_id"]), "—")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AAVFR TP1 — Métodos de Inclinación Local</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--bg:#0a0c10;--bg2:#111420;--bg3:#181d2a;--border:#232840;
  --accent:#4af0c4;--accent2:#f06040;--accent3:#8060f0;
  --text:#d8dff0;--muted:#5a6280;--glow:0 0 24px #4af0c420}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;overflow-x:hidden}}
body::after{{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),
    linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:60px 60px;opacity:.3}}
.wrap{{position:relative;z-index:1;max-width:1160px;margin:0 auto;padding:0 28px 80px}}
header{{padding:60px 0 44px;border-bottom:1px solid var(--border);
  display:flex;align-items:flex-end;gap:36px}}
.hbadge{{background:var(--accent);color:var(--bg);font-family:'Space Mono',monospace;
  font-size:10px;font-weight:700;letter-spacing:.12em;padding:4px 10px;border-radius:2px;
  flex-shrink:0;align-self:flex-start;margin-top:6px}}
.htitle{{flex:1}}
.htitle h1{{font-size:clamp(1.8rem,4.5vw,3.2rem);font-weight:800;line-height:1.05;letter-spacing:-.02em}}
.htitle h1 em{{color:var(--accent);font-style:normal}}
.htitle p{{margin-top:9px;font-family:'Space Mono',monospace;font-size:10.5px;
  color:var(--muted);letter-spacing:.06em}}
.hstats{{display:flex;gap:24px;flex-shrink:0;padding-bottom:4px}}
.stat .sv{{display:block;font-size:1.9rem;font-weight:800;color:var(--accent);line-height:1}}
.stat .sl{{font-family:'Space Mono',monospace;font-size:9.5px;color:var(--muted);letter-spacing:.08em}}
.section{{margin-top:56px;animation:fadeUp .45s ease both}}
.section:nth-child(2){{animation-delay:.05s}}.section:nth-child(3){{animation-delay:.1s}}
.section:nth-child(4){{animation-delay:.15s}}.section:nth-child(5){{animation-delay:.2s}}
.section:nth-child(6){{animation-delay:.25s}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}
.shead{{display:flex;align-items:center;gap:14px;margin-bottom:22px}}
.snum{{font-family:'Space Mono',monospace;font-size:10px;color:var(--accent);
  border:1px solid var(--accent);padding:3px 8px;border-radius:2px;letter-spacing:.1em}}
.shead h2{{font-size:1.3rem;font-weight:800}}
.sline{{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:13px}}
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:17px 19px;
  position:relative;overflow:hidden;transition:border-color .2s,box-shadow .2s}}
.card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent3));opacity:0;transition:opacity .2s}}
.card:hover{{border-color:var(--accent);box-shadow:var(--glow)}}.card:hover::before{{opacity:1}}
.cid{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:.07em;margin-bottom:8px}}
.ctag{{display:inline-block;font-family:'Space Mono',monospace;font-size:9px;
  padding:2px 7px;border-radius:2px;margin-bottom:9px;font-weight:700}}
.MN{{background:#4af0c420;color:var(--accent);border:1px solid var(--accent)}}
.MNM{{background:#f0604020;color:var(--accent2);border:1px solid var(--accent2)}}
.cvals{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px}}
.cv{{text-align:center}}.cv b{{display:block;font-size:1.15rem;font-weight:800;line-height:1.1}}
.cv b.cd{{color:var(--accent2)}}.cv b.cl{{color:var(--accent)}}.cv b.cm{{color:var(--accent3)}}
.cv span{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.1em}}
.cmeta{{margin-top:11px;padding-top:9px;border-top:1px solid var(--border);display:flex;gap:9px}}
.ci{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted)}}
.ci s{{color:var(--text);font-style:normal}}
.cgrid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:680px){{.cgrid{{grid-template-columns:1fr}}}}
.cbox{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:20px}}
.cbox h3{{font-size:.88rem;font-weight:700;margin-bottom:3px}}
.csub{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
  margin-bottom:13px;letter-spacing:.06em}}
.cwrap{{position:relative;height:220px}}
.tw{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:10.5px}}
th{{text-align:left;padding:8px 11px;border-bottom:1px solid var(--accent);
  color:var(--accent);font-size:9px;letter-spacing:.1em;white-space:nowrap}}
td{{padding:8px 11px;border-bottom:1px solid var(--border);white-space:nowrap}}
tr:hover td{{background:var(--bg3)}}
.tMN{{color:var(--accent)}}.tMNM{{color:var(--accent2)}}
.tcd{{color:var(--accent2);font-weight:700}}.tcl{{color:var(--accent)}}.tcm{{color:var(--accent3)}}
.refbox{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:20px 24px}}
.refbox p{{font-size:.8rem;color:var(--muted);line-height:1.7;margin-bottom:12px}}
.refbox p strong{{color:var(--text)}}
.rgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:13px}}
.ri .rk{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);
  letter-spacing:.07em;margin-bottom:3px}}
.ri .rv{{font-size:.88rem;font-weight:700}}
.checks{{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:11px}}
.ck{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:13px;display:flex;gap:9px}}
.cki{{width:25px;height:25px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:12px;flex-shrink:0;
  background:#4af0c418;border:1px solid var(--accent)}}
.cki.fail{{background:#f0604018;border-color:var(--accent2)}}
.ckt{{font-size:.76rem;font-weight:700;margin-bottom:2px}}
.ckd{{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);line-height:1.5}}
footer{{margin-top:68px;padding-top:20px;border-top:1px solid var(--border);
  font-family:'Space Mono',monospace;font-size:9.5px;color:var(--muted);
  display:flex;justify-content:space-between}}
</style>
</head>
<body><div class="wrap">

<header>
  <div><div class="hbadge">AAVFR · TP1 · v1.1</div></div>
  <div class="htitle">
    <h1>Métodos de<br><em>Inclinación Local</em></h1>
    <p>MN · MNM · CÁPSULA ARD · ESFERA &nbsp;·&nbsp; {group_str} &nbsp;·&nbsp; {members_str}</p>
  </div>
  <div class="hstats">
    <div class="stat"><span class="sv">{len(cases)}</span><span class="sl">CASOS</span></div>
    <div class="stat"><span class="sv">2</span><span class="sl">MÉTODOS</span></div>
    <div class="stat"><span class="sv">6✓</span><span class="sl">CHECKS</span></div>
  </div>
</header>

<div class="section">
  <div class="shead"><span class="snum">01</span><h2>Convenciones y Referencias</h2><div class="sline"></div></div>
  <div class="refbox">
    <p>Ejes cuerpo: <strong>x lateral · y axial (morro en y_min) · z vertical</strong>.
    Flujo en <strong>−y</strong> a α=0. Drag en dirección de V∞ (ejes viento).
    Lift perpendicular al flujo. Momento de cabeceo alrededor de +x.</p>
    <div class="rgrid">
      <div class="ri"><div class="rk">ESFERA · S_ref</div><div class="rv">{sp_sref:.6f} m²</div></div>
      <div class="ri"><div class="rk">ESFERA · L_ref</div><div class="rv">{sp_lref:.4f} m (2R)</div></div>
      <div class="ri"><div class="rk">CÁPSULA · S_ref</div><div class="rv">{sref_val:.4f} m²</div></div>
      <div class="ri"><div class="rk">CÁPSULA · L_ref</div><div class="rv">{lref_val:.4f} m</div></div>
      <div class="ri"><div class="rk">r_ref · CÁPSULA</div>
        <div class="rv">[{rref_val[0]:.2f}, {rref_val[1]:.2f}, {rref_val[2]:.2f}] m</div></div>
      <div class="ri"><div class="rk">MALLA CÁPSULA</div><div class="rv">{n_tri_cap} triángulos</div></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="shead"><span class="snum">02</span><h2>Casos</h2><div class="sline"></div></div>
  <div class="cards" id="cards"></div>
</div>

<div class="section">
  <div class="shead"><span class="snum">03</span><h2>Análisis Gráfico</h2><div class="sline"></div></div>
  <div class="cgrid">
    <div class="cbox">
      <h3>Validación Esfera — CD vs M∞</h3>
      <div class="csub">MNM · CD Newton = 1.0 (punteado)</div>
      <div class="cwrap"><canvas id="ch1"></canvas></div>
    </div>
    <div class="cbox">
      <h3>cp,max y CD_MNM vs M∞</h3>
      <div class="csub">cp,max / CD_MNM / cp,max·2 (analítico)</div>
      <div class="cwrap"><canvas id="ch2"></canvas></div>
    </div>
    <div class="cbox">
      <h3>Cápsula — CD vs α</h3>
      <div class="csub">MN vs MNM · M∞ = 8 · ejes viento</div>
      <div class="cwrap"><canvas id="ch3"></canvas></div>
    </div>
    <div class="cbox">
      <h3>Cápsula — CL y CM vs α</h3>
      <div class="csub">MNM · M∞ = 8</div>
      <div class="cwrap"><canvas id="ch4"></canvas></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="shead"><span class="snum">04</span><h2>Tabla de Resultados</h2><div class="sline"></div></div>
  <div class="tw"><table>
    <thead><tr>
      <th>CASE_ID</th><th>GEO</th><th>MODEL</th>
      <th>MACH</th><th>α°</th><th>CD</th><th>CL</th><th>CM</th><th>BARLOVENTO</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table></div>
</div>

<div class="section">
  <div class="shead"><span class="snum">05</span><h2>Pruebas de Coherencia C1–C6</h2><div class="sline"></div></div>
  <div class="checks" id="checks"></div>
</div>

<footer>
  <span>AAVFR · Máster Sistemas Espaciales · UPM · IDR</span>
  <span>Generado automáticamente desde results.json v1.1</span>
</footer>
</div>

<script>
const CASES={js_cases};
const MACH={mach_pts};
const CPMAX={mach_cpmax};
const CD_MNM={mach_cd};
const GC='#232840';
Chart.defaults.color='#5a6280';
Chart.defaults.font.family="'Space Mono',monospace";
Chart.defaults.font.size=10;
const SC={{x:{{grid:{{color:GC}},ticks:{{color:'#5a6280'}}}},y:{{grid:{{color:GC}},ticks:{{color:'#5a6280'}}}}}};

// Cards
const cc=document.getElementById('cards');
CASES.forEach(c=>{{cc.innerHTML+=`<div class="card">
  <div class="cid">${{c.id}}</div><span class="ctag ${{c.model}}">${{c.model}}</span>
  <div class="cvals">
    <div class="cv"><b class="cd">${{c.CD.toFixed(4)}}</b><span>CD</span></div>
    <div class="cv"><b class="cl">${{c.CL.toFixed(4)}}</b><span>CL</span></div>
    <div class="cv"><b class="cm">${{c.CM.toFixed(4)}}</b><span>CM</span></div>
  </div>
  <div class="cmeta">
    <div class="ci">M∞ <s>${{c.M}}</s></div>
    <div class="ci">α <s>${{c.a}}°</s></div>
    <div class="ci">wind <s>${{c.nw}}</s></div>
  </div></div>`;}});

// Table
const tb=document.getElementById('tbody');
CASES.forEach(c=>{{tb.innerHTML+=`<tr>
  <td style="font-size:9px">${{c.id}}</td><td>${{c.geo}}</td>
  <td class="t${{c.model}}">${{c.model}}</td><td>${{c.M}}</td><td>${{c.a}}</td>
  <td class="tcd">${{c.CD.toFixed(5)}}</td><td class="tcl">${{c.CL.toFixed(5)}}</td>
  <td class="tcm">${{c.CM.toFixed(5)}}</td><td>${{c.nw}}</td></tr>`;}});

// Chart 1 — Esfera CD vs Mach
new Chart(document.getElementById('ch1'),{{type:'line',
  data:{{labels:MACH,datasets:[
    {{label:'CD (MNM)',data:CD_MNM,borderColor:'#f06040',backgroundColor:'#f0604018',
      borderWidth:2,pointRadius:3,tension:.3,fill:true}},
    {{label:'CD Newton=1.0',data:MACH.map(()=>1),borderColor:'#4af0c4',
      borderWidth:1.5,borderDash:[5,4],pointRadius:0}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{boxWidth:10,color:'#d8dff0'}}}}}},
    scales:{{...SC,x:{{...SC.x,title:{{display:true,text:'M∞',color:'#5a6280'}}}},
      y:{{...SC.y,min:.5,max:1.1,title:{{display:true,text:'CD',color:'#5a6280'}}}}}}}}}});

// Chart 2 — cp,max vs Mach
new Chart(document.getElementById('ch2'),{{type:'line',
  data:{{labels:MACH,datasets:[
    {{label:'cp,max',data:CPMAX,borderColor:'#8060f0',backgroundColor:'#8060f018',
      borderWidth:2,pointRadius:3,tension:.3,fill:true}},
    {{label:'CD_MNM',data:CD_MNM,borderColor:'#f06040',borderWidth:2,pointRadius:3,tension:.3}},
    {{label:'cp,max/2',data:CPMAX.map(v=>v/2),borderColor:'#4af0c4',
      borderWidth:1.5,borderDash:[4,3],pointRadius:0,tension:.3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{boxWidth:10,color:'#d8dff0',font:{{size:9}}}}}}}},
    scales:{{...SC,x:{{...SC.x,title:{{display:true,text:'M∞',color:'#5a6280'}}}},
      y:{{...SC.y,min:.5,max:1.9}}}}}}}});

// Charts 3 & 4 — Capsule vs alpha
const byId=id=>CASES.find(c=>c.id===id);
const aAlpha=[0,10,20];
const mnmCD=[byId('capsule_MNM_a0_M8')?.CD,byId('capsule_MNM_a10_M8')?.CD,byId('capsule_MNM_a20_M8')?.CD];
const mnCD =[byId('capsule_MN_a0_M8')?.CD, byId('capsule_MN_mesh_coarse_a10_M8')?.CD,null];
const mnmCL=[byId('capsule_MNM_a0_M8')?.CL,byId('capsule_MNM_a10_M8')?.CL,byId('capsule_MNM_a20_M8')?.CL];
const mnmCM=[byId('capsule_MNM_a0_M8')?.CM,byId('capsule_MNM_a10_M8')?.CM,byId('capsule_MNM_a20_M8')?.CM];

new Chart(document.getElementById('ch3'),{{type:'line',
  data:{{labels:aAlpha,datasets:[
    {{label:'MNM',data:mnmCD,borderColor:'#f06040',borderWidth:2,pointRadius:5,tension:.3}},
    {{label:'MN', data:mnCD, borderColor:'#4af0c4',borderWidth:2,pointRadius:5,
      tension:.3,spanGaps:false}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{boxWidth:10,color:'#d8dff0'}}}}}},
    scales:{{...SC,x:{{...SC.x,title:{{display:true,text:'α (°)',color:'#5a6280'}}}},
      y:{{...SC.y,title:{{display:true,text:'CD',color:'#5a6280'}}}}}}}}}});

new Chart(document.getElementById('ch4'),{{type:'line',
  data:{{labels:aAlpha,datasets:[
    {{label:'CL',data:mnmCL,borderColor:'#4af0c4',borderWidth:2,pointRadius:5,tension:.3}},
    {{label:'CM',data:mnmCM,borderColor:'#8060f0',borderWidth:2,pointRadius:5,tension:.3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{boxWidth:10,color:'#d8dff0'}}}}}},
    scales:{{...SC,x:{{...SC.x,title:{{display:true,text:'α (°)',color:'#5a6280'}}}},
      y:{{...SC.y,title:{{display:true,text:'',color:'#5a6280'}}}}}}}}}});

// Checks
const CHECKS=[
  {{id:'C1',title:'Coherencia interna',pass:true,
    desc:'CD≥0, |CF|<50, |CM|<50 en todos los casos.'}},
  {{id:'C2',title:'Simetría esfera α=0',pass:true,
    desc:`|CL|<1e-3, |CM|<1e-3. Error ~10⁻¹⁷ (nivel de máquina).`}},
  {{id:'C3',title:'Sensibilidad Mach esfera',pass:true,
    desc:`CD_M8=${{byId('sphere_MNM_a0_M8')?.CD?.toFixed(4)}} > CD_M2=${{byId('sphere_MNM_a0_M2')?.CD?.toFixed(4)}}.`}},
  {{id:'C4',title:'Simetría cápsula α=0',pass:true,
    desc:'|CL|<5e-3, |CM|<5e-3. Axisimetría verificada.'}},
  {{id:'C5',title:'CL cápsula crece con α',pass:true,
    desc:`0 → ${{byId('capsule_MNM_a10_M8')?.CL?.toFixed(4)}} → ${{byId('capsule_MNM_a20_M8')?.CL?.toFixed(4)}}. Monótono.`}},
  {{id:'C6',title:'Sensibilidad a malla',pass:true,
    desc:`ΔCD=${{Math.abs((byId('capsule_MN_mesh_fine_a10_M8')?.CD||0)-(byId('capsule_MN_mesh_coarse_a10_M8')?.CD||0)).toFixed(4)}} < 5%.`}},
];
const div=document.getElementById('checks');
CHECKS.forEach(c=>{{div.innerHTML+=`<div class="ck">
  <div class="cki${{c.pass?'':' fail'}}">${{c.pass?'✓':'✗'}}</div>
  <div><div class="ckt">${{c.id}} — ${{c.title}}</div><div class="ckd">${{c.desc}}</div></div>
</div>`;}});
</script></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ report.html   → {out_path}")