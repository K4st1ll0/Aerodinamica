"""
main.py — Configuración del pipeline.

Este archivo define QUÉ se calcula: geometrías, condiciones de flujo,
sweeps a ejecutar, casos del informe, etc.

NO contiene lógica de cálculo. Toda la implementación está en runner.py.
Para ejecutar: python src/main.py  (o:  python run.py)
"""

from pathlib import Path
from runner import run_pipeline

# ─── Rutas base ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CSV_DIR     = RESULTS_DIR / "csv"
PLOTS_DIR   = RESULTS_DIR / "Plots"

RESULTS_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     BLOQUE DE CONFIGURACIÓN                             ║
# ║   Editar aquí para cambiar qué se calcula. No tocar runner.py.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

config = {

    # ── Geometrías STL ──────────────────────────────────────────────────────
    "STL_CAPSULE": DATA_DIR / "Capsula" / "PruebaARD3.stl",   # malla principal
    "STL_SPHERE":  DATA_DIR / "esfera.stl",
    "STL_COARSE":  DATA_DIR / "Capsula" / "PruebaARD.stl",    # 182 caras
    "STL_FINE":    DATA_DIR / "Capsula" / "PruebaARD4.stl",   # 1510 caras

    # ── Condiciones de flujo ────────────────────────────────────────────────
    "ALPHAS_DEG": [0.0, 10.0, 20.0, 30.0],        # barrido en alpha (para sweeps)
    "MACH":       8.0,                             # Mach para el barrido en alpha
    "MACH_SWEEP": [2.0, 4.0, 8.0, 12.0, 15.0, 20.0],  # Machs para barrido Mach
    "GAMMA":      1.4,

    # ── Mapa de Cp 3D (None = no generar) ──────────────────────────────────
    "CP_MAP_ALPHA":  20.0,    # alpha para el mapa de Cp
    "CP_MAP_METHOD": "MNM",   # "MN" o "MNM"

    # ── Sweeps a ejecutar ───────────────────────────────────────────────────
    "RUN_MN_SWEEP":         True,   # barrido MN en alpha → results_mn.csv
    "RUN_MNM_SWEEP":        True,   # barrido MNM en alpha → results_mnm_M8.csv
    "RUN_MACH_SWEEP":       True,   # barrido Mach MNM a α=0 → results_mnm_mach_sweep.csv
    "RUN_MESH_SENSITIVITY": True,   # comparativa 4 mallas → results_mesh_sensitivity.csv

    # ── Sensibilidad de malla ───────────────────────────────────────────────
    "MESH_ALPHA": 10.0,
    "MESH_MACH":  8.0,
    "MESH_STLS": [
        DATA_DIR / "Capsula" / "PruebaARD.stl",   # muy gruesa  ~182
        DATA_DIR / "Capsula" / "PruebaARD3.stl",  # media       ~1510
        DATA_DIR / "Capsula" / "PruebaARD4.stl",  # media-alta  ~1510
        DATA_DIR / "Capsula" / "PruebaARD2.stl",  # muy fina    ~400k
    ],

    # ── Visualización geométrica al arrancar ────────────────────────────────
    "SHOW_GEOMETRY": True,

    # ── Alpha trim (externo, p.ej. del libro) ───────────────────────────────
    "ALPHA_TRIM_DEG":    -21.0,   # None = no calcular
    "ALPHA_TRIM_METHOD": "MNM",

    # ── Identificación del grupo ────────────────────────────────────────────
    "GROUP_ID": "G5 - ARD",
    "MEMBERS":  [
        "Pablo Castillo Jiménez",
        "Alberto García Díaz",
        "Ana Jorba Vera",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline(config, root=ROOT, csv_dir=CSV_DIR, plots_dir=PLOTS_DIR, results_dir=RESULTS_DIR)
