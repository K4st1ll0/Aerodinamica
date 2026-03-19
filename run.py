"""
run.py — Punto de entrada del trabajo TP1 de Aerodinámica.

Uso:
    python run.py

Ejecuta src/main.py desde la raíz del proyecto, garantizando que las rutas
relativas (data/, results/) se resuelvan correctamente.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "src" / "main.py"

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, str(MAIN)],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)
