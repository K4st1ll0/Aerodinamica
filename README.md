# Trabajo 1 — Métodos de Inclinación Local (MN / MNM)

**Grupo G5 · ARD** — Pablo Castillo Jiménez · Alberto García Díaz · Ana Jorba Vera

---

## 1. Cómo ejecutar

```bash
# Desde la raíz del repositorio:
python src/main.py
```

Los resultados se generan en `results/`: `results.json`, `report.html`, `report_3d.html`,
subcarpetas `csv/{capsule}/` y `Plots/{capsule}/`.

Para analizar una nueva cápsula, añadir su ruta STL a la lista `CAPSULES` en `src/main.py`
(cada cápsula genera automáticamente su propia subcarpeta de resultados).

---

## 2. Dependencias

| | Versión |
|---|---|
| Python | 3.11 |
| numpy | ≥ 1.26 |
| matplotlib | ≥ 3.10 |
| trimesh | ≥ 4.11 |

```bash
pip install numpy matplotlib trimesh
```

---

## 3. Convenciones

**Ejes del cuerpo** — `x` lateral, `y` axial (apunta hacia la popa), `z` ascendente.

**Ángulo de ataque α** — `α = 0°`: flujo frontal sobre el escudo térmico.
`α > 0°`: flujo inclinado hacia `+z` (morro sube).

**Dirección de V∞** — A `α = 0°` el flujo llega en `+y`.
Los ejes viento rotan con α:
`eD = [0, cos α, sin α]` (arrastre), `eL = eM × eD` (sustentación), `eM = [1, 0, 0]` (cabeceo).

**Signos** — `CD > 0` oponiéndose al avance; `CL > 0` con α > 0; `CM > 0` en `+x`.
Fuerza por cara: `CF_i = −(cp_i · Aᵢ / S_ref) · n̂ᵢ` (presión actúa hacia el interior).

**S_ref y L_ref** — Cápsula: `S_ref = ext_x · ext_z` (sección frontal xz), `L_ref = ext_y` (longitud axial).
Esfera: `S_ref = π R²`, `L_ref = 2R`.

**Punto de referencia de momentos** — Centroide geométrico de las caras ponderado por área.
Eje de momento: `eM = [1, 0, 0]`.

---

## 4. Casos en `results.json`

| `case_id` | Geometría | Modelo | α (°) | M∞ |
|---|---|---|---|---|
| `sphere_MN_a0_M8` | Esfera | MN | 0 | 8 |
| `sphere_MNM_a0_M2` | Esfera | MNM | 0 | 2 |
| `sphere_MNM_a0_M8` | Esfera | MNM | 0 | 8 |
| `capsule_MN_a0_M8` | ARD chillfine | MN | 0 | 8 |
| `capsule_MNM_a0_M8` | ARD chillfine | MNM | 0 | 8 |
| `capsule_MNM_a10_M8` | ARD chillfine | MNM | 10 | 8 |
| `capsule_MNM_a20_M8` | ARD chillfine | MNM | 20 | 8 |
| `capsule_MN_mesh_coarse_a10_M8` | ARD coarse | MN | 10 | 8 |
| `capsule_MN_mesh_fine_a10_M8` | ARD fine | MN | 10 | 8 |

CD, CL, CM, CF y CM_total de cada caso en `results/results.json`.
