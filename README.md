# TP1 — Métodos de Inclinación Local (MN / MNM)

Trabajo práctico de AAVFR centrado en la implementación y análisis de los Métodos de Inclinación Local para el cálculo de distribuciones de coeficiente de presión y coeficientes aerodinámicos globales sobre geometrías trianguladas en formato STL.

El programa permite:
- leer geometrías STL,
- calcular normales, áreas y centros de triángulos,
- obtener distribuciones de \(C_p\) con los modelos **MN** y **MNM**,
- integrar fuerzas y momentos globales,
- generar figuras en `results/`,
- exportar los casos requeridos al archivo `results.json`.

---

## 1. Cómo ejecutar el trabajo

### Estructura esperada
```text
.
├── src/
├── data/
├── results/
├── run
└── README.md