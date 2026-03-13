from __future__ import annotations

import numpy as np


def unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0.0:
        raise ValueError("No se puede normalizar un vector nulo.")
    return v / n


def flow_direction_from_alpha(alpha_deg: float) -> np.ndarray:
    """
    Convención:
    - alpha = 0 deg  -> flujo en -y
    - alpha > 0      -> el flujo se inclina hacia -z
    """
    alpha = np.deg2rad(alpha_deg)
    v = np.array([
        0.0,
        -np.cos(alpha),
        -np.sin(alpha),
    ], dtype=float)
    return unit_vector(v)


def compute_mu(normals: np.ndarray, Vinf_hat: np.ndarray) -> np.ndarray:
    """
    mu = n · s, donde s = -Vinf_hat apunta hacia donde viene el aire.
    """
    normals = np.asarray(normals, dtype=float)
    s_hat = -unit_vector(Vinf_hat)
    return normals @ s_hat


def compute_cpmax_modified_newton(Mach: float, gamma: float = 1.4) -> float:
    """
    Calcula cp_max del Método de Newton Modificado (MNM).

    cp_max = (2 / (gamma * M^2)) * (p02/pinf - 1)

    con:
    p02/pinf =
    [ ((gamma + 1)/2) * M^2 ]^(gamma/(gamma+1))
    *
    [ (1 - gamma + 2*gamma*M^2)/(gamma + 1) ]^(1/(gamma+1))
    """
    M = float(Mach)
    g = float(gamma)

    if M <= 0.0:
        raise ValueError("Mach debe ser > 0.")
    if g <= 1.0:
        raise ValueError("gamma debe ser > 1.")

    term1 = ((g + 1.0) / 2.0 * M**2) ** (g / (g + 1.0))
    term2 = ((1.0 - g + 2.0 * g * M**2) / (g + 1.0)) ** (1.0 / (g + 1.0))
    p02_over_pinf = term1 * term2

    cp_max = (2.0 / (g * M**2)) * (p02_over_pinf - 1.0)
    return float(cp_max)


def compute_cp_modified_newton(
    mu: np.ndarray,
    Mach: float,
    gamma: float = 1.4,
) -> tuple[np.ndarray, float]:
    """
    Método de Newton Modificado:
    cp = cp_max * mu^2 en barlovento
    cp = 0 en sotavento
    """
    mu = np.asarray(mu, dtype=float)
    cp = np.zeros_like(mu)

    cp_max = compute_cpmax_modified_newton(Mach=Mach, gamma=gamma)

    mask = mu > 0.0
    cp[mask] = cp_max * mu[mask] ** 2

    return cp, cp_max


def compute_force_coeff_faces(
    cp: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    S_ref: float,
) -> np.ndarray:
    """
    Coeficiente de fuerza por cara:
    CF_i = -(cp_i * A_i / S_ref) * n_i
    """
    if S_ref <= 0.0:
        raise ValueError("S_ref debe ser > 0.")

    cp = np.asarray(cp, dtype=float)
    areas = np.asarray(areas, dtype=float)
    normals = np.asarray(normals, dtype=float)

    return -(cp[:, None] * areas[:, None] * normals) / S_ref


def compute_moment_coeff_faces(
    cp: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
    r_ref: np.ndarray,
    S_ref: float,
    L_ref: float,
) -> np.ndarray:
    """
    Coeficiente de momento por cara:
    CM_i = cross(r_i - r_ref, -(cp_i * A_i * n_i)) / (S_ref * L_ref)
    """
    if S_ref <= 0.0:
        raise ValueError("S_ref debe ser > 0.")
    if L_ref <= 0.0:
        raise ValueError("L_ref debe ser > 0.")

    cp = np.asarray(cp, dtype=float)
    areas = np.asarray(areas, dtype=float)
    normals = np.asarray(normals, dtype=float)
    centers = np.asarray(centers, dtype=float)
    r_ref = np.asarray(r_ref, dtype=float)

    arm = centers - r_ref
    force_like = -(cp[:, None] * areas[:, None] * normals)

    return np.cross(arm, force_like) / (S_ref * L_ref)


def project_global_coefficients(
    CF_total: np.ndarray,
    CM_total: np.ndarray,
    eD: np.ndarray,
    eL: np.ndarray,
    eM: np.ndarray,
) -> dict:
    """
    Proyecta fuerza y momento globales en ejes de drag, lift y momento.
    """
    eD = unit_vector(eD)
    eL = unit_vector(eL)
    eM = unit_vector(eM)

    return {
        "CD": float(np.dot(CF_total, eD)),
        "CL": float(np.dot(CF_total, eL)),
        "CM": float(np.dot(CM_total, eM)),
    }


def solve_modified_newton_case(
    centers: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    alpha_deg: float,
    Mach: float,
    S_ref: float,
    L_ref: float,
    r_ref: np.ndarray,
    eD: np.ndarray,
    eL: np.ndarray,
    eM: np.ndarray,
    gamma: float = 1.4,
) -> dict:
    """
    Resuelve un caso con Método de Newton Modificado (MNM).
    """
    centers = np.asarray(centers, dtype=float)
    areas = np.asarray(areas, dtype=float)
    normals = np.asarray(normals, dtype=float)

    Vinf_hat = flow_direction_from_alpha(alpha_deg)
    mu = compute_mu(normals, Vinf_hat)
    cp, cp_max = compute_cp_modified_newton(mu, Mach=Mach, gamma=gamma)

    CF_faces = compute_force_coeff_faces(cp, areas, normals, S_ref)
    CM_faces = compute_moment_coeff_faces(
        cp, areas, normals, centers, r_ref, S_ref, L_ref
    )

    CF_total = CF_faces.sum(axis=0)
    CM_total = CM_faces.sum(axis=0)

    scalars = project_global_coefficients(CF_total, CM_total, eD, eL, eM)

    return {
        "alpha_deg": float(alpha_deg),
        "Mach": float(Mach),
        "gamma": float(gamma),
        "cp_max": float(cp_max),
        "Vinf_hat": Vinf_hat,
        "mu": mu,
        "cp": cp,
        "CF_faces": CF_faces,
        "CM_faces": CM_faces,
        "CF_total": CF_total,
        "CM_total": CM_total,
        "n_windward": int(np.sum(mu > 0.0)),
        "n_leeward": int(np.sum(mu <= 0.0)),
        **scalars,
    }