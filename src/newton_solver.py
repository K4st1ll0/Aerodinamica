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
    - alpha = 0 deg  -> flujo en -x
    - alpha > 0      -> el flujo se inclina hacia -z
    """
    alpha = np.deg2rad(alpha_deg)
    v = np.array([
        -np.cos(alpha),
        0.0,
        -np.sin(alpha),
    ], dtype=float)
    return unit_vector(v)


def compute_mu(normals: np.ndarray, Vinf_hat: np.ndarray) -> np.ndarray:
    """
    mu = n · s, con s = -Vinf_hat apuntando hacia donde viene el aire.
    """
    s_hat = -unit_vector(Vinf_hat)
    return normals @ s_hat


def compute_cp_newton(mu: np.ndarray) -> np.ndarray:
    """
    Método de Newton:
    cp = 2 * mu^2 en barlovento
    cp = 0 en sotavento
    """
    cp = np.zeros_like(mu)
    mask = mu > 0.0
    cp[mask] = 2.0 * mu[mask] ** 2
    return cp


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

    r_ref = np.asarray(r_ref, dtype=float)
    arm = centers - r_ref
    force_like = -(cp[:, None] * areas[:, None] * normals)

    return np.cross(arm, force_like) / (S_ref * L_ref)


def solve_newton_case(
    centers: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    alpha_deg: float,
    S_ref: float,
    L_ref: float,
    r_ref: np.ndarray,
    eD: np.ndarray,
    eL: np.ndarray,
    eM: np.ndarray,
) -> dict:
    """
    Resuelve un caso con Método de Newton (MN).
    """
    Vinf_hat = flow_direction_from_alpha(alpha_deg)
    mu = compute_mu(normals, Vinf_hat)
    cp = compute_cp_newton(mu)

    CF_faces = compute_force_coeff_faces(cp, areas, normals, S_ref)
    CM_faces = compute_moment_coeff_faces(
        cp, areas, normals, centers, r_ref, S_ref, L_ref
    )

    CF_total = CF_faces.sum(axis=0)
    CM_total = CM_faces.sum(axis=0)

    scalars = project_global_coefficients(CF_total, CM_total, eD, eL, eM)

    return {
        "alpha_deg": float(alpha_deg),
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

def project_global_coefficients(
    CF_total: np.ndarray,
    CM_total: np.ndarray,
    eD: np.ndarray,
    eL: np.ndarray,
    eM: np.ndarray,
) -> dict:
    """
    Proyecta los vectores globales de fuerza y momento
    sobre los ejes definidos de drag, lift y momento.
    """
    eD = unit_vector(eD)
    eL = unit_vector(eL)
    eM = unit_vector(eM)

    return {
        "CD": float(np.dot(CF_total, eD)),
        "CL": float(np.dot(CF_total, eL)),
        "CM": float(np.dot(CM_total, eM)),
    }