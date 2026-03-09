# src/stl_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def load_stl(filepath: str | Path, process: bool = True) -> trimesh.Trimesh:
    """
    Carga un archivo STL y devuelve una malla trimesh.Trimesh.

    Parameters
    ----------
    filepath : str | Path
        Ruta al archivo STL.
    process : bool, optional
        Si True, trimesh intentará limpiar/procesar la geometría al cargarla.

    Returns
    -------
    trimesh.Trimesh
        Malla cargada.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el archivo no contiene una malla triangular válida.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo STL: {filepath}")

    mesh = trimesh.load_mesh(filepath, process=process)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"El archivo no contiene una única malla triangular válida: {filepath}"
        )

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"La malla no contiene caras: {filepath}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"La malla no contiene vértices: {filepath}")

    return mesh


def validate_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Devuelve un diagnóstico básico de la malla.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Malla a validar.

    Returns
    -------
    dict
        Diccionario con información útil sobre la calidad/topología de la malla.
    """
    areas = mesh.area_faces
    invalid_area_faces = int(np.sum(areas <= 0.0))

    info = {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "has_degenerate_faces": bool(invalid_area_faces > 0),
        "n_degenerate_faces": invalid_area_faces,
        "bounds_min": mesh.bounds[0].tolist(),
        "bounds_max": mesh.bounds[1].tolist(),
        "extents": mesh.extents.tolist(),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_volume else None,
    }

    return info


def get_vertices_faces(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve arrays de vértices y caras.

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    vertices : (N, 3) ndarray
    faces : (M, 3) ndarray
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return vertices, faces


def get_face_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Devuelve las coordenadas de los vértices de cada triángulo.

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    face_vertices : (M, 3, 3) ndarray
        Para cada cara:
        - eje 0: índice de cara
        - eje 1: vértice local (0,1,2)
        - eje 2: coordenadas xyz
    """
    vertices, faces = get_vertices_faces(mesh)
    return vertices[faces]


def compute_face_centers(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Calcula el centroide de cada triángulo.

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    centers : (M, 3) ndarray
    """
    face_vertices = get_face_vertices(mesh)
    centers = np.mean(face_vertices, axis=1)
    return centers


def compute_face_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Devuelve el área de cada triángulo.

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    areas : (M,) ndarray
    """
    return np.asarray(mesh.area_faces, dtype=float)


def compute_face_normals(mesh: trimesh.Trimesh, fix_orientation: bool = False) -> np.ndarray:
    """
    Devuelve las normales unitarias por cara.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    fix_orientation : bool, optional
        Si True, intenta reorientar la malla para coherencia en el winding.

    Returns
    -------
    normals : (M, 3) ndarray
        Normales unitarias por cara.
    """
    if fix_orientation:
        mesh = mesh.copy()
        mesh.fix_normals()

    normals = np.array(mesh.face_normals, dtype=float, copy=True)

    # Seguridad extra: renormalización
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 0.0
    normals[valid] /= norms[valid]

    return normals


def compute_face_geometry(mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
    """
    Calcula la geometría elemental por triángulo:
    - centros
    - áreas
    - normales
    - vértices locales

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    dict
        {
            "face_vertices": (M,3,3),
            "centers": (M,3),
            "areas": (M,),
            "normals": (M,3)
        }
    """
    face_vertices = get_face_vertices(mesh)
    centers = compute_face_centers(mesh)
    areas = compute_face_areas(mesh)
    normals = compute_face_normals(mesh)

    return {
        "face_vertices": face_vertices,
        "centers": centers,
        "areas": areas,
        "normals": normals,
    }


def mesh_reference_lengths(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """
    Calcula longitudes características simples a partir del bounding box.

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    dict
        {
            "lx": ...,
            "ly": ...,
            "lz": ...,
            "diag": ...
        }
    """
    extents = np.asarray(mesh.extents, dtype=float)
    diag = float(np.linalg.norm(extents))

    return {
        "lx": float(extents[0]),
        "ly": float(extents[1]),
        "lz": float(extents[2]),
        "diag": diag,
    }


def print_mesh_summary(mesh: trimesh.Trimesh) -> None:
    """
    Imprime un resumen rápido de la malla.
    """
    info = validate_mesh(mesh)
    refs = mesh_reference_lengths(mesh)

    print("\n=== Mesh summary ===")
    print(f"Vertices           : {info['n_vertices']}")
    print(f"Faces              : {info['n_faces']}")
    print(f"Watertight         : {info['is_watertight']}")
    print(f"Winding consistent : {info['is_winding_consistent']}")
    print(f"Degenerate faces   : {info['n_degenerate_faces']}")
    print(f"Bounds min         : {info['bounds_min']}")
    print(f"Bounds max         : {info['bounds_max']}")
    print(f"Extents            : {info['extents']}")
    print(f"Surface area       : {info['surface_area']:.6f}")
    print(f"Volume             : {info['volume']}")
    print(f"Lx, Ly, Lz         : {refs['lx']:.6f}, {refs['ly']:.6f}, {refs['lz']:.6f}")
    print(f"Box diagonal       : {refs['diag']:.6f}")


if __name__ == "__main__":
    # Ejemplo rápido de uso:
    # python src/stl_utils.py
    stl_path = Path("data/esfera.stl")

    mesh = load_stl(stl_path)
    print_mesh_summary(mesh)

    geom = compute_face_geometry(mesh)

    print("\n=== Face geometry ===")
    print(f"Centers shape : {geom['centers'].shape}")
    print(f"Areas shape   : {geom['areas'].shape}")
    print(f"Normals shape : {geom['normals'].shape}")

    print("\nPrimeras 5 áreas:")
    print(geom["areas"][:5])

    print("\nPrimeros 5 centros:")
    print(geom["centers"][:5])

    print("\nPrimeras 5 normales:")
    print(geom["normals"][:5])

def plot_geom(
    geom: dict,
    show_mesh: bool = True,
    show_centers: bool = True,
    show_normals: bool = False,
    normal_scale: float = 10.0,
    color_by: str | None = None,
    alpha: float = 0.7,
    edgecolor: str = "k",
    linewidth: float = 0.3,
    title: str = "Mesh visualization",
):
    """
    Visualiza la geometría a partir del diccionario geom.

    Parameters
    ----------
    geom : dict
        Salida de compute_face_geometry(mesh), con:
        - face_vertices : (M, 3, 3)
        - centers       : (M, 3)
        - areas         : (M,)
        - normals       : (M, 3)
    show_mesh : bool
        Si True, pinta los triángulos.
    show_centers : bool
        Si True, pinta los centroides.
    show_normals : bool
        Si True, pinta las normales.
    normal_scale : float
        Escala visual de las flechas de normales.
    color_by : str | None
        Puede ser:
        - None   -> color uniforme
        - "area" -> colorea por área
    alpha : float
        Transparencia de la malla.
    """

    face_vertices = geom["face_vertices"]
    centers = geom["centers"]
    areas = geom["areas"]
    normals = geom["normals"]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # -------------------------
    # Colores por cara
    # -------------------------
    if color_by is None:
        facecolors = "lightblue"
    elif color_by == "area":
        a_min = np.min(areas)
        a_max = np.max(areas)
        if np.isclose(a_max, a_min):
            values = np.zeros_like(areas)
        else:
            values = (areas - a_min) / (a_max - a_min)
        cmap = plt.cm.viridis
        facecolors = cmap(values)
    else:
        raise ValueError(f"color_by='{color_by}' no soportado")

    # -------------------------
    # Malla
    # -------------------------
    if show_mesh:
        poly = Poly3DCollection(
            face_vertices,
            facecolors=facecolors,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection3d(poly)

    # -------------------------
    # Centros
    # -------------------------
    if show_centers:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            s=8,
            label="Centers",
        )

    # -------------------------
    # Normales
    # -------------------------
    if show_normals:
        ax.quiver(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=normal_scale,
            normalize=True,
        )

    # -------------------------
    # Ajuste de ejes
    # -------------------------
    all_pts = face_vertices.reshape(-1, 3)
    x = all_pts[:, 0]
    y = all_pts[:, 1]
    z = all_pts[:, 2]

    xmid = 0.5 * (x.min() + x.max())
    ymid = 0.5 * (y.min() + y.max())
    zmid = 0.5 * (z.min() + z.max())

    max_range = 0.5 * max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())

    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    ax.set_zlim(zmid - max_range, zmid + max_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    if show_centers:
        ax.legend()

    plt.tight_layout()
    plt.show()  