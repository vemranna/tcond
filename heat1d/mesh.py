from __future__ import annotations

from dataclasses import dataclass

import fipy
import numpy as np

from .schema import Geometry


@dataclass(slots=True)
class MeshInfo:
    mesh: fipy.Grid1D
    layer_id: np.ndarray
    interface_faces: list[int]
    interface_map: dict[tuple[str, str], int]
    dx: np.ndarray


def _graded_dx(thickness: float, nodes: int, grading: float) -> np.ndarray:
    if abs(grading - 1.0) < 1.0e-12:
        return np.full(nodes, thickness / nodes, dtype=float)
    dx0 = thickness * (grading - 1.0) / (grading**nodes - 1.0)
    return dx0 * grading ** np.arange(nodes, dtype=float)


def build_mesh(geometry: Geometry) -> MeshInfo:
    dx_segments: list[np.ndarray] = []
    layer_id_segments: list[np.ndarray] = []
    interface_faces: list[int] = []
    interface_map: dict[tuple[str, str], int] = {}

    cell_offset = 0
    for index, layer in enumerate(geometry.layers):
        if layer.dx_list is not None:
            dx = np.asarray(layer.dx_list, dtype=float)
        elif layer.grading is not None:
            dx = _graded_dx(layer.thickness, layer.nodes, layer.grading)
        else:
            dx = np.full(layer.nodes, layer.thickness / layer.nodes, dtype=float)

        dx_segments.append(dx)
        layer_id_segments.append(np.full(dx.size, index, dtype=int))
        cell_offset += dx.size
        if index < len(geometry.layers) - 1:
            interface_faces.append(cell_offset)
            interface_map[(layer.name, geometry.layers[index + 1].name)] = cell_offset

    dx_full = np.concatenate(dx_segments)
    mesh = fipy.Grid1D(dx=dx_full)
    layer_ids = np.concatenate(layer_id_segments)
    return MeshInfo(
        mesh=mesh,
        layer_id=layer_ids,
        interface_faces=interface_faces,
        interface_map=interface_map,
        dx=dx_full,
    )
