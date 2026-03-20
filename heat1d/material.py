from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fipy
import numpy as np

from .functions import make_scalar_function
from .mesh import MeshInfo
from .schema import Geometry, MaterialProperties, SimulationInput


@dataclass(slots=True)
class LayerMaterialEvaluator:
    density: callable
    specific_heat: callable
    conductivity: callable
    heat_generation: callable


@dataclass(slots=True)
class MaterialFields:
    rho_cp: fipy.CellVariable
    k_cell: fipy.CellVariable
    k: fipy.FaceVariable
    Q: fipy.CellVariable
    layer_id: np.ndarray
    dx: np.ndarray
    evaluators: list[LayerMaterialEvaluator]
    interface_resistances: dict[int, float]

    def update(self, temperature: fipy.CellVariable) -> None:
        t_values = np.asarray(temperature.value, dtype=float)
        rho_cp_values = np.zeros_like(t_values)
        k_cell_values = np.zeros_like(t_values)
        q_values = np.zeros_like(t_values)

        for layer_index, evaluator in enumerate(self.evaluators):
            mask = self.layer_id == layer_index
            if not np.any(mask):
                continue
            layer_temperatures = t_values[mask]
            density = np.asarray(evaluator.density(layer_temperatures), dtype=float)
            specific_heat = np.asarray(evaluator.specific_heat(layer_temperatures), dtype=float)
            conductivity = np.asarray(evaluator.conductivity(layer_temperatures), dtype=float)
            q_values[mask] = np.asarray(evaluator.heat_generation(layer_temperatures), dtype=float)
            rho_cp_values[mask] = density * specific_heat
            k_cell_values[mask] = conductivity

        self.rho_cp.setValue(rho_cp_values)
        self.k_cell.setValue(k_cell_values)
        self.Q.setValue(q_values)
        self.k.setValue(self._face_conductivity(k_cell_values))

    def _face_conductivity(self, k_cell_values: np.ndarray) -> np.ndarray:
        face_values = np.zeros(self.k.globalValue.shape, dtype=float)
        if k_cell_values.size == 1:
            face_values[:] = k_cell_values[0]
            return face_values

        face_values[0] = k_cell_values[0]
        face_values[-1] = k_cell_values[-1]
        left = k_cell_values[:-1]
        right = k_cell_values[1:]
        harmonic = 2.0 * left * right / np.maximum(left + right, 1.0e-30)
        face_values[1:-1] = harmonic

        for face_index, resistance in self.interface_resistances.items():
            left_index = face_index - 1
            right_index = face_index
            dx_left = 0.5 * self.dx[left_index]
            dx_right = 0.5 * self.dx[right_index]
            dx_harmonic = 2.0 * dx_left * dx_right / (dx_left + dx_right)
            k_harmonic = face_values[face_index]
            face_values[face_index] = dx_harmonic / ((dx_harmonic / max(k_harmonic, 1.0e-30)) + resistance)

        return face_values


def make_material_fields(
    config: SimulationInput,
    mesh_info: MeshInfo,
    base_dir: Path,
) -> MaterialFields:
    mesh = mesh_info.mesh
    evaluators: list[LayerMaterialEvaluator] = []
    for layer in config.geometry.layers:
        material = config.materials[layer.material]
        evaluators.append(
            LayerMaterialEvaluator(
                density=make_scalar_function(material.density, base_dir),
                specific_heat=make_scalar_function(material.specific_heat, base_dir),
                conductivity=make_scalar_function(material.thermal_conductivity, base_dir),
                heat_generation=make_scalar_function(material.volumetric_heat_generation, base_dir),
            )
        )

    interface_resistances: dict[int, float] = {}
    iface_lookup = {(iface.between[0], iface.between[1]): iface.resistance for iface in config.geometry.interfaces}
    for pair, face_index in mesh_info.interface_map.items():
        interface_resistances[face_index] = iface_lookup.get(pair, 0.0)

    rho_cp = fipy.CellVariable(mesh=mesh, name="rho_cp", value=1.0)
    k_cell = fipy.CellVariable(mesh=mesh, name="k_cell", value=1.0)
    k_face = fipy.FaceVariable(mesh=mesh, name="k", value=1.0)
    heat_generation = fipy.CellVariable(mesh=mesh, name="Q", value=0.0)

    fields = MaterialFields(
        rho_cp=rho_cp,
        k_cell=k_cell,
        k=k_face,
        Q=heat_generation,
        layer_id=mesh_info.layer_id,
        dx=mesh_info.dx,
        evaluators=evaluators,
        interface_resistances=interface_resistances,
    )
    return fields
