from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .material import MaterialFields
from .mesh import MeshInfo
from .schema import Outputs, OutputInterval, SimulationInput


def _should_write_interval(interval: OutputInterval, t: float, step: int) -> bool:
    if interval.type == "every_step":
        return True
    if interval.type == "every_n_steps":
        return step % interval.n == 0
    return abs((t / interval.dt) - round(t / interval.dt)) < 1.0e-9


def _write_table(path: Path, header: str, row: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if not exists:
            handle.write(header + "\n")
        handle.write(row + "\n")


def write_node_history(path: Path, t: float, positions: list[tuple[str, int]], temperature_values: np.ndarray, flux_values: np.ndarray) -> None:
    header = ["time_s"]
    row = [f"{t:.12g}"]
    for label, index in positions:
        header.extend([f"T_{label}", f"q_{label}"])
        row.extend([f"{temperature_values[index]:.12g}", f"{flux_values[index]:.12g}"])
    _write_table(path, "\t".join(header), "\t".join(row))


def write_spatial_profile(path: Path, x: np.ndarray, temperature_values: np.ndarray, flux_values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("x_m\tT_K\tq_W_m2\n")
        for xi, ti, qi in zip(x, temperature_values, flux_values):
            handle.write(f"{xi:.12g}\t{ti:.12g}\t{qi:.12g}\n")


def write_energy_balance(path: Path, t: float, q_left: float, q_right: float, stored: float, generated: float) -> None:
    residual = q_left + q_right + generated - stored
    row = f"{t:.12g}\t{q_left:.12g}\t{q_right:.12g}\t{stored:.12g}\t{generated:.12g}\t{residual:.12g}"
    _write_table(path, "time_s\tq_left\tq_right\tdU_dt\tQ_gen\tresidual", row)


def write_tecplot_zone(path: Path, t: float, x: np.ndarray, temperature_values: np.ndarray, flux_values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if path.stat().st_size == 0:
            handle.write('TITLE = "heat1d"\nVARIABLES = "x", "T", "q"\n')
        handle.write(f'ZONE T="t={t:.12g}", I={len(x)}, F=POINT\n')
        for xi, ti, qi in zip(x, temperature_values, flux_values):
            handle.write(f"{xi:.12g} {ti:.12g} {qi:.12g}\n")


@dataclass(slots=True)
class OutputHandler:
    base_dir: Path
    outputs: Outputs
    mesh_info: MeshInfo
    node_positions: list[tuple[str, int]]
    last_temperature: np.ndarray | None = None
    next_profile_dt: float | None = None

    def write(self, t: float, step: int, temperature, fields: MaterialFields, boundary_fluxes: tuple[float, float], dt: float) -> None:
        output_dir = self.base_dir / self.outputs.directory
        x_cells = np.asarray(self.mesh_info.mesh.cellCenters[0], dtype=float)
        temperature_values = np.asarray(temperature.value, dtype=float)
        flux_faces = -np.asarray(fields.k.value, dtype=float) * np.asarray(temperature.faceGrad.value[0], dtype=float)
        flux_cells = 0.5 * (flux_faces[:-1] + flux_faces[1:])

        if self.outputs.node_history.enabled and _should_write_interval(self.outputs.node_history.interval, t, step):
            write_node_history(output_dir / self.outputs.node_history.file, t, self.node_positions, temperature_values, flux_cells)

        if self.outputs.spatial_profiles.enabled and self._should_write_profile(t, step):
            path = output_dir / f"{self.outputs.spatial_profiles.file_prefix}{step:05d}.tsv"
            write_spatial_profile(path, x_cells, temperature_values, flux_cells)

        if self.outputs.energy_balance.enabled and _should_write_interval(self.outputs.energy_balance.interval, t, step):
            generated = float(np.sum(np.asarray(fields.Q.value) * np.asarray(self.mesh_info.mesh.cellVolumes)))
            if self.last_temperature is None:
                stored = 0.0
            else:
                stored = float(np.sum(np.asarray(fields.rho_cp.value) * np.asarray(self.mesh_info.mesh.cellVolumes) * (temperature_values - self.last_temperature) / max(dt, 1.0e-30)))
            write_energy_balance(output_dir / self.outputs.energy_balance.file, t, boundary_fluxes[0], boundary_fluxes[1], stored, generated)

        if self.outputs.tecplot.enabled and _should_write_interval(self.outputs.tecplot.interval, t, step):
            write_tecplot_zone(output_dir / self.outputs.tecplot.file, t, x_cells, temperature_values, flux_cells)

        self.last_temperature = temperature_values.copy()

    def _should_write_profile(self, t: float, step: int) -> bool:
        spec = self.outputs.spatial_profiles
        if spec.at_times is not None:
            return any(abs(t - sample_time) < 1.0e-9 for sample_time in spec.at_times)
        if spec.every_n_steps is not None:
            return step % spec.every_n_steps == 0
        if spec.every_dt is not None:
            if self.next_profile_dt is None:
                self.next_profile_dt = spec.every_dt
            if t + 1.0e-9 >= self.next_profile_dt:
                while self.next_profile_dt <= t + 1.0e-9:
                    self.next_profile_dt += spec.every_dt
                return True
        return False


def make_output_handler(config: SimulationInput, mesh_info: MeshInfo, base_dir: Path) -> OutputHandler:
    positions: list[tuple[str, int]] = []
    mesh_x = np.asarray(mesh_info.mesh.cellCenters[0], dtype=float)
    if config.outputs.node_history.include_boundaries:
        positions.extend([("left_boundary", 0), ("right_boundary", len(mesh_x) - 1)])
    if config.outputs.node_history.include_interfaces:
        for face_index in mesh_info.interface_faces:
            cell_index = max(face_index - 1, 0)
            positions.append((f"interface_{face_index}", cell_index))
    for node in config.outputs.node_history.nodes:
        index = int(np.argmin(np.abs(mesh_x - node.x)))
        positions.append((node.label, index))
    return OutputHandler(base_dir=base_dir, outputs=config.outputs, mesh_info=mesh_info, node_positions=positions)
