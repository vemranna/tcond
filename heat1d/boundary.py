from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fipy
import numpy as np

from .functions import make_scalar_function
from .schema import (
    AdiabaticBC,
    BoundaryCondition,
    BoundaryConditions,
    CombinedBC,
    ConvectionBC,
    FluxBC,
    RadiationBC,
    ScheduledBC,
    TemperatureBC,
)


@dataclass(slots=True)
class BoundaryState:
    value_left: float = 0.0
    value_right: float = 0.0


@dataclass(slots=True)
class BoundaryHandler:
    mesh: fipy.Grid1D
    temperature: fipy.CellVariable
    left_spec: BoundaryCondition
    right_spec: BoundaryCondition
    base_dir: Path
    left_value: fipy.Variable
    right_value: fipy.Variable
    left_flux: fipy.Variable
    right_flux: fipy.Variable
    state: BoundaryState

    def update(self, t: float) -> None:
        self._apply_side("left", self.left_spec, t)
        self._apply_side("right", self.right_spec, t)

    def _apply_side(self, side: str, spec: BoundaryCondition, t: float) -> None:
        temperature = self._boundary_temperature(side)
        fixed_value, fixed_flux = self._evaluate_bc(spec, t, temperature)
        if side == "left":
            self.left_value.setValue(fixed_value)
            self.left_flux.setValue(fixed_flux)
            self.state.value_left = fixed_flux
        else:
            self.right_value.setValue(fixed_value)
            self.right_flux.setValue(fixed_flux)
            self.state.value_right = fixed_flux

    def _boundary_temperature(self, side: str) -> float:
        values = np.asarray(self.temperature.value, dtype=float)
        return float(values[0] if side == "left" else values[-1])

    def _evaluate_bc(self, spec: BoundaryCondition, t: float, wall_temperature: float) -> tuple[float, float]:
        inf = 1.0e300
        if isinstance(spec, AdiabaticBC):
            return inf, 0.0
        if isinstance(spec, TemperatureBC):
            value = float(make_scalar_function(spec.value, self.base_dir)(np.asarray([t]))[0])
            return value, 0.0
        if isinstance(spec, FluxBC):
            flux = float(make_scalar_function(spec.value, self.base_dir)(np.asarray([t]))[0])
            return inf, flux
        if isinstance(spec, ConvectionBC):
            htc = float(make_scalar_function(spec.htc, self.base_dir)(np.asarray([t]))[0])
            t_bulk = float(make_scalar_function(spec.T_bulk, self.base_dir)(np.asarray([t]))[0])
            return inf, htc * (t_bulk - wall_temperature)
        if isinstance(spec, RadiationBC):
            t_env = float(make_scalar_function(spec.T_environment, self.base_dir)(np.asarray([t]))[0])
            flux = spec.emissivity * spec.stefan_boltzmann * (t_env**4 - wall_temperature**4)
            return inf, flux
        if isinstance(spec, CombinedBC):
            flux_total = 0.0
            for component in spec.components:
                _, component_flux = self._evaluate_bc(component, t, wall_temperature)
                flux_total += component_flux
            return inf, flux_total
        if isinstance(spec, ScheduledBC):
            for entry in spec.schedule:
                if entry.until == "end" or t <= entry.until:
                    return self._evaluate_bc(entry.bc, t, wall_temperature)
        raise TypeError(f"Unsupported boundary condition: {type(spec)!r}")


def make_boundary_conditions(
    mesh: fipy.Grid1D,
    bc_config: BoundaryConditions,
    temperature: fipy.CellVariable,
    base_dir: Path,
) -> tuple[BoundaryHandler, list[fipy.Variable], list[fipy.Term]]:
    left_value = fipy.Variable(value=1.0e300)
    right_value = fipy.Variable(value=1.0e300)
    left_flux = fipy.Variable(value=0.0)
    right_flux = fipy.Variable(value=0.0)

    temperature.constrain(left_value, mesh.facesLeft)
    temperature.constrain(right_value, mesh.facesRight)

    left_term = (mesh.facesLeft * left_flux).divergence
    right_term = -(mesh.facesRight * right_flux).divergence
    handler = BoundaryHandler(
        mesh=mesh,
        temperature=temperature,
        left_spec=bc_config.left,
        right_spec=bc_config.right,
        base_dir=base_dir,
        left_value=left_value,
        right_value=right_value,
        left_flux=left_flux,
        right_flux=right_flux,
        state=BoundaryState(),
    )
    return handler, [left_flux, right_flux], [left_term, right_term]
