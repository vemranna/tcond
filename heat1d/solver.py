from __future__ import annotations

from pathlib import Path

import fipy
import numpy as np

from .boundary import make_boundary_conditions
from .functions import make_scalar_function
from .loader import load
from .material import make_material_fields
from .mesh import build_mesh
from .output import make_output_handler
from .restart import load_checkpoint, save_checkpoint
from .schema import PiecewiseIC, PolynomialIC, RestartIC, SimulationInput, UniformIC


def _make_initial_temperature(config: SimulationInput, mesh_x: np.ndarray, base_dir: Path) -> np.ndarray:
    spec = config.initial_conditions
    if isinstance(spec, UniformIC):
        return np.full(mesh_x.shape, spec.value, dtype=float)
    if isinstance(spec, PolynomialIC):
        polynomial = np.polynomial.Polynomial(spec.coefficients)
        return np.asarray(polynomial(mesh_x), dtype=float)
    if isinstance(spec, PiecewiseIC):
        function = make_scalar_function(spec, base_dir)
        return np.asarray(function(mesh_x), dtype=float)
    if isinstance(spec, RestartIC):
        checkpoint = load_checkpoint(spec.file if spec.file.is_absolute() else base_dir / spec.file)
        return np.asarray(checkpoint["T_values"], dtype=float)
    raise TypeError(f"Unsupported initial condition: {type(spec)!r}")


def _norm(values: np.ndarray, norm_type: str) -> float:
    if norm_type == "L1":
        return float(np.linalg.norm(values, ord=1))
    if norm_type == "Linf":
        return float(np.linalg.norm(values, ord=np.inf))
    return float(np.linalg.norm(values, ord=2))


def _make_solver(control) -> fipy.solvers.solver.Solver | None:
    if control.type == "LinearLUSolver":
        return fipy.LinearLUSolver(tolerance=control.tolerance)
    if control.type == "LinearPCGSolver":
        return fipy.LinearPCGSolver(tolerance=control.tolerance)
    if control.type == "LinearGMRESSolver":
        return fipy.LinearGMRESSolver(tolerance=control.tolerance)
    return None


def run(config: SimulationInput, config_path: Path | None = None) -> dict[str, float]:
    base_dir = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
    mesh_info = build_mesh(config.geometry)
    fields = make_material_fields(config, mesh_info, base_dir)
    mesh = mesh_info.mesh
    mesh_x = np.asarray(mesh.cellCenters[0], dtype=float)

    initial_temperature = _make_initial_temperature(config, mesh_x, base_dir)
    if initial_temperature.shape[0] != mesh.numberOfCells:
        raise ValueError("Initial temperature field length does not match mesh")

    temperature = fipy.CellVariable(mesh=mesh, name="T", value=initial_temperature, hasOld=True)
    fields.update(temperature)
    boundary_handler, boundary_flux_variables, boundary_terms = make_boundary_conditions(
        mesh,
        config.boundary_conditions,
        temperature,
        base_dir,
    )
    output_handler = make_output_handler(config, mesh_info, base_dir)

    equation = (
        fipy.TransientTerm(coeff=fields.rho_cp)
        == fipy.DiffusionTerm(coeff=fields.k) + fields.Q + sum(boundary_terms)
    )
    linear_solver = _make_solver(config.solution.solver)

    t = 0.0
    step = 0
    dt = config.solution.time.dt_initial
    restart_path = None
    if isinstance(config.initial_conditions, RestartIC):
        restart_path = config.initial_conditions.file if config.initial_conditions.file.is_absolute() else base_dir / config.initial_conditions.file
        checkpoint = load_checkpoint(restart_path)
        if not np.allclose(checkpoint["mesh_dx"], mesh_info.dx):
            raise ValueError("Restart checkpoint mesh does not match current mesh")
        t = checkpoint["t"]
        step = checkpoint["step"]
        dt = checkpoint["dt"]
        temperature.setValue(checkpoint["T_values"])

    total_time = config.solution.time.total
    max_steps = config.solution.time.max_steps

    while True:
        if total_time is not None and t >= total_time - 1.0e-12:
            break
        if max_steps is not None and step >= max_steps:
            break

        temperature.updateOld()
        step_start = np.asarray(temperature.value, dtype=float).copy()
        trial_dt = dt
        if total_time is not None:
            trial_dt = min(trial_dt, max(total_time - t, 0.0))
        if trial_dt <= 0.0:
            break

        previous_iterate = np.asarray(temperature.value, dtype=float).copy()
        for _ in range(config.solution.nonlinear.max_iterations):
            fields.update(temperature)
            boundary_handler.update(t + trial_dt)
            equation.solve(var=temperature, dt=trial_dt, solver=linear_solver)
            current = np.asarray(temperature.value, dtype=float).copy()
            delta = current - previous_iterate
            if config.solution.nonlinear.tolerance_type == "relative":
                denominator = max(_norm(previous_iterate, config.solution.nonlinear.tolerance_norm), 1.0e-30)
                residual = _norm(delta, config.solution.nonlinear.tolerance_norm) / denominator
            else:
                residual = _norm(delta, config.solution.nonlinear.tolerance_norm)
            if residual < config.solution.nonlinear.tolerance:
                break
            relaxed = (
                config.solution.nonlinear.under_relaxation * current
                + (1.0 - config.solution.nonlinear.under_relaxation) * previous_iterate
            )
            temperature.setValue(relaxed)
            previous_iterate = relaxed.copy()

        t += trial_dt
        step += 1
        output_handler.write(
            t,
            step,
            temperature,
            fields,
            (float(boundary_flux_variables[0].value), float(boundary_flux_variables[1].value)),
            trial_dt,
        )

        if config.solution.time.adaptive.enabled:
            dT_max = float(np.max(np.abs(np.asarray(temperature.value) - step_start)))
            if dT_max > 0.0:
                dt = np.clip(
                    trial_dt * config.solution.time.adaptive.target_dT_per_step / dT_max * config.solution.time.adaptive.safety_factor,
                    config.solution.time.dt_min,
                    config.solution.time.dt_max,
                )
            else:
                dt = min(config.solution.time.dt_max, trial_dt * 2.0)
        else:
            dt = trial_dt

        if config.outputs.checkpoint.enabled and step % config.outputs.checkpoint.interval.n == 0:
            checkpoint_dir = base_dir / config.outputs.checkpoint.directory
            checkpoint_path = checkpoint_dir / f"checkpoint_{step:05d}.hdf5"
            save_checkpoint(checkpoint_path, t, step, dt, temperature, Path(config_path or "input.yaml"), mesh_info.dx)
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.hdf5"))
            for old_path in checkpoints[:-config.outputs.checkpoint.keep_last_n]:
                old_path.unlink()

    return {"time": t, "step": step, "dt": dt, "restart_path": str(restart_path) if restart_path else ""}


def run_from_file(config_path: Path) -> dict[str, float]:
    config = load(config_path)
    return run(config, config_path=config_path)
