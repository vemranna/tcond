from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

from .schema import ConstantFunction, PiecewiseFunction, PolynomialFunction, ScalarFunction


def _as_array(values: np.ndarray | float | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _load_tsv_columns(path: Path, x_col: int | str, y_col: int | str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, comments="#", names=True, dtype=float)
    if data.dtype.names is None:
        matrix = np.genfromtxt(path, comments="#", dtype=float)
        if matrix.ndim == 1:
            matrix = np.atleast_2d(matrix)
        x_values = matrix[:, int(x_col) - 1 if isinstance(x_col, int) else 0]
        y_values = matrix[:, int(y_col) - 1 if isinstance(y_col, int) else 1]
        return _as_array(x_values), _as_array(y_values)

    def get_column(selector: int | str) -> np.ndarray:
        if isinstance(selector, int):
            name = data.dtype.names[selector - 1]
        else:
            name = selector
        return _as_array(data[name])

    return get_column(x_col), get_column(y_col)


def make_scalar_function(spec: ScalarFunction, base_dir: Path | None = None) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(spec, ConstantFunction):
        return lambda x: np.full_like(_as_array(x), spec.value, dtype=float)

    if isinstance(spec, PolynomialFunction):
        polynomial = np.polynomial.Polynomial(spec.coefficients)
        return lambda x: _as_array(polynomial(_as_array(x)))

    if isinstance(spec, PiecewiseFunction):
        path = spec.file if spec.file.is_absolute() else (base_dir or Path.cwd()) / spec.file
        x_values, y_values = _load_tsv_columns(path, spec.x_col, spec.y_col)
        interpolator = interp1d(
            x_values,
            y_values,
            kind=spec.interpolation,
            bounds_error=False,
            fill_value=(y_values[0], y_values[-1]),
            assume_sorted=False,
        )
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))

        def evaluate(x: np.ndarray) -> np.ndarray:
            points = _as_array(x)
            if spec.extrapolation == "error" and (np.any(points < x_min) or np.any(points > x_max)):
                raise ValueError(f"Input out of interpolation bounds for '{path}'")
            if spec.extrapolation == "warn+clamp" and (np.any(points < x_min) or np.any(points > x_max)):
                warnings.warn(f"Clamping extrapolated values for '{path}'", stacklevel=2)
            if spec.extrapolation in {"clamp", "warn+clamp"}:
                points = np.clip(points, x_min, x_max)
            return _as_array(interpolator(points))

        return evaluate

    raise TypeError(f"Unsupported scalar function spec: {type(spec)!r}")
