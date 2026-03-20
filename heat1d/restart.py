from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def save_checkpoint(path: Path, t: float, step: int, dt: float, temperature, config_path: Path, mesh_dx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["t"] = t
        handle.attrs["step"] = step
        handle.attrs["dt"] = dt
        handle.attrs["config_path"] = str(config_path)
        handle.create_dataset("T_values", data=np.asarray(temperature.value, dtype=float))
        handle.create_dataset("mesh_dx", data=np.asarray(mesh_dx, dtype=float))


def load_checkpoint(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "t": float(handle.attrs["t"]),
            "step": int(handle.attrs["step"]),
            "dt": float(handle.attrs["dt"]),
            "config_path": Path(handle.attrs["config_path"]),
            "T_values": np.asarray(handle["T_values"], dtype=float),
            "mesh_dx": np.asarray(handle["mesh_dx"], dtype=float),
        }
