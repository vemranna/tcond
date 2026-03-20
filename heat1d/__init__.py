"""1D transient heat conduction solver package."""

from .loader import load
from .solver import run
from .schema import SimulationInput

__all__ = ["SimulationInput", "load", "run"]
