"""
Pydantic v2 models for the 1D heat conduction input file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    """Forbid extra keys in all models — catches typos in the input file."""

    model_config = ConfigDict(extra="forbid")


class ConstantFunction(_StrictModel):
    type: Literal["constant"]
    value: float


class PolynomialFunction(_StrictModel):
    type: Literal["polynomial"]
    coefficients: list[float] = Field(min_length=1)


class PiecewiseFunction(_StrictModel):
    type: Literal["piecewise"]
    file: Path
    x_col: int | str = 1
    y_col: int | str = 2
    interpolation: Literal["linear", "cubic", "nearest"] = "linear"
    extrapolation: Literal["clamp", "error", "warn+clamp"] = "clamp"

    @field_validator("x_col", "y_col", mode="before")
    @classmethod
    def _col_positive(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 1:
            raise ValueError("Column indices are 1-based — must be >= 1")
        return value


ScalarFunction = Annotated[
    Union[ConstantFunction, PolynomialFunction, PiecewiseFunction],
    Field(discriminator="type"),
]


class Layer(_StrictModel):
    name: str
    material: str
    thickness: float | None = Field(default=None, gt=0)
    nodes: int | None = Field(default=None, ge=2)
    dx_list: list[float] | None = None
    grading: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _mesh_spec(self) -> "Layer":
        has_uniform = self.thickness is not None and self.nodes is not None
        has_explicit = self.dx_list is not None
        if not has_uniform and not has_explicit:
            raise ValueError(
                f"Layer '{self.name}': specify either (thickness + nodes) or dx_list"
            )
        if has_uniform and has_explicit:
            raise ValueError(
                f"Layer '{self.name}': specify (thickness + nodes) OR dx_list, not both"
            )
        if has_explicit and self.grading is not None:
            raise ValueError(
                f"Layer '{self.name}': grading is only valid with thickness+nodes, not dx_list"
            )
        if self.dx_list is not None:
            if len(self.dx_list) < 2:
                raise ValueError(f"Layer '{self.name}': dx_list must contain at least 2 cells")
            if any(dx <= 0 for dx in self.dx_list):
                raise ValueError(f"Layer '{self.name}': dx_list values must be > 0")
            if self.thickness is not None and abs(sum(self.dx_list) - self.thickness) > 1.0e-12:
                raise ValueError(f"Layer '{self.name}': dx_list must sum to thickness")
        return self


class Interface(_StrictModel):
    between: list[str] = Field(min_length=2, max_length=2)
    contact_resistance: float | None = Field(default=None, ge=0)
    contact_conductance: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _one_or_neither(self) -> "Interface":
        if self.contact_resistance is not None and self.contact_conductance is not None:
            raise ValueError("Specify contact_resistance OR contact_conductance, not both")
        return self

    @property
    def resistance(self) -> float:
        if self.contact_resistance is not None:
            return self.contact_resistance
        if self.contact_conductance is not None:
            return 1.0 / self.contact_conductance
        return 0.0


class Geometry(_StrictModel):
    layers: list[Layer] = Field(min_length=1)
    interfaces: list[Interface] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_layer_names(self) -> "Geometry":
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("Layer names must be unique")
        return self

    @model_validator(mode="after")
    def _interface_names_valid(self) -> "Geometry":
        layer_names = {layer.name for layer in self.layers}
        for iface in self.interfaces:
            for name in iface.between:
                if name not in layer_names:
                    raise ValueError(f"Interface references unknown layer '{name}'")
        return self


class MaterialProperties(_StrictModel):
    density: ScalarFunction
    specific_heat: ScalarFunction
    thermal_conductivity: ScalarFunction
    volumetric_heat_generation: ScalarFunction = ConstantFunction(type="constant", value=0.0)


class UniformIC(_StrictModel):
    type: Literal["uniform"]
    value: float


class PolynomialIC(_StrictModel):
    type: Literal["polynomial"]
    coefficients: list[float] = Field(min_length=1)


class PiecewiseIC(_StrictModel):
    type: Literal["piecewise"]
    file: Path
    x_col: int | str = 1
    y_col: int | str = 2
    interpolation: Literal["linear", "cubic", "nearest"] = "linear"


class RestartIC(_StrictModel):
    type: Literal["restart"]
    file: Path


InitialCondition = Annotated[
    Union[UniformIC, PolynomialIC, PiecewiseIC, RestartIC],
    Field(discriminator="type"),
]


class TemperatureBC(_StrictModel):
    type: Literal["temperature"]
    value: ScalarFunction


class FluxBC(_StrictModel):
    type: Literal["flux"]
    value: ScalarFunction


class ConvectionBC(_StrictModel):
    type: Literal["convection"]
    htc: ScalarFunction
    T_bulk: ScalarFunction


class RadiationBC(_StrictModel):
    type: Literal["radiation"]
    emissivity: float = Field(gt=0, le=1)
    T_environment: ScalarFunction
    stefan_boltzmann: float = 5.67e-8


class AdiabaticBC(_StrictModel):
    type: Literal["adiabatic"]


_CombinableBC = Union[FluxBC, ConvectionBC, RadiationBC]


class CombinedBC(_StrictModel):
    type: Literal["combined"]
    components: list[_CombinableBC] = Field(min_length=1)


class ScheduleEntry(_StrictModel):
    until: float | Literal["end"]
    bc: Annotated[
        Union[TemperatureBC, FluxBC, ConvectionBC, RadiationBC, AdiabaticBC, CombinedBC],
        Field(discriminator="type"),
    ]


class ScheduledBC(_StrictModel):
    type: Literal["scheduled"]
    schedule: list[ScheduleEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _last_entry_is_end(self) -> "ScheduledBC":
        if self.schedule[-1].until != "end":
            raise ValueError("Last schedule entry must have until: end")
        return self


BoundaryCondition = Annotated[
    Union[
        TemperatureBC,
        FluxBC,
        ConvectionBC,
        RadiationBC,
        AdiabaticBC,
        CombinedBC,
        ScheduledBC,
    ],
    Field(discriminator="type"),
]


class BoundaryConditions(_StrictModel):
    left: BoundaryCondition
    right: BoundaryCondition


class AdaptiveStepping(_StrictModel):
    enabled: bool = True
    target_dT_per_step: float = Field(default=5.0, gt=0)
    safety_factor: float = Field(default=0.9, gt=0, le=1)


class TimeControl(_StrictModel):
    total: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    dt_initial: float = Field(gt=0)
    dt_min: float = Field(default=1.0e-9, gt=0)
    dt_max: float = Field(default=1.0, gt=0)
    adaptive: AdaptiveStepping = AdaptiveStepping()

    @model_validator(mode="after")
    def _termination_specified(self) -> "TimeControl":
        if self.total is None and self.max_steps is None:
            raise ValueError("Specify either time.total or time.max_steps")
        if self.total is not None and self.max_steps is not None:
            raise ValueError("Specify only one of time.total or time.max_steps")
        if self.dt_min > self.dt_initial:
            raise ValueError("dt_min must be <= dt_initial")
        if self.dt_initial > self.dt_max:
            raise ValueError("dt_initial must be <= dt_max")
        return self


class NonlinearControl(_StrictModel):
    max_iterations: int = Field(default=50, ge=1)
    tolerance: float = Field(default=1.0e-6, gt=0)
    tolerance_norm: Literal["L1", "L2", "Linf"] = "L2"
    tolerance_type: Literal["relative", "absolute"] = "relative"
    under_relaxation: float = Field(default=0.7, gt=0, le=1)


class SolverControl(_StrictModel):
    type: Literal["LinearLUSolver", "LinearPCGSolver", "LinearGMRESSolver"] = "LinearLUSolver"
    tolerance: float = Field(default=1.0e-10, gt=0)
    preconditioner: str | None = None


class SolutionControls(_StrictModel):
    time: TimeControl
    nonlinear: NonlinearControl = NonlinearControl()
    solver: SolverControl = SolverControl()


class NodeSpec(_StrictModel):
    label: str
    x: float = Field(ge=0)


class OutputInterval(_StrictModel):
    type: Literal["every_step", "every_n_steps", "every_dt"]
    n: int | None = Field(default=None, ge=1)
    dt: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _params_consistent(self) -> "OutputInterval":
        if self.type == "every_n_steps" and self.n is None:
            raise ValueError("every_n_steps requires n")
        if self.type == "every_dt" and self.dt is None:
            raise ValueError("every_dt requires dt")
        return self


class NodeHistoryOutput(_StrictModel):
    enabled: bool = True
    file: str = "node_history.tsv"
    nodes: list[NodeSpec] = Field(default_factory=list)
    include_interfaces: bool = True
    include_boundaries: bool = True
    interval: OutputInterval = OutputInterval(type="every_step")


class SpatialProfileOutput(_StrictModel):
    enabled: bool = True
    file_prefix: str = "profile_"
    at_times: list[float] | None = None
    every_n_steps: int | None = Field(default=None, ge=1)
    every_dt: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _trigger_specified(self) -> "SpatialProfileOutput":
        triggers = [
            bool(self.at_times),
            self.every_n_steps is not None,
            self.every_dt is not None,
        ]
        if sum(triggers) != 1:
            raise ValueError(
                "spatial_profiles: specify exactly one of at_times, every_n_steps, every_dt"
            )
        return self


class EnergyBalanceOutput(_StrictModel):
    enabled: bool = True
    file: str = "energy_balance.tsv"
    interval: OutputInterval = OutputInterval(type="every_step")


class TecplotOutput(_StrictModel):
    enabled: bool = False
    file: str = "solution.dat"
    format: Literal["ascii", "binary"] = "ascii"
    interval: OutputInterval = OutputInterval(type="every_n_steps", n=10)


class CheckpointOutput(_StrictModel):
    enabled: bool = True
    directory: str = "restart/"
    interval: OutputInterval = OutputInterval(type="every_n_steps", n=100)
    keep_last_n: int = Field(default=3, ge=1)


class Outputs(_StrictModel):
    directory: str = "results/"
    node_history: NodeHistoryOutput = NodeHistoryOutput()
    spatial_profiles: SpatialProfileOutput = SpatialProfileOutput(at_times=[0.0])
    energy_balance: EnergyBalanceOutput = EnergyBalanceOutput()
    tecplot: TecplotOutput = TecplotOutput()
    checkpoint: CheckpointOutput = CheckpointOutput()


class Metadata(_StrictModel):
    model_config = ConfigDict(extra="allow")
    title: str = ""
    author: str = ""
    date: str = ""
    description: str = ""


class SimulationInput(_StrictModel):
    metadata: Metadata = Metadata()
    geometry: Geometry
    materials: dict[str, MaterialProperties]
    initial_conditions: InitialCondition
    boundary_conditions: BoundaryConditions
    solution: SolutionControls
    outputs: Outputs = Outputs()

    @model_validator(mode="after")
    def _materials_cover_all_layers(self) -> "SimulationInput":
        for layer in self.geometry.layers:
            if layer.material not in self.materials:
                raise ValueError(
                    f"Layer '{layer.name}' references material '{layer.material}' "
                    "which is not defined in materials"
                )
        return self
