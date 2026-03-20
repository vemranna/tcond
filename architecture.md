# Architecture Document
## 1D Transient Heat Conduction Solver — FiPy-based

---

### 1. Overview

The solver reads a YAML input file, builds a FiPy mesh and equation, integrates
forward in time using a fully implicit scheme, and writes requested outputs.
The design is intentionally flat: a small number of plain Python modules,
minimal class hierarchy, and no framework beyond FiPy and Pydantic.

---

### 2. Module Layout

```
heat1d/
├── main.py            # entry point — parse args, call run()
├── schema.py          # Pydantic input models (the spec document)
├── loader.py          # YAML → validated SimulationInput
├── functions.py       # ScalarFunction → callable f(x)
├── mesh.py            # build FiPy mesh from geometry config
├── material.py        # assemble cell-wise ρ, Cp, k, Q fields
├── solver.py          # time-stepping loop
├── boundary.py        # apply / update boundary conditions
├── output.py          # write TSV, Tecplot, HDF5 checkpoint
└── restart.py         # save and load HDF5 checkpoints
```

Each module has one clear responsibility. There are no circular imports.

---

### 3. Module Responsibilities

#### 3.1  `loader.py`

```python
def load(yaml_path: Path) -> SimulationInput
```

- Reads YAML with `ruamel.yaml` (preserves comments, useful for restart metadata)
- Resolves all `file:` paths relative to the YAML file's directory
- Calls `SimulationInput.model_validate(raw_dict)`
- Raises on validation errors with Pydantic's built-in messages

Nothing else. No physics here.

---

#### 3.2  `functions.py`

The central utility. Every material property, BC parameter, and IC that is
a `ScalarFunction` is converted into a plain Python callable.

```python
def make_scalar_function(spec: ScalarFunction, base_dir: Path) -> Callable[[float], float]
```

- `ConstantFunction`  → `lambda x: spec.value`
- `PolynomialFunction` → `numpy.polynomial.Polynomial(spec.coefficients)`
- `PiecewiseFunction`  → loads TSV, selects columns, builds `scipy.interpolate.interp1d`

Column selection from TSV (1-based index or name):

```python
def _load_tsv_columns(path, x_col, y_col) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=...)
    # if x_col/y_col are int: use iloc[:, col-1]
    # if str: use df[col_name]
```

All functions returned by `make_scalar_function` have the same signature
`f(scalar: float) -> float`. The caller is responsible for passing the right
physical quantity (T, t, or x).

**Extrapolation** is handled inside the returned callable by a thin wrapper
that clamps/warns/raises before calling `interp1d`.

---

#### 3.3  `mesh.py`

```python
def build_mesh(geometry: Geometry) -> fipy.CellVariable  # returns mesh + layer map
```

Returns two things:
- `mesh`: a `fipy.Grid1D` constructed from the concatenated `dx` array of all layers
- `layer_id`: a numpy integer array of length `n_cells`, value = layer index

Building the `dx` array:

```python
dx_segments = []
for layer in geometry.layers:
    if layer.dx_list:
        dx_segments.append(np.array(layer.dx_list))
    else:
        dx = layer.thickness / layer.nodes
        if layer.grading:
            # geometric series: dx[i] = dx0 * grading^i, sum = thickness
            dx0 = thickness * (grading - 1) / (grading**nodes - 1)
            dx_segments.append(dx0 * grading**np.arange(nodes))
        else:
            dx_segments.append(np.full(layer.nodes, dx))

mesh = fipy.Grid1D(dx=np.concatenate(dx_segments))
```

Interface positions (face indices) are recorded for output and contact
resistance handling.

---

#### 3.4  `material.py`

```python
def make_material_fields(mesh, layer_id, materials_config, geometry) -> MaterialFields
```

Returns a `MaterialFields` dataclass:
```python
@dataclass
class MaterialFields:
    rho_cp: fipy.CellVariable    # ρ·Cp — updated each nonlinear iteration
    k:      fipy.FaceVariable    # thermal conductivity — updated each iteration
    Q:      fipy.CellVariable    # volumetric heat generation
```

At each nonlinear iteration, `update(T)` re-evaluates all property functions
at current cell temperatures and writes the values into the FiPy variables.

**Interface conductivity** between cells of different layers uses harmonic mean
of the two adjacent cell conductivities — FiPy's default face interpolation.
For non-zero contact resistance, the interface face conductivity is overridden:

```
k_face_interface = dx_harmonic_mean / (dx_harmonic_mean/k_harmonic + R_contact)
```

where `dx_harmonic_mean` is the harmonic mean of the two adjacent cell
half-widths. This ensures the interface resistance is correctly folded into
the diffusion coefficient seen by FiPy.

---

#### 3.5  `boundary.py`

```python
def make_boundary_conditions(mesh, bc_config, T_var) -> BoundaryHandler
```

`BoundaryHandler` holds two `fipy.FixedValue` / `fipy.FixedFlux` constraints
(or equivalent) and exposes:

```python
def update(t: float, T_boundary: float)
```

which re-evaluates all time- and temperature-dependent BC parameters and
applies them to the FiPy equation terms.

BC implementation in FiPy terms:

| Type        | FiPy implementation                                  |
|-------------|------------------------------------------------------|
| temperature | `FixedValue` constraint on boundary face             |
| adiabatic   | natural BC (do nothing — FiPy default is zero flux)  |
| flux        | `FixedFlux` or explicit source term on boundary cell |
| convection  | explicit source term: `coeff*(T_bulk - T_face)`      |
| radiation   | explicit source term: `ε·σ·(T_env⁴ - T_face⁴)`      |
| combined    | sum of individual source terms                       |
| scheduled   | delegates to sub-BC based on current time            |

Convection and radiation are nonlinear and are treated as explicit source
terms on the boundary cell, re-evaluated at the start of each nonlinear
iteration. This avoids modifying FiPy's internal matrix structure.

---

#### 3.6  `solver.py`

The time-stepping loop. This is the core of the solver.

```python
def run(config: SimulationInput):
    mesh, layer_id        = build_mesh(config.geometry)
    fields                = make_material_fields(...)
    T                     = fipy.CellVariable(mesh, value=initial_T)
    boundary_handler      = make_boundary_conditions(...)
    output_handler        = make_output_handler(...)

    t, step, dt = 0.0, 0, config.solution.time.dt_initial

    while not done(t, step, config):

        # --- nonlinear (Picard) iteration ---
        T_old = T.value.copy()
        for inner in range(max_inner):
            fields.update(T)
            boundary_handler.update(t + dt, T_boundary(T))
            eq.solve(var=T, dt=dt)                    # FiPy implicit solve
            residual = norm(T.value - T_old, ...)
            if residual < tolerance:
                break
            T_old = T.value.copy() * alpha + T_old * (1 - alpha)   # under-relax

        # --- adaptive time step ---
        if config.solution.time.adaptive.enabled:
            dT_max = np.max(np.abs(T.value - T_at_start_of_step))
            dt = clip(dt * target_dT / dT_max * safety, dt_min, dt_max)

        t += dt
        step += 1
        output_handler.write(t, step, T, fields)
```

The FiPy equation is assembled once and reused:

```python
eq = (fipy.TransientTerm(coeff=fields.rho_cp)
      == fipy.DiffusionTerm(coeff=fields.k)
      + fields.Q
      + boundary_source_terms)
```

---

#### 3.7  `output.py`

```python
class OutputHandler:
    def write(self, t, step, T, fields): ...
```

Checks each output spec against `t` and `step`, then calls the appropriate
writer. All writers are simple standalone functions — no class hierarchy.

```python
def write_node_history(path, t, nodes, T, flux): ...       # appends one row to TSV
def write_spatial_profile(path, mesh, T, flux): ...        # writes one snapshot TSV
def write_energy_balance(path, t, q_left, q_right, dU): ...
def write_tecplot_zone(path, t, mesh, T, flux): ...        # appends ZONE to .dat
```

**Energy balance** calculation:
```
Q_in  = q_left_face * A + q_right_face * A        (face fluxes, W)
dU/dt = sum(rho_cp * cell_volume * dT/dt)          (W)
residual = Q_in - Q_stored + Q_generated           (should be ~0)
```

**Heat flux** at faces is computed as:
```python
flux_faces = -fields.k * T.faceGrad   # FiPy idiom
```
Cell-centred flux (for output) is interpolated from face values.

---

#### 3.8  `restart.py`

```python
def save_checkpoint(path: Path, t, step, dt, T, config_path): ...
def load_checkpoint(path: Path) -> dict: ...
```

Uses `h5py`. The checkpoint stores:
- `t`, `step`, `dt` — solver state scalars
- `T_values` — numpy array of cell temperatures
- `config_path` — path to the original YAML (for verification on restart)
- `mesh_dx` — dx array (verified against rebuilt mesh on restart to catch mismatches)

Restart is initiated by setting `initial_conditions.type: restart` in the
YAML. The loader detects this and `solver.py` calls `load_checkpoint` before
entering the time loop.

---

### 4. Data Flow

```
YAML file
    │
    ▼
loader.py  ──►  SimulationInput (validated, all paths resolved)
    │
    ├──► mesh.py        →  fipy.Grid1D mesh + layer_id array
    │
    ├──► functions.py   →  callable f(x) for every ScalarFunction
    │
    ├──► material.py    →  MaterialFields (FiPy CellVariable / FaceVariable)
    │
    ├──► boundary.py    →  BoundaryHandler
    │
    ├──► output.py      →  OutputHandler
    │
    └──► solver.py      →  time loop → writes all outputs
```

---

### 5. Dependencies

| Package         | Purpose                              | Notes                          |
|-----------------|--------------------------------------|--------------------------------|
| `fipy`          | PDE discretisation and solve         | core                           |
| `pydantic >= 2` | input validation                     | core                           |
| `ruamel.yaml`   | YAML parsing                         | preserves comments             |
| `numpy`         | arrays, polynomial evaluation        | core                           |
| `scipy`         | `interp1d` for piecewise functions   | core                           |
| `pandas`        | TSV loading with flexible column sel | could be replaced with numpy   |
| `h5py`          | HDF5 checkpoint read/write           | only needed if restart enabled |

No other dependencies. In particular, no `sympy` or `numexpr` — the
`expression` function type mentioned as advanced in the YAML template is
deferred until needed.

---

### 6. Key Design Decisions

**Single scalar function interface.** Every property — whether constant,
polynomial, or tabulated — is converted to a `Callable[[float], float]` by
`functions.py`. The rest of the code never inspects the function type. This
makes adding new function types (e.g. `expression`) a one-file change.

**FiPy fields updated in-place.** `MaterialFields.update(T)` writes directly
into the existing `CellVariable` and `FaceVariable` objects rather than
creating new ones each iteration. FiPy's equation object holds references to
these variables, so updates are automatically seen by the solver.

**Boundary source terms, not constraints.** Convection and radiation BCs are
implemented as explicit source terms added to the equation rather than as
FiPy constraint objects. This avoids complexity in switching BC types and
makes the `scheduled` BC trivial to implement — just change which source term
is active.

**No inheritance for BCs or outputs.** The BC and output types are plain
dataclasses / Pydantic models with a `type` discriminator field. Behaviour
is selected with `if/elif` on the type, not polymorphism. For the number of
types involved this is simpler to read and debug.

**Adaptive time stepping is post-step.** The new `dt` is computed after each
accepted step based on the actual temperature change. The step is never
rejected and rerun — this keeps the logic simple and is sufficient for the
smoothly varying problems targeted.
