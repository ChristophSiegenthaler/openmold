# openmold

Compares injection-moulded part variants with open-source tools: wall thickness, filling, cooling, sink marks,
warpage and draft. One YAML file describes the project, one command runs it, and the result is a single HTML report
that works offline, including a 3D fill animation.

It is meant for **ranking design variants early**. It does not replace Moldflow or Moldex3D: see [Limits](#limits).

## Setup

You need Python 3.10+ and [CalculiX](http://www.calculix.de/) (`ccx` on the PATH). gmsh comes in through pip.
FreeCAD is only needed for `exports.step`, and Chromium only for `report.pdf`.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # a few minutes
```

## Try it

This writes two synthetic boxes and compares them, in about 5 minutes:

```bash
.venv/bin/python examples/make_demo_parts.py examples/local
.venv/bin/openmold run examples/demo.yaml
# open examples/local/run/report.html
```

## Your own project

Keep project files outside this repository. Copy `examples/demo.yaml` next to your STEP files and adapt it.

```bash
openmold check project.yaml             # validate the config and look for the tools
openmold run project.yaml --jobs 2      # parts in parallel
openmold run project.yaml --parts B     # only one part
openmold run project.yaml --only fill   # only one step; earlier results must exist
openmold report project.yaml            # rebuild figures and report
```

Finished steps are skipped. A step reruns when its inputs change (STEP file, mesh size, material, process values),
and so do the steps that depend on it. `--force` reruns everything selected.

Each part must be **one closed solid, in mm**. Check these before trusting a result:

1. **Gate position.** It is your input, and it decides the fill pattern and the weld lines.
2. **Mesh size.** `mesh.thermal` should be about a third of the wall, otherwise the run stops with a message.
3. **Material card.** The generic card only supports a ranking.

## Configuration

Everything except `project`, `parts`, `material` and `gates` has a default, listed in `openmold/config.py`.

| Section | What it sets |
|---|---|
| `project` | `name`, and `output` directory relative to the config file |
| `parts` | list of `id`, `step` and `label` |
| `material` | a packaged card (`generic_tpu_65d`) or a path to a YAML card |
| `demould_axis` | used for the draft check and the camera. Results use the CAD coordinates as they are |
| `gates` | list of `point: [x, y, z]` and `radius`. Several gates are allowed |
| `process` | `melt_temp`, `mold_temp`, `room_temp`, `fill_time`, `htc` (W/m²K), `eject_times` |
| `mesh` | element size in mm for `fill`, `thermal`, `warp` and `surface` (draft check and exports) |
| `cooling.steps` | `[increment, duration]` pairs in seconds |
| `cooling.htc_zones` | areas that cool worse, such as long core pins: `htc` and `cylinders` (`base`, `axis`, `radius`, `start`, `length`) |
| `cooling.wall_probe` | box around the middle of a typical wall, for the assumed end of packing. Found automatically if omitted |
| `fill.air_trap_exclude` | boxes that vent, such as the parting line |
| `warp.supports` | 3 points for the 3-2-1 support. Does not affect the distortion unless the points are badly placed |
| `warp.flatness` | named planes: `box` to select nodes, `normal` of the plane |
| `warp.bows` | relative movement of `mid` against `ref` along `direction` |
| `warp.lateral` | largest in-plane shift of `points`, for example screw holes |
| `exports` | `parts`, `eject_time`, `scales` (1 is true scale), `coloured_ply`, `step` |
| `plots` | `warp_scale`, `section` plane, `section_times`, `detail` view (`center`, `radius`), `curve_start` |
| `report` | `title`, `notes_html` (your conclusions, added at the top), `pdf` |
| `tools` | binary names and `ccx_threads` |

Boxes are `{min: [x, y, z], max: [x, y, z]}`, and `null` leaves that side open.

## Material cards

Start from `openmold/materials/template.yaml`. The flow model is `cross` or `cross_wlf`; thermal properties are a
number or a `[[T, value], ...]` table. A table with a latent-heat peak in `specific_heat` makes the run slower, because
CalculiX then chooses its own increments. Halve `cooling.steps` once to check that the cooling times hold.

**Do not commit restricted data.** Lab reports and supplier cards often forbid reproduction: keep them next to the project.
`.gitignore` already blocks STEP, PDF, UDB, meshes, results and any `local/` directory.

## Results

```
<output>/report.html          the report, one self-contained file
<output>/summary.csv, .json   all key numbers
<output>/exports/             deformed geometry as STL and PLY
<output>/plots/               figures as PNG
<output>/<part>/              raw results of each part
```

Exports come in two versions. `as-molded` keeps the simulated shrinkage. `warp-only` has it removed, so it overlays the
CAD and shows only the distortion. Scales other than 1 exaggerate the shape and carry `_x20` in the file name; do not
use those for fit checks.

## How it works

| Step | Tool | Module |
|---|---|---|
| Meshes | gmsh | `geometry.py` |
| Wall thickness (inscribed sphere) | NumPy, trimesh | `thickness.py` |
| Fill: isothermal Hele-Shaw, control volume, Cross or Cross-WLF at melt temperature | SciPy | `fill.py` |
| Cooling (DC3D4) and sink estimate | CalculiX | `thermal.py` |
| Warpage after ejection (C3D10), split into shrinkage and distortion | CalculiX | `warp.py` |
| Draft check | NumPy | `draft.py` |
| Figures, report, animation | PyVista, three.js | `plots.py`, `report.py`, `assets/` |

## Limits

- **Filling is isothermal:** no frozen layer and no shear heating, so absolute pressures come out low.
- **No packing simulation.** The sink depth is a rule of thumb: melt still molten when the nominal wall freezes shrinks by
  `sink_volumetric` without compensation.
- **Warpage covers only the free cooling after ejection.** Shrinkage while still in the mould, and flow-induced anisotropy,
  are missing. The simulated shrinkage is therefore much smaller than the real one and **is not a tool shrink factor**.
- **The mould is not modelled:** heat transfer coefficients replace steel and cooling channels.
- **The draft check cannot detect undercuts,** because it does not know the parting line.
- **Numerical overshoot:** linear tets with a consistent mass matrix swing far above melt temperature and below mould
  temperature near the surface in the first seconds. It has died out by the times reported, and the cooling curves start
  at `plots.curve_start`.
- **Not validated against real parts.** Each solver matches an analytical solution in `tests/test_physics.py` (Hele-Shaw strip
  pressure, slab cooling, free shrinkage), but the pipeline has never been compared with moulded parts or a commercial solver.

A natural next step is a real fill and pack simulation with [openInjMoldSim](https://github.com/krebeljk/openInjMoldSim),
which uses the same Cross-WLF and Tait data a measured material card provides.

## Solver threads

Keep `tools.ccx_threads` at 1. With CalculiX 2.23 and SPOOLES (Arch package `calculix-ccx` 2.23-1), runs with several threads
sometimes return a wrong displacement field with no error message. The same input deck gave 7 wrong results in 300 runs on
4 threads (up to 0.67 mm of false warpage) and 0 wrong results in 300 runs on 1 thread. To speed up a project, process parts
in parallel with `--jobs` instead; each run needs several GB of memory on large meshes.

## License

GPL-3.0-or-later, see `LICENSE`. This software comes without any warranty.

## Third-party code

`openmold/assets/` contains three.js r147 (MIT), see `THREE_LICENSE.txt`.
