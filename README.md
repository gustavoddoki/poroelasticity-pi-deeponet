# PI-MIONet for One-Dimensional Biot Consolidation

This repository contains a corrected and reproducible research implementation
for solving the one-dimensional Biot consolidation model with a
physics-informed multi-input operator network (PI-MIONet), historically
described in the thesis as a PI-DeepONet. A finite-volume Gauss-Seidel solver is
included as the classical baseline.

Biot consolidation describes the coupled interaction between deformation and
fluid pressure in porous media. The model studied here is a stiff coupled PDE
system with displacement `u(x, t)` and pressure `p(x, t)`.

## Why This Project Exists

Classical numerical methods can solve the model accurately, but repeated
simulations become expensive when the mesh is refined or when source terms and
initial conditions vary. Physics-informed operator networks use a costly
training phase to enable fast inference for new inputs.

The project combines:

- a finite-volume Gauss-Seidel solver with implicit Euler time stepping;
- a PI-MIONet with branches for `U`, `P`, `u0`, and `p0`;
- manufactured analytical solutions for physics and accuracy validation;
- deterministic sampling, validation, checkpoints, and resumed neural runs;
- the original manuscript and selected result figures.

## Repository Layout

```text
|-- experiments/
|   |-- run_gauss_seidel.py
|   `-- train_pi_deeponet.py
|-- paper/
|   |-- manuscript.tex
|   `-- figures/
|-- results/
|   `-- figures/
|-- src/poroelasticity/
|   |-- analytical.py
|   |-- numerical/
|   `-- neural/
|-- tests/
|-- REPRODUCIBILITY.md
|-- TRAINING.md
`-- pyproject.toml
```

## Mathematical Model

The one-dimensional model is defined by

```math
-E\frac{\partial^2u}{\partial x^2} + \frac{\partial p}{\partial x} = U(x,t)
```

```math
\frac{\partial}{\partial t}\left(\frac{\partial u}{\partial x}\right)
- K\frac{\partial^2p}{\partial x^2} = P(x,t)
```

where `E` is the elastic modulus, `K` is the hydraulic conductivity, `U` is the
body-force density source term, and `P` is the fluid injection/extraction source
term.

## Result Status

The thesis reported approximately 30 ms of neural inference after training and
compared the neural model with highly refined Gauss-Seidel runs. Those numbers
are historical results, not fresh reproductions: the original checkpoints,
complete logs, seeds, and sampled parameter sets were not preserved.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for corrected numerical results
and [`TRAINING.md`](TRAINING.md) for the neural experiment protocol and its
scientific limitations.

## Selected Figures

PI-MIONet training flow:

![PI-MIONet flowchart](results/figures/pi_deeponet_flowchart.png)

Multi-input operator architecture for the Biot model:

![MIONet architecture](results/figures/deeponet_biot_architecture.png)

Historical comparison with the analytical solution:

![Historical realistic-case comparison](results/figures/real_results_comparison.png)

## Quick Start

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,neural]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,neural]"
```

Run the tests and numerical baseline:

```bash
pytest
python experiments/run_gauss_seidel.py --spatial-cells 32 --time-steps 64
```

Run a short neural smoke test:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 10 \
  --num-functions 4 \
  --points-per-function 4 \
  --output-dir runs/smoke
```

The original thesis used 100,000 epochs. Long runs should only start after the
smoke test has produced finite losses, checkpoints, and validation metrics.

## Origin

This repository consolidates two earlier projects:

- `Poroelasticity-NumericalMethods`
- `Poroelasticity-DeepLearning`

The consolidated version presents the model, numerical baseline, neural
operator, evaluation assets, tests, and manuscript as one research-engineering
project.
