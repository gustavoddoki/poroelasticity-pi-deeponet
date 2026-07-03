# Neural Training Guide

The neural model is a physics-informed multi-input operator network (MIONet):
four branch encoders process `U`, `P`, `u0`, and `p0`, while a trunk encoder
processes `(x, t)`. Separate operator heads predict displacement, pressure,
and the normalized Darcy flux.

## Reproducible Smoke Test

Install the optional neural dependencies and run a small experiment:

```bash
pip install -e ".[dev,neural]"
python experiments/train_pi_deeponet.py \
  --epochs 10 \
  --num-functions 4 \
  --points-per-function 4 \
  --validation-functions 8 \
  --validation-points 16 \
  --checkpoint-every 5 \
  --output-dir runs/smoke
```

The output directory contains the complete configuration, metrics, best
weights, and resumable TensorFlow checkpoints. Resume an interrupted run with:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 100000 \
  --output-dir runs/hypothetical \
  --resume
```

## Experiment Configurations

Hypothetical material parameters:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 100000 \
  --num-functions 16 \
  --points-per-function 16 \
  --elastic-modulus 1 \
  --hydraulic-conductivity 1 \
  --output-dir runs/hypothetical
```

Realistic material parameters use the physics-only mixed Darcy formulation:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 100000 \
  --num-functions 16 \
  --points-per-function 16 \
  --validation-functions 32 \
  --validation-points 64 \
  --elastic-modulus 1e8 \
  --hydraulic-conductivity 1e-5 \
  --displacement-data-weight 0 \
  --pressure-data-weight 0 \
  --joint-warmup-epochs 100000 \
  --displacement-steps 1 \
  --pressure-steps 1 \
  --dtype float32 \
  --validate-every 100 \
  --log-every 10 \
  --output-dir runs/realistic-mixed-physics-only
```

Define the Darcy flux as `q = -K p_x`. The original mass equation is then
represented by the equivalent first-order system

```text
u_xt + q_x - P = 0
q + K p_x = 0
```

The network predicts the normalized flux `q_hat = q / q_c`, with
`q_c = K p_c / L` and `p_c` obtained from the RMS initial-pressure branch.
Dividing Darcy's identity by `q_c` gives an order-one pressure gradient without
dividing the mass residual or any displacement error by `K`. The impermeable
right boundary is imposed as `q_hat(L,t) = 0`.

The loss contains only momentum balance, mass balance, Darcy's identity,
boundary conditions, and initial conditions when both data weights are zero.
Analytical `u`, `p`, and `q` values are used only for validation metrics and
checkpoint selection, never as optimization targets in this configuration.

Two-output checkpoints are incompatible with this three-output model. Start a
new run directory. `--initial-weights` may only reference weights produced by
the mixed model; `--initial-weights` and `--resume` are mutually exclusive.

An attempted auxiliary residual that divided the mass equation by `K` was
removed after it amplified early displacement error and caused pressure
divergence. Alternating displacement and pressure updates also failed to
improve pressure identifiability. Both are retained only as negative ablations.

The trainer stores two weight files. `best.weights.h5` minimizes the mean
relative validation error across displacement and pressure.
`best.objective.weights.h5` minimizes the validation training objective. Keeping
both makes disagreement between physical residuals and field accuracy visible.

Start with `float32`, which is substantially faster on common free GPUs. The
mixed residual removes the second pressure derivative and normalizes the Darcy
identity; compare `float64` as a precision ablation rather than a requirement.

## Free GPU Runtimes

Lightning AI and Modal both provide recurring free compute credits. Free
runtimes can be interrupted, so use short checkpoint intervals and preserve the
entire run directory in persistent storage. Google Colab is suitable for smoke
tests but its free GPU availability and runtime duration are not guaranteed.

## Scientific Scope

The manufactured source terms and initial conditions are generated from the
same six random parameters. Consequently, this experiment evaluates
generalization within that parametric family; it does not establish
generalization to arbitrary source functions.

For `E=1e8` and `K=1e-5`, pressure has weak influence on the physics residuals.
An observed 100,000-epoch pure-physics run reduced validation loss below
`4e-4` while pressure relative error remained above `70%`. This is an
identifiability and scaling failure: a small residual is not evidence of an
accurate pressure field in this regime.

Use `--displacement-data-weight` and `--pressure-data-weight` to add separately
reported manufactured-solution supervision. The legacy `--data-weight` option
sets both to the same value. Any nonzero value means the experiment is hybrid,
not physics-only.

Mixed runs must report all three residuals, both learning rates, precision,
flux normalization, random seed, and relative validation errors for `u`, `p`,
and `q`.

## Non-Dimensionalization

The displacement equation is multiplied by ``E`` while the pressure equation
is multiplied by ``K``, so the two fields interact with coefficients that
differ by thirteen orders of magnitude when ``E = 1e8`` and ``K = 1e-5``. The
mass residual then becomes effectively decoupled from ``p`` because the
``K * p_xx`` term vanishes relative to ``u_xt`` in the original second-order
form.

The trainer applies the characteristic scales

```
p_c = E * u_c / L
t_c = L**2 / (K * E)
```

so that the non-dimensional equations read

```
-u_xx + p_x = U_tilde
u_xt - p_xx = P_tilde
```

with every PDE coefficient equal to one. The flags `--displacement-scale` and
`--no-non-dimensionalization` control the choice; the default behaviour keeps
the scaling on, and the trainer prints the chosen scales at the start of
every run. Pass `--no-non-dimensionalization` only when comparing directly to
the legacy raw-residual form.

### Loss Decomposition

The scaling is applied **only to the displacement and mass conservation PDE
residuals**. The Darcy identity is already balanced by the internal
``flux_scale = K * p_c / L`` normalization. The boundary, initial-condition,
and data losses stay in physical units because dividing them by ``p_c``
would push typical pressure errors (order ``1``) below fp32 precision once
``p_c`` is on the order of ``E``, masking the pressure gradient entirely.
