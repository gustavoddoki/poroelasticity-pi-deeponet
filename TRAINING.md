# Neural Training Guide

The neural model is a physics-informed multi-input operator network (MIONet):
four branch encoders process `U`, `P`, `u0`, and `p0`, while a trunk encoder
processes `(x, t)`. Separate operator heads predict displacement and pressure.

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

Realistic material parameters require an explicitly hybrid pressure constraint:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 100000 \
  --num-functions 16 \
  --points-per-function 16 \
  --validation-functions 32 \
  --validation-points 64 \
  --elastic-modulus 1e8 \
  --hydraulic-conductivity 1e-5 \
  --pressure-data-weight 1 \
  --validate-every 100 \
  --log-every 10 \
  --output-dir runs/realistic-hybrid
```

This is a physics-informed hybrid experiment: pressure labels from the
manufactured solution supplement the PDE, boundary, and initial residuals.
Report the nonzero pressure data weight with every result. A pure-physics run
with these dimensional coefficients is retained as a diagnostic, not as the
recommended final experiment.

Pressure-residual preconditioning can recover a stronger physical gradient
without changing the zero of the mass-balance equation:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 1000 \
  --num-functions 16 \
  --points-per-function 16 \
  --validation-functions 32 \
  --validation-points 64 \
  --learning-rate 1e-5 \
  --elastic-modulus 1e8 \
  --hydraulic-conductivity 1e-5 \
  --pressure-data-weight 0.1 \
  --pressure-precondition-weight 1 \
  --pressure-precondition-warmup 1000 \
  --dtype float64 \
  --output-dir runs/realistic-preconditioned-smoke
```

The auxiliary residual is `((u_xt - P) / K - p_xx) / (p_c / L^2)`, where
`p_c` is the RMS initial-pressure scale for each sampled function. Its target is
detached from the displacement graph, so it preconditions the pressure head
without sending the `1 / K` amplification into the displacement head. The
original dimensional residual remains in the objective and metrics. The target
weight is ramped linearly during `--pressure-precondition-warmup` epochs.

Because division by a small conductivity magnifies cancellation error, use
`float64` for this experiment. The trainer clips displacement and pressure
gradients independently and records both pre-clipping norms.

The trainer stores two weight files. `best.weights.h5` minimizes the mean
relative validation error across displacement and pressure.
`best.objective.weights.h5` minimizes the validation training objective. Keeping
both makes disagreement between physical residuals and field accuracy visible.

Start with `float32`, which is substantially faster on common free GPUs. For a
pure-physics dimensional experiment, use `float64` and first run a short
comparison because recovering pressure requires resolving differences between
terms with widely separated magnitudes.

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

`--pressure-precondition-weight` changes only optimization conditioning: it
multiplies a residual algebraically equivalent to the original mass-balance
equation for `K > 0`. It does not add solution labels. Runs must report its
weight, warmup, precision, and the original pressure-equation residual.
