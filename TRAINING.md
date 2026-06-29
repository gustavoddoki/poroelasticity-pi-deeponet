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

Realistic material parameters:

```bash
python experiments/train_pi_deeponet.py \
  --epochs 100000 \
  --num-functions 16 \
  --points-per-function 16 \
  --elastic-modulus 1e8 \
  --hydraulic-conductivity 1e-5 \
  --output-dir runs/realistic
```

Start with `float32`, which is substantially faster on common free GPUs. Run a
short `float64` comparison before the final experiment if the PDE residuals or
gradients indicate that additional precision is necessary.

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
The optional `--data-weight` argument can add manufactured-solution supervision
for an explicitly hybrid experiment. A nonzero value must be reported because
the resulting model is no longer trained from physics alone.
