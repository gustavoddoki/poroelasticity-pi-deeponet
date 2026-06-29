# Reproducibility Notes

This repository preserves the original thesis manuscript under `paper/` while
maintaining a corrected, testable implementation under `src/`.

## Corrections Applied

- The manufactured pressure source uses the positive contribution from
  `u_xt`, consistent with the governing pressure equation.
- The stabilized implicit-Euler pressure equation uses
  `c3 = K * dt / dx**2 + 1 / (4 * E)` and `c4 = -1 / (4 * E)` after the
  equation is multiplied by `dt`.
- Random PI-DeepONet collocation points evaluate source terms at the sampled
  time rather than at a fixed final time.
- Percentage errors use a finite denominator near exact zeros.
- The neural architecture is identified as a multi-input operator network
  (MIONet), matching its four branch encoders and tensor-product reduction.
- Neural training normalizes the large `U` input, uses multiple collocation
  points per sampled function, and records deterministic validation metrics.
- Training runs preserve configuration, metrics, best weights, and resumable
  checkpoints; incompatible configurations cannot resume the same run.

These changes correct implementation and transcription issues found while
consolidating the two legacy repositories. The historical manuscript is left
unchanged so the research record remains explicit.

## Result Status

The thesis tables and figures are historical results. They are not claimed as
fresh reproductions because the original trained checkpoints, complete logs,
random seeds, and exact sampled parameter sets are unavailable.

The numerical implementation is validated against the manufactured analytical
solution by automated PDE-residual and Gauss-Seidel tests. Run them with:

```bash
pytest
```

Using the default manufactured parameters, tolerance `1e-5`, and the corrected
Gauss-Seidel implementation produced:

| Spatial cells | Time steps | Iterations | MAPE displacement | MAPE pressure |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 728 | 2.1863% | 0.2198% |
| 8 | 16 | 5,230 | 0.7233% | 0.6194% |
| 16 | 32 | 40,534 | 0.2690% | 0.4322% |
| 32 | 64 | 297,344 | 0.1111% | 0.2458% |

The pressure error is not monotonic on the coarsest meshes, but both fields are
below 0.25% at `32 x 64`. The displacement result is close to the historical
`0.1200%` thesis value. Exact agreement is not expected because the random
manufactured parameters used for the thesis run were not preserved.

For a numerical experiment with explicit mesh settings:

```bash
python experiments/run_gauss_seidel.py --spatial-cells 32 --time-steps 64
```

Neural results should only be reported as reproduced after training a new model
and preserving its configuration, seed, checkpoint, logs, and evaluation data.
The current training pipeline preserves all of those artifacts under its run
directory. See `TRAINING.md` for the protocol.
