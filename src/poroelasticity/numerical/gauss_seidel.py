from dataclasses import dataclass

import numpy as np

from poroelasticity.analytical import BiotConfig, ManufacturedParameters, initial_conditions, source_terms


@dataclass(frozen=True)
class GaussSeidelResult:
    x: np.ndarray
    t: np.ndarray
    displacement: np.ndarray
    pressure: np.ndarray
    converged: bool
    iterations: int


def _coefficients(config: BiotConfig, dx: float, dt: float):
    e = config.elastic_modulus
    k = config.hydraulic_conductivity
    c1 = e / dx**2
    c2 = 1.0 / (2.0 * dx)
    c3 = k / dx**2
    c4 = 1.0 / (4.0 * e * dt)
    return c1, c2, c3, c4


def _residual(u, p, u_prev, p_prev, source_u, source_p, c1, c2, c3, c4, dt):
    ru = np.zeros_like(u)
    rp = np.zeros_like(p)
    n = len(u) - 2

    for i in range(n + 2):
        if i == 0:
            ru[i] = u[i + 1] - u[i]
            rp[i] = -p[i + 1] - p[i]
        elif i == n + 1:
            ru[i] = -u[i - 1] - u[i]
            rp[i] = p[i - 1] - p[i]
        else:
            ru[i] = c1 * (u[i - 1] - 2.0 * u[i] + u[i + 1]) + c2 * (p[i - 1] - p[i + 1]) + source_u[i]
            rp[i] = (
                c2 * (u[i - 1] - u[i + 1])
                + c3 * (p[i - 1] - 2.0 * p[i] + p[i + 1])
                + c2 * (u_prev[i + 1] - u_prev[i - 1])
                + c4 * (p_prev[i - 1] - 2.0 * p_prev[i] + p_prev[i + 1])
                + source_p[i] * dt
            )

    return ru, rp


def _sweep(u, p, u_prev, p_prev, source_u, source_p, c1, c2, c3, c4, dt):
    n = len(u) - 2

    for i in range(n + 2):
        if i == 0:
            u[i] = u[i + 1]
        elif i == n + 1:
            u[i] = -u[i - 1]
        else:
            u[i] = (c1 * (u[i - 1] + u[i + 1]) + c2 * (p[i - 1] - p[i + 1]) + source_u[i]) / (2.0 * c1)

    for i in range(n + 2):
        if i == 0:
            p[i] = -p[i + 1]
        elif i == n + 1:
            p[i] = p[i - 1]
        else:
            p[i] = (
                c2 * (u[i - 1] - u[i + 1])
                + c3 * (p[i - 1] + p[i + 1])
                + c2 * (u_prev[i + 1] - u_prev[i - 1])
                + c4 * (p_prev[i - 1] - 2.0 * p_prev[i] + p_prev[i + 1])
                + source_p[i] * dt
            ) / (2.0 * c3)

    return u, p


def solve_gauss_seidel(
    config: BiotConfig,
    params: ManufacturedParameters | None = None,
    spatial_cells: int = 64,
    time_steps: int = 128,
    tolerance: float = 1e-5,
    max_iterations_per_step: int = 10000,
) -> GaussSeidelResult:
    """Solve the manufactured one-dimensional Biot problem with Gauss-Seidel sweeps."""

    params = params or ManufacturedParameters()
    dx = config.length / spatial_cells
    dt = config.final_time / time_steps
    x = np.arange(-dx / 2.0, config.length + dx, dx)
    t = np.linspace(0.0, config.final_time, time_steps + 1)
    c1, c2, c3, c4 = _coefficients(config, dx, dt)

    u_prev, p_prev = initial_conditions(config, x, params)
    u = u_prev.copy()
    p = p_prev.copy()
    total_iterations = 0
    converged = True

    for step in range(1, time_steps + 1):
        source_u, source_p = source_terms(config, x, t[step], params)
        baseline_ru, baseline_rp = _residual(u, p, u_prev, p_prev, source_u, source_p, c1, c2, c3, c4, dt)
        baseline_norm = np.linalg.norm(baseline_ru, np.inf) + np.linalg.norm(baseline_rp, np.inf)
        baseline_norm = max(float(baseline_norm), 1e-12)

        for iteration in range(1, max_iterations_per_step + 1):
            u, p = _sweep(u, p, u_prev, p_prev, source_u, source_p, c1, c2, c3, c4, dt)
            ru, rp = _residual(u, p, u_prev, p_prev, source_u, source_p, c1, c2, c3, c4, dt)
            residual_ratio = (np.linalg.norm(ru, np.inf) + np.linalg.norm(rp, np.inf)) / baseline_norm
            total_iterations += 1
            if residual_ratio < tolerance:
                break
        else:
            converged = False
            break

        u_prev = u.copy()
        p_prev = p.copy()

    return GaussSeidelResult(x=x, t=t, displacement=u, pressure=p, converged=converged, iterations=total_iterations)
