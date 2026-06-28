from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiotConfig:
    """Physical and domain parameters for the one-dimensional Biot model."""

    length: float = 0.5
    final_time: float = 1.0
    elastic_modulus: float = 1.0
    hydraulic_conductivity: float = 1.0


@dataclass(frozen=True)
class ManufacturedParameters:
    """Parameters for the manufactured analytical solution used in experiments."""

    displacement_amplitude: float = 1.0
    pressure_amplitude: float = 1.0
    displacement_decay: float = 1.0
    pressure_decay: float = 1.0
    displacement_shift: float = 0.0
    pressure_shift: float = 0.0


def analytical_solution(config: BiotConfig, x, t, params: ManufacturedParameters):
    """Return the manufactured displacement and pressure fields."""

    wave = np.pi * np.asarray(x) / (2.0 * config.length)
    u = (
        params.displacement_amplitude
        * np.cos(wave)
        * np.exp(-np.asarray(t) * params.displacement_decay + params.displacement_shift)
    )
    p = (
        params.pressure_amplitude
        * np.sin(wave)
        * np.exp(-np.asarray(t) * params.pressure_decay + params.pressure_shift)
    )
    return u, p


def initial_conditions(config: BiotConfig, x, params: ManufacturedParameters):
    """Return analytical initial conditions at t=0."""

    return analytical_solution(config, x, 0.0, params)


def source_terms(config: BiotConfig, x, t, params: ManufacturedParameters):
    """Return source terms U(x,t) and P(x,t) for the manufactured solution."""

    wave = np.pi * np.asarray(x) / (2.0 * config.length)
    frequency = np.pi / (2.0 * config.length)
    exp_u = np.exp(-np.asarray(t) * params.displacement_decay + params.displacement_shift)
    exp_p = np.exp(-np.asarray(t) * params.pressure_decay + params.pressure_shift)

    U = (
        config.elastic_modulus * params.displacement_amplitude * frequency**2 * np.cos(wave) * exp_u
        + params.pressure_amplitude * frequency * np.cos(wave) * exp_p
    )
    P = (
        -params.displacement_amplitude * params.displacement_decay * frequency * np.sin(wave) * exp_u
        + config.hydraulic_conductivity
        * params.pressure_amplitude
        * frequency**2
        * np.sin(wave)
        * exp_p
    )
    return U, P


def mean_absolute_percentage_error(y_true, y_pred, eps: float = 1e-12) -> float:
    """Return MAPE while avoiding division by exact zeros."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)
