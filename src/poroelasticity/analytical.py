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


@dataclass(frozen=True)
class NonDimensionalScales:
    """Characteristic scales that balance the Biot PDE coefficients."""

    length: float
    displacement: float
    pressure: float
    time: float


def non_dimensional_scales(config: BiotConfig, displacement_scale: float = 1.0) -> NonDimensionalScales:
    """Return characteristic scales that make every term in the Biot system O(1).

    The scaling choice targets the dominant balance of the coupled equations:

    - pressure scale balances the elastic and pressure-gradient terms in the
      displacement equation, ``p_c = E * u_c / L``;
    - time scale balances the storage and conduction terms in the mass
      conservation equation, ``t_c = L**2 / (K * E)``.

    With these choices the non-dimensional equations read

    .. math::
        -\\partial_{\\tilde{x}\\tilde{x}} \\tilde{u} + \\partial_{\\tilde{x}} \\tilde{p} = \\tilde{U}
        \\partial_{\\tilde{x}\\tau} \\tilde{u} - \\partial_{\\tilde{x}\\tilde{x}} \\tilde{p} = \\tilde{P}

    so the pressure field stops being weakly identifiable because the
    ``K * p_xx`` gradient damping is removed from the residual.
    """

    if displacement_scale <= 0.0:
        raise ValueError("displacement_scale must be positive")

    length = config.length
    elastic_modulus = config.elastic_modulus
    hydraulic_conductivity = config.hydraulic_conductivity
    pressure_scale = elastic_modulus * displacement_scale / length
    time_scale = length**2 / (hydraulic_conductivity * elastic_modulus)
    return NonDimensionalScales(
        length=length,
        displacement=displacement_scale,
        pressure=pressure_scale,
        time=time_scale,
    )


def normalize_displacement(scales: NonDimensionalScales, displacement) -> np.ndarray:
    """Return the non-dimensional displacement ``u_tilde = u / u_c``."""

    return np.asarray(displacement) / scales.displacement


def normalize_pressure(scales: NonDimensionalScales, pressure) -> np.ndarray:
    """Return the non-dimensional pressure ``p_tilde = p / p_c``."""

    return np.asarray(pressure) / scales.pressure


def denormalize_displacement(scales: NonDimensionalScales, displacement) -> np.ndarray:
    """Return the physical displacement ``u = u_c * u_tilde``."""

    return scales.displacement * np.asarray(displacement)


def denormalize_pressure(scales: NonDimensionalScales, pressure) -> np.ndarray:
    """Return the physical pressure ``p = p_c * p_tilde``."""

    return scales.pressure * np.asarray(pressure)


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


def darcy_flux(config: BiotConfig, x, t, params: ManufacturedParameters):
    """Return the manufactured Darcy flux q = -K p_x."""

    wave = np.pi * np.asarray(x) / (2.0 * config.length)
    frequency = np.pi / (2.0 * config.length)
    exp_p = np.exp(-np.asarray(t) * params.pressure_decay + params.pressure_shift)
    return (
        -config.hydraulic_conductivity
        * params.pressure_amplitude
        * frequency
        * np.cos(wave)
        * exp_p
    )


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
        params.displacement_amplitude * params.displacement_decay * frequency * np.sin(wave) * exp_u
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
