import numpy as np
import pytest

from poroelasticity.analytical import (
    BiotConfig,
    ManufacturedParameters,
    analytical_solution,
    denormalize_displacement,
    denormalize_pressure,
    non_dimensional_scales,
    normalize_displacement,
    normalize_pressure,
    source_terms,
)


def test_scales_balance_physical_coefficients():
    config = BiotConfig(elastic_modulus=1e8, hydraulic_conductivity=1e-5, length=0.5)

    scales = non_dimensional_scales(config, displacement_scale=1.0)

    assert scales.pressure == pytest.approx(config.elastic_modulus / config.length)
    assert scales.time == pytest.approx(config.length**2 / (config.hydraulic_conductivity * config.elastic_modulus))
    assert scales.displacement == 1.0


def test_scales_reject_non_positive_displacement():
    config = BiotConfig()

    with pytest.raises(ValueError):
        non_dimensional_scales(config, displacement_scale=0.0)


def test_normalize_round_trip_recovers_physical_values():
    config = BiotConfig()
    scales = non_dimensional_scales(config, displacement_scale=2.0)
    physical_u = np.array([-1.5, 0.0, 3.0])
    physical_p = np.array([0.0, 1.0, -2.0])

    non_dim_u = normalize_displacement(scales, physical_u)
    non_dim_p = normalize_pressure(scales, physical_p)

    np.testing.assert_allclose(denormalize_displacement(scales, non_dim_u), physical_u)
    np.testing.assert_allclose(denormalize_pressure(scales, non_dim_p), physical_p)


def test_manufactured_displacement_residual_is_o1_in_nondimensional_form():
    config = BiotConfig(elastic_modulus=3.5, hydraulic_conductivity=0.2, length=0.5)
    params = ManufacturedParameters(
        displacement_amplitude=1.7,
        pressure_amplitude=0.8,
        displacement_decay=1.4,
        pressure_decay=0.6,
        displacement_shift=-0.3,
        pressure_shift=-0.7,
    )
    scales = non_dimensional_scales(config, displacement_scale=params.displacement_amplitude)

    x = np.linspace(0.05, config.length - 0.05, 24)
    t_array = np.full_like(x, 0.37)
    _, _ = analytical_solution(config, x, t_array, params)
    source_u, _ = source_terms(config, x, t_array, params)

    frequency = np.pi / (2.0 * config.length)
    wave = frequency * x
    exp_u = np.exp(-t_array * params.displacement_decay + params.displacement_shift)
    exp_p = np.exp(-t_array * params.pressure_decay + params.pressure_shift)
    u_xx = -params.displacement_amplitude * frequency**2 * np.cos(wave) * exp_u
    p_x = params.pressure_amplitude * frequency * np.cos(wave) * exp_p

    residual_u_factor = config.length**2 / (config.elastic_modulus * scales.displacement)
    non_dim_residual_u = (-config.elastic_modulus * u_xx + p_x - source_u) * residual_u_factor

    assert np.max(np.abs(non_dim_residual_u)) < 1e-9


def test_manufactured_mass_residual_is_o1_in_nondimensional_form():
    config = BiotConfig(elastic_modulus=3.5, hydraulic_conductivity=0.2, length=0.5)
    params = ManufacturedParameters(
        displacement_amplitude=1.7,
        pressure_amplitude=0.8,
        displacement_decay=1.4,
        pressure_decay=0.6,
        displacement_shift=-0.3,
        pressure_shift=-0.7,
    )
    scales = non_dimensional_scales(config, displacement_scale=params.displacement_amplitude)

    x = np.linspace(0.05, config.length - 0.05, 24)
    t_array = np.full_like(x, 0.37)
    _, _ = analytical_solution(config, x, t_array, params)
    _, source_p = source_terms(config, x, t_array, params)

    frequency = np.pi / (2.0 * config.length)
    wave = frequency * x
    exp_u = np.exp(-t_array * params.displacement_decay + params.displacement_shift)
    u_xt = params.displacement_amplitude * params.displacement_decay * frequency * np.sin(wave) * exp_u
    p_xx = -params.pressure_amplitude * frequency**2 * np.sin(wave) * np.exp(
        -t_array * params.pressure_decay + params.pressure_shift
    )

    residual_p_factor = config.length**3 / (config.hydraulic_conductivity * config.elastic_modulus * scales.displacement)
    non_dim_residual_p = (u_xt - config.hydraulic_conductivity * p_xx - source_p) * residual_p_factor

    assert np.max(np.abs(non_dim_residual_p)) < 1e-9