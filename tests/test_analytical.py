import numpy as np

from poroelasticity.analytical import (
    BiotConfig,
    ManufacturedParameters,
    analytical_solution,
    darcy_flux,
    initial_conditions,
    mean_absolute_percentage_error,
    source_terms,
)


def test_initial_conditions_match_analytical_solution_at_t_zero():
    config = BiotConfig()
    params = ManufacturedParameters()
    x = np.linspace(0.0, config.length, 8)

    expected_u, expected_p = analytical_solution(config, x, 0.0, params)
    actual_u, actual_p = initial_conditions(config, x, params)

    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_p, expected_p)


def test_source_terms_preserve_input_shape():
    config = BiotConfig()
    params = ManufacturedParameters()
    x = np.linspace(0.0, config.length, 8)

    source_u, source_p = source_terms(config, x, 0.25, params)

    assert source_u.shape == x.shape
    assert source_p.shape == x.shape


def test_manufactured_solution_satisfies_biot_equations():
    config = BiotConfig(elastic_modulus=3.5, hydraulic_conductivity=0.2)
    params = ManufacturedParameters(
        displacement_amplitude=1.7,
        pressure_amplitude=0.8,
        displacement_decay=1.4,
        pressure_decay=0.6,
        displacement_shift=-0.3,
        pressure_shift=-0.7,
    )
    x = np.linspace(0.01, config.length - 0.01, 16)
    t = 0.37
    frequency = np.pi / (2.0 * config.length)
    wave = frequency * x
    exp_u = np.exp(-t * params.displacement_decay + params.displacement_shift)
    exp_p = np.exp(-t * params.pressure_decay + params.pressure_shift)

    u_xx = -params.displacement_amplitude * frequency**2 * np.cos(wave) * exp_u
    u_xt = params.displacement_amplitude * params.displacement_decay * frequency * np.sin(wave) * exp_u
    p_x = params.pressure_amplitude * frequency * np.cos(wave) * exp_p
    p_xx = -params.pressure_amplitude * frequency**2 * np.sin(wave) * exp_p
    source_u, source_p = source_terms(config, x, t, params)

    np.testing.assert_allclose(-config.elastic_modulus * u_xx + p_x, source_u)
    np.testing.assert_allclose(u_xt - config.hydraulic_conductivity * p_xx, source_p)


def test_manufactured_flux_satisfies_mixed_biot_equations():
    config = BiotConfig(elastic_modulus=3.5, hydraulic_conductivity=0.2)
    params = ManufacturedParameters(pressure_amplitude=0.8, pressure_decay=0.6, pressure_shift=-0.7)
    x = np.linspace(0.01, config.length - 0.01, 16)
    t = 0.37
    frequency = np.pi / (2.0 * config.length)
    wave = frequency * x
    exp_u = np.exp(-t * params.displacement_decay + params.displacement_shift)
    exp_p = np.exp(-t * params.pressure_decay + params.pressure_shift)
    p_x = params.pressure_amplitude * frequency * np.cos(wave) * exp_p
    u_xt = params.displacement_amplitude * params.displacement_decay * frequency * np.sin(wave) * exp_u
    q_x = (
        config.hydraulic_conductivity
        * params.pressure_amplitude
        * frequency**2
        * np.sin(wave)
        * exp_p
    )
    _, source_p = source_terms(config, x, t, params)

    np.testing.assert_allclose(darcy_flux(config, x, t, params), -config.hydraulic_conductivity * p_x)
    np.testing.assert_allclose(u_xt + q_x, source_p)


def test_mape_remains_finite_at_exact_zeros():
    actual = np.array([0.0, 1.0])
    predicted = np.array([0.0, 0.9])

    error = mean_absolute_percentage_error(actual, predicted)

    assert np.isfinite(error)
    assert np.isclose(error, 5.0)
