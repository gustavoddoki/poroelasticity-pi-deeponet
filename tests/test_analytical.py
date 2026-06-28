import numpy as np

from poroelasticity.analytical import BiotConfig, ManufacturedParameters, analytical_solution, initial_conditions, source_terms


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
