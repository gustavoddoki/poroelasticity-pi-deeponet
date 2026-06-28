from poroelasticity.analytical import (
    BiotConfig,
    ManufacturedParameters,
    analytical_solution,
    mean_absolute_percentage_error,
)
from poroelasticity.numerical.gauss_seidel import solve_gauss_seidel


def test_gauss_seidel_matches_manufactured_solution():
    config = BiotConfig()
    params = ManufacturedParameters()

    result = solve_gauss_seidel(
        config,
        params,
        spatial_cells=8,
        time_steps=16,
        tolerance=1e-6,
    )
    expected_u, expected_p = analytical_solution(config, result.x, config.final_time, params)

    assert result.converged
    assert mean_absolute_percentage_error(expected_u, result.displacement) < 1.0
    assert mean_absolute_percentage_error(expected_p, result.pressure) < 1.0
