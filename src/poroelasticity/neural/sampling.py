import numpy as np

from poroelasticity.analytical import (
    BiotConfig,
    ManufacturedParameters,
    analytical_solution,
    initial_conditions,
    source_terms,
)


def random_manufactured_parameters(rng: np.random.Generator) -> ManufacturedParameters:
    return ManufacturedParameters(
        displacement_amplitude=rng.uniform(0.5, 3.0),
        pressure_amplitude=rng.uniform(0.5, 3.0),
        displacement_decay=rng.uniform(0.5, 2.0),
        pressure_decay=rng.uniform(0.5, 2.0),
        displacement_shift=rng.uniform(-2.0, 0.0),
        pressure_shift=rng.uniform(-2.0, 0.0),
    )


def create_sample(
    config: BiotConfig,
    size_x: int,
    size_t: int,
    num_functions: int,
    points_per_function: int,
    x_grid,
    t_grid,
    rng: np.random.Generator | None = None,
    dtype=np.float32,
):
    """Create a reproducible MIONet batch with several points per function."""

    if num_functions < 1 or points_per_function < 1:
        raise ValueError("num_functions and points_per_function must be positive")

    rng = rng or np.random.default_rng()
    dtype = np.dtype(dtype)
    sample_count = num_functions * points_per_function
    source_sensor_count = size_x * size_t

    source_u_branch = np.empty((sample_count, source_sensor_count), dtype=dtype)
    source_p_branch = np.empty((sample_count, source_sensor_count), dtype=dtype)
    initial_u_branch = np.empty((sample_count, size_x), dtype=dtype)
    initial_p_branch = np.empty((sample_count, size_x), dtype=dtype)
    x_random = np.empty((sample_count, 1), dtype=dtype)
    t_random = np.empty((sample_count, 1), dtype=dtype)
    source_u_random = np.empty((sample_count, 1), dtype=dtype)
    source_p_random = np.empty((sample_count, 1), dtype=dtype)
    initial_u_random = np.empty((sample_count, 1), dtype=dtype)
    initial_p_random = np.empty((sample_count, 1), dtype=dtype)
    real_u = np.empty((sample_count, 1), dtype=dtype)
    real_p = np.empty((sample_count, 1), dtype=dtype)

    for function_index in range(num_functions):
        params = random_manufactured_parameters(rng)
        source_u, source_p = source_terms(config, x_grid, t_grid, params)
        initial_u, initial_p = initial_conditions(config, np.asarray(x_grid)[0], params)
        start = function_index * points_per_function
        stop = start + points_per_function

        source_u_branch[start:stop] = np.asarray(source_u).reshape(source_sensor_count)
        source_p_branch[start:stop] = np.asarray(source_p).reshape(source_sensor_count)
        initial_u_branch[start:stop] = initial_u
        initial_p_branch[start:stop] = initial_p

        x_values = rng.uniform(0.0, config.length, size=points_per_function)
        t_values = rng.uniform(0.0, config.final_time, size=points_per_function)
        source_u_values, source_p_values = source_terms(config, x_values, t_values, params)
        initial_u_values, initial_p_values = initial_conditions(config, x_values, params)
        real_u_values, real_p_values = analytical_solution(config, x_values, t_values, params)

        x_random[start:stop, 0] = x_values
        t_random[start:stop, 0] = t_values
        source_u_random[start:stop, 0] = source_u_values
        source_p_random[start:stop, 0] = source_p_values
        initial_u_random[start:stop, 0] = initial_u_values
        initial_p_random[start:stop, 0] = initial_p_values
        real_u[start:stop, 0] = real_u_values
        real_p[start:stop, 0] = real_p_values

    branch_inputs = [source_u_branch, source_p_branch, initial_u_branch, initial_p_branch]
    random_inputs = [
        x_random,
        t_random,
        source_u_random,
        source_p_random,
        initial_u_random,
        initial_p_random,
    ]
    return branch_inputs, random_inputs, [real_u, real_p]
