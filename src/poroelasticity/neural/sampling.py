import random

import numpy as np

from poroelasticity.analytical import BiotConfig, ManufacturedParameters, analytical_solution, initial_conditions, source_terms


def random_manufactured_parameters() -> ManufacturedParameters:
    return ManufacturedParameters(
        displacement_amplitude=random.uniform(0.5, 3.0),
        pressure_amplitude=random.uniform(0.5, 3.0),
        displacement_decay=random.uniform(0.5, 2.0),
        pressure_decay=random.uniform(0.5, 2.0),
        displacement_shift=random.uniform(-2.0, 0.0),
        pressure_shift=random.uniform(-2.0, 0.0),
    )


def create_sample(config: BiotConfig, size_x: int, size_t: int, batch_size: int, x_grid, t_grid):
    """Create one PI-DeepONet training batch."""

    u_branch = []
    p_branch = []
    u0_branch = []
    p0_branch = []
    x_random = []
    t_random = []
    source_u_random = []
    source_p_random = []
    u0_random = []
    p0_random = []
    u_real = []
    p_real = []

    for _ in range(batch_size):
        params = random_manufactured_parameters()
        source_u, source_p = source_terms(config, x_grid, t_grid, params)
        initial_u, initial_p = initial_conditions(config, x_grid[0], params)

        u_branch.append(source_u.reshape(size_x * size_t))
        p_branch.append(source_p.reshape(size_x * size_t))
        u0_branch.append(initial_u)
        p0_branch.append(initial_p)

        x_value = random.uniform(0.0, config.length)
        t_value = random.uniform(0.0, config.final_time)
        source_u_value, source_p_value = source_terms(config, x_value, t_value, params)
        initial_u_value, initial_p_value = initial_conditions(config, x_value, params)
        real_u_value, real_p_value = analytical_solution(config, x_value, t_value, params)

        x_random.append([x_value])
        t_random.append([t_value])
        source_u_random.append([source_u_value])
        source_p_random.append([source_p_value])
        u0_random.append([initial_u_value])
        p0_random.append([initial_p_value])
        u_real.append([real_u_value])
        p_real.append([real_p_value])

    branch_inputs = [np.array(u_branch), np.array(p_branch), np.array(u0_branch), np.array(p0_branch)]
    random_inputs = [
        np.array(x_random),
        np.array(t_random),
        np.array(source_u_random),
        np.array(source_p_random),
        np.array(u0_random),
        np.array(p0_random),
    ]
    real_solution = [np.array(u_real), np.array(p_real)]
    return branch_inputs, random_inputs, real_solution
