import numpy as np

from poroelasticity.analytical import BiotConfig
from poroelasticity.neural.sampling import create_sample


def test_sample_reuses_each_function_across_collocation_points():
    config = BiotConfig()
    size_x = 3
    size_t = 4
    x_domain = np.linspace(0.0, config.length, size_x)
    t_domain = np.linspace(0.0, config.final_time, size_t).reshape((size_t, 1))
    x_grid = np.tile(x_domain, (size_t, 1))

    batch = create_sample(
        config,
        size_x,
        size_t,
        num_functions=2,
        points_per_function=3,
        x_grid=x_grid,
        t_grid=t_domain,
        rng=np.random.default_rng(7),
        dtype=np.float32,
    )
    branch_inputs, random_inputs, real_solution = batch

    assert branch_inputs[0].shape == (6, size_x * size_t)
    assert branch_inputs[2].shape == (6, size_x)
    assert random_inputs[0].shape == (6, 1)
    assert real_solution[0].shape == (6, 1)
    assert branch_inputs[0].dtype == np.float32
    np.testing.assert_array_equal(branch_inputs[0][0], branch_inputs[0][1])
    np.testing.assert_array_equal(branch_inputs[0][1], branch_inputs[0][2])
    assert not np.array_equal(branch_inputs[0][2], branch_inputs[0][3])


def test_sample_is_reproducible_from_seed():
    config = BiotConfig()
    x_domain = np.linspace(0.0, config.length, 3)
    t_domain = np.linspace(0.0, config.final_time, 4).reshape((4, 1))
    x_grid = np.tile(x_domain, (4, 1))

    first = create_sample(config, 3, 4, 2, 2, x_grid, t_domain, rng=np.random.default_rng(11))
    second = create_sample(config, 3, 4, 2, 2, x_grid, t_domain, rng=np.random.default_rng(11))

    for first_group, second_group in zip(first, second):
        for first_value, second_value in zip(first_group, second_group):
            np.testing.assert_array_equal(first_value, second_value)
