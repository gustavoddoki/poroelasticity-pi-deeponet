import numpy as np
import pytest

from poroelasticity.analytical import BiotConfig
from poroelasticity.neural.losses import compute_loss
from poroelasticity.neural.model import create_model
from poroelasticity.neural.sampling import create_sample


tf = pytest.importorskip("tensorflow")


def test_mionet_forward_and_physics_loss_are_finite_for_realistic_scales():
    tf.keras.backend.set_floatx("float32")
    tf.keras.utils.set_random_seed(5)
    config = BiotConfig(elastic_modulus=1e8, hydraulic_conductivity=1e-5)
    size_x = 3
    size_t = 4
    x_domain = np.linspace(0.0, config.length, size_x, dtype=np.float32)
    t_domain = np.linspace(0.0, config.final_time, size_t, dtype=np.float32).reshape((size_t, 1))
    x_grid = np.tile(x_domain, (size_t, 1))
    batch = create_sample(
        config,
        size_x,
        size_t,
        num_functions=2,
        points_per_function=2,
        x_grid=x_grid,
        t_grid=t_domain,
        rng=np.random.default_rng(5),
        dtype=np.float32,
    )
    model = create_model(config, size_x, size_t, neurons_per_layer=8, hidden_layers=2)

    loss, components, relative_errors = compute_loss(config, model, *batch)

    assert model.name == "physics_informed_mionet_biot"
    assert model.count_params() > 0
    assert np.isfinite(float(loss))
    assert all(np.isfinite(float(value)) for value in components)
    assert len(components) == 8
    assert all(np.isfinite(float(value)) for value in relative_errors)

    weighted_loss, weighted_components, _ = compute_loss(
        config,
        model,
        *batch,
        displacement_data_weight=0.5,
        pressure_data_weight=2.0,
    )
    expected = loss + 0.5 * weighted_components[6] + 2.0 * weighted_components[7]
    np.testing.assert_allclose(float(weighted_loss), float(expected), rtol=1e-5)


def test_mionet_weights_round_trip(tmp_path):
    tf.keras.backend.set_floatx("float32")
    config = BiotConfig()
    model = create_model(config, size_x=3, size_t=4, neurons_per_layer=4, hidden_layers=1)
    path = tmp_path / "model.weights.h5"

    model.save_weights(path)
    restored = create_model(config, size_x=3, size_t=4, neurons_per_layer=4, hidden_layers=1)
    restored.load_weights(path)

    for expected, actual in zip(model.get_weights(), restored.get_weights()):
        np.testing.assert_array_equal(expected, actual)
