import numpy as np
import pytest

from poroelasticity.analytical import BiotConfig, non_dimensional_scales
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
    scales = non_dimensional_scales(config)
    model = create_model(config, size_x, size_t, neurons_per_layer=8, hidden_layers=2)

    loss, components, relative_errors = compute_loss(config, model, *batch, scales=scales)

    assert model.name == "mixed_physics_informed_mionet_biot"
    assert model.count_params() > 0
    assert np.isfinite(float(loss))
    assert all(np.isfinite(float(value)) for value in components)
    assert len(components) == 9
    assert len(relative_errors) == 3
    assert all(np.isfinite(float(value)) for value in relative_errors)

    weighted_loss, weighted_components, _ = compute_loss(
        config,
        model,
        *batch,
        displacement_data_weight=0.5,
        pressure_data_weight=2.0,
        scales=scales,
    )
    expected = loss + 0.5 * weighted_components[7] + 2.0 * weighted_components[8]
    np.testing.assert_allclose(float(weighted_loss), float(expected), rtol=1e-5)


def test_nondimensional_loss_uses_expected_residual_factors():
    tf.keras.backend.set_floatx("float32")
    tf.keras.utils.set_random_seed(11)
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
        rng=np.random.default_rng(11),
        dtype=np.float32,
    )

    scales = non_dimensional_scales(config)
    model = create_model(config, size_x, size_t, neurons_per_layer=8, hidden_layers=2)
    _, components, _ = compute_loss(config, model, *batch, scales=scales)
    displacement_loss, mass_loss = float(components[0]), float(components[1])

    assert np.isfinite(displacement_loss)
    assert np.isfinite(mass_loss)
    assert displacement_loss > 0.0
    assert mass_loss > 0.0

    displacement_factor = (config.length**2) / (config.elastic_modulus * scales.displacement)
    mass_factor = (config.length**3) / (config.hydraulic_conductivity * config.elastic_modulus * scales.displacement)
    assert mass_factor / displacement_factor == pytest.approx(config.length / config.hydraulic_conductivity)


def test_displacement_and_pressure_flux_heads_update_independently():
    tf.keras.backend.set_floatx("float32")
    tf.keras.utils.set_random_seed(7)
    config = BiotConfig()
    size_x = 3
    size_t = 4
    x_domain = np.linspace(0.0, config.length, size_x, dtype=np.float32)
    t_domain = np.linspace(0.0, config.final_time, size_t, dtype=np.float32).reshape((size_t, 1))
    batch = create_sample(
        config,
        size_x,
        size_t,
        num_functions=2,
        points_per_function=2,
        x_grid=np.tile(x_domain, (size_t, 1)),
        t_grid=t_domain,
        rng=np.random.default_rng(7),
        dtype=np.float32,
    )
    model = create_model(config, size_x, size_t, neurons_per_layer=4, hidden_layers=1)
    displacement_variables = tuple(
        variable
        for layer in model.layers
        if layer.name.startswith("u_net_")
        for variable in layer.trainable_variables
    )
    pressure_flux_variables = tuple(
        variable
        for layer in model.layers
        if layer.name.startswith(("p_net_", "q_net_"))
        for variable in layer.trainable_variables
    )
    assert len(displacement_variables) + len(pressure_flux_variables) == len(model.trainable_variables)

    pressure_flux_before = [variable.numpy().copy() for variable in pressure_flux_variables]
    with tf.GradientTape() as tape:
        loss, _, _ = compute_loss(config, model, *batch, training=True)
    gradients = tape.gradient(loss, displacement_variables)
    assert all(gradient is not None for gradient in gradients)
    tf.keras.optimizers.Adam(1e-5).apply_gradients(zip(gradients, displacement_variables))
    for expected, actual in zip(pressure_flux_before, pressure_flux_variables):
        np.testing.assert_array_equal(expected, actual.numpy())

    displacement_before = [variable.numpy().copy() for variable in displacement_variables]
    with tf.GradientTape() as tape:
        loss, _, _ = compute_loss(config, model, *batch, training=True)
    gradients = tape.gradient(loss, pressure_flux_variables)
    assert all(gradient is not None for gradient in gradients)
    tf.keras.optimizers.Adam(1e-5).apply_gradients(zip(gradients, pressure_flux_variables))
    for expected, actual in zip(displacement_before, displacement_variables):
        np.testing.assert_array_equal(expected, actual.numpy())


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
