from poroelasticity.analytical import BiotConfig


def create_model(
    config: BiotConfig,
    size_x: int,
    size_t: int,
    neurons_per_layer: int,
    hidden_layers: int = 5,
):
    """Create a mixed physics-informed MIONet for displacement, pressure, and flux."""

    import tensorflow as tf

    if hidden_layers < 1:
        raise ValueError("hidden_layers must be positive")

    branch_input_u0 = tf.keras.Input(shape=(size_x,), name="branch_input_u0")
    branch_input_p0 = tf.keras.Input(shape=(size_x,), name="branch_input_p0")
    branch_input_u = tf.keras.Input(shape=(size_t * size_x,), name="branch_input_U")
    branch_input_p = tf.keras.Input(shape=(size_t * size_x,), name="branch_input_P")
    trunk_input = tf.keras.Input(shape=(2,), name="trunk_input")

    source_u_scale = max(abs(config.elastic_modulus), 1.0)
    normalized_source_u = tf.keras.layers.Rescaling(
        1.0 / source_u_scale,
        name="normalize_source_U",
    )(branch_input_u)
    normalized_trunk = tf.keras.layers.Normalization(
        axis=-1,
        mean=[config.length / 2.0, config.final_time / 2.0],
        variance=[(config.length / 2.0) ** 2, (config.final_time / 2.0) ** 2],
        name="normalize_coordinates",
    )(trunk_input)

    def encoder(value, prefix: str):
        encoded = value
        for layer in range(hidden_layers):
            encoded = tf.keras.layers.Dense(
                neurons_per_layer,
                activation="tanh",
                name=f"{prefix}_{layer}",
            )(encoded)
        return tf.keras.layers.Dense(
            neurons_per_layer,
            activation=None,
            name=f"{prefix}_latent",
        )(encoded)

    def operator_output(prefix: str):
        branch_u0 = encoder(branch_input_u0, f"{prefix}_branch_u0")
        branch_p0 = encoder(branch_input_p0, f"{prefix}_branch_p0")
        branch_u = encoder(normalized_source_u, f"{prefix}_branch_U")
        branch_p = encoder(branch_input_p, f"{prefix}_branch_P")
        trunk = encoder(normalized_trunk, f"{prefix}_trunk")
        branch_product = tf.keras.layers.Multiply(name=f"{prefix}_branch_product")(
            [branch_u0, branch_p0, branch_u, branch_p]
        )
        operator_value = tf.keras.layers.Dot(axes=1, name=f"{prefix}_operator_dot")(
            [branch_product, trunk]
        )
        return tf.keras.layers.Dense(
            1,
            kernel_initializer="ones",
            bias_initializer="zeros",
            name=f"{prefix}_output",
        )(operator_value)

    displacement = operator_output("u_net")
    pressure = operator_output("p_net")
    normalized_flux = operator_output("q_net")

    return tf.keras.Model(
        inputs=[branch_input_u0, branch_input_p0, branch_input_u, branch_input_p, trunk_input],
        outputs=[displacement, pressure, normalized_flux],
        name="mixed_physics_informed_mionet_biot",
    )
