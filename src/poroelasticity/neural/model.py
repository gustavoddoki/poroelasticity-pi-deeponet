def create_model(size_x: int, size_t: int, neurons_per_layer: int):
    """Create the two-output DeepONet architecture used for displacement and pressure."""

    import tensorflow as tf

    branch_input_u0 = tf.keras.Input(shape=(size_x,), name="branch_input_u0")
    branch_input_p0 = tf.keras.Input(shape=(size_x,), name="branch_input_p0")
    branch_input_u = tf.keras.Input(shape=(size_t * size_x,), name="branch_input_U")
    branch_input_p = tf.keras.Input(shape=(size_t * size_x,), name="branch_input_P")
    trunk_input = tf.keras.Input(shape=(2,), name="trunk_input")

    def deep_operator_output(prefix: str):
        branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_0_u0")(branch_input_u0)
        branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_0_p0")(branch_input_p0)
        branch_u = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_0_U")(branch_input_u)
        branch_p = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_0_P")(branch_input_p)
        trunk = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_trunk_0")(trunk_input)

        for i in range(4):
            layer = i + 1
            branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_{layer}_u0")(branch_u0)
            branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_{layer}_p0")(branch_p0)
            branch_u = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_{layer}_U")(branch_u)
            branch_p = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_{layer}_P")(branch_p)
            trunk = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_trunk_{layer}")(trunk)

        branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_out_u0")(branch_u0)
        branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_out_p0")(branch_p0)
        branch_u = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_out_U")(branch_u)
        branch_p = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_branch_out_P")(branch_p)
        trunk = tf.keras.layers.Dense(neurons_per_layer, activation="tanh", name=f"{prefix}_trunk_out")(trunk)
        merged = tf.keras.layers.Multiply(name=f"{prefix}_operator_product")([branch_u0, branch_p0, branch_u, branch_p, trunk])
        return tf.keras.layers.Lambda(
            lambda tensor: tf.reduce_sum(tensor, axis=1, keepdims=True),
            name=f"{prefix}_operator_output",
        )(merged)

    displacement = deep_operator_output("u_net")
    pressure = deep_operator_output("p_net")

    return tf.keras.Model(
        inputs=[branch_input_u0, branch_input_p0, branch_input_u, branch_input_p, trunk_input],
        outputs=[displacement, pressure],
        name="pi_deeponet_biot",
    )
