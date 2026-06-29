def compute_loss(
    config,
    model,
    branch_inputs,
    random_inputs,
    real_solution,
    data_weight=0.0,
    training=False,
):
    """Compute dimensionless physics, boundary, initial, and optional data losses."""

    import tensorflow as tf

    dtype = tf.as_dtype(model.compute_dtype)
    branch_inputs = [tf.convert_to_tensor(value, dtype=dtype) for value in branch_inputs]
    random_inputs = [tf.convert_to_tensor(value, dtype=dtype) for value in random_inputs]
    real_solution = [tf.convert_to_tensor(value, dtype=dtype) for value in real_solution]
    source_u_branch, source_p_branch, initial_u_branch, initial_p_branch = branch_inputs
    x, t, source_u, source_p, initial_u, initial_p = random_inputs

    e = tf.cast(config.elastic_modulus, dtype)
    k = tf.cast(config.hydraulic_conductivity, dtype)
    length = tf.cast(config.length, dtype)
    epsilon = tf.cast(tf.keras.backend.epsilon(), dtype)

    model_inputs = {
        "branch_input_u0": initial_u_branch,
        "branch_input_p0": initial_p_branch,
        "branch_input_U": source_u_branch,
        "branch_input_P": source_p_branch,
    }

    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch([x, t])
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch([x, t])
            u, p = model(
                {**model_inputs, "trunk_input": tf.concat([x, t], axis=-1)},
                training=training,
            )
        u_x = inner_tape.gradient(u, x)
        p_x = inner_tape.gradient(p, x)

    u_xx = outer_tape.gradient(u_x, x)
    u_xt = outer_tape.gradient(u_x, t)
    p_xx = outer_tape.gradient(p_x, x)

    residual_u = -e * u_xx + p_x - source_u
    residual_p = u_xt - k * p_xx - source_p
    scale_u = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(source_u))), 1.0))
    scale_p = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(source_p))), 1.0))
    loss_displacement_equation = tf.reduce_mean(tf.square(residual_u / scale_u))
    loss_pressure_equation = tf.reduce_mean(tf.square(residual_p / scale_p))

    sample_count = tf.shape(x)[0]
    x_left = tf.zeros((sample_count, 1), dtype=dtype)
    with tf.GradientTape() as tape:
        tape.watch(x_left)
        u_left, p_left = model(
            {**model_inputs, "trunk_input": tf.concat([x_left, t], axis=-1)},
            training=training,
        )
    u_x_left = tape.gradient(u_left, x_left)
    loss_left_boundary = tf.reduce_mean(tf.square(length * u_x_left)) + tf.reduce_mean(tf.square(p_left))

    x_right = tf.fill((sample_count, 1), length)
    with tf.GradientTape() as tape:
        tape.watch(x_right)
        u_right, p_right = model(
            {**model_inputs, "trunk_input": tf.concat([x_right, t], axis=-1)},
            training=training,
        )
    p_x_right = tape.gradient(p_right, x_right)
    loss_right_boundary = tf.reduce_mean(tf.square(u_right)) + tf.reduce_mean(tf.square(length * p_x_right))

    t_initial = tf.zeros((sample_count, 1), dtype=dtype)
    u_initial, p_initial = model(
        {**model_inputs, "trunk_input": tf.concat([x, t_initial], axis=-1)},
        training=training,
    )
    loss_initial = tf.reduce_mean(tf.square(u_initial - initial_u)) + tf.reduce_mean(tf.square(p_initial - initial_p))

    true_u, true_p = real_solution
    scale_real_u = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(true_u))), epsilon))
    scale_real_p = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(true_p))), epsilon))
    loss_data = tf.reduce_mean(tf.square((u - true_u) / scale_real_u)) + tf.reduce_mean(
        tf.square((p - true_p) / scale_real_p)
    )
    relative_l2_u = tf.sqrt(tf.reduce_sum(tf.square(u - true_u))) / tf.maximum(
        tf.sqrt(tf.reduce_sum(tf.square(true_u))), epsilon
    )
    relative_l2_p = tf.sqrt(tf.reduce_sum(tf.square(p - true_p))) / tf.maximum(
        tf.sqrt(tf.reduce_sum(tf.square(true_p))), epsilon
    )

    loss_physics = loss_displacement_equation + loss_pressure_equation
    loss_total = loss_physics + loss_left_boundary + loss_right_boundary + loss_initial + data_weight * loss_data
    components = [
        loss_displacement_equation,
        loss_pressure_equation,
        loss_left_boundary,
        loss_right_boundary,
        loss_initial,
        loss_data,
    ]
    return loss_total, components, [relative_l2_u, relative_l2_p]
