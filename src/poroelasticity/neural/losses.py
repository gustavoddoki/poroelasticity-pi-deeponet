def compute_loss(config, model, branch_inputs, random_inputs, batch_size, real_solution):
    """Compute physics-informed DeepONet losses for the Biot model."""

    import tensorflow as tf

    source_u_branch = tf.convert_to_tensor(branch_inputs[0], dtype=tf.float64)
    source_p_branch = tf.convert_to_tensor(branch_inputs[1], dtype=tf.float64)
    initial_u_branch = tf.convert_to_tensor(branch_inputs[2], dtype=tf.float64)
    initial_p_branch = tf.convert_to_tensor(branch_inputs[3], dtype=tf.float64)
    x = tf.convert_to_tensor(random_inputs[0], dtype=tf.float64)
    t = tf.convert_to_tensor(random_inputs[1], dtype=tf.float64)

    e = tf.cast(config.elastic_modulus, tf.float64)
    k = tf.cast(config.hydraulic_conductivity, tf.float64)

    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch([x, t])
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch([x, t])
            trunk_input = tf.concat([x, t], axis=-1)
            u, p = model(
                {
                    "branch_input_u0": initial_u_branch,
                    "branch_input_p0": initial_p_branch,
                    "branch_input_U": source_u_branch,
                    "branch_input_P": source_p_branch,
                    "trunk_input": trunk_input,
                }
            )
        u_x = inner_tape.gradient(u, x)
        p_x = inner_tape.gradient(p, x)

    u_xx = outer_tape.gradient(u_x, x)
    u_xt = outer_tape.gradient(u_x, t)
    p_xx = outer_tape.gradient(p_x, x)

    loss_displacement_equation = tf.reduce_mean(tf.square(-e * u_xx + p_x - random_inputs[2]))
    loss_pressure_equation = tf.reduce_mean(tf.square(u_xt - k * p_xx - random_inputs[3]))
    loss_equations = loss_displacement_equation / e**2 + loss_pressure_equation

    epsilon = tf.cast(tf.keras.backend.epsilon(), tf.float64)
    denominator_u = tf.maximum(tf.abs(real_solution[0]), epsilon)
    denominator_p = tf.maximum(tf.abs(real_solution[1]), epsilon)
    loss_u_real = tf.reduce_mean(tf.abs(u - real_solution[0]) / denominator_u)
    loss_p_real = tf.reduce_mean(tf.abs(p - real_solution[1]) / denominator_p)

    x_left = tf.zeros((batch_size, 1), dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_left, t])
        trunk_input = tf.concat([x_left, t], axis=-1)
        u_left, p_left = model(
            {
                "branch_input_u0": initial_u_branch,
                "branch_input_p0": initial_p_branch,
                "branch_input_U": source_u_branch,
                "branch_input_P": source_p_branch,
                "trunk_input": trunk_input,
            }
        )
    u_x_left = tape.gradient(u_left, x_left)
    loss_left_boundary = tf.reduce_mean(tf.square(u_x_left)) + tf.reduce_mean(tf.square(p_left))

    x_right = tf.fill((batch_size, 1), tf.cast(config.length, tf.float64))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_right, t])
        trunk_input = tf.concat([x_right, t], axis=-1)
        u_right, p_right = model(
            {
                "branch_input_u0": initial_u_branch,
                "branch_input_p0": initial_p_branch,
                "branch_input_U": source_u_branch,
                "branch_input_P": source_p_branch,
                "trunk_input": trunk_input,
            }
        )
    p_x_right = tape.gradient(p_right, x_right)
    loss_right_boundary = tf.reduce_mean(tf.square(u_right)) + tf.reduce_mean(tf.square(p_x_right))

    t_initial = tf.zeros((batch_size, 1), dtype=tf.float64)
    trunk_input = tf.concat([x, t_initial], axis=-1)
    u_initial, p_initial = model(
        {
            "branch_input_u0": initial_u_branch,
            "branch_input_p0": initial_p_branch,
            "branch_input_U": source_u_branch,
            "branch_input_P": source_p_branch,
            "trunk_input": trunk_input,
        }
    )
    loss_initial = tf.reduce_mean(tf.square(u_initial - random_inputs[4])) + tf.reduce_mean(tf.square(p_initial - random_inputs[5]))
    loss_total = loss_equations + loss_left_boundary + loss_right_boundary + loss_initial

    return (
        loss_total,
        [loss_displacement_equation / e**2, loss_pressure_equation, loss_left_boundary, loss_right_boundary, loss_initial],
        [loss_u_real, loss_p_real],
    )
