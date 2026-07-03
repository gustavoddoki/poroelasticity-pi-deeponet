from typing import Optional

from poroelasticity.analytical import NonDimensionalScales


def compute_loss(
    config,
    model,
    branch_inputs,
    random_inputs,
    real_solution,
    data_weight=0.0,
    displacement_data_weight=None,
    pressure_data_weight=None,
    training=False,
    scales: Optional[NonDimensionalScales] = None,
):
    """Compute the mixed Biot physics, boundary, initial, and optional data losses.

    When ``scales`` is provided, only the displacement and mass conservation
    PDE residuals are rescaled by the characteristic factors
    ``L**2 / (E * u_c)`` and ``L**3 / (K * E * u_c)`` so that the two
    contributions to each residual have unit order of magnitude. This
    removes the gradient damping caused by the small hydraulic conductivity
    on the pressure side and the large elastic modulus on the displacement
    side.

    The boundary, initial-condition, and data losses stay in physical units
    so that direct supervision remains visible even when ``p_c`` is much
    larger than the physical pressure magnitudes: dividing those losses by
    ``p_c`` would push them below fp32 precision and prevent the network
    from learning the pressure field.
    """

    import tensorflow as tf

    dtype = tf.as_dtype(model.compute_dtype)
    branch_inputs = [tf.convert_to_tensor(value, dtype=dtype) for value in branch_inputs]
    random_inputs = [tf.convert_to_tensor(value, dtype=dtype) for value in random_inputs]
    real_solution = [tf.convert_to_tensor(value, dtype=dtype) for value in real_solution]
    if displacement_data_weight is None:
        displacement_data_weight = data_weight
    if pressure_data_weight is None:
        pressure_data_weight = data_weight
    source_u_branch, source_p_branch, initial_u_branch, initial_p_branch = branch_inputs
    x, t, source_u, source_p, initial_u, initial_p = random_inputs

    e = tf.cast(config.elastic_modulus, dtype)
    k = tf.cast(config.hydraulic_conductivity, dtype)
    length = tf.cast(config.length, dtype)
    epsilon = tf.cast(tf.keras.backend.epsilon(), dtype)
    pressure_scale = tf.stop_gradient(
        tf.maximum(
            tf.sqrt(tf.reduce_mean(tf.square(initial_p_branch), axis=1, keepdims=True)),
            epsilon,
        )
    )
    flux_scale = tf.stop_gradient(tf.maximum(tf.abs(k) * pressure_scale / length, epsilon))

    if scales is None:
        displacement_factor = tf.cast(1.0, dtype)
        mass_factor = tf.cast(1.0, dtype)
    else:
        displacement_factor = (length**2) / (e * tf.cast(scales.displacement, dtype))
        mass_factor = (length**3) / (k * e * tf.cast(scales.displacement, dtype))

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
            u, p, normalized_q = model(
                {**model_inputs, "trunk_input": tf.concat([x, t], axis=-1)},
                training=training,
            )
        u_x = inner_tape.gradient(u, x)
        p_x = inner_tape.gradient(p, x)
        normalized_q_x = inner_tape.gradient(normalized_q, x)

    u_xx = outer_tape.gradient(u_x, x)
    u_xt = outer_tape.gradient(u_x, t)
    q = flux_scale * normalized_q
    q_x = flux_scale * normalized_q_x

    residual_u = -e * u_xx + p_x - source_u
    residual_mass = u_xt + q_x - source_p
    residual_darcy = normalized_q + k * p_x / flux_scale
    residual_u_normalized = residual_u * displacement_factor
    residual_mass_normalized = residual_mass * mass_factor
    scale_u = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(source_u))), 1.0))
    scale_p = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(source_p))), 1.0))
    loss_displacement_equation = tf.reduce_mean(tf.square(residual_u_normalized / scale_u))
    loss_mass_equation = tf.reduce_mean(tf.square(residual_mass_normalized / scale_p))
    loss_darcy_equation = tf.reduce_mean(tf.square(residual_darcy))

    sample_count = tf.shape(x)[0]
    x_left = tf.zeros((sample_count, 1), dtype=dtype)
    with tf.GradientTape() as tape:
        tape.watch(x_left)
        u_left, p_left, _ = model(
            {**model_inputs, "trunk_input": tf.concat([x_left, t], axis=-1)},
            training=training,
        )
    u_x_left = tape.gradient(u_left, x_left)
    loss_left_boundary = tf.reduce_mean(tf.square(length * u_x_left)) + tf.reduce_mean(tf.square(p_left))

    x_right = tf.fill((sample_count, 1), length)
    u_right, _, normalized_q_right = model(
        {**model_inputs, "trunk_input": tf.concat([x_right, t], axis=-1)},
        training=training,
    )
    loss_right_boundary = tf.reduce_mean(tf.square(u_right)) + tf.reduce_mean(tf.square(normalized_q_right))

    t_initial = tf.zeros((sample_count, 1), dtype=dtype)
    u_initial, p_initial, _ = model(
        {**model_inputs, "trunk_input": tf.concat([x, t_initial], axis=-1)},
        training=training,
    )
    loss_initial = tf.reduce_mean(tf.square(u_initial - initial_u)) + tf.reduce_mean(tf.square(p_initial - initial_p))

    true_u, true_p, true_q = real_solution
    scale_real_u = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(true_u))), epsilon))
    scale_real_p = tf.stop_gradient(tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(true_p))), epsilon))
    loss_data_u = tf.reduce_mean(tf.square((u - true_u) / scale_real_u))
    loss_data_p = tf.reduce_mean(tf.square((p - true_p) / scale_real_p))
    loss_data = loss_data_u + loss_data_p
    relative_l2_u = tf.sqrt(tf.reduce_sum(tf.square(u - true_u))) / tf.maximum(
        tf.sqrt(tf.reduce_sum(tf.square(true_u))), epsilon
    )
    relative_l2_p = tf.sqrt(tf.reduce_sum(tf.square(p - true_p))) / tf.maximum(
        tf.sqrt(tf.reduce_sum(tf.square(true_p))), epsilon
    )
    relative_l2_q = tf.sqrt(tf.reduce_sum(tf.square(q - true_q))) / tf.maximum(
        tf.sqrt(tf.reduce_sum(tf.square(true_q))), epsilon
    )

    loss_physics = loss_displacement_equation + loss_mass_equation + loss_darcy_equation
    loss_total = (
        loss_physics
        + loss_left_boundary
        + loss_right_boundary
        + loss_initial
        + tf.cast(displacement_data_weight, dtype) * loss_data_u
        + tf.cast(pressure_data_weight, dtype) * loss_data_p
    )
    components = [
        loss_displacement_equation,
        loss_mass_equation,
        loss_darcy_equation,
        loss_left_boundary,
        loss_right_boundary,
        loss_initial,
        loss_data,
        loss_data_u,
        loss_data_p,
    ]
    return loss_total, components, [relative_l2_u, relative_l2_p, relative_l2_q]
