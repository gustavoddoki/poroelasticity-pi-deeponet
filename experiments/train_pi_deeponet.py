import argparse

import numpy as np

from poroelasticity.analytical import BiotConfig
from poroelasticity.neural.losses import compute_loss
from poroelasticity.neural.model import create_model
from poroelasticity.neural.sampling import create_sample


def train_step(config, model, optimizer, branch_inputs, random_inputs, batch_size, real_solution):
    import tensorflow as tf

    with tf.GradientTape() as tape:
        loss_total, loss_components, real_errors = compute_loss(
            config, model, branch_inputs, random_inputs, batch_size, real_solution
        )
    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_total, loss_components, real_errors


def main():
    parser = argparse.ArgumentParser(description="Train the PI-DeepONet model for the Biot consolidation problem.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size-x", type=int, default=51)
    parser.add_argument("--size-t", type=int, default=101)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    import tensorflow as tf

    tf.keras.backend.set_floatx("float64")
    config = BiotConfig()
    x_domain = np.linspace(0.0, config.length, num=args.size_x, dtype=np.float64)
    t_domain = np.linspace(0.0, config.final_time, num=args.size_t, dtype=np.float64).reshape((args.size_t, 1))
    x_grid = np.tile(x_domain, (args.size_t, 1))

    model = create_model(args.size_x, args.size_t, args.neurons)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        branch_inputs, random_inputs, real_solution = create_sample(
            config, args.size_x, args.size_t, args.batch_size, x_grid, t_domain
        )
        loss_total, _, real_errors = train_step(
            config, model, optimizer, branch_inputs, random_inputs, args.batch_size, real_solution
        )
        print(
            f"epoch={epoch} loss={float(loss_total):.6e} "
            f"mape_u={float(real_errors[0]) * 100:.4f}% mape_p={float(real_errors[1]) * 100:.4f}%"
        )


if __name__ == "__main__":
    main()
