import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

from poroelasticity.analytical import BiotConfig
from poroelasticity.neural.losses import compute_loss
from poroelasticity.neural.model import create_model
from poroelasticity.neural.sampling import create_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Train the physics-informed MIONet for Biot consolidation.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-functions", type=int, default=8)
    parser.add_argument("--points-per-function", type=int, default=4)
    parser.add_argument("--validation-functions", type=int, default=16)
    parser.add_argument("--validation-points", type=int, default=64)
    parser.add_argument("--size-x", type=int, default=51)
    parser.add_argument("--size-t", type=int, default=101)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--displacement-learning-rate", type=float, default=None)
    parser.add_argument("--pressure-learning-rate", type=float, default=None)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=0.0, help="Legacy shared weight for both data losses.")
    parser.add_argument("--displacement-data-weight", type=float, default=None)
    parser.add_argument("--pressure-data-weight", type=float, default=None)
    parser.add_argument("--joint-warmup-epochs", type=int, default=100000)
    parser.add_argument("--displacement-steps", type=int, default=1)
    parser.add_argument("--pressure-steps", type=int, default=3)
    parser.add_argument("--elastic-modulus", type=float, default=1.0)
    parser.add_argument("--hydraulic-conductivity", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=0.5)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pi_mionet"))
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def append_metrics(path: Path, row):
    fieldnames = list(row)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def validate_resume_configuration(saved, current):
    immutable_keys = (
        "num_functions",
        "points_per_function",
        "validation_functions",
        "validation_points",
        "size_x",
        "size_t",
        "neurons",
        "hidden_layers",
        "learning_rate",
        "displacement_learning_rate",
        "pressure_learning_rate",
        "clip_norm",
        "data_weight",
        "displacement_data_weight",
        "pressure_data_weight",
        "joint_warmup_epochs",
        "displacement_steps",
        "pressure_steps",
        "elastic_modulus",
        "hydraulic_conductivity",
        "length",
        "final_time",
        "dtype",
        "seed",
        "formulation",
    )
    mismatches = [key for key in immutable_keys if saved.get(key) != current.get(key)]
    if mismatches:
        details = ", ".join(f"{key}: {saved.get(key)!r} != {current.get(key)!r}" for key in mismatches)
        raise ValueError(f"resume configuration does not match the saved run: {details}")


def main():
    args = parse_args()

    import tensorflow as tf

    if args.displacement_data_weight is None:
        args.displacement_data_weight = args.data_weight
    if args.pressure_data_weight is None:
        args.pressure_data_weight = args.data_weight
    if args.displacement_learning_rate is None:
        args.displacement_learning_rate = args.learning_rate
    if args.pressure_learning_rate is None:
        args.pressure_learning_rate = args.learning_rate
    if min(args.displacement_data_weight, args.pressure_data_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if min(args.learning_rate, args.displacement_learning_rate, args.pressure_learning_rate) <= 0:
        raise ValueError("learning rates must be positive")
    if args.joint_warmup_epochs < 0:
        raise ValueError("joint warmup epochs must be non-negative")
    if min(args.displacement_steps, args.pressure_steps) < 1:
        raise ValueError("displacement and pressure steps must be positive")
    if args.resume and args.initial_weights is not None:
        raise ValueError("--initial-weights cannot be combined with --resume")
    if args.initial_weights is not None and not args.initial_weights.exists():
        raise FileNotFoundError(f"initial weights not found: {args.initial_weights}")

    tf.keras.backend.set_floatx(args.dtype)
    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    config = BiotConfig(
        length=args.length,
        final_time=args.final_time,
        elastic_modulus=args.elastic_modulus,
        hydraulic_conductivity=args.hydraulic_conductivity,
    )
    if config.length <= 0:
        raise ValueError("length must be positive")
    if config.hydraulic_conductivity <= 0:
        raise ValueError("the mixed Darcy formulation requires positive hydraulic conductivity")
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    x_domain = np.linspace(0.0, config.length, num=args.size_x, dtype=np_dtype)
    t_domain = np.linspace(0.0, config.final_time, num=args.size_t, dtype=np_dtype).reshape((args.size_t, 1))
    x_grid = np.tile(x_domain, (args.size_t, 1))

    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration["formulation"] = "mixed_darcy"
    configuration["tensorflow_version"] = tf.__version__
    configuration["gpu_devices"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    config_path = args.output_dir / "config.json"
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"no saved configuration found under {args.output_dir}")
        saved_configuration = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_configuration(saved_configuration, configuration)
    elif config_path.exists() or (args.output_dir / "metrics.csv").exists():
        raise FileExistsError(f"run directory already contains results: {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    model = create_model(
        config,
        args.size_x,
        args.size_t,
        args.neurons,
        hidden_layers=args.hidden_layers,
    )
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
    if len(displacement_variables) + len(pressure_flux_variables) != len(model.trainable_variables):
        raise RuntimeError("unable to partition model variables into displacement and pressure/flux heads")
    if args.initial_weights is not None:
        model.load_weights(args.initial_weights)
        print(f"initialized_weights={args.initial_weights}")
    optimizer_u = tf.keras.optimizers.Adam(args.displacement_learning_rate)
    optimizer_p = tf.keras.optimizers.Adam(args.pressure_learning_rate)
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int64),
        best_validation=tf.Variable(float("inf"), dtype=tf.float32),
        best_validation_score=tf.Variable(float("inf"), dtype=tf.float32),
        optimizer_u=optimizer_u,
        optimizer_p=optimizer_p,
        model=model,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(args.output_dir / "checkpoints"), max_to_keep=3)
    if args.resume:
        if not manager.latest_checkpoint:
            raise FileNotFoundError(f"no checkpoint found under {args.output_dir}")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"restored={manager.latest_checkpoint} epoch={int(checkpoint.epoch.numpy())}")

    validation_batch = create_sample(
        config,
        args.size_x,
        args.size_t,
        args.validation_functions,
        args.validation_points,
        x_grid,
        t_domain,
        rng=np.random.default_rng(args.seed + 1_000_000),
        dtype=np_dtype,
    )
    validation_initial_u = validation_batch[0][2]
    validation_initial_p = validation_batch[0][3]
    displacement_scale = max(float(np.sqrt(np.mean(validation_initial_u**2))), np.finfo(float).tiny)
    pressure_scale = max(float(np.sqrt(np.mean(validation_initial_p**2))), np.finfo(float).tiny)
    mechanics_scale = max(abs(config.elastic_modulus) * displacement_scale, np.finfo(float).tiny)
    pressure_mechanics_ratio = pressure_scale * config.length / mechanics_scale
    pressure_flow_ratio = (
        abs(config.hydraulic_conductivity)
        * pressure_scale
        * config.final_time
        / (displacement_scale * config.length)
    )
    if (
        args.pressure_data_weight == 0 and max(pressure_mechanics_ratio, pressure_flow_ratio) < 1e-3
    ):
        warnings.warn(
            "pressure is weakly identifiable from the physics residuals at these scales; "
            "the mixed Darcy formulation improves conditioning but does not change the physical coupling; "
            "report pressure and flux validation across multiple seeds",
            stacklevel=2,
        )

    def evaluate_training_loss(branch_inputs, random_inputs, real_solution, training=True):
        return compute_loss(
            config,
            model,
            branch_inputs,
            random_inputs,
            real_solution,
            data_weight=args.data_weight,
            displacement_data_weight=args.displacement_data_weight,
            pressure_data_weight=args.pressure_data_weight,
            training=training,
        )

    @tf.function(reduce_retracing=True)
    def joint_train_step(branch_inputs, random_inputs, real_solution):
        with tf.GradientTape() as tape:
            loss_total, loss_components, relative_errors = evaluate_training_loss(
                branch_inputs, random_inputs, real_solution
            )
        variables = displacement_variables + pressure_flux_variables
        gradients = tape.gradient(loss_total, variables)
        displacement_count = len(displacement_variables)
        displacement_gradients, displacement_gradient_norm = tf.clip_by_global_norm(
            gradients[:displacement_count], args.clip_norm
        )
        pressure_flux_gradients, pressure_flux_gradient_norm = tf.clip_by_global_norm(
            gradients[displacement_count:], args.clip_norm
        )
        gradient_norm = tf.linalg.global_norm(gradients)
        tf.debugging.check_numerics(loss_total, "non-finite training loss")
        optimizer_u.apply_gradients(zip(displacement_gradients, displacement_variables))
        optimizer_p.apply_gradients(zip(pressure_flux_gradients, pressure_flux_variables))
        return (
            loss_total,
            loss_components,
            relative_errors,
            gradient_norm,
            displacement_gradient_norm,
            pressure_flux_gradient_norm,
        )

    @tf.function(reduce_retracing=True)
    def displacement_train_step(branch_inputs, random_inputs, real_solution):
        with tf.GradientTape() as tape:
            loss_total, loss_components, relative_errors = evaluate_training_loss(
                branch_inputs, random_inputs, real_solution
            )
        gradients = tape.gradient(loss_total, displacement_variables)
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, args.clip_norm)
        tf.debugging.check_numerics(loss_total, "non-finite displacement-step loss")
        optimizer_u.apply_gradients(zip(clipped_gradients, displacement_variables))
        return loss_total, loss_components, relative_errors, gradient_norm

    @tf.function(reduce_retracing=True)
    def pressure_train_step(branch_inputs, random_inputs, real_solution):
        with tf.GradientTape() as tape:
            loss_total, loss_components, relative_errors = evaluate_training_loss(
                branch_inputs, random_inputs, real_solution
            )
        gradients = tape.gradient(loss_total, pressure_flux_variables)
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, args.clip_norm)
        tf.debugging.check_numerics(loss_total, "non-finite pressure-step loss")
        optimizer_p.apply_gradients(zip(clipped_gradients, pressure_flux_variables))
        return loss_total, loss_components, relative_errors, gradient_norm

    start_epoch = int(checkpoint.epoch.numpy()) + 1
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            training_batch = create_sample(
                config,
                args.size_x,
                args.size_t,
                args.num_functions,
                args.points_per_function,
                x_grid,
                t_domain,
                rng=np.random.default_rng(args.seed + epoch),
                dtype=np_dtype,
            )
            if epoch <= args.joint_warmup_epochs:
                training_phase = "joint"
                displacement_updates = 1
                pressure_updates = 1
                (
                    loss_total,
                    components,
                    relative_errors,
                    gradient_norm,
                    displacement_gradient_norm,
                    pressure_flux_gradient_norm,
                ) = joint_train_step(*training_batch)
            else:
                training_phase = "alternating"
                displacement_updates = args.displacement_steps
                pressure_updates = args.pressure_steps
                for _ in range(args.displacement_steps):
                    loss_total, components, relative_errors, displacement_gradient_norm = displacement_train_step(
                        *training_batch
                    )
                for _ in range(args.pressure_steps):
                    loss_total, components, relative_errors, pressure_flux_gradient_norm = pressure_train_step(
                        *training_batch
                    )
                gradient_norm = tf.sqrt(
                    tf.square(displacement_gradient_norm) + tf.square(pressure_flux_gradient_norm)
                )
            checkpoint.epoch.assign(epoch)

            validation_loss = None
            validation_errors = None
            validation_score = None
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                validation_loss, _, validation_errors = compute_loss(
                    config,
                    model,
                    *validation_batch,
                    data_weight=args.data_weight,
                    displacement_data_weight=args.displacement_data_weight,
                    pressure_data_weight=args.pressure_data_weight,
                )
                if validation_loss < tf.cast(checkpoint.best_validation, validation_loss.dtype):
                    checkpoint.best_validation.assign(tf.cast(validation_loss, tf.float32))
                    model.save_weights(args.output_dir / "best.objective.weights.h5")
                validation_score = 0.5 * (validation_errors[0] + validation_errors[1])
                if validation_score < tf.cast(checkpoint.best_validation_score, validation_score.dtype):
                    checkpoint.best_validation_score.assign(tf.cast(validation_score, tf.float32))
                    model.save_weights(args.output_dir / "best.weights.h5")

            if epoch % args.log_every == 0 or validation_loss is not None or epoch == args.epochs:
                row = {
                    "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - started,
                    "loss": float(loss_total),
                    "loss_pde_u": float(components[0]),
                    "loss_mass": float(components[1]),
                    "loss_darcy": float(components[2]),
                    "loss_boundary_left": float(components[3]),
                    "loss_boundary_right": float(components[4]),
                    "loss_initial": float(components[5]),
                    "loss_data": float(components[6]),
                    "loss_data_u": float(components[7]),
                    "loss_data_p": float(components[8]),
                    "training_phase": training_phase,
                    "displacement_updates": displacement_updates,
                    "pressure_updates": pressure_updates,
                    "relative_l2_u": float(relative_errors[0]),
                    "relative_l2_p": float(relative_errors[1]),
                    "relative_l2_q": float(relative_errors[2]),
                    "gradient_norm": float(gradient_norm),
                    "displacement_gradient_norm": float(displacement_gradient_norm),
                    "pressure_flux_gradient_norm": float(pressure_flux_gradient_norm),
                    "validation_loss": "" if validation_loss is None else float(validation_loss),
                    "validation_score": "" if validation_score is None else float(validation_score),
                    "validation_relative_l2_u": "" if validation_errors is None else float(validation_errors[0]),
                    "validation_relative_l2_p": "" if validation_errors is None else float(validation_errors[1]),
                    "validation_relative_l2_q": "" if validation_errors is None else float(validation_errors[2]),
                }
                append_metrics(args.output_dir / "metrics.csv", row)
                print(
                    f"epoch={epoch} loss={float(loss_total):.6e} "
                    f"rel_l2_u={float(relative_errors[0]):.4f} "
                    f"rel_l2_p={float(relative_errors[1]):.4f} "
                    f"rel_l2_q={float(relative_errors[2]):.4f}"
                )

            if epoch % args.checkpoint_every == 0:
                manager.save(checkpoint_number=epoch)
    finally:
        manager.save(checkpoint_number=int(checkpoint.epoch.numpy()))


if __name__ == "__main__":
    main()
