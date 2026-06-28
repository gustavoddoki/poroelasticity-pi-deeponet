import argparse

from poroelasticity.analytical import BiotConfig, ManufacturedParameters, analytical_solution, mean_absolute_percentage_error
from poroelasticity.numerical.gauss_seidel import solve_gauss_seidel


def main():
    parser = argparse.ArgumentParser(description="Run the Gauss-Seidel baseline for the Biot consolidation problem.")
    parser.add_argument("--spatial-cells", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    config = BiotConfig(length=0.5, final_time=1.0, elastic_modulus=1.0, hydraulic_conductivity=1.0)
    params = ManufacturedParameters()
    result = solve_gauss_seidel(
        config,
        params,
        spatial_cells=args.spatial_cells,
        time_steps=args.time_steps,
        tolerance=args.tolerance,
    )
    expected_u, expected_p = analytical_solution(config, result.x, config.final_time, params)

    print(f"converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"MAPE displacement: {mean_absolute_percentage_error(expected_u, result.displacement):.4f}%")
    print(f"MAPE pressure: {mean_absolute_percentage_error(expected_p, result.pressure):.4f}%")


if __name__ == "__main__":
    main()
