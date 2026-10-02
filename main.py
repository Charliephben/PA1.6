import argparse
from scenarios.runner import run_monte_carlo, run_scenario

def main():
    parser = argparse.ArgumentParser(description="Run a room temperature scenario")
    parser.add_argument("--scenario", required=True, help="Path to YAML scenario file")
    parser.add_argument(
        "--monte-carlo",
        type=int,
        metavar="RUNS",
        help="Run the scenario RUNS times with consecutive seeds",
    )
    args = parser.parse_args()
    if args.monte_carlo is None:
        run_scenario(args.scenario)
    else:
        run_monte_carlo(args.scenario, args.monte_carlo)

if __name__ == "__main__":
    main()
