
"""
compare_models.py — Unified comparison across Greedy, Simulated Annealing, LP, CP-SAT, and QUBO.

This orchestrator:
  - Loads warehouses/customers/distance matrix
  - Runs available solvers (toggle via CLI flags)
  - Harmonizes metrics: total_cost, total_distance_metric, feasibility, runtime
  - Produces a bar chart and a printed summary.

Greedy & SA hooks:
  - If your repo has 'route_greedy.py' exposing a function
      run(warehouses_df, customers_df, distances_df, objective) -> results_dict
    with keys like in LP output, we will call it.
  - If you have a 'sa_solver.py' exposing
      run_sa(warehouses_df, customers_df, distances_df, objective, **kwargs) -> results_dict
    we will call it.
  - Otherwise we skip and warn.

QUBO:
  - If --run_qubo, we will call qubo_model.build_qubo and (optionally) sample with neal if installed.

Usage:
  python compare_models.py \
      --warehouses warehouses.csv \
      --customers customers.csv \
      --distances distance_matrix.csv \
      --objective distance_times_demand \
      --run_greedy --run_sa --run_lp --run_cp --run_qubo \
      --time_limit_s 30 \
      --outdir output

Dependencies: pandas, numpy, matplotlib; plus respective solver deps.
"""
from __future__ import annotations
print(">>> THIS IS THE CORRECT compare_models.py VERSION <<<")

import argparse
import importlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_entities(warehouses_path: Path, customers_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wh = pd.read_csv(warehouses_path).sort_values("Warehouse_ID").reset_index(drop=True)
    cu = pd.read_csv(customers_path).sort_values("Customer_ID").reset_index(drop=True)
    return wh, cu


def load_distance_matrix(distances_path: Path, n_w: int, n_c: int) -> pd.DataFrame:
    try:
        raw = pd.read_csv(distances_path, header=None)
        if raw.shape == (n_w, n_c):
            return raw.astype(float)
    except Exception:
        pass
    df = pd.read_csv(distances_path, header=0)
    if not np.issubdtype(df.dtypes.iloc[0], np.number):
        df = df.drop(df.columns[0], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape != (n_w, n_c):
        raise ValueError(f"distance_matrix.csv shape mismatch. Expected ({n_w} x {n_c}), got {df.shape}.")
    return df.astype(float)


def ensure_metrics_dict(method_name: str, feasible: bool, total_cost: float, total_distance: float, runtime_sec: float) -> Dict:
    return {
        "method": method_name,
        "feasible": bool(feasible),
        "total_cost": float(total_cost),
        "total_distance": float(total_distance),
        "runtime_sec": float(runtime_sec),
    }


def try_run_greedy(wh: pd.DataFrame, cu: pd.DataFrame, D: pd.DataFrame, objective: str) -> Dict | None:
    try:
        rg = importlib.import_module("route_greedy")
        t0 = time.time()
        res = rg.run(wh, cu, D, objective=objective)  # expected to return dict like LP
        runtime = time.time() - t0
        # Harmonize fields
        return {
            "method_name": "Greedy",
            "feasible": True,  # greedy typically respects capacity if implemented so; override if provided
            "total_cost": float(res.get("total_cost", np.nan)),
            "total_distance": float(res.get("total_distance", np.nan)),
            "assignment": res.get("assignment"),
            "capacity_usage": res.get("capacity_usage"),
            "runtime_sec": float(res.get("runtime_sec", runtime)),
        }
    except Exception as e:
        logging.warning(f"Greedy module not available or incompatible: {e}")
        return None


def try_run_sa(wh: pd.DataFrame, cu: pd.DataFrame, D: pd.DataFrame, objective: str, sa_kwargs: Dict) -> Dict | None:
    # Expect sa_solver.run_sa(...)
    try:
        sa = importlib.import_module("sa_solver")
        t0 = time.time()
        res = sa.run_sa(wh, cu, D, objective=objective, **sa_kwargs)
        runtime = time.time() - t0
        return {
            "method_name": "SimulatedAnnealing",
            "feasible": bool(res.get("feasible", True)),
            "total_cost": float(res.get("total_cost", np.nan)),
            "total_distance": float(res.get("total_distance", np.nan)),
            "assignment": res.get("assignment"),
            "capacity_usage": res.get("capacity_usage"),
            "runtime_sec": float(res.get("runtime_sec", runtime)),
        }
    except Exception as e:
        logging.warning(f"SA module not available or incompatible: {e}")
        return None


def run_lp(wh: pd.DataFrame, cu: pd.DataFrame, D: pd.DataFrame, objective: str, time_limit_s: int | None) -> Dict:
    from benchmark_lp import solve_lp  # local import to avoid heavy deps if not used
    return solve_lp(wh, cu, D, objective=objective, solver="CBC", time_limit_s=time_limit_s)


def run_cp(wh: pd.DataFrame, cu: pd.DataFrame, D: pd.DataFrame, objective: str, time_limit_s: int | None) -> Dict:
    from benchmark_constraint_solver import solve_cp_sat
    return solve_cp_sat(wh, cu, D, objective=objective, time_limit_s=time_limit_s)


def run_qubo(
    wh: pd.DataFrame,
    cu: pd.DataFrame,
    D: pd.DataFrame,
    objective: str,
    lambda_assign: float,
    lambda_capacity: float,
    granularity: float,
    solve_qubo: bool,
    sa_reads: int,
    seed: int,
) -> Dict:
    from qubo_model import build_qubo, decode_solution_to_assignment
    import dimod

    t0 = time.time()
    bqm, meta = build_qubo(
        warehouses=wh,
        customers=cu,
        distances=D,
        lambda_assign=lambda_assign,
        lambda_capacity=lambda_capacity,
        granularity=granularity,
        objective=objective,
    )
    build_time = time.time() - t0

    if not solve_qubo:
        return {
            "method_name": "QUBO(unsolved)",
            "feasible": False,
            "total_cost": np.nan,
            "total_distance": np.nan,
            "assignment": None,
            "capacity_usage": None,
            "runtime_sec": build_time,
        }

    # Try neal sampler
    try:
        import neal  # type: ignore
        sampler = neal.SimulatedAnnealingSampler()
        t1 = time.time()
        sampleset = sampler.sample(bqm, num_reads=sa_reads, seed=seed)
        solve_time = time.time() - t1
        best = sampleset.first
        decoded = decode_solution_to_assignment(bqm, best.sample, wh, cu, D, objective)
        return {
            "method_name": "QUBO+SA",
            "feasible": decoded["feasible"],
            "total_cost": decoded["total_cost"],
            "total_distance": decoded["total_distance"],
            "assignment": decoded["assignment"],
            "capacity_usage": decoded["capacity_usage"],
            "runtime_sec": build_time + solve_time,
        }
    except Exception as e:
        logging.warning(f"neal not available for QUBO sampling: {e}")
        return {
            "method_name": "QUBO(unsolved)",
            "feasible": False,
            "total_cost": np.nan,
            "total_distance": np.nan,
            "assignment": None,
            "capacity_usage": None,
            "runtime_sec": build_time,
        }


def plot_bars(metrics: List[Dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [m["method"] if "method" in m else m["method_name"] for m in metrics]
    cost_vals = [m["total_cost"] for m in metrics]
    dist_vals = [m["total_distance"] for m in metrics]
    time_vals = [m["runtime_sec"] for m in metrics]

    # One chart per metric
    plt.figure()
    plt.bar(labels, cost_vals)
    plt.ylabel("Total Cost (objective)")
    plt.title("Model Comparison — Total Cost")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "compare_total_cost.png", dpi=200)

    plt.figure()
    plt.bar(labels, dist_vals)
    plt.ylabel("Total Distance Metric")
    plt.title("Model Comparison — Total Distance Metric")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "compare_total_distance.png", dpi=200)

    plt.figure()
    plt.bar(labels, time_vals)
    plt.ylabel("Runtime (sec)")
    plt.title("Model Comparison — Runtime")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "compare_runtime.png", dpi=200)


def main():
    print(">>> INSIDE main()")
    parser = argparse.ArgumentParser(description="Compare Greedy, SA, LP, CP-SAT, and QUBO assignment models.")
    parser.add_argument("--warehouses", type=Path, required=True)
    parser.add_argument("--customers", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--objective", choices=["distance_only", "distance_times_demand"], default="distance_times_demand")

    parser.add_argument("--run_greedy", action="store_true")
    parser.add_argument("--run_sa", action="store_true")
    parser.add_argument("--run_lp", action="store_true")
    parser.add_argument("--run_cp", action="store_true")
    parser.add_argument("--run_qubo", action="store_true")
    parser.add_argument("--solve_qubo", action="store_true")

    parser.add_argument("--lambda_assign", type=float, default=4.0)
    parser.add_argument("--lambda_capacity", type=float, default=4.0)
    parser.add_argument("--granularity", type=float, default=10.0)
    parser.add_argument("--sa_reads", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--time_limit_s", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    configure_logging(args.verbose)

    wh, cu = load_entities(args.warehouses, args.customers)
    D = load_distance_matrix(args.distances, len(wh), len(cu))

    results: List[Dict] = []

    if args.run_greedy:
        r = try_run_greedy(wh, cu, D, args.objective)
        if r:
            results.append(r)

    if args.run_sa:
        r = try_run_sa(wh, cu, D, args.objective, sa_kwargs={"seed": args.seed})
        if r:
            results.append(r)

    if args.run_lp:
        try:
            r = run_lp(wh, cu, D, args.objective, args.time_limit_s)
            results.append(r)
        except Exception as e:
            logging.error(f"LP run failed: {e}")

    if args.run_cp:
        try:
            r = run_cp(wh, cu, D, args.objective, args.time_limit_s)
            results.append(r)
        except Exception as e:
            logging.error(f"CP-SAT run failed: {e}")

    if args.run_qubo:
        try:
            r = run_qubo(
                wh,
                cu,
                D,
                args.objective,
                args.lambda_assign,
                args.lambda_capacity,
                args.granularity,
                args.solve_qubo,
                args.sa_reads,
                args.seed,
            )
            results.append(r)
        except Exception as e:
            logging.error(f"QUBO build/solve failed: {e}")

    if not results:
        print("No models were executed. Enable flags such as --run_lp, --run_cp, --run_qubo, etc.")
        return

    # Summarize
    rows = []
    for r in results:
        rows.append(
            {
                "method": r.get("method_name", "Unknown"),
                "feasible": r.get("feasible", False),
                "total_cost": r.get("total_cost", np.nan),
                "total_distance": r.get("total_distance", np.nan),
                "runtime_sec": r.get("runtime_sec", np.nan),
            }
        )
    summary = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.outdir / "comparison_summary.csv", index=False)

    # Plots
    plot_bars(rows, args.outdir)

    # Console
    print("\n=== Model Comparison Summary ===")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()