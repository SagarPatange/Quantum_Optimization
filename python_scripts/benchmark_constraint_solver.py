
"""
benchmark_constraint_solver.py — Constraint Programming baseline (OR-Tools CP-SAT).

Same model as LP baseline, solved with CP-SAT (binary variables + linear constraints).

Inputs / Outputs mirror benchmark_lp.py.

Usage:
  python benchmark_constraint_solver.py \
      --warehouses warehouses.csv \
      --customers customers.csv \
      --distances distance_matrix.csv \
      --objective distance_times_demand \
      --time_limit_s 30 \
      --outdir output

Dependencies: pandas, numpy, ortools
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
except ImportError as e:
    raise SystemExit("Missing dependency 'ortools'. Install with: pip install ortools") from e


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


def build_cost_matrix(D: np.ndarray, demand: np.ndarray, objective: str) -> np.ndarray:
    if objective == "distance_only":
        return D.copy()
    elif objective == "distance_times_demand":
        return D * demand.reshape((1, -1))
    else:
        raise ValueError("objective must be one of {'distance_only','distance_times_demand'}")


def solve_cp_sat(
    warehouses: pd.DataFrame,
    customers: pd.DataFrame,
    distances: pd.DataFrame,
    objective: str,
    time_limit_s: int | None = None,
) -> Dict:
    start = time.time()
    n_w, n_c = len(warehouses), len(customers)
    capacities = warehouses["Capacity"].to_numpy(dtype=float)
    demand = customers["Demand"].to_numpy(dtype=float)
    D = distances.to_numpy(dtype=float)
    C = build_cost_matrix(D, demand, objective)

    # Scale costs to integers (CP-SAT wants integer objective). Use 1e6 precision cap.
    scale = 1000  # meters if distance in km + demand multiplier -> adjust as needed
    C_int = np.rint(C * scale).astype(int)

    model = cp_model.CpModel()
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n_w) for j in range(n_c)}

    # Each customer assigned exactly once
    for j in range(n_c):
        model.Add(sum(x[(i, j)] for i in range(n_w)) == 1)

    # Capacity constraints
    for i in range(n_w):
        # demand can be float; CP-SAT needs integers -> scale demand too
        # Use same scale so that demand*scale and capacity*scale match units.
        demand_int = np.rint(demand * scale).astype(int)
        cap_int = int(round(capacities[i] * scale))
        model.Add(sum(demand_int[j] * x[(i, j)] for j in range(n_c)) <= cap_int)

    # Objective
    model.Minimize(sum(C_int[i, j] * x[(i, j)] for i in range(n_w) for j in range(n_c)))

    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    runtime = time.time() - start

    status_map = {
        cp_model.OPTIMAL: "Optimal",
        cp_model.FEASIBLE: "Feasible",
        cp_model.INFEASIBLE: "Infeasible",
        cp_model.MODEL_INVALID: "ModelInvalid",
        cp_model.UNKNOWN: "Unknown",
    }
    status_str = status_map.get(status, str(status))
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    wh_ids = warehouses["Warehouse_ID"].tolist()
    cu_ids = customers["Customer_ID"].tolist()

    assignments = []
    total_cost_int = 0
    total_distance_metric = 0.0

    for j in range(n_c):
        chosen_i = None
        for i in range(n_w):
            if solver.BooleanValue(x[(i, j)]):
                chosen_i = i
                break
        if chosen_i is None:
            assignments.append(
                {"Customer_ID": cu_ids[j], "Assigned_Warehouse_ID": -1, "Demand": demand[j], "DistanceKM": np.nan, "Cost": np.nan}
            )
        else:
            dist_km = float(D[chosen_i, j])
            cost = float(C[chosen_i, j])
            total_cost_int += C_int[chosen_i, j]
            total_distance_metric += dist_km * (demand[j] if objective == "distance_times_demand" else 1.0)
            assignments.append(
                {
                    "Customer_ID": cu_ids[j],
                    "Assigned_Warehouse_ID": wh_ids[chosen_i],
                    "Demand": demand[j],
                    "DistanceKM": dist_km,
                    "Cost": cost,
                }
            )

    assignment_df = pd.DataFrame(assignments)

    used = np.zeros(n_w, dtype=float)
    for _, row in assignment_df.dropna(subset=["Assigned_Warehouse_ID"]).iterrows():
        if row["Assigned_Warehouse_ID"] == -1:
            continue
        i = wh_ids.index(row["Assigned_Warehouse_ID"])
        used[i] += row["Demand"]
    cap_usage = pd.DataFrame(
        {
            "Warehouse_ID": wh_ids,
            "Capacity": capacities,
            "Used": used,
            "Residual": capacities - used,
        }
    )

    return {
        "method_name": "CP-SAT",
        "status": status_str,
        "feasible": feasible,
        "total_cost": float(total_cost_int) / scale,  # unscale to original magnitude
        "total_distance": float(total_distance_metric),
        "assignment": assignment_df,
        "capacity_usage": cap_usage,
        "runtime_sec": runtime,
    }


def write_outputs(results: Dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    results["assignment"].to_csv(outdir / "CP_assignment.csv", index=False)
    results["capacity_usage"].to_csv(outdir / "CP_capacity_usage.csv", index=False)
    with open(outdir / "CP_summary.txt", "w") as f:
        f.write(
            f"Method: {results['method_name']}\n"
            f"Status: {results['status']}\n"
            f"Feasible: {results['feasible']}\n"
            f"Objective(total_cost): {results['total_cost']:.6f}\n"
            f"Total_distance_metric: {results['total_distance']:.6f}\n"
            f"Runtime_sec: {results['runtime_sec']:.3f}\n"
        )
    logging.info(f"Wrote outputs to {outdir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="CP-SAT baseline for warehouse–customer assignment.")
    parser.add_argument("--warehouses", type=Path, required=True)
    parser.add_argument("--customers", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--objective", choices=["distance_only", "distance_times_demand"], default="distance_times_demand")
    parser.add_argument("--time_limit_s", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    configure_logging(args.verbose)
    wh, cu = load_entities(args.warehouses, args.customers)
    D = load_distance_matrix(args.distances, len(wh), len(cu))

    results = solve_cp_sat(wh, cu, D, args.objective, args.time_limit_s)
    write_outputs(results, args.outdir)
    print(
        f"[CP-SAT] Status={results['status']} Feasible={results['feasible']} "
        f"Objective={results['total_cost']:.6f} DistanceMetric={results['total_distance']:.6f} "
        f"Runtime={results['runtime_sec']:.3f}s"
    )


if __name__ == "__main__":
    main()