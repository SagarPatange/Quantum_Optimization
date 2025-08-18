
"""
benchmark_lp.py — Linear Programming baseline for warehouse–customer assignment.

Minimize total delivery distance (or cost) while:
  1) Assigning each customer to exactly one warehouse.
  2) Not exceeding any warehouse's capacity.

Objective supports:
  - distance_only
  - distance_times_demand  (default; approximates total km-weighted flow)

Inputs (CSV):
  - warehouses.csv:  Warehouse_ID,Latitude,Longitude,Capacity
  - customers.csv:   Customer_ID,Latitude,Longitude,Demand
  - distance_matrix.csv: shape (n_warehouses x n_customers); units = km
                         (can be headerless; row/col order must correspond to
                         warehouses/customers sorted by *_ID unless headers
                         explicitly match IDs.)

Outputs (to --outdir, default ./output):
  - LP_assignment.csv: Customer_ID,Assigned_Warehouse_ID,Demand,DistanceKM,Cost
  - LP_capacity_usage.csv: Warehouse_ID,Capacity,Used,Residual
  - LP_summary.txt: objective, feasibility, runtime

Usage:
  python benchmark_lp.py \
      --warehouses warehouses.csv \
      --customers customers.csv \
      --distances distance_matrix.csv \
      --objective distance_times_demand \
      --outdir output

Dependencies: pandas, numpy, pulp
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
    import pulp
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'pulp'. Install with: pip install pulp"
    ) from e


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_entities(warehouses_path: Path, customers_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wh = pd.read_csv(warehouses_path).copy()
    cu = pd.read_csv(customers_path).copy()

    required_wh = {"Warehouse_ID", "Latitude", "Longitude", "Capacity"}
    required_cu = {"Customer_ID", "Latitude", "Longitude", "Demand"}

    if not required_wh.issubset(wh.columns):
        raise ValueError(f"warehouses.csv missing columns: {required_wh - set(wh.columns)}")
    if not required_cu.issubset(cu.columns):
        raise ValueError(f"customers.csv missing columns: {required_cu - set(cu.columns)}")

    wh = wh.sort_values("Warehouse_ID").reset_index(drop=True)
    cu = cu.sort_values("Customer_ID").reset_index(drop=True)
    return wh, cu


def load_distance_matrix(distances_path: Path, n_w: int, n_c: int) -> pd.DataFrame:
    """
    Loads distance matrix robustly:
      - If headerless with exact shape (n_w x n_c), use as-is.
      - Else try header=0 and attempt to align via ID columns if present.
    """
    try:
        raw = pd.read_csv(distances_path, header=None)
    except Exception:
        raw = None

    if raw is not None and raw.shape == (n_w, n_c):
        logging.info("Loaded distance_matrix.csv as headerless matrix.")
        return raw.astype(float)

    # Try header=0 path
    logging.info("Attempting to load distance_matrix.csv with header=0...")
    df = pd.read_csv(distances_path, header=0)

    # If first column looks like warehouse identifiers and columns like customer IDs
    # try to strip identifiers and return numeric block.
    # Otherwise, validate shape directly.
    candidates = df.copy()
    # Try to drop non-numeric first col if needed:
    if not np.issubdtype(candidates.dtypes.iloc[0], np.number):
        candidates = candidates.drop(candidates.columns[0], axis=1)

    # Now ensure numeric and shape:
    candidates = candidates.apply(pd.to_numeric, errors="coerce")
    if candidates.shape != (n_w, n_c):
        raise ValueError(
            f"distance_matrix.csv shape mismatch. Expected ({n_w} x {n_c}), got {candidates.shape}."
        )

    return candidates.astype(float)


def build_cost_matrix(
    distances_km: np.ndarray,
    customer_demand: np.ndarray,
    objective: str = "distance_times_demand",
) -> np.ndarray:
    """
    Returns cost matrix C[i,j].
    """
    if objective == "distance_only":
        return distances_km.copy()
    elif objective == "distance_times_demand":
        return distances_km * customer_demand.reshape((1, -1))
    else:
        raise ValueError("objective must be one of {'distance_only','distance_times_demand'}")


def solve_lp(
    warehouses: pd.DataFrame,
    customers: pd.DataFrame,
    distances: pd.DataFrame,
    objective: str,
    solver: str = "CBC",
    time_limit_s: int | None = None,
) -> Dict:
    """
    Formulates and solves the LP as a Binary Integer Program (assignment).
    """
    start = time.time()
    n_w, n_c = len(warehouses), len(customers)
    capacities = warehouses["Capacity"].to_numpy(dtype=float)
    demand = customers["Demand"].to_numpy(dtype=float)
    D = distances.to_numpy(dtype=float)

    C = build_cost_matrix(D, demand, objective)
    logging.info(f"Problem size: warehouses={n_w}, customers={n_c}, binary vars={n_w*n_c}")

    # Decision variables x[i,j] in {0,1}
    model = pulp.LpProblem("WarehouseCustomerAssignment", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n_w), range(n_c)), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective
    model += pulp.lpSum(C[i, j] * x[i][j] for i in range(n_w) for j in range(n_c)), "TotalCost"

    # Each customer assigned to exactly one warehouse
    for j in range(n_c):
        model += pulp.lpSum(x[i][j] for i in range(n_w)) == 1, f"assign_once_c{j}"

    # Warehouse capacity not exceeded: sum_j demand_j * x[i,j] <= Capacity_i
    for i in range(n_w):
        model += pulp.lpSum(demand[j] * x[i][j] for j in range(n_c)) <= capacities[i], f"cap_w{i}"

    # Solve
    if solver.upper() == "CBC":
        pulp_solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    elif solver.upper() == "GLPK":
        pulp_solver = pulp.GLPK_CMD(msg=False, options=["--tmlim", str(time_limit_s or 0)])
    else:
        pulp_solver = None  # use default

    result_status = model.solve(pulp_solver)
    runtime = time.time() - start

    status_str = pulp.LpStatus[model.status]
    feasible = status_str in {"Optimal", "Feasible"}

    # Extract solution
    assignments = []
    total_cost = pulp.value(model.objective) if feasible else float("inf")
    total_distance = 0.0

    # Precompute for reporting
    wh_ids = warehouses["Warehouse_ID"].tolist()
    cu_ids = customers["Customer_ID"].tolist()

    for j in range(n_c):
        chosen_i = None
        for i in range(n_w):
            if pulp.value(x[i][j]) > 0.5:
                chosen_i = i
                break
        if chosen_i is None:
            # If infeasible or not assigned, assign sentinel -1
            assignments.append(
                {
                    "Customer_ID": cu_ids[j],
                    "Assigned_Warehouse_ID": -1,
                    "Demand": demand[j],
                    "DistanceKM": np.nan,
                    "Cost": np.nan,
                }
            )
        else:
            dist_km = float(D[chosen_i, j])
            # Compute cost consistent with objective
            cost = float(C[chosen_i, j])
            total_distance += dist_km * (demand[j] if objective == "distance_times_demand" else 1.0)
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

    # Capacity usage
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
        "method_name": "LP",
        "status": status_str,
        "feasible": feasible,
        "total_cost": float(total_cost),
        "total_distance": float(total_distance),
        "assignment": assignment_df,
        "capacity_usage": cap_usage,
        "runtime_sec": runtime,
    }


def write_outputs(results: Dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    results["assignment"].to_csv(outdir / "LP_assignment.csv", index=False)
    results["capacity_usage"].to_csv(outdir / "LP_capacity_usage.csv", index=False)
    with open(outdir / "LP_summary.txt", "w") as f:
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
    parser = argparse.ArgumentParser(description="LP baseline for warehouse–customer assignment.")
    parser.add_argument("--warehouses", type=Path, required=True)
    parser.add_argument("--customers", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--objective", choices=["distance_only", "distance_times_demand"], default="distance_times_demand")
    parser.add_argument("--solver", choices=["CBC", "GLPK", "DEFAULT"], default="CBC")
    parser.add_argument("--time_limit_s", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    configure_logging(args.verbose)

    warehouses, customers = load_entities(args.warehouses, args.customers)
    distances = load_distance_matrix(args.distances, len(warehouses), len(customers))

    results = solve_lp(
        warehouses=warehouses,
        customers=customers,
        distances=distances,
        objective=args.objective,
        solver=args.solver,
        time_limit_s=args.time_limit_s,
    )
    write_outputs(results, args.outdir)

    # Also print a short summary
    print(
        f"[LP] Status={results['status']} Feasible={results['feasible']} "
        f"Objective={results['total_cost']:.6f} DistanceMetric={results['total_distance']:.6f} "
        f"Runtime={results['runtime_sec']:.3f}s"
    )


if __name__ == "__main__":
    main()