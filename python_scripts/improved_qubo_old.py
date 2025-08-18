
"""
qubo_model.py — QUBO builder for warehouse–customer assignment.

Binary variables:
  x_{i,j} = 1 if warehouse i serves customer j, else 0

Objective (to minimize):
  sum_{i,j} cost_{i,j} * x_{i,j}
  where cost_{i,j} = distance[i,j] * Demand[j] (default) OR distance only.

Hard constraints via penalty terms:
  (A) One-warehouse-per-customer:
      For each customer j: (sum_i x_{i,j} - 1)^2
      Weight: lambda_assign

  (B) Capacity not exceeded:
      We convert <= into equality with non-negative slack using binary expansion:
      For each warehouse i:
         sum_j Demand[j] * x_{i,j} + granularity * sum_k (2^k) * s_{i,k} = Capacity[i]
      Penalty term: ( ... )^2 with weight lambda_capacity
      - granularity controls discretization (e.g., 10 units).
      - Number of slack bits per warehouse is ceil(Capacity[i] / granularity). 
        (You may cap/adjust for performance.)

Outputs:
  - A dimod.BinaryQuadraticModel (BQM) object
  - Serialized JSON (bqm.to_serializable()) to output/qubo_bqm.json
  - Optional sampling with dimod's neal.SimulatedAnnealingSampler (if --solve)

Usage:
  python qubo_model.py \
      --warehouses warehouses.csv \
      --customers customers.csv \
      --distances distance_matrix.csv \
      --objective distance_times_demand \
      --lambda_assign 4.0 \
      --lambda_capacity 4.0 \
      --granularity 10 \
      --outdir output \
      [--solve] [--sa_reads 2000] [--seed 123]

Dependencies: pandas, numpy, dimod (and optionally neal for SA sampling)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import dimod
except ImportError as e:
    raise SystemExit("Missing dependency 'dimod'. Install with: pip install dimod") from e


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


def build_qubo(
    warehouses: pd.DataFrame,
    customers: pd.DataFrame,
    distances: pd.DataFrame,
    lambda_assign: float,
    lambda_capacity: float,
    granularity: float,
    objective: str = "distance_times_demand",
) -> Tuple[dimod.BinaryQuadraticModel, Dict]:
    """
    Constructs a BQM with:
      - Linear terms from the objective
      - Quadratic terms from assignment and capacity penalties
    Returns (bqm, metadata)
    """
    n_w, n_c = len(warehouses), len(customers)
    capacities = warehouses["Capacity"].to_numpy(dtype=float)
    demand = customers["Demand"].to_numpy(dtype=float)
    D = distances.to_numpy(dtype=float)
    C = build_cost_matrix(D, demand, objective)  # cost matrix

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    var_x = {}  # (i,j) -> label
    for i in range(n_w):
        for j in range(n_c):
            label = f"x_{i}_{j}"
            var_x[(i, j)] = label
            # Objective linear term
            bqm.add_variable(label, C[i, j])

    # (A) One-warehouse-per-customer: for each j, (sum_i x_{i,j} - 1)^2
    for j in range(n_c):
        vars_j = [var_x[(i, j)] for i in range(n_w)]
        # Expand: sum_i x_i^2 + 2*sum_{i<k} x_i x_k - 2*sum_i x_i + 1
        # Since x^2 = x, linear coeff += (1 - 2) = -1 per var; quadratic += 2 for pairs.
        # Multiply by lambda_assign.
        for label in vars_j:
            bqm.add_linear(label, -lambda_assign)  # -1 * lambda_assign

        for idx_i in range(len(vars_j)):
            for idx_k in range(idx_i + 1, len(vars_j)):
                bqm.add_quadratic(vars_j[idx_i], vars_j[idx_k], 2.0 * lambda_assign)

        # constant term +lambda_assign is omitted (doesn't affect optimization)

    # (B) Capacity with binary slack:
    # sum_j demand[j]*x_{i,j} + granularity * sum_k 2^k s_{i,k} = capacities[i]
    # Penalty: lambda_capacity * ( ... - capacities[i])^2
    var_s = {}  # (i,k) -> label
    max_bits_per_i = []
    for i in range(n_w):
        bits = max(1, int(math.ceil(capacities[i] / float(granularity))))
        max_bits_per_i.append(bits)
        for k in range(bits.bit_length()):  # binary expansion up to cover 'bits' count of units
            label = f"s_{i}_{k}"
            var_s[(i, k)] = label
            bqm.add_variable(label, 0.0)

        # Build linear/quadratic contribution of (A_i + B_i - C_i)^2:
        # where A_i = sum_j d_j * x_{i,j}, B_i = g * sum_k 2^k s_{i,k}, C_i = capacity_i
        # Expand: (A+B-C)^2 = A^2 + B^2 + C^2 + 2AB - 2AC - 2BC
        # Constants dropped. Need to add terms for A^2, B^2, 2AB, -2AC, -2BC.

        # A^2: (sum_j d_j x_ij)^2 = sum_j d_j^2 x_ij + 2 * sum_{j<l} d_j d_l x_ij x_il
        for j in range(n_c):
            bqm.add_linear(var_x[(i, j)], lambda_capacity * (demand[j] ** 2))
        for j in range(n_c):
            for l in range(j + 1, n_c):
                bqm.add_quadratic(var_x[(i, j)], var_x[(i, l)], 2.0 * lambda_capacity * demand[j] * demand[l])

        # B^2: (g * sum_k 2^k s_ik)^2 = g^2 * [ sum_k (2^{2k} s_ik) + 2 sum_{k<m} 2^{k+m} s_ik s_im ]
        g = float(granularity)
        # Determine how many slack bits really needed to represent capacities up to ceil(C_i/g)
        slack_units_needed = int(math.ceil(capacities[i] / g))
        K = max(1, slack_units_needed.bit_length())  # number of bits
        for k in range(K):
            label_k = var_s[(i, k)]
            bqm.add_linear(label_k, lambda_capacity * (g ** 2) * (2 ** (2 * k)))
        for k in range(K):
            for m in range(k + 1, K):
                bqm.add_quadratic(var_s[(i, k)], var_s[(i, m)], 2.0 * lambda_capacity * (g ** 2) * (2 ** (k + m)))

        # 2AB: 2 * (sum_j d_j x_ij) * (g * sum_k 2^k s_ik)
        for j in range(n_c):
            for k in range(K):
                bqm.add_quadratic(
                    var_x[(i, j)],
                    var_s[(i, k)],
                    2.0 * lambda_capacity * demand[j] * g * (2 ** k),
                )

        # -2AC: -2 * capacity_i * (sum_j d_j x_ij)
        for j in range(n_c):
            bqm.add_linear(var_x[(i, j)], -2.0 * lambda_capacity * capacities[i] * demand[j])

        # -2BC: -2 * capacity_i * (g * sum_k 2^k s_ik)
        for k in range(K):
            bqm.add_linear(var_s[(i, k)], -2.0 * lambda_capacity * capacities[i] * g * (2 ** k))

        # Note: constant term (capacity_i^2 * lambda_capacity) ignored.

    meta = {
        "objective": objective,
        "lambda_assign": lambda_assign,
        "lambda_capacity": lambda_capacity,
        "granularity": granularity,
        "n_warehouses": len(warehouses),
        "n_customers": len(customers),
        "notes": "Capacity handled via binary slack with granularity; increase lambda_capacity to enforce stricter adherence.",
    }
    return bqm, meta


def decode_solution_to_assignment(
    bqm: "dimod.BinaryQuadraticModel",
    sample: Dict[str, int],
    warehouses: pd.DataFrame,
    customers: pd.DataFrame,
    distances: pd.DataFrame,
    objective: str,
) -> Dict:
    """
    Converts a binary sample into an assignment table and metrics.
    """
    n_w, n_c = len(warehouses), len(customers)
    wh_ids = warehouses["Warehouse_ID"].tolist()
    cu_ids = customers["Customer_ID"].tolist()
    demand = customers["Demand"].to_numpy(float)
    D = distances.to_numpy(float)

    # For each customer j, pick i with x_{i,j}=1; if multiple, choose minimal cost; if none, argmin cost
    assignments = []
    total_distance_metric = 0.0
    for j in range(n_c):
        chosen_i = None
        active = []
        for i in range(n_w):
            if sample.get(f"x_{i}_{j}", 0) >= 1:
                active.append(i)
        if len(active) == 1:
            chosen_i = active[0]
        elif len(active) > 1:
            # choose the cheapest among actives
            costs = [(i, D[i, j] * (demand[j] if objective == "distance_times_demand" else 1.0)) for i in active]
            chosen_i = min(costs, key=lambda t: t[1])[0]
        else:
            # no active assignment -> pick cheapest i
            costs = [(i, D[i, j] * (demand[j] if objective == "distance_times_demand" else 1.0)) for i in range(n_w)]
            chosen_i = min(costs, key=lambda t: t[1])[0]

        dist_km = float(D[chosen_i, j])
        cost = dist_km * (demand[j] if objective == "distance_times_demand" else 1.0)
        total_distance_metric += cost
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

    # Capacity usage & feasibility checks
    used_by_w = assignment_df.groupby("Assigned_Warehouse_ID")["Demand"].sum().reindex(wh_ids, fill_value=0.0)
    cap = warehouses.set_index("Warehouse_ID")["Capacity"].astype(float)
    capacity_usage = pd.DataFrame(
        {"Warehouse_ID": wh_ids, "Capacity": cap.values, "Used": used_by_w.values, "Residual": cap.values - used_by_w.values}
    )

    feasible = (assignment_df.groupby("Customer_ID").size() == 1).all() and (capacity_usage["Used"] <= capacity_usage["Capacity"]).all()

    total_cost = assignment_df["Cost"].sum()
    return {
        "assignment": assignment_df,
        "capacity_usage": capacity_usage,
        "feasible": bool(feasible),
        "total_cost": float(total_cost),
        "total_distance": float(total_distance_metric),
    }


def main():
    parser = argparse.ArgumentParser(description="Build (and optionally solve) QUBO for assignment.")
    parser.add_argument("--warehouses", type=Path, required=True)
    parser.add_argument("--customers", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--objective", choices=["distance_only", "distance_times_demand"], default="distance_times_demand")
    parser.add_argument("--lambda_assign", type=float, default=4.0)
    parser.add_argument("--lambda_capacity", type=float, default=4.0)
    parser.add_argument("--granularity", type=float, default=10.0)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--sa_reads", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    configure_logging(args.verbose)

    wh, cu = load_entities(args.warehouses, args.customers)
    D = load_distance_matrix(args.distances, len(wh), len(cu))

    start = time.time()
    bqm, meta = build_qubo(
        warehouses=wh,
        customers=cu,
        distances=D,
        lambda_assign=args.lambda_assign,
        lambda_capacity=args.lambda_capacity,
        granularity=args.granularity,
        objective=args.objective,
    )
    build_time = time.time() - start

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Serialize BQM
    serial = bqm.to_serializable()
    with open(args.outdir / "qubo_bqm.json", "w") as f:
        json.dump({"bqm": serial, "meta": meta, "build_time_sec": build_time}, f)
    print(f"[QUBO] Built BQM with {len(bqm.variables)} variables; wrote to {args.outdir / 'qubo_bqm.json'} in {build_time:.3f}s")

    if args.solve:
        try:
            import neal  # type: ignore
        except Exception as e:
            print("neal (SimulatedAnnealingSampler) not available; install with: pip install dimod neal")
            return

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=args.sa_reads, seed=args.seed)
        best = sampleset.first
        decoded = decode_solution_to_assignment(bqm, best.sample, wh, cu, D, args.objective)

        # Write decoded solution
        decoded["assignment"].to_csv(args.outdir / "QUBO_assignment.csv", index=False)
        decoded["capacity_usage"].to_csv(args.outdir / "QUBO_capacity_usage.csv", index=False)
        with open(args.outdir / "QUBO_summary.txt", "w") as f:
            f.write(
                f"Feasible(heuristic check): {decoded['feasible']}\n"
                f"TotalCost: {decoded['total_cost']:.6f}\n"
                f"TotalDistanceMetric: {decoded['total_distance']:.6f}\n"
                f"Energy(best): {best.energy:.6f}\n"
                f"BuildTime_sec: {build_time:.3f}\n"
            )
        print(
            f"[QUBO] Decoded solution -> Feasible={decoded['feasible']} "
            f"Cost={decoded['total_cost']:.6f} DistMetric={decoded['total_distance']:.6f}"
        )


if __name__ == "__main__":
    main()