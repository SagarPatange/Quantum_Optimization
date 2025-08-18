import pandas as pd
import numpy as np
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Greedy warehouse-customer assignment")
parser.add_argument("--warehouses", required=True, help="Path to warehouses.csv")
parser.add_argument("--customers", required=True, help="Path to customers.csv")
parser.add_argument("--distances", required=True, help="Path to distance_matrix.csv")
parser.add_argument("--outdir", default="output", help="Output directory")
args = parser.parse_args()

# --- Load Data ---
df_warehouses = pd.read_csv(args.warehouses)
df_customers = pd.read_csv(args.customers)
distance_matrix = pd.read_csv(args.distances).values

# --- Initialization ---
num_warehouses = len(df_warehouses)
num_customers = len(df_customers)
warehouse_capacities = df_warehouses['Capacity'].values
customer_demands = df_customers['Demand'].values

# --- Assign customers greedily to nearest warehouse ---
assignments = [-1] * num_customers
capacity_used = np.zeros(num_warehouses)
total_distance = 0.0

for c_id in range(num_customers):
    demand = customer_demands[c_id]
    sorted_warehouses = np.argsort(distance_matrix[:, c_id])  # Closest first

    for w_id in sorted_warehouses:
        if capacity_used[w_id] + demand <= warehouse_capacities[w_id]:
            assignments[c_id] = w_id
            capacity_used[w_id] += demand
            total_distance += distance_matrix[w_id, c_id]
            break

# --- Build output DataFrame ---
results = pd.DataFrame({
    'Customer_ID': df_customers['Customer_ID'],
    'Assigned_Warehouse': [df_warehouses.iloc[w]['Warehouse_ID'] if w >= 0 else -1 for w in assignments],
    'Customer_Demand': customer_demands,
    'Distance_km': [distance_matrix[w, c] if w >= 0 else np.nan for c, w in enumerate(assignments)]
})

# --- Metrics ---
total_cost = total_distance
total_capacity_used = capacity_used.sum()
total_capacity = warehouse_capacities.sum()
unassigned_customers = sum([1 for a in assignments if a == -1])

print("Total Cost (km):", round(total_cost, 2))
print("Total Capacity Used:", int(total_capacity_used), "/", int(total_capacity))
print("Utilization:", round(total_capacity_used / total_capacity * 100, 2), "%")
print("Unassigned Customers:", unassigned_customers)

# --- Save Route Plan ---
os.makedirs(args.outdir, exist_ok=True)
output_path = os.path.join(args.outdir, "greedy_route_plan.csv")
results.to_csv(output_path, index=False)
print(f"✅ Route plan saved to '{output_path}'")