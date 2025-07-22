import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

# --- Load Data ---
df_warehouses = pd.read_csv("warehouses.csv")
df_customers = pd.read_csv("customers.csv")
distance_matrix = pd.read_csv("distance_matrix.csv").values

# --- Initialization ---
num_warehouses = len(df_warehouses)
num_customers = len(df_customers)
warehouse_capacities = df_warehouses['Capacity'].values
customer_demands = df_customers['Demand'].values

# --- Assign customers greedily to nearest warehouse ---
assignments = [-1] * num_customers  # Customer -> warehouse mapping
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
total_cost = total_distance  # Assuming cost ∝ distance
total_capacity_used = capacity_used.sum()
total_capacity = warehouse_capacities.sum()
unassigned_customers = sum([1 for a in assignments if a == -1])

print("Total Cost (km):", round(total_cost, 2))
print("Total Capacity Used:", int(total_capacity_used), "/", int(total_capacity))
print("Utilization:", round(total_capacity_used / total_capacity * 100, 2), "%")
print("Unassigned Customers:", unassigned_customers)

# --- Save Route Plan ---
results.to_csv("greedy_route_plan.csv", index=False)
print("✅ Route plan saved to 'greedy_route_plan.csv'")