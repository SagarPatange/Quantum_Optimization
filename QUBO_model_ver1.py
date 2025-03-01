import pandas as pd
import dimod
from dwave_qbsolv import QBSolv

# Load datasets
df_warehouses = pd.read_csv("warehouses.csv")
df_suppliers = pd.read_csv("suppliers.csv")
df_customers = pd.read_csv("customers.csv")
transport_cost = pd.read_csv("transport_cost.csv").values

# Parameters
total_warehouses = len(df_warehouses)
total_customers = len(df_customers)
penalty = 1000  # Large penalty to enforce constraints

# Binary Decision Variables for QUBO
qubo = {}

# Variable mapping: x_ij represents sending supply from Warehouse i to Customer j
for i in range(total_warehouses):
    for j in range(total_customers):
        qubo[(f'x_{i}_{j}', f'x_{i}_{j}')] = transport_cost[i, j]  # Minimize cost

# Demand constraint: Ensure customer demand is met
for j in range(total_customers):
    constraint = sum([f'x_{i}_{j}' for i in range(total_warehouses)])
    qubo[(constraint, constraint)] = penalty  # Penalize deviation from demand

# Warehouse capacity constraint: Ensure warehouses do not exceed capacity
for i in range(total_warehouses):
    constraint = sum([f'x_{i}_{j}' for j in range(total_customers)])
    qubo[(constraint, constraint)] = penalty  # Penalize exceeding capacity

# Convert to BQM (Binary Quadratic Model)
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

# Solve using QBSolv (simulated annealing approach)
solution = QBSolv().sample(bqm)

# Extract results
qubo_solution = solution.first.sample

# Print optimal warehouse-customer assignments
print("Optimal Supply Assignments:")
for key, value in qubo_solution.items():
    if value == 1:
        print(key, "-> Assigned")

print("QUBO optimization complete.")
