import numpy as np
import pandas as pd
import dimod
from dimod import BinaryQuadraticModel
from dimod.reference.samplers import SimulatedAnnealingSampler

# 1. Load data
#   - distance_matrix.csv: full 50×100 geodesic distances
#   - warehouses.csv: includes columns ['Warehouse_ID','Latitude','Longitude','Capacity']
#   - customers.csv: includes columns ['Customer_ID','Latitude','Longitude','Demand']

df_dist = pd.read_csv(r'data\distance_matrix.csv', header=0)
distances = df_dist.values

df_wh = pd.read_csv(r'data\warehouses.csv', header=0)
df_cust = pd.read_csv(r'data\\customers.csv', header=0)

capacities = df_wh['Capacity'].values  # shape: (num_warehouses,)
demands    = df_cust['Demand'].values   # shape: (num_customers,)

num_wh, num_cust = distances.shape

# 2. Decision variables: y_{w,c} = 1 if warehouse w serves customer c
#    Flattened into dict for BQM indexing

y = {(w, c): f'y_{w}_{c}'
     for w in range(num_wh)
     for c in range(num_cust)}

# 3. Build Binary Quadratic Model
bqm = BinaryQuadraticModel(vartype=dimod.BINARY)

# Objective: minimize total travel distance
distance_weight = 1.0
for (w, c), var in y.items():
    bqm.add_linear(var, distance_weight * distances[w, c])

# 4. Penalty weights
A = 10.0  # assignment (each customer exactly one warehouse)
B = 1.0   # capacity (warehouse capacity constraints)

# 5. Constraint: each customer must be assigned to exactly one warehouse
for c in range(num_cust):
    vars_c = [y[(w, c)] for w in range(num_wh)]
    # penalty: (sum_w y_{w,c} - 1)^2 = sum_i x_i + 2 sum_{i<j} x_i x_j - 2 sum_i x_i + 1
    # linear terms
    for var in vars_c:
        bqm.add_linear(var, -A)
    # quadratic terms
    for i in range(len(vars_c)):
        for j in range(i + 1, len(vars_c)):
            bqm.add_quadratic(vars_c[i], vars_c[j], 2 * A)
    # constant offset
    bqm.offset += A

# 6. Constraint: warehouse capacity not exceeded
#    penalty: (sum_c demand[c] * y_{w,c} - capacity[w])^2
for w in range(num_wh):
    vars_w = [y[(w, c)] for c in range(num_cust)]
    cap = capacities[w]
    # linear terms: demand[i]^2 - 2*cap*demand[i]
    for i, var in enumerate(vars_w):
        d_i = demands[i]
        bqm.add_linear(var, B * (d_i**2 - 2 * cap * d_i))
    # quadratic terms: 2 * demand[i] * demand[j]
    for i in range(len(vars_w)):
        for j in range(i + 1, len(vars_w)):
            bqm.add_quadratic(vars_w[i], vars_w[j], 2 * B * demands[i] * demands[j])
    # constant offset
    bqm.offset += B * (cap**2)

# 7. Solve with Simulated Annealing
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# 8. Extract best solution
best = sampleset.first.sample
assignments = [(w, c) for (w, c), var in y.items() if best[var] == 1]
total_cost = sum(distances[w, c] for (w, c) in assignments)

print(f"Assignments (w,c): {assignments}")
print(f"Total distance cost: {total_cost:.2f} km")
