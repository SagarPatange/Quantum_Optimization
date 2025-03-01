import numpy as np
import pandas as pd


np.random.seed(42)

# Parameters for dataset
total_warehouses = 10
total_suppliers = 5
total_customers = 15

# Generate synthetic warehouses
df_warehouses = pd.DataFrame({
    'Warehouse_ID': range(1, total_warehouses + 1),
    'Capacity': np.random.randint(500, 2000, total_warehouses),
    'Fixed_Cost': np.random.randint(2000, 10000, total_warehouses)
})

# Generate synthetic suppliers
df_suppliers = pd.DataFrame({
    'Supplier_ID': range(1, total_suppliers + 1),
    'Supply_Capacity': np.random.randint(1000, 5000, total_suppliers),
    'Cost_Per_Unit': np.random.randint(5, 20, total_suppliers)
})

# Generate synthetic customer demand
df_customers = pd.DataFrame({
    'Customer_ID': range(1, total_customers + 1),
    'Demand': np.random.randint(100, 500, total_customers)
})

# Generate synthetic transportation costs (Warehouse → Customer)
transport_cost = np.random.randint(10, 50, (total_warehouses, total_customers))

# Save datasets to CSV
df_warehouses.to_csv("warehouses.csv", index=False)
df_suppliers.to_csv("suppliers.csv", index=False)
df_customers.to_csv("customers.csv", index=False)
pd.DataFrame(transport_cost).to_csv("transport_cost.csv", index=False)


