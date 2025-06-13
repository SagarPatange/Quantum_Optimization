import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data
df_warehouses = pd.read_csv("warehouses.csv")
df_customers = pd.read_csv("customers.csv")
df_distances = pd.read_csv("distance_matrix_google_sampled.csv")

# Filter out extreme outliers for better visualization
df_distances = df_distances[df_distances["Distance_km"] < df_distances["Distance_km"].quantile(0.95)]

# Create Graph
G = nx.Graph()

# Add nodes with attributes
for _, row in df_warehouses.iterrows():
    G.add_node(f"Warehouse {row['Warehouse_ID']}", node_type="warehouse")

for _, row in df_customers.iterrows():
    G.add_node(f"Customer {row['Customer_ID']}", node_type="customer")

# Add edges with weights
for _, row in df_distances.iterrows():
    warehouse_id = f"Warehouse {row['Warehouse_Index']}"
    customer_id = f"Customer {row['Customer_Index']}"
    distance = row["Distance_km"]

    G.add_edge(warehouse_id, customer_id, weight=distance)

# Use a better layout that spreads nodes apart
pos = nx.fruchterman_reingold_layout(G, seed=42, k=1.5)  # Increase k to push nodes apart

# Define separate node lists
warehouse_nodes = [node for node in G.nodes if G.nodes[node].get("node_type") == "warehouse"]
customer_nodes = [node for node in G.nodes if G.nodes[node].get("node_type") == "customer"]

# Create figure
plt.figure(figsize=(12, 8))

# Draw nodes with better scaling
nx.draw_networkx_nodes(G, pos, nodelist=warehouse_nodes, node_size=1000, node_color="blue", alpha=0.8, label="Warehouses")
nx.draw_networkx_nodes(G, pos, nodelist=customer_nodes, node_size=200, node_color="green", alpha=0.6, label="Customers")

# Draw edges with better transparency
nx.draw_networkx_edges(G, pos, alpha=0.4, width=1, edge_color="gray")

# Label only a **few** warehouses instead of all
selected_warehouses = {node: node.split()[-1] for node in warehouse_nodes[:10]}  # Show only first 10 warehouses
nx.draw_networkx_labels(G, pos, labels=selected_warehouses, font_size=12, font_color="white", font_weight="bold")

# Title and legend
plt.legend(loc="upper right", fontsize=10)
plt.title("Warehouse-Customer Distance Network (Improved Layout)", fontsize=14, fontweight="bold")
plt.axis("off")

# Show plot
plt.show()