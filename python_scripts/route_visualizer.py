import pandas as pd
import folium
from folium import plugins
from branca.colormap import linear
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Visualize warehouse-customer routes")
parser.add_argument("--warehouses", required=True, help="Path to warehouses.csv")
parser.add_argument("--customers", required=True, help="Path to customers.csv")
parser.add_argument("--routes", required=True, help="Path to route_plan.csv (e.g., greedy_route_plan.csv)")
parser.add_argument("--outfile", default="route_map.html", help="Path to save the output HTML map")
args = parser.parse_args()

# --- Load data ---
# Load data
# Load data
df_warehouses = pd.read_csv(args.warehouses)
df_customers = pd.read_csv(args.customers)
df_routes = pd.read_csv(args.routes)

# Rename lat/lon columns to avoid conflict
df_customers = df_customers.rename(columns={"Latitude": "Customer_Lat", "Longitude": "Customer_Lon"})
df_warehouses = df_warehouses.rename(columns={"Latitude": "Warehouse_Lat", "Longitude": "Warehouse_Lon"})

# Merge enriched customer + warehouse data into routes
df_routes = df_routes.merge(df_customers, on='Customer_ID')
df_routes = df_routes.merge(df_warehouses, left_on='Assigned_Warehouse', right_on='Warehouse_ID')
print("ROUTES COLUMNS:", df_routes.columns)
# --- Create Map ---
center_lat = df_warehouses['Warehouse_Lat'].mean()
center_lon = df_warehouses['Warehouse_Lon'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

# Add warehouse markers
for _, row in df_warehouses.iterrows():
    folium.CircleMarker(
        location=[row['Warehouse_Lat'], row['Warehouse_Lon']],
        radius=7,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=f"Warehouse {row['Warehouse_ID']} (Capacity: {row['Capacity']})"
    ).add_to(m)

# Add customer markers
for _, row in df_customers.iterrows():
    folium.CircleMarker(
        location=[row['Customer_Lat'], row['Customer_Lon']],
        radius=4,
        color='green',
        fill=True,
        fill_opacity=0.5,
        popup=f"Customer {row['Customer_ID']} (Demand: {row['Demand']})"
    ).add_to(m)

# Draw arcs: Warehouse → Customer
color_map = linear.Set1_09.scale(0, df_warehouses.shape[0])
warehouse_colors = {wid: color_map(i) for i, wid in enumerate(df_warehouses['Warehouse_ID'])}

for _, row in df_routes.iterrows():
    if row['Assigned_Warehouse'] == -1:
        continue  # Skip unassigned
    customer_loc = [row['Customer_Lat'], row['Customer_Lon']]
    warehouse_loc = [row['Warehouse_Lat'], row['Warehouse_Lon']]
    folium.PolyLine(
        locations=[warehouse_loc, customer_loc],
        color=warehouse_colors[row['Assigned_Warehouse']],
        weight=2,
        opacity=0.5,
        tooltip=f"W{row['Assigned_Warehouse']} → C{row['Customer_ID']} ({round(row['Distance_km'], 2)} km)"
    ).add_to(m)

# Save to file
m.save(args.outfile)
print(f"✅ Interactive map saved to {args.outfile}")