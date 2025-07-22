import pandas as pd
import folium
from folium import plugins
from branca.colormap import linear
import random

# Load data
df_warehouses = pd.read_csv("warehouses.csv")
df_customers = pd.read_csv("customers.csv")
df_routes = pd.read_csv("greedy_route_plan.csv")  # Output from greedy script

# Merge customer data with assigned warehouse info
df_routes = df_routes.merge(df_customers, on='Customer_ID', suffixes=('', '_cust'))
df_routes = df_routes.merge(df_warehouses, left_on='Assigned_Warehouse', right_on='Warehouse_ID', suffixes=('', '_wh'))

# Center of the map
center_lat = df_warehouses['Latitude'].mean()
center_lon = df_warehouses['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

# Add warehouse markers
for _, row in df_warehouses.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=7,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=f"Warehouse {row['Warehouse_ID']} (Capacity: {row['Capacity']})"
    ).add_to(m)

# Add customer markers
for _, row in df_customers.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
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
        continue  # skip unassigned

    customer_loc = [row['Latitude_cust'], row['Longitude_cust']]
    warehouse_loc = [row['Latitude'], row['Longitude']]

    folium.PolyLine(
        locations=[warehouse_loc, customer_loc],
        color=warehouse_colors[row['Assigned_Warehouse']],
        weight=2,
        opacity=0.5,
        tooltip=f"W{row['Assigned_Warehouse']} → C{row['Customer_ID']} ({round(row['Distance_km'], 2)} km)"
    ).add_to(m)

# Save to file
m.save("route_map.html")
print("✅ Interactive map saved to route_map.html")