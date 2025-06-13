import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Set reproducibility
np.random.seed(42)

# Define dataset size
total_warehouses = 50
total_customers = 100

# Generate random warehouse locations
df_warehouses = pd.DataFrame({
    'Warehouse_ID': range(1, total_warehouses + 1),
    'Latitude': np.random.uniform(30, 50, total_warehouses),  # Random US-based latitudes
    'Longitude': np.random.uniform(-120, -80, total_warehouses),  # Random US-based longitudes
    'Capacity': np.random.randint(500, 5000, total_warehouses)
})

# Generate random customer locations and demands
df_customers = pd.DataFrame({
    'Customer_ID': range(1, total_customers + 1),
    'Latitude': np.random.uniform(30, 50, total_customers),
    'Longitude': np.random.uniform(-120, -80, total_customers),
    'Demand': np.random.randint(100, 1000, total_customers)
})

# Create Distance Matrix
def calculate_distance_matrix(warehouses, customers):
    matrix = np.zeros((len(warehouses), len(customers)))
    for i, (w_lat, w_lon) in enumerate(zip(warehouses['Latitude'], warehouses['Longitude'])):
        for j, (c_lat, c_lon) in enumerate(zip(customers['Latitude'], customers['Longitude'])):
            matrix[i, j] = geodesic((w_lat, w_lon), (c_lat, c_lon)).km  # Distance in km
    return matrix

distance_matrix = calculate_distance_matrix(df_warehouses, df_customers)

# Save datasets

pd.DataFrame(distance_matrix).to_csv("distance_matrix.csv", index=False)

print("Dataset generation complete: warehouses.csv, customers.csv, distance_matrix.csv")


import googlemaps
import requests
import time
# Load warehouse and customer data
df_warehouses = pd.read_csv("warehouses.csv")
df_customers = pd.read_csv("customers.csv")

API_KEY = "API KEY"  # Replace with your valid Google API key
URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Format locations
warehouse_locs = df_warehouses[['Latitude', 'Longitude']].apply(tuple, axis=1).tolist()
customer_locs = df_customers[['Latitude', 'Longitude']].apply(tuple, axis=1).tolist()

def get_google_distance(origin, destination):
    payload = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin[0], "longitude": origin[1]}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": destination[0], "longitude": destination[1]}
            }
        },
        "travelMode": "DRIVE"
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.distanceMeters"
    }

    try:
        response = requests.post(URL, json=payload, headers=headers)
        result = response.json()

        print("RAW API RESPONSE:", result)  # Debugging line

        if "routes" in result and len(result["routes"]) > 0:
            return result["routes"][0]["distanceMeters"] / 1000  # Convert meters to km

    except Exception as e:
        print(f"Error retrieving distance: {e}")
    
    return np.nan  # Return NaN if API call fails

# Iterate over all warehouse-customer pairs
import random

num_samples = min(200, len(warehouse_locs) * len(customer_locs))  # Limit to 200 samples
sample_pairs = random.sample([(i, j) for i in range(len(warehouse_locs)) for j in range(len(customer_locs))], num_samples)

for i, j in sample_pairs:
    distance_matrix[i, j] = get_google_distance(warehouse_locs[i], customer_locs[j])
    time.sleep(0.2)  # Avoid API rate limits
# Save only the sampled distances
df_distance_sampled = pd.DataFrame(sample_pairs, columns=["Warehouse_Index", "Customer_Index"])
df_distance_sampled["Distance_km"] = [distance_matrix[i, j] for i, j in sample_pairs]
df_distance_sampled.to_csv("distance_matrix_google_sampled.csv", index=False)

print("Google API sampled distance matrix saved successfully!")
print("Google API distance matrix saved successfully!")


import pandas as pd

# Load the sampled distance matrix
df = pd.read_csv("distance_matrix_google_sampled.csv")

# Display first few rows
print(df.head())
print(df.shape)