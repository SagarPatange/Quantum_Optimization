#!/bin/bash

# Set paths
PYTHON_PATH="/Users/abdur-rahmanibn-bilalwaajid/miniconda3/bin/python"
SCRIPT_PATH="/Users/abdur-rahmanibn-bilalwaajid/Desktop/Quantum_Optimization/python scripts/compare_models.py"
WAREHOUSES_PATH="/Users/abdur-rahmanibn-bilalwaajid/Desktop/Quantum_Optimization/data/warehouses.csv"
CUSTOMERS_PATH="/Users/abdur-rahmanibn-bilalwaajid/Desktop/Quantum_Optimization/data/customers.csv"
DISTANCES_PATH="/Users/abdur-rahmanibn-bilalwaajid/Desktop/Quantum_Optimization/data/distance_matrix.csv"
OUTPUT_DIR="/Users/abdur-rahmanibn-bilalwaajid/Desktop/Quantum_Optimization/output"

# Run the script
"$PYTHON_PATH" "$SCRIPT_PATH" \
  --warehouses "$WAREHOUSES_PATH" \
  --customers "$CUSTOMERS_PATH" \
  --distances "$DISTANCES_PATH" \
  --run_greedy \
  --run_sa \
  --run_lp \
  --run_cp \
  --outdir "$OUTPUT_DIR" \
  -v

