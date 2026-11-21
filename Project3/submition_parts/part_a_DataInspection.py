import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from data.oslo_prices import (
    load_oslo_prices,
    audit_price_panel
)

# ============================================================
#                  OSLO DATA INSPECTION
# ============================================================

# 1. Load data
prices = load_oslo_prices(
    "oslo-stocks-2023-2025.xlsx"
)

# 2. Audit data
report = audit_price_panel(prices)

print("\nDATA AUDIT")
for key, value in report.items():
    print(f"{key}: {value}")

# 3. Inspect dimensions
print("\nDATA SHAPE")
print(prices.shape)

# 4. Inspect instruments
print("\nINSTRUMENTS")
print(prices.columns.tolist())

# 5. Inspect first observations
print("\nFIRST OBSERVATIONS")
print(prices.head())

# 6. Inspect historical coverage
print("\nOBSERVATION COVERAGE")

coverage = prices.notna().sum().sort_values(
    ascending=False
)

print(coverage)

# 7. Inspect missing observations
print("\nMISSING OBSERVATIONS")

missing = prices.isna().sum().sort_values(
    ascending=False
)

print(missing)