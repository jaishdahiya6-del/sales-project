"""
generate_dataset.py
--------------------
Generates a realistic synthetic retail sales dataset used throughout this
project for demonstration purposes. Run once to create data/raw/sales_data.csv

Author: Data Analytics Team
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000

categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Beauty", "Toys"]
regions = ["North", "South", "East", "West", "Central"]
products = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Tablet"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers"],
    "Home & Kitchen": ["Blender", "Cookware Set", "Vacuum Cleaner", "Air Fryer", "Bedsheet"],
    "Sports": ["Yoga Mat", "Dumbbells", "Cycling Helmet", "Running Shoes", "Football"],
    "Beauty": ["Moisturizer", "Perfume", "Lipstick", "Shampoo", "Sunscreen"],
    "Toys": ["Building Blocks", "Action Figure", "Puzzle", "Remote Car", "Board Game"],
}

dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
rows = []

for i in range(N):
    date = np.random.choice(dates)
    category = np.random.choice(categories)
    product = np.random.choice(products[category])
    region = np.random.choice(regions)

    base_price = {
        "Electronics": 250, "Clothing": 45, "Home & Kitchen": 80,
        "Sports": 60, "Beauty": 30, "Toys": 35,
    }[category]

    unit_price = round(np.random.normal(base_price, base_price * 0.2), 2)
    unit_price = max(unit_price, 5)

    quantity = np.random.poisson(4) + 1
    discount_pct = np.random.choice([0, 0, 0, 5, 10, 15, 20], p=[0.4, 0.15, 0.1, 0.15, 0.1, 0.06, 0.04])

    sales = round(unit_price * quantity * (1 - discount_pct / 100), 2)
    cost = round(unit_price * quantity * np.random.uniform(0.55, 0.75), 2)
    profit = round(sales - cost, 2)

    customer_age = int(np.clip(np.random.normal(35, 12), 18, 75))
    satisfaction = int(np.clip(np.random.normal(4, 0.9), 1, 5))

    # Introduce some missing values and light noise to make cleaning meaningful
    if np.random.rand() < 0.03:
        customer_age = np.nan
    if np.random.rand() < 0.02:
        satisfaction = np.nan
    if np.random.rand() < 0.015:
        region = None

    # Introduce a few outliers deliberately
    if np.random.rand() < 0.005:
        sales *= 12
        quantity *= 8

    rows.append({
        "OrderID": f"ORD{100000 + i}",
        "Date": pd.Timestamp(date),
        "Category": category,
        "Product": product,
        "Region": region,
        "UnitPrice": unit_price,
        "Quantity": quantity,
        "DiscountPct": discount_pct,
        "Sales": sales,
        "Cost": cost,
        "Profit": profit,
        "CustomerAge": customer_age,
        "SatisfactionScore": satisfaction,
    })

df = pd.DataFrame(rows)

# Add a handful of exact duplicate rows to simulate real-world data issues
dupes = df.sample(15, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df = df.sort_values("Date").reset_index(drop=True)
df.to_csv("/home/claude/Project/data/raw/sales_data.csv", index=False)
print(f"Generated dataset with {len(df)} rows -> data/raw/sales_data.csv")
