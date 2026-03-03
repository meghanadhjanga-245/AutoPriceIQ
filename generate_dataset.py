"""
Generate a realistic synthetic used car dataset (8,000 records)
incorporating traditional cars AND 2026 EVs with battery/range features.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

np.random.seed(42)
N = 8000

BRANDS = {
    "Toyota": (18000, 6000, 0.10),
    "Honda": (16500, 5500, 0.09),
    "Ford": (17000, 6000, 0.10),
    "BMW": (45000, 15000, 0.05),
    "Mercedes": (52000, 18000, 0.05),
    "Audi": (40000, 13000, 0.05),
    "Tesla": (55000, 18000, 0.07),
    "Chevrolet": (15000, 5000, 0.08),
    "Hyundai": (14500, 4500, 0.08),
    "Kia": (13500, 4000, 0.07),
    "Volkswagen": (20000, 7000, 0.06),
    "Nissan": (15000, 5000, 0.07),
    "Rivian": (60000, 12000, 0.04),
    "Lucid": (80000, 20000, 0.03),
    "Polestar": (50000, 14000, 0.03),
    "Subaru": (22000, 7000, 0.03),
}

EV_BRANDS = {"Tesla", "Rivian", "Lucid", "Polestar"}
EV_COMPATIBLE = {"Hyundai", "Kia", "Volkswagen", "BMW", "Mercedes", "Audi", "Chevrolet", "Nissan"}

FUEL_TYPES = {"EV": 0, "Hybrid": 1, "Petrol": 2, "Diesel": 3}
TRANSMISSIONS = ["Automatic", "Manual", "CVT"]
BODY_TYPES = ["Sedan", "SUV", "Truck", "Coupe", "Hatchback", "Van", "Convertible"]
CONDITIONS = ["Excellent", "Good", "Fair", "Poor"]
COLORS = ["White", "Black", "Silver", "Gray", "Blue", "Red", "Green", "Brown"]

brands = np.random.choice(list(BRANDS.keys()), N,
                          p=[v[2] for v in BRANDS.values()])

years = np.random.randint(2010, 2026, N)
age = 2026 - years

fuel_types = []
for brand in brands:
    if brand in EV_BRANDS:
        fuel_types.append("EV")
    elif brand in EV_COMPATIBLE:
        f = np.random.choice(["EV", "Hybrid", "Petrol", "Diesel"], p=[0.20, 0.25, 0.35, 0.20])
        fuel_types.append(f)
    else:
        f = np.random.choice(["Hybrid", "Petrol", "Diesel"], p=[0.15, 0.60, 0.25])
        fuel_types.append(f)
fuel_types = np.array(fuel_types)

mileage = np.where(
    fuel_types == "EV",
    np.random.randint(0, 80000, N),
    np.random.randint(5000, 250000, N)
)
mileage = mileage + age * np.random.randint(5000, 15000, N)
mileage = np.clip(mileage, 0, 350000)

battery_range_km = np.where(
    fuel_types == "EV",
    np.random.randint(200, 650, N),
    np.nan
)
battery_health_pct = np.where(
    fuel_types == "EV",
    np.clip(100 - age * np.random.uniform(1, 3, N), 60, 100),
    np.nan
)
charging_speed_kw = np.where(
    fuel_types == "EV",
    np.random.choice([11, 22, 50, 100, 150, 250, 350], N),
    np.nan
)
ev_incentive_2026 = np.where(
    fuel_types == "EV",
    np.random.choice([0, 3500, 5000, 7500], N, p=[0.1, 0.2, 0.3, 0.4]),
    0
)

engine_size = np.where(
    fuel_types == "EV",
    np.nan,
    np.random.choice([1.0, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], N)
)
horsepower = np.where(
    fuel_types == "EV",
    np.random.randint(200, 1100, N),
    np.random.randint(80, 550, N)
)

transmissions = np.where(
    fuel_types == "EV",
    "Automatic",
    np.random.choice(TRANSMISSIONS, N, p=[0.60, 0.25, 0.15])
)
body_types = np.random.choice(BODY_TYPES, N, p=[0.25, 0.30, 0.15, 0.10, 0.10, 0.05, 0.05])
conditions = np.random.choice(CONDITIONS, N, p=[0.35, 0.40, 0.15, 0.10])
colors = np.random.choice(COLORS, N)
num_owners = np.random.choice([1, 2, 3, 4, 5], N, p=[0.40, 0.30, 0.15, 0.10, 0.05])
accident_history = np.random.choice([0, 1], N, p=[0.75, 0.25])
service_records = np.random.choice([0, 1], N, p=[0.30, 0.70])

# ── Price Computation ──────────────────────────────────────────────────────────
prices = []
for i in range(N):
    brand = brands[i]
    base, spread, _ = BRANDS[brand]
    price = np.random.normal(base, spread)
    price -= age[i] * np.random.uniform(1200, 2800)
    price -= mileage[i] * np.random.uniform(0.02, 0.06)
    cond_mult = {"Excellent": 1.12, "Good": 1.00, "Fair": 0.85, "Poor": 0.70}
    price *= cond_mult[conditions[i]]
    if fuel_types[i] == "EV":
        price += 8000
        price += battery_range_km[i] * 25 if not np.isnan(battery_range_km[i]) else 0
        price += ev_incentive_2026[i] * 0.5
    elif fuel_types[i] == "Hybrid":
        price += 3500
    if accident_history[i]:
        price *= 0.88
    if service_records[i]:
        price *= 1.05
    price -= (num_owners[i] - 1) * np.random.uniform(400, 900)
    price += horsepower[i] * 15
    price += np.random.normal(0, 1500)
    prices.append(max(price, 2000))

df = pd.DataFrame({
    "brand": brands,
    "year": years,
    "age_years": age,
    "mileage_km": mileage.astype(int),
    "fuel_type": fuel_types,
    "transmission": transmissions,
    "body_type": body_types,
    "condition": conditions,
    "color": colors,
    "num_owners": num_owners,
    "accident_history": accident_history,
    "service_records": service_records,
    "engine_size_L": engine_size,
    "horsepower": horsepower.astype(int),
    "battery_range_km": battery_range_km,
    "battery_health_pct": battery_health_pct,
    "charging_speed_kw": charging_speed_kw,
    "ev_incentive_2026_usd": ev_incentive_2026.astype(int),
    "is_ev": (fuel_types == "EV").astype(int),
    "price_usd": np.round(prices, 2),
})

out_path = os.path.join(os.path.dirname(__file__), "data", "used_cars.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"[OK] Dataset generated: {out_path}  ({N} rows, {df.shape[1]} columns)")
print(df.describe())
