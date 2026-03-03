"""
train_model.py    Full ML pipeline:
   EDA summaries saved to reports/
   Feature engineering + preprocessing
   Linear Regression (baseline), Random Forest, XGBoost (tuned)
   RMSE / MAE / R evaluation
   SHAP feature importance
   Saves model artefacts to models/
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE, "data", "used_cars.csv")
MODEL_DIR  = os.path.join(BASE, "models")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

#  0. Load data 
print("=" * 65)
print(" CAR PRICE PREDICTION    ML Pipeline")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n  Loaded: {df.shape[0]:,} rows  {df.shape[1]} columns")
print(df.dtypes.to_string())

#  1. EDA 
print("\n 1. EDA ")

# 1a Price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["price_usd"], bins=60, kde=True, color="#6366F1", ax=axes[0])
axes[0].set_title("Price Distribution (USD)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Price (USD)")
sns.histplot(np.log1p(df["price_usd"]), bins=60, kde=True, color="#10B981", ax=axes[1])
axes[1].set_title("Log-Price Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(1 + Price)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "01_price_distribution.png"), dpi=150)
plt.close()

# 1b Correlation heatmap (numeric only)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "02_correlation_matrix.png"), dpi=150)
plt.close()

# 1c EV vs ICE price boxplot
fig, ax = plt.subplots(figsize=(10, 5))
df["fuel_label"] = df["fuel_type"]
sns.boxplot(data=df, x="fuel_label", y="price_usd",
            palette=["#6366F1", "#10B981", "#F59E0B", "#EF4444"], ax=ax)
ax.set_title("Price by Fuel Type", fontsize=13, fontweight="bold")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Price (USD)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "03_price_by_fuel.png"), dpi=150)
plt.close()

# 1d Mileage vs Price scatter
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["mileage_km"], df["price_usd"],
                     c=df["age_years"], cmap="plasma", alpha=0.4, s=10)
plt.colorbar(scatter, ax=ax, label="Age (years)")
ax.set_title("Mileage vs Price (coloured by Age)", fontsize=13, fontweight="bold")
ax.set_xlabel("Mileage (km)")
ax.set_ylabel("Price (USD)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "04_mileage_vs_price.png"), dpi=150)
plt.close()

# 1e Brand avg price bar
brand_avg = df.groupby("brand")["price_usd"].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 6))
colors_bar = ["#6366F1" if b in {"Tesla","Rivian","Lucid","Polestar"} else "#94A3B8"
              for b in brand_avg.index]
ax.bar(brand_avg.index, brand_avg.values, color=colors_bar, edgecolor="white")
ax.set_title("Median Price by Brand  (purple = EV-first)", fontsize=13, fontweight="bold")
ax.set_xticklabels(brand_avg.index, rotation=45, ha="right")
ax.set_ylabel("Median Price (USD)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "05_brand_price.png"), dpi=150)
plt.close()

print("   [OK]  EDA plots saved to reports/")

#  2. Feature Engineering 
print("\n 2. Feature Engineering ")

df["price_per_mile"]  = (df["price_usd"] / (df["mileage_km"] + 1)).round(4)
df["ev_score"]        = df["battery_range_km"].fillna(0) * df["battery_health_pct"].fillna(0) / 100
df["luxury_flag"]     = df["brand"].isin(["BMW","Mercedes","Audi","Lucid","Polestar"]).astype(int)
df["ev_adj_price"]    = df["price_usd"] - df["ev_incentive_2026_usd"]
df["power_density"]   = df["horsepower"] / (df["engine_size_L"].fillna(1) + 0.01)
df["cond_score"]      = df["condition"].map({"Excellent":4,"Good":3,"Fair":2,"Poor":1})
df["age_mileage"]     = df["age_years"] * df["mileage_km"]   # interaction term

TARGET  = "price_usd"
DROPS   = ["price_usd", "fuel_label", "price_per_mile", "ev_adj_price"]
FEATURES = [c for c in df.columns if c not in DROPS]

CAT_COLS = ["brand", "fuel_type", "transmission", "body_type", "condition", "color"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

print(f"   Features: {len(FEATURES)}  |  Categorical: {len(CAT_COLS)}  |  Numeric: {len(NUM_COLS)}")

X = df[FEATURES]
y = np.log1p(df[TARGET])   # log-transform target  better RMSE behaviour

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

#  3. Preprocessing pipeline 
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
preprocessor = ColumnTransformer([
    ("num", num_transformer, NUM_COLS),
    ("cat", cat_transformer, CAT_COLS),
])

#  4. Models 
print("\n 3. Model Training ")

results = {}

def evaluate(name, pipeline, X_tr, X_te, y_tr, y_te):
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    # back-transform
    preds_orig = np.expm1(preds)
    y_orig     = np.expm1(y_te)
    rmse = np.sqrt(mean_squared_error(y_orig, preds_orig))
    mae  = mean_absolute_error(y_orig, preds_orig)
    r2   = r2_score(y_orig, preds_orig)
    cv   = cross_val_score(pipeline, X_tr, y_tr, cv=5,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"   [{name}]  RMSE=${rmse:,.0f}  MAE=${mae:,.0f}  R={r2:.4f}  CV-RMSE={-cv.mean():.4f}{cv.std():.4f}")
    results[name] = {"RMSE": round(rmse,2), "MAE": round(mae,2), "R2": round(r2,4)}
    return pipeline

# 4a Linear Regression (baseline)
lr_pipe = Pipeline([("pre", preprocessor),
                    ("model", Ridge(alpha=10))])
lr_pipe = evaluate("Ridge Regression (baseline)", lr_pipe, X_train, X_test, y_train, y_test)

# 4b Random Forest
rf_pipe = Pipeline([("pre", preprocessor),
                    ("model", RandomForestRegressor(
                        n_estimators=300, max_depth=20,
                        min_samples_leaf=2, n_jobs=-1, random_state=42))])
rf_pipe = evaluate("Random Forest", rf_pipe, X_train, X_test, y_train, y_test)

# 4c XGBoost with GridSearchCV
xgb_base = Pipeline([("pre", preprocessor),
                      ("model", XGBRegressor(
                          objective="reg:squarederror",
                          eval_metric="rmse",
                          n_jobs=-1,
                          random_state=42,
                          tree_method="hist"))])

param_grid = {
    "model__n_estimators":   [400, 600],
    "model__max_depth":      [6, 8],
    "model__learning_rate":  [0.05, 0.10],
    "model__subsample":      [0.8],
    "model__colsample_bytree": [0.8],
}

gs = GridSearchCV(xgb_base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                  n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
best_xgb = gs.best_estimator_
print(f"\n   XGB Best Params: {gs.best_params_}")
best_xgb = evaluate("XGBoost (tuned)", best_xgb, X_train, X_test, y_train, y_test)

#  5. Feature Importance (XGBoost) 
print("\n 4. Feature Importance (XGBoost) ")

xgb_model   = best_xgb.named_steps["model"]
pre_fitted   = best_xgb.named_steps["pre"]
feat_names   = (NUM_COLS +
                pre_fitted.named_transformers_["cat"]["encoder"].get_feature_names_out(CAT_COLS).tolist())
fi = pd.Series(xgb_model.feature_importances_, index=feat_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
top20 = fi.head(20)
colors_fi = ["#6366F1" if "ev" in c.lower() or "battery" in c.lower() or "is_ev" in c
             else "#10B981" for c in top20.index]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors_fi[::-1])
ax.set_title("Top-20 Feature Importances  XGBoost  (purple = EV features)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "06_feature_importance.png"), dpi=150)
plt.close()
print("   [OK]  Feature importance plot saved")

#  6. SHAP 
print("\n 5. SHAP Explainability ")
X_test_transformed = pre_fitted.transform(X_test)
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed[:500])

shap.summary_plot(shap_values, X_test_transformed[:500],
                  feature_names=feat_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "07_shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   [OK]  SHAP summary plot saved")

#  7. Actual vs Predicted 
y_pred_log = best_xgb.predict(X_test)
y_pred_raw = np.expm1(y_pred_log)
y_true_raw = np.expm1(y_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_true_raw, y_pred_raw, alpha=0.3, s=8, color="#6366F1")
lo, hi = y_true_raw.min(), y_true_raw.max()
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Price (USD)", fontsize=11)
axes[0].set_ylabel("Predicted Price (USD)", fontsize=11)
axes[0].set_title("Actual vs Predicted  (XGBoost)", fontsize=13, fontweight="bold")
axes[0].legend()

residuals = y_true_raw - y_pred_raw
axes[1].scatter(y_pred_raw, residuals, alpha=0.3, s=8, color="#EF4444")
axes[1].axhline(0, color="black", lw=1.2)
axes[1].set_xlabel("Predicted Price (USD)")
axes[1].set_ylabel("Residual (USD)")
axes[1].set_title("Residual Plot", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "08_actual_vs_predicted.png"), dpi=150)
plt.close()
print("   [OK]  Actual vs Predicted + Residual plots saved")

#  8. Save artefacts 
print("\n 6. Saving Artefacts ")
joblib.dump(best_xgb, os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
joblib.dump(lr_pipe,   os.path.join(MODEL_DIR, "ridge_pipeline.pkl"))
joblib.dump(rf_pipe,   os.path.join(MODEL_DIR, "rf_pipeline.pkl"))

meta = {
    "model_results": results,
    "num_cols":  NUM_COLS,
    "cat_cols":  CAT_COLS,
    "features":  FEATURES,
    "target":    TARGET,
    "xgb_best_params": gs.best_params_,
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("   [OK]  Models + metadata saved to models/")
print("\n" + "=" * 65)
print(" [DONE]  Pipeline Complete!")
print(" Model Results Summary:")
for name, m in results.items():
    print(f"   {name:<35}  R={m['R2']:.4f}  RMSE=${m['RMSE']:,.0f}  MAE=${m['MAE']:,.0f}")
print("=" * 65)
