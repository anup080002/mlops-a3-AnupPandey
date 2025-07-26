"""
train.py
========
1. Loads the California‑housing tabular dataset
2. Fits a scikit‑learn LinearRegression model
3. Evaluates R² and stores the `.joblib` artifact
4. Converts weights to an *identical* PyTorch `SingleLayer`
   and saves state_dict
5. Writes metrics to JSON for later CI visibility
"""

import json
import pathlib

import joblib
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from torch_model import SingleLayer

# ---------- File system helpers ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---------- 1. Data ----------
X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 2. Fit scikit‑learn ----------
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

joblib.dump(sk_model, ARTIFACT_DIR / "linreg.joblib")

# ---------- 3. Metric ----------
r2_val = r2_score(y_test, sk_model.predict(X_test))
print(f"Scikit‑learn R²: {r2_val:.4f}")

# ---------- 4. Map → PyTorch ----------
pt_model = SingleLayer(in_dim=X.shape[1])
with torch.no_grad():
    pt_model.linear.weight.copy_(
        torch.tensor(sk_model.coef_, dtype=torch.float32).unsqueeze(0)
    )
    pt_model.linear.bias.copy_(
        torch.tensor(sk_model.intercept_, dtype=torch.float32)
    )
torch.save(pt_model.state_dict(), ARTIFACT_DIR / "linreg_torch.pth")

# ---------- 5. Metrics ----------
json.dump({"r2_sklearn": r2_val}, open(ARTIFACT_DIR / "metrics.json", "w"))
