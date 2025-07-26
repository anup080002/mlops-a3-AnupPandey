"""
Manual symmetric‑affine UINT8 quantisation of the linear‑regression
weights & bias. Also writes a Torch model with de‑quant weights so
metric parity can be checked.

Produces:
    models/unquant_params.joblib
    models/quant_params.joblib
    models/quant_metrics.json
    models/quant_torch.pth
"""

import json
import pathlib

import joblib
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

from torch_model import SingleLayer

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "models"
ART_DIR.mkdir(exist_ok=True)

# ---------- Load original ----------
sk_model = joblib.load(ART_DIR / "linreg.joblib")
W = sk_model.coef_.astype(np.float32)
b = np.array([sk_model.intercept_], dtype=np.float32)

joblib.dump({"W": W, "b": b}, ART_DIR / "unquant_params.joblib")

# ---------- Quant helpers ----------
def quant_affine_fp32_to_u8(arr: np.ndarray):
    qmin, qmax = 0, 255
    rmin, rmax = float(arr.min()), float(arr.max())
    if rmax == rmin:  # avoid div‑by‑zero for pathological case
        scale, zp = 1.0, 0
        q_arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        scale = (rmax - rmin) / (qmax - qmin)
        zp = int(round(qmin - rmin / scale))
        zp = max(qmin, min(qmax, zp))
        q_arr = np.round(arr / scale + zp).clip(qmin, qmax).astype(np.uint8)
    return q_arr, scale, zp


def dequant_u8(q_arr, scale, zp):
    return scale * (q_arr.astype(np.float32) - zp)


# ---------- Quantise ----------
qW, w_scale, w_zp = quant_affine_fp32_to_u8(W)
qb, b_scale, b_zp = quant_affine_fp32_to_u8(b)

joblib.dump(
    {
        "qW": qW,
        "w_scale": w_scale,
        "w_zp": int(w_zp),
        "qb": qb,
        "b_scale": b_scale,
        "b_zp": int(b_zp),
    },
    ART_DIR / "quant_params.joblib",
)

# ---------- De‑quant for eval ----------
W_dq = dequant_u8(qW, w_scale, w_zp)
b_dq = dequant_u8(qb, b_scale, b_zp)

torch_model = SingleLayer(in_dim=W_dq.shape[0])
with torch.no_grad():
    torch_model.linear.weight.copy_(torch.tensor(W_dq).unsqueeze(0))
    torch_model.linear.bias.copy_(torch.tensor(b_dq))

torch.save(torch_model.state_dict(), ART_DIR / "quant_torch.pth")

# ---------- Compare metrics ----------
X, y = fetch_california_housing(return_X_y=True, as_frame=False)
pred_orig = sk_model.predict(X)
pred_quant = (
    torch_model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
)

metrics = {
    "r2_orig": float(r2_score(y, pred_orig)),
    "r2_quant": float(r2_score(y, pred_quant)),
    "size_unquant_kb": round(
        (ART_DIR / "unquant_params.joblib").stat().st_size / 1024, 2
    ),
    "size_quant_kb": round(
        (ART_DIR / "quant_params.joblib").stat().st_size / 1024, 2
    ),
}

json.dump(metrics, open(ART_DIR / "quant_metrics.json", "w"))
print("[quantize.py] Metrics\n", json.dumps(metrics, indent=2))
