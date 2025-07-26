"""
Minimal runtime used INSIDE the Docker image to prove that:

• the model file was copied correctly
• dependencies import correctly
• inference runs without GPU

It **does not** expose any REST API – that would be out of scope for
the assignment’s “smoke‑test” requirement.
"""

import pathlib
import joblib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "linreg.joblib"

if not MODEL_PATH.exists():
    raise SystemExit("ERROR: model artifact not found inside container!")

model = joblib.load(MODEL_PATH)

# deterministic random sample so CI logs are consistent
rng = np.random.default_rng(seed=2025)
sample = rng.random((1, 8))  # California‑housing has 8 features

pred = model.predict(sample)[0]
print(f"[predict.py]  sample={sample.tolist()[0]}  →  ŷ={pred:.3f}")
