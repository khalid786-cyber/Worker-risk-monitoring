"""Shared config + helper functions for the Migrant Worker Risk-Tier project."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

# ---- features used for clustering (disease outcomes are NOT used) ----------
NUMERIC = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "hemoglobin", "glucose", "spo2",
    "years_in_job", "dust_exposure", "chemical_exposure",
    "clinic_visits_12m", "missed_followups", "days_since_last_visit",
]
CATEGORICAL = ["occupation", "housing"]
FEATURES = NUMERIC + CATEGORICAL

# Direction in which each feature raises risk (+1 higher = riskier, -1 lower = riskier).
# Used ONLY to rank the discovered clusters into ordered tiers (Low -> High).
RISK_SIGNS = {
    "age": +1, "systolic_bp": +1, "diastolic_bp": +1, "glucose": +1,
    "hemoglobin": -1, "spo2": -1, "years_in_job": +1,
    "dust_exposure": +1, "chemical_exposure": +1,
    "missed_followups": +1, "days_since_last_visit": +1,
}

NICE = {
    "age": "age", "bmi": "BMI", "systolic_bp": "systolic BP", "diastolic_bp": "diastolic BP",
    "hemoglobin": "hemoglobin", "glucose": "glucose", "spo2": "SpO2",
    "years_in_job": "years in job", "dust_exposure": "dust exposure",
    "chemical_exposure": "chemical exposure", "missed_followups": "missed follow-ups",
    "days_since_last_visit": "days since last visit",
}

TIER_NAMES = {
    3: ["Low", "Medium", "High"],
    4: ["Low", "Moderate", "High", "Critical"],
    5: ["Very Low", "Low", "Medium", "High", "Critical"],
}


def tier_names(k):
    return TIER_NAMES.get(k, [f"Tier {i + 1}" for i in range(k)])


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)],
                             sparse_threshold=0.0)


def raw_risk(Xt, feature_names):
    """Mean signed z-score over risk features. Higher = riskier."""
    cols, signs = [], []
    for f, s in RISK_SIGNS.items():
        cols.append(list(feature_names).index(f"num__{f}"))
        signs.append(s)
    return (Xt[:, cols] * np.array(signs)).mean(axis=1)


def score_workers(df, bundle):
    """Take raw worker records -> add cluster, tier, risk_index (0-100), reasons."""
    pre, km = bundle["pre"], bundle["km"]
    Xt = pre.transform(df[FEATURES])
    names = bundle["feature_names"]
    out = df.copy()
    out["cluster"] = km.predict(Xt)
    rank = out["cluster"].map(bundle["cluster_to_rank"]).astype(int)
    out["tier_rank"] = rank
    out["risk_tier"] = rank.map(lambda r: bundle["tier_names"][r])
    s = raw_risk(Xt, names)
    lo, hi = bundle["score_min"], bundle["score_max"]
    out["risk_index"] = np.clip(100 * (s - lo) / (hi - lo), 0, 100).round(1)

    # plain-language reasons: top 2 features pushing this worker's risk up
    idx = {f: list(names).index(f"num__{f}") for f in RISK_SIGNS}
    reasons = []
    for i in range(len(out)):
        contrib = {f: RISK_SIGNS[f] * Xt[i, j] for f, j in idx.items()}
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:2]
        txt = [f"{NICE[f]} ({_fmt(df.iloc[i][f])})" for f, c in top if c > 0.8]
        reasons.append(", ".join(txt) if txt else "no major risk drivers")
    out["top_risk_drivers"] = reasons
    return out


def _fmt(v):
    if pd.isna(v):
        return "n/a"
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:g}"
