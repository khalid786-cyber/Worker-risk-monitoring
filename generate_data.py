"""Generate a realistic SYNTHETIC migrant-worker health-screening dataset.

NOTE: This is simulated data (no public dataset of individual migrant-worker
records exists). Workers are drawn from three hidden sub-populations with
different vitals / exposure / care-access patterns; disease outcomes are drawn
afterwards from clinically plausible risk relationships. Document this as a
prototype dataset in your report.
"""
import argparse
import os
import numpy as np
import pandas as pd

OCC = ["Construction", "Factory", "Mining", "Agriculture", "Domestic", "Hospitality"]
HOUSING = ["Rented room", "Shared dormitory", "Worksite shed", "Informal settlement"]
ORIGIN = ["Bihar", "Uttar Pradesh", "Odisha", "Jharkhand", "West Bengal", "Rajasthan"]
WORK = ["Punjab", "Maharashtra", "Gujarat", "Karnataka", "Delhi"]

OCC_P = {0: [.15, .15, .02, .30, .25, .13],
         1: [.30, .30, .08, .12, .10, .10],
         2: [.35, .25, .25, .05, .03, .07]}
HOUSE_P = {0: [.55, .20, .05, .20], 1: [.30, .30, .15, .25], 2: [.10, .25, .25, .40]}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=8000, seed=42, with_outcomes=True):
    rng = np.random.default_rng(seed)
    g = rng.choice([0, 1, 2], size=n, p=[.50, .30, .20])  # hidden sub-population
    pick = lambda table: np.array([rng.choice(len(table[k]), p=table[k]) for k in g])
    occ_i, house_i = pick(OCC_P), pick(HOUSE_P)

    by = lambda a, b, c: np.choose(g, [a, b, c])
    age = np.clip(rng.normal(by(32, 37, 42), 8.5), 18, 65)
    bmi = np.clip(rng.normal(by(23.5, 25, 26.5), by(3, 3.5, 4.5)), 14, 42)
    sbp = np.clip(rng.normal(by(118, 127, 139), by(10, 12, 15)), 90, 200)
    dbp = np.clip(sbp * 0.62 + rng.normal(0, 6, n), 55, 120)
    hb = np.clip(rng.normal(by(13.5, 12.5, 11.0), by(1.2, 1.4, 1.6)), 6, 17)
    glu = np.clip(rng.normal(by(92, 106, 128), by(12, 18, 30)), 60, 350)
    spo2 = np.clip(rng.normal(by(98, 97, 94.5), by(.9, 1.3, 2.3)), 80, 100)
    yrs = np.clip(rng.gamma(2.0, by(2.0, 3.0, 3.5)), 0, 35)
    dust = rng.binomial(1, by(.20, .50, .80))
    chem = rng.binomial(1, by(.05, .20, .45))
    visits = rng.poisson(by(3.5, 2.2, 1.2))
    missed = rng.poisson(by(.4, 1.2, 2.8))
    since = np.clip(rng.gamma(2.0, by(40, 80, 150)), 5, 900)

    df = pd.DataFrame({
        "age": age.round(0), "bmi": bmi.round(1), "systolic_bp": sbp.round(0),
        "diastolic_bp": dbp.round(0), "hemoglobin": hb.round(1), "glucose": glu.round(0),
        "spo2": spo2.round(0), "years_in_job": yrs.round(1), "dust_exposure": dust,
        "chemical_exposure": chem, "clinic_visits_12m": visits, "missed_followups": missed,
        "days_since_last_visit": since.round(0),
        "occupation": np.array(OCC)[occ_i], "housing": np.array(HOUSING)[house_i],
    })
    # realistic missingness in lab / pulse-ox fields
    for col in ["hemoglobin", "glucose", "spo2"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan

    df.insert(0, "worker_id", [f"MW{i:05d}" for i in range(1, n + 1)])
    df["state_of_origin"] = rng.choice(ORIGIN, n)
    df["work_state"] = rng.choice(WORK, n)

    if with_outcomes:  # reported disease incidence (used ONLY for validation)
        informal = (house_i == 3).astype(float)
        tb = rng.random(n) < sigmoid(-3.8 + 0.9 * g + 0.5 * informal)
        resp = rng.random(n) < sigmoid(-3.2 + 0.8 * g + 0.8 * dust)
        anaemia = rng.random(n) < sigmoid(-2.8 + 0.7 * g + 0.6 * (hb < 12))
        htn = rng.random(n) < sigmoid(-3.3 + 0.9 * g + 1.1 * (sbp > 140))
        diab = rng.random(n) < sigmoid(-3.8 + 0.8 * g + 1.4 * (glu > 126))
        df["tb"], df["respiratory_illness"] = tb.astype(int), resp.astype(int)
        df["anaemia"], df["hypertension"], df["diabetes"] = anaemia.astype(int), htn.astype(int), diab.astype(int)
        df["any_disease"] = (tb | resp | anaemia | htn | diab).astype(int)
        df["_sim_group"] = g  # simulation ground truth; never used for training
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    d = generate(a.n, a.seed)
    d.to_csv("data/migrant_health_synthetic.csv", index=False)
    print(f"Saved data/migrant_health_synthetic.csv  shape={d.shape}")
    print(d.head())
