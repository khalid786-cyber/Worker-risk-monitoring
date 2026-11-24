"""Full pipeline: preprocess -> choose k -> k-Means -> risk tiers -> validation -> outputs."""
import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from common import (CATEGORICAL, FEATURES, NUMERIC, RISK_SIGNS, SEED,
                    build_preprocessor, raw_risk, score_workers, tier_names)

warnings.filterwarnings("ignore")
OUT = "outputs"
DISEASES = ["tb", "respiratory_illness", "anaemia", "hypertension", "diabetes", "any_disease"]


def internal_metrics(X, labels):
    ok = labels >= 0
    if len(set(labels[ok])) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(
        silhouette=silhouette_score(X[ok], labels[ok], sample_size=min(4000, ok.sum()), random_state=SEED),
        davies_bouldin=davies_bouldin_score(X[ok], labels[ok]),
        calinski_harabasz=calinski_harabasz_score(X[ok], labels[ok]))


def incidence_gap(labels, disease, score):
    """Order clusters by mean risk score; return (highest - lowest) incidence + Spearman trend."""
    d = pd.DataFrame({"c": labels, "y": disease, "s": score})
    d = d[d.c >= 0]
    if d.c.nunique() < 2:
        return np.nan, np.nan
    order = d.groupby("c").s.mean().sort_values().index
    rank = d.c.map({c: i for i, c in enumerate(order)})
    inc = d.groupby(rank).y.mean()
    rho, _ = stats.spearmanr(rank, d.y)
    return inc.iloc[-1] - inc.iloc[0], rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/migrant_health_synthetic.csv")
    ap.add_argument("--k", type=int, default=None, help="force number of tiers (default: auto in 3..6)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} workers")

    # 1. standardize ---------------------------------------------------------
    pre = build_preprocessor()
    X = pre.fit_transform(df[FEATURES])
    fnames = pre.get_feature_names_out()
    print(f"Feature matrix: {X.shape}")

    # 2. scan k --------------------------------------------------------------
    rows = []
    for k in range(2, 9):
        km = KMeans(k, init="k-means++", n_init=10, random_state=SEED).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_, **internal_metrics(X, km.labels_)))
    scan = pd.DataFrame(rows)
    scan.to_csv(f"{OUT}/k_selection.csv", index=False)
    cand = scan[(scan.k >= 3) & (scan.k <= 6)].copy()   # a risk scheme needs >= 3 tiers
    cand["avg_rank"] = (cand.silhouette.rank(ascending=False) + cand.davies_bouldin.rank()
                        + cand.calinski_harabasz.rank(ascending=False)) / 3
    auto_k = int(cand.sort_values(["avg_rank", "k"]).iloc[0].k)
    K = args.k or auto_k
    print(f"\nk selection (3..6, rank agreement of silhouette/DB/CH) -> k={auto_k}; using k={K}")
    print(scan.round(3).to_string(index=False))

    # 3. final k-Means -------------------------------------------------------
    km = KMeans(K, init="k-means++", n_init=20, random_state=SEED).fit(X)
    labels = km.labels_

    # 4. rank clusters into ordered risk tiers ------------------------------
    s_worker = raw_risk(X, fnames)
    cluster_score = pd.Series(s_worker).groupby(labels).mean().sort_values()
    cluster_to_rank = {int(c): i for i, c in enumerate(cluster_score.index)}
    names = tier_names(K)
    bundle = dict(pre=pre, km=km, feature_names=list(fnames), k=K, tier_names=names,
                  cluster_to_rank=cluster_to_rank,
                  score_min=float(np.percentile(s_worker, 0.5)),
                  score_max=float(np.percentile(s_worker, 99.5)))
    joblib.dump(bundle, "models/bundle.joblib")

    scored = score_workers(df, bundle)
    scored.to_csv(f"{OUT}/workers_scored.csv", index=False)

    # tier profile (original units)
    prof = scored.groupby("risk_tier")[NUMERIC + ["risk_index"]].mean().round(2)
    prof.insert(0, "n_workers", scored.groupby("risk_tier").size())
    prof = prof.reindex(names)
    prof.to_csv(f"{OUT}/tier_profile.csv")
    print("\nTier profile:\n", prof[["n_workers", "age", "systolic_bp", "hemoglobin", "glucose",
                                    "spo2", "missed_followups", "risk_index"]].to_string())

    # 5. validation vs disease incidence -------------------------------------
    rank = scored.tier_rank
    inc = scored.groupby("risk_tier")[DISEASES].mean().reindex(names) * 100
    inc.round(2).to_csv(f"{OUT}/incidence_by_tier.csv")
    print("\nDisease incidence (%) per tier:\n", inc.round(1).to_string())

    chi2, p_chi, dof, _ = stats.chi2_contingency(pd.crosstab(scored.risk_tier, scored.any_disease))
    rho, p_rho = stats.spearmanr(rank, scored.any_disease)
    kw = {}
    for f in ["systolic_bp", "hemoglobin", "glucose", "spo2", "missed_followups"]:
        H, p = stats.kruskal(*[g[f].dropna() for _, g in scored.groupby("risk_tier")])
        kw[f] = dict(H=float(H), p=float(p))
    print(f"\nChi-square (tier x any_disease): chi2={chi2:.1f}, dof={dof}, p={p_chi:.2e}")
    print(f"Spearman (tier rank vs any_disease): rho={rho:.3f}, p={p_rho:.2e}")
    for f, v in kw.items():
        print(f"Kruskal-Wallis {f}: H={v['H']:.1f}, p={v['p']:.2e}")

    # 6. stability -----------------------------------------------------------
    rng = np.random.default_rng(SEED)
    seed_ari = [adjusted_rand_score(labels, KMeans(K, n_init=10, random_state=s).fit(X).labels_)
                for s in range(10)]
    boot_ari = []
    for _ in range(10):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        b = KMeans(K, n_init=10, random_state=SEED).fit(X[idx])
        boot_ari.append(adjusted_rand_score(labels, b.predict(X)))
    sim_ari = adjusted_rand_score(df["_sim_group"], labels) if "_sim_group" in df else None
    print(f"\nStability ARI  seeds: {np.mean(seed_ari):.3f}±{np.std(seed_ari):.3f}"
          f"  bootstrap(80%): {np.mean(boot_ari):.3f}±{np.std(boot_ari):.3f}")
    if sim_ari is not None:
        print(f"ARI vs hidden simulated sub-populations (sanity check only): {sim_ari:.3f}")

    # 7. baselines -----------------------------------------------------------
    base = {"k-Means (ours)": labels,
            "Hierarchical (Ward)": AgglomerativeClustering(K).fit_predict(X),
            "Gaussian Mixture": GaussianMixture(K, random_state=SEED, n_init=3).fit_predict(X),
            "DBSCAN": DBSCAN(eps=3.0, min_samples=20).fit_predict(X)}
    brow = []
    for name, lab in base.items():
        m = internal_metrics(X, lab)
        gap, r = incidence_gap(lab, scored.any_disease.values, s_worker)
        brow.append(dict(method=name, n_clusters=len(set(lab[lab >= 0])),
                         noise_pct=round(100 * (lab < 0).mean(), 1),
                         **{k: round(v, 3) for k, v in m.items()},
                         incidence_gap_pct=round(100 * gap, 1) if gap == gap else np.nan,
                         spearman_trend=round(r, 3) if r == r else np.nan))
    bdf = pd.DataFrame(brow)
    bdf.to_csv(f"{OUT}/baseline_comparison.csv", index=False)
    print("\nBaseline comparison:\n", bdf.to_string(index=False))

    # 8. outreach list -------------------------------------------------------
    cols = ["worker_id", "risk_tier", "risk_index", "top_risk_drivers", "occupation",
            "housing", "work_state", "missed_followups", "days_since_last_visit"]
    out = scored.sort_values("risk_index", ascending=False)
    out[out.tier_rank == K - 1][cols].to_csv(f"{OUT}/outreach_list_high_risk.csv", index=False)

    # 9. plots ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for a, (c, t, best) in zip(ax, [("inertia", "Elbow (inertia)", None), ("silhouette", "Silhouette ↑", "max"),
                                    ("davies_bouldin", "Davies-Bouldin ↓", "min"),
                                    ("calinski_harabasz", "Calinski-Harabasz ↑", "max")]):
        a.plot(scan.k, scan[c], "o-")
        a.axvline(K, color="r", ls="--", lw=1)
        a.set_title(t); a.set_xlabel("k")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_k_selection.png", dpi=140); plt.close()

    pc = PCA(2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(6.5, 5))
    colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#9b2226", "#6a040f"]
    for r in range(K):
        m = (rank == r).values
        plt.scatter(pc[m, 0], pc[m, 1], s=6, alpha=.5, c=colors[r % 5], label=names[r])
    plt.legend(markerscale=3); plt.title("Risk tiers (PCA projection)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_pca_tiers.png", dpi=140); plt.close()

    plt.figure(figsize=(7, 4.2))
    w = 0.14
    for i, d in enumerate(DISEASES):
        plt.bar(np.arange(K) + i * w, inc[d], w, label=d)
    plt.xticks(np.arange(K) + w * 2.5, names); plt.ylabel("Incidence (%)")
    plt.title("Disease incidence by risk tier"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_incidence_by_tier.png", dpi=140); plt.close()

    z = pd.DataFrame(km.cluster_centers_, columns=fnames)[[f"num__{c}" for c in NUMERIC]]
    z.columns = NUMERIC
    z = z.loc[cluster_score.index]; z.index = names
    plt.figure(figsize=(9, 3.2))
    plt.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2); plt.colorbar(label="z-score")
    plt.xticks(range(len(NUMERIC)), NUMERIC, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(K), names); plt.title("Cluster centroids (standardized)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_centroid_heatmap.png", dpi=140); plt.close()

    json.dump(dict(k=K, auto_k=auto_k, n=len(df),
                   chi2=dict(stat=float(chi2), dof=int(dof), p=float(p_chi)),
                   spearman=dict(rho=float(rho), p=float(p_rho)), kruskal=kw,
                   stability_ari=dict(seeds_mean=float(np.mean(seed_ari)), bootstrap_mean=float(np.mean(boot_ari))),
                   ari_vs_sim_groups=sim_ari), open(f"{OUT}/validation.json", "w"), indent=2)
    print(f"\nDone. Model -> models/bundle.joblib ; results & figures -> {OUT}/")


if __name__ == "__main__":
    main()
