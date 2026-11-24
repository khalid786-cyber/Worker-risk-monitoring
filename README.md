# K-Means Risk-Tier Segmentation of Migrant Worker Health Records (SIH26006)

## Run it (about 5 minutes)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python generate_data.py        # 1. creates data/migrant_health_synthetic.csv
python train.py                # 2. trains, validates, writes outputs/ + models/
streamlit run app.py           # 3. opens the dashboard at http://localhost:8501

`python train.py --k 4` forces a different number of tiers. The `outputs/` and `models/` folders
already contain a trained run, so `streamlit run app.py` works immediately.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic migrant-worker health records (vitals, exposure, visits, disease outcomes) |
| `common.py` | Features, preprocessing (impute + z-score + one-hot), risk ranking, scoring function |
| `train.py` | k selection, k-Means, tiers, validation tests, stability, baselines, plots |
| `app.py` | Streamlit app: dashboard, single-worker scoring, batch CSV upload, live monitor |
| `outputs/` | Result tables + figures for your report |

## Objective mapping (for your report)
1. **Standardize features**: median imputation, z-score scaling, one-hot encoding (`common.build_preprocessor`).
2. **k-Means risk tiers**: k-means++ seeding, 20 restarts; k chosen from 3-6 by agreement of silhouette, Davies-Bouldin and Calinski-Harabasz, with the elbow plot; clusters ranked into Low/Medium/High by a composite risk score.
3. **Validate vs disease incidence**: incidence per tier, chi-square, Spearman trend, Kruskal-Wallis, stability (ARI), baselines (Ward, GMM, DBSCAN).

## Important honesty note
The dataset is SIMULATED. No public dataset of individual migrant-worker records exists. State this in your
report ("prototype evaluated on a synthetic dataset; real health-department data can be dropped in with the
same column names"). To use real data, make a CSV with the columns in `common.FEATURES` plus the disease
columns (tb, respiratory_illness, anaemia, hypertension, diabetes, any_disease) and run
`python train.py --data your_file.csv`.

## Known limitation to mention
Silhouette is low (~0.12) because real-world health features overlap heavily; k=2 scores higher on internal
indices, but at least 3 ordered tiers are needed for a Low/Medium/High outreach scheme. The tiers are
justified by external validation (incidence rises monotonically) and stability (ARI > 0.97).
