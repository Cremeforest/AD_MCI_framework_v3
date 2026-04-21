import pandas as pd
import numpy as np

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# ------------------------------------------------------------
# Simulate missing state module on the TEST set only
# ------------------------------------------------------------
STATE_COLS = [
    "state_MMSE",
    "state_CDR_global",
    "state_CDRSB",
    "state_ADAS",
    "state_FAQ",
]

# 1. load modules
baseline = pd.read_csv("data_processed/modules/adni_module_baseline.csv")
state = pd.read_csv("data_processed/modules/adni_module_state.csv")
structure = pd.read_csv("data_processed/modules/adni_module_structure.csv")
dynamics = pd.read_csv("data_processed/modules/adni_module_dynamics.csv")

# 2. merge
X = (
    baseline.merge(state, on="RID")
    .merge(structure, on="RID")
    .merge(dynamics, on="RID")
)

# 3. labels
y = pd.read_csv("data_processed/labels/adni_survival_labels.csv")
df = X.merge(y, on="RID")

# 4. split
train_ids = pd.read_csv("data_processed/split/adni_train.csv")
test_ids = pd.read_csv("data_processed/split/adni_test.csv")

df_train = df[df["RID"].isin(train_ids["RID"])].copy()
df_test = df[df["RID"].isin(test_ids["RID"])].copy()

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# 5. feature columns
feature_cols = [c for c in df.columns if c not in ["RID", "time", "event"]]

print("\nFeature columns used:")
print(feature_cols)

# 6. clean data
df_train[feature_cols] = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
df_test[feature_cols] = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# 7. FORCE drop_state on TEST set only
# ------------------------------------------------------------
for col in STATE_COLS:
    if col in df_test.columns:
        df_test[col] = np.nan

print("\nForced state-module missingness on TEST set:")
print(STATE_COLS)

# 8. drop all-NaN columns in train
missing_counts = df_train[feature_cols].isna().sum()
all_nan_cols = missing_counts[missing_counts == len(df_train)].index.tolist()

if len(all_nan_cols) > 0:
    print("\nDropping all-NaN columns:", all_nan_cols)
    feature_cols = [c for c in feature_cols if c not in all_nan_cols]

# 9. impute using TRAIN medians
train_median = df_train[feature_cols].median()
df_train[feature_cols] = df_train[feature_cols].fillna(train_median)
df_test[feature_cols] = df_test[feature_cols].fillna(train_median)

# 10. build survival format
y_train = Surv.from_arrays(
    event=df_train["event"].astype(bool),
    time=df_train["time"]
)

y_test = Surv.from_arrays(
    event=df_test["event"].astype(bool),
    time=df_test["time"]
)

X_train = df_train[feature_cols].values
X_test = df_test[feature_cols].values

# 11. train RSF
rsf = RandomSurvivalForest(
    n_estimators=200,
    min_samples_split=10,
    min_samples_leaf=15,
    random_state=42,
    n_jobs=-1
)

rsf.fit(X_train, y_train)

# 12. predict risk
surv_funcs = rsf.predict_survival_function(X_test)
risk_scores = np.array([1 - fn(fn.x[-1]) for fn in surv_funcs])

# 13. C-index
c_index = concordance_index_censored(
    df_test["event"].astype(bool),
    df_test["time"],
    risk_scores
)[0]

full_data_cindex = 0.811
delta = c_index - full_data_cindex

print("\nFinal number of features used:", len(feature_cols))
print(f"RSF C-index (drop_state): {c_index:.4f}")
print(f"Delta vs full-data RSF (0.811): {delta:.4f}")