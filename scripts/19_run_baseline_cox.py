import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# 1. load modules
baseline = pd.read_csv("data_processed/modules/adni_module_baseline.csv")
state = pd.read_csv("data_processed/modules/adni_module_state.csv")
structure = pd.read_csv("data_processed/modules/adni_module_structure.csv")
dynamics = pd.read_csv("data_processed/modules/adni_module_dynamics.csv")

# 2. merge module features
X = baseline.merge(state, on="RID") \
            .merge(structure, on="RID") \
            .merge(dynamics, on="RID")

# 3. load survival labels
y = pd.read_csv("data_processed/labels/adni_survival_labels.csv")

# 4. merge features + labels
df = X.merge(y, on="RID")

# 5. load train / val / test split
train_ids = pd.read_csv("data_processed/split/adni_train.csv")
val_ids = pd.read_csv("data_processed/split/adni_val.csv")
test_ids = pd.read_csv("data_processed/split/adni_test.csv")

# 6. split train / val / test
df_train = df[df["RID"].isin(train_ids["RID"])].copy()
df_val = df[df["RID"].isin(val_ids["RID"])].copy()
df_test = df[df["RID"].isin(test_ids["RID"])].copy()

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)

# 7. define columns
required_cols = ["time", "event"]
feature_cols = [c for c in df.columns if c not in ["RID", "time", "event"]]
print("\nFeature columns used:")
print(feature_cols)

# 8. replace inf with NaN
df_train[feature_cols] = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
df_test[feature_cols] = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)

# 9. inspect missing
missing_counts = df_train[feature_cols].isna().sum()
print("\nMissing values in train features:")
print(missing_counts[missing_counts > 0])

# 10. drop columns that are entirely NaN in train
all_nan_cols = missing_counts[missing_counts == len(df_train)].index.tolist()
if len(all_nan_cols) > 0:
    print("\nDropping all-NaN columns:", all_nan_cols)
    feature_cols = [c for c in feature_cols if c not in all_nan_cols]

# 11. impute using train medians
train_medians = df_train[feature_cols].median()
df_train[feature_cols] = df_train[feature_cols].fillna(train_medians)
df_test[feature_cols] = df_test[feature_cols].fillna(train_medians)

# 12. final check for remaining NaN
remaining_nan_train = df_train[feature_cols].isna().sum()
remaining_nan_train = remaining_nan_train[remaining_nan_train > 0]

if len(remaining_nan_train) > 0:
    print("\nStill NaN after imputation, dropping columns:")
    print(remaining_nan_train)
    bad_cols = remaining_nan_train.index.tolist()
    feature_cols = [c for c in feature_cols if c not in bad_cols]

# 13. build Cox data
cox_train = df_train[feature_cols + ["time", "event"]].copy()
cox_test = df_test[feature_cols + ["time", "event"]].copy()

# 14. fit CoxPH with mild penalizer for stability
cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_train, duration_col="time", event_col="event")

# 15. predict risk on test
risk = cph.predict_partial_hazard(cox_test)

# 16. compute C-index
c_index = concordance_index(
    cox_test["time"],
    -risk,
    cox_test["event"]
)

print("\nFinal number of features used:", len(feature_cols))
print("Cox C-index:", c_index)