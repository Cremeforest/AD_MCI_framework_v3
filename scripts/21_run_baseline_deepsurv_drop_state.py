import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

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

# -----------------------------
# 1. Reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Load modules
# -----------------------------
baseline = pd.read_csv("data_processed/modules/adni_module_baseline.csv")
state = pd.read_csv("data_processed/modules/adni_module_state.csv")
structure = pd.read_csv("data_processed/modules/adni_module_structure.csv")
dynamics = pd.read_csv("data_processed/modules/adni_module_dynamics.csv")

X = (
    baseline.merge(state, on="RID")
    .merge(structure, on="RID")
    .merge(dynamics, on="RID")
)

y = pd.read_csv("data_processed/labels/adni_survival_labels.csv")
df = X.merge(y, on="RID")

# -----------------------------
# 3. Load splits
# -----------------------------
train_ids = pd.read_csv("data_processed/split/adni_train.csv")
val_ids = pd.read_csv("data_processed/split/adni_val.csv")
test_ids = pd.read_csv("data_processed/split/adni_test.csv")

df_train = df[df["RID"].isin(train_ids["RID"])].copy()
df_val = df[df["RID"].isin(val_ids["RID"])].copy()
df_test = df[df["RID"].isin(test_ids["RID"])].copy()

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)

# -----------------------------
# 4. Feature columns
# -----------------------------
feature_cols = [c for c in df.columns if c not in ["RID", "time", "event"]]
print("\nFeature columns used:")
print(feature_cols)

# Replace inf with NaN
df_train[feature_cols] = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
df_val[feature_cols] = df_val[feature_cols].replace([np.inf, -np.inf], np.nan)
df_test[feature_cols] = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# 5. FORCE drop_state on TEST set only
# ------------------------------------------------------------
for col in STATE_COLS:
    if col in df_test.columns:
        df_test[col] = np.nan

print("\nForced state-module missingness on TEST set:")
print(STATE_COLS)

# Drop all-NaN columns in train
missing_counts = df_train[feature_cols].isna().sum()
all_nan_cols = missing_counts[missing_counts == len(df_train)].index.tolist()
if len(all_nan_cols) > 0:
    print("\nDropping all-NaN columns:", all_nan_cols)
    feature_cols = [c for c in feature_cols if c not in all_nan_cols]

# Median imputation using TRAIN only
train_medians = df_train[feature_cols].median()
df_train[feature_cols] = df_train[feature_cols].fillna(train_medians)
df_val[feature_cols] = df_val[feature_cols].fillna(train_medians)
df_test[feature_cols] = df_test[feature_cols].fillna(train_medians)

# Standardization using TRAIN only
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols])
X_val = scaler.transform(df_val[feature_cols])
X_test = scaler.transform(df_test[feature_cols])

time_train = df_train["time"].values.astype(np.float32)
event_train = df_train["event"].values.astype(np.float32)

time_val = df_val["time"].values.astype(np.float32)
event_val = df_val["event"].values.astype(np.float32)

time_test = df_test["time"].values.astype(np.float32)
event_test = df_test["event"].values.astype(np.float32)

print("\nFinal number of features used:", len(feature_cols))

# -----------------------------
# 6. DeepSurv model
# -----------------------------
class DeepSurvNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -----------------------------
# 7. Cox partial likelihood loss
# -----------------------------
def cox_ph_loss(risk_scores, times, events):
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    log_cum_sum_exp = torch.logcumsumexp(risk_scores, dim=0)
    diff = risk_scores - log_cum_sum_exp
    loss = -torch.sum(diff * events) / (torch.sum(events) + 1e-8)
    return loss

# -----------------------------
# 8. Tensor conversion
# -----------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

time_train_t = torch.tensor(time_train, dtype=torch.float32).to(device)
event_train_t = torch.tensor(event_train, dtype=torch.float32).to(device)

time_val_t = torch.tensor(time_val, dtype=torch.float32).to(device)
event_val_t = torch.tensor(event_val, dtype=torch.float32).to(device)

# -----------------------------
# 9. Train
# -----------------------------
model = DeepSurvNet(input_dim=X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_cindex = -np.inf
best_state = None
patience = 25
patience_counter = 0
num_epochs = 300

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    train_risk = model(X_train_t)
    loss = cox_ph_loss(train_risk, time_train_t, event_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_risk = model(X_val_t).cpu().numpy()

    val_cindex = concordance_index(
        time_val,
        -val_risk,
        event_val
    )

    if val_cindex > best_val_cindex:
        best_val_cindex = val_cindex
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val C-index: {val_cindex:.4f}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# -----------------------------
# 10. Test evaluation
# -----------------------------
if best_state is None:
    raise ValueError("best_state is None. Training did not save a valid checkpoint.")

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    test_risk = model(X_test_t).cpu().numpy()

test_cindex = concordance_index(
    time_test,
    -test_risk,
    event_test
)

full_data_cindex = 0.811
delta = test_cindex - full_data_cindex

print("\nBest Val C-index:", best_val_cindex)
print(f"DeepSurv Test C-index (drop_state): {test_cindex:.4f}")
print(f"Delta vs full-data DeepSurv (0.811): {delta:.4f}")