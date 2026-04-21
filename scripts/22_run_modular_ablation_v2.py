import os
import copy
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

# =========================================================
# 0. Config
# =========================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

RUN_MODES = [
    "full",
    "drop_baseline",
    "drop_structure",
    "drop_state",
    "drop_dynamics",
]

MAX_EPOCHS = 300
PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4

LAMBDA_SURV = 1.0
LAMBDA_EVENT3Y = 0.5
LAMBDA_HIGHRISK = 0.5

EMBED_DIM = 32
DROPOUT = 0.2
NUM_HEADS = 4
MODULE_MASK_PROB = 0.15  # random module masking during training

# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print("Using device:", DEVICE)

# =========================================================
# 2. Column definitions
# =========================================================
MODULE_COLS = {
    "baseline": [
        "age", "sex", "education", "APOE4", "baseline_MMSE", "baseline_CDR_global"
    ],
    "structure": [
        "n_visits_total", "followup_span_days", "visits_per_year",
        "median_visit_gap_days", "sd_visit_gap_days"
    ],
    "state": [
        "state_MMSE", "state_CDR_global", "state_CDRSB", "state_ADAS", "state_FAQ"
    ],
    "dynamics": [
        "MMSE_delta_6m", "MMSE_delta_12m",
        "CDRSB_delta_6m", "CDRSB_delta_12m",
        "ADAS_delta_6m", "ADAS_delta_12m",
        "FAQ_delta_6m", "FAQ_delta_12m"
    ]
}

MODULE_ORDER = ["baseline", "structure", "state", "dynamics"]
MODULE_INDEX = {name: i for i, name in enumerate(MODULE_ORDER)}

# =========================================================
# 3. Utilities
# =========================================================
def safe_numeric_df(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def derive_auxiliary_labels(df_train, df_val, df_test):
    """
    event_3y: event within 36 months
    highrisk: among event-positive train subjects, event time <= train median event time
    """
    for df in [df_train, df_val, df_test]:
        df["event_3y"] = ((df["event"] == 1) & (df["time"] <= 36)).astype(np.float32)

    event_times_train = df_train.loc[df_train["event"] == 1, "time"].values
    if len(event_times_train) == 0:
        raise ValueError("No positive events in training set.")

    highrisk_threshold = float(np.median(event_times_train))
    print(f"\nTrain-derived highrisk threshold (median event time): {highrisk_threshold:.4f} months")

    for df in [df_train, df_val, df_test]:
        df["highrisk"] = ((df["event"] == 1) & (df["time"] <= highrisk_threshold)).astype(np.float32)

    return df_train, df_val, df_test, highrisk_threshold

def prepare_one_module(module_name, module_df, cols, train_ids, val_ids, test_ids):
    """
    Returns:
        train_arr, val_arr, test_arr
        train_mask, val_mask, test_mask   # shape (N, 1)
        keep_cols
    """
    existing_cols = [c for c in cols if c in module_df.columns]
    if len(existing_cols) == 0:
        raise ValueError(f"No expected columns found for module '{module_name}'.")

    train_m = train_ids[["RID"]].merge(module_df[["RID"] + existing_cols], on="RID", how="left")
    val_m = val_ids[["RID"]].merge(module_df[["RID"] + existing_cols], on="RID", how="left")
    test_m = test_ids[["RID"]].merge(module_df[["RID"] + existing_cols], on="RID", how="left")

    train_x_raw = safe_numeric_df(train_m[existing_cols])
    val_x_raw = safe_numeric_df(val_m[existing_cols])
    test_x_raw = safe_numeric_df(test_m[existing_cols])

    all_nan_cols = train_x_raw.columns[train_x_raw.isna().all()].tolist()
    keep_cols = [c for c in existing_cols if c not in all_nan_cols]

    if len(all_nan_cols) > 0:
        print(f"[{module_name}] Dropping all-NaN columns: {all_nan_cols}")

    if len(keep_cols) == 0:
        print(f"[{module_name}] All columns are NaN in train. Creating dummy zero feature.")
        train_arr = np.zeros((len(train_ids), 1), dtype=np.float32)
        val_arr = np.zeros((len(val_ids), 1), dtype=np.float32)
        test_arr = np.zeros((len(test_ids), 1), dtype=np.float32)

        train_mask = np.zeros((len(train_ids), 1), dtype=np.float32)
        val_mask = np.zeros((len(val_ids), 1), dtype=np.float32)
        test_mask = np.zeros((len(test_ids), 1), dtype=np.float32)

        return train_arr, val_arr, test_arr, train_mask, val_mask, test_mask, []

    train_x = train_x_raw[keep_cols].copy()
    val_x = val_x_raw[keep_cols].copy()
    test_x = test_x_raw[keep_cols].copy()

    # availability before imputation
    train_mask = (~train_x.isna().all(axis=1)).astype(np.float32).values.reshape(-1, 1)
    val_mask = (~val_x.isna().all(axis=1)).astype(np.float32).values.reshape(-1, 1)
    test_mask = (~test_x.isna().all(axis=1)).astype(np.float32).values.reshape(-1, 1)

    # median impute using train only
    medians = train_x.median()
    train_x = train_x.fillna(medians)
    val_x = val_x.fillna(medians)
    test_x = test_x.fillna(medians)

    # final fill
    train_x = train_x.fillna(0)
    val_x = val_x.fillna(0)
    test_x = test_x.fillna(0)

    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_x).astype(np.float32)
    val_arr = scaler.transform(val_x).astype(np.float32)
    test_arr = scaler.transform(test_x).astype(np.float32)

    print(f"[{module_name}] kept columns ({len(keep_cols)}): {keep_cols}")
    print(f"[{module_name}] availability train/val/test: "
          f"{train_mask.mean():.3f} / {val_mask.mean():.3f} / {test_mask.mean():.3f}")

    return (
        train_arr, val_arr, test_arr,
        train_mask.astype(np.float32),
        val_mask.astype(np.float32),
        test_mask.astype(np.float32),
        keep_cols
    )

def clone_dict_of_arrays(d):
    return {k: v.copy() for k, v in d.items()}

def apply_module_ablation(module_dict, mask_dict, mode):
    """
    hard ablation at module level:
    - zero input
    - zero availability mask
    """
    x = clone_dict_of_arrays(module_dict)
    m = clone_dict_of_arrays(mask_dict)

    if mode == "full":
        return x, m

    target = mode.replace("drop_", "")
    if target not in x:
        raise ValueError(f"Unknown ablation target: {target}")

    x[target][:] = 0.0
    m[target][:] = 0.0
    return x, m

def cox_ph_loss(risk_scores, times, events):
    """
    Negative Cox partial log-likelihood
    """
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
    diff = risk_scores - log_cumsum
    loss = -(diff * events).sum() / (events.sum() + 1e-8)
    return loss

def binary_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == labels).float().mean().item()

def masked_softmax(logits, mask, dim=-1, eps=1e-8):
    """
    logits: [N, M]
    mask:   [N, M] with 1=available, 0=masked
    """
    logits = logits.masked_fill(mask <= 0, -1e9)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask
    probs = probs / (probs.sum(dim=dim, keepdim=True) + eps)
    return probs

def random_module_masking(mask_dict, drop_prob=0.15):
    """
    Randomly drop available modules during training, but keep at least one active module per sample.
    mask_dict values are torch tensors [N, 1].
    """
    out = {k: v.clone() for k, v in mask_dict.items()}

    mask_stack = torch.cat([out[k] for k in MODULE_ORDER], dim=1)  # [N,4]
    N, M = mask_stack.shape

    for i in range(N):
        available = torch.where(mask_stack[i] > 0)[0]
        if len(available) <= 1:
            continue

        drop_flags = torch.rand(len(available), device=mask_stack.device) < drop_prob
        keep_available = available[~drop_flags]

        # ensure at least one remains
        if len(keep_available) == 0:
            keep_one = available[torch.randint(0, len(available), (1,), device=mask_stack.device)]
            keep_available = keep_one

        new_row = torch.zeros(M, device=mask_stack.device)
        new_row[keep_available] = 1.0
        mask_stack[i] = new_row

    out = {
        name: mask_stack[:, idx:idx+1]
        for idx, name in enumerate(MODULE_ORDER)
    }
    return out

# =========================================================
# 4. Model
# =========================================================
class ModuleEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=32, dropout=0.2):
        super().__init__()
        hidden = max(16, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerLiteFusion(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, dropout=0.2, num_modules=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modules = num_modules

        # learnable module-type embeddings
        self.module_type_embeddings = nn.Parameter(torch.randn(num_modules, embed_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, tokens, availability_mask):
        """
        tokens: [N, M, E]
        availability_mask: [N, M] with 1=available, 0=masked
        """
        N, M, E = tokens.shape

        # add module-type embeddings
        type_emb = self.module_type_embeddings.unsqueeze(0).expand(N, -1, -1)  # [N,M,E]
        x = tokens + type_emb

        # MultiheadAttention key_padding_mask: True means ignore
        key_padding_mask = (availability_mask <= 0)

        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        x = self.attn_norm(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        gate_logits = self.gate(x).squeeze(-1)  # [N,M]
        weights = masked_softmax(gate_logits, availability_mask, dim=1)  # [N,M]

        fused = (x * weights.unsqueeze(-1)).sum(dim=1)  # [N,E]
        return fused, weights

class ModularFrameworkV2(nn.Module):
    def __init__(self, input_dims, embed_dim=32, dropout=0.2, num_heads=4):
        super().__init__()
        self.module_order = MODULE_ORDER

        self.encoders = nn.ModuleDict({
            name: ModuleEncoder(input_dims[name], embed_dim=embed_dim, dropout=dropout)
            for name in self.module_order
        })

        self.fusion = TransformerLiteFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_modules=len(self.module_order)
        )

        self.post_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
        )

        self.surv_head = nn.Linear(embed_dim, 1)
        self.event3y_head = nn.Linear(embed_dim, 1)
        self.highrisk_head = nn.Linear(embed_dim, 1)

    def forward(self, x_dict, mask_dict):
        tokens = []
        masks = []

        for name in self.module_order:
            z = self.encoders[name](x_dict[name])   # [N,E]
            tokens.append(z.unsqueeze(1))           # [N,1,E]
            masks.append(mask_dict[name])           # [N,1]

        tokens = torch.cat(tokens, dim=1)          # [N,M,E]
        availability = torch.cat(masks, dim=1)     # [N,M]

        fused, weights = self.fusion(tokens, availability)
        fused = self.post_fusion(fused)

        risk = self.surv_head(fused).squeeze(-1)
        event3y_logits = self.event3y_head(fused).squeeze(-1)
        highrisk_logits = self.highrisk_head(fused).squeeze(-1)

        return risk, event3y_logits, highrisk_logits, weights

# =========================================================
# 5. Load data
# =========================================================
baseline_df = pd.read_csv("data_processed/modules/adni_module_baseline.csv")
structure_df = pd.read_csv("data_processed/modules/adni_module_structure.csv")
state_df = pd.read_csv("data_processed/modules/adni_module_state.csv")
dynamics_df = pd.read_csv("data_processed/modules/adni_module_dynamics.csv")

label_df = pd.read_csv("data_processed/labels/adni_survival_labels.csv")
train_ids = pd.read_csv("data_processed/split/adni_train.csv")
val_ids = pd.read_csv("data_processed/split/adni_val.csv")
test_ids = pd.read_csv("data_processed/split/adni_test.csv")

label_df = label_df[["RID", "time", "event"]].copy()

train_meta = train_ids[["RID"]].merge(label_df, on="RID", how="inner")
val_meta = val_ids[["RID"]].merge(label_df, on="RID", how="inner")
test_meta = test_ids[["RID"]].merge(label_df, on="RID", how="inner")

print("\nSplit sizes:")
print("Train:", train_meta.shape)
print("Val  :", val_meta.shape)
print("Test :", test_meta.shape)

train_meta, val_meta, test_meta, highrisk_threshold = derive_auxiliary_labels(
    train_meta, val_meta, test_meta
)

# =========================================================
# 6. Prepare module arrays
# =========================================================
module_dfs = {
    "baseline": baseline_df,
    "structure": structure_df,
    "state": state_df,
    "dynamics": dynamics_df,
}

x_train_all = {}
x_val_all = {}
x_test_all = {}

m_train_all = {}
m_val_all = {}
m_test_all = {}

used_columns = {}

for name in MODULE_ORDER:
    tr_arr, va_arr, te_arr, tr_mask, va_mask, te_mask, keep_cols = prepare_one_module(
        module_name=name,
        module_df=module_dfs[name],
        cols=MODULE_COLS[name],
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    x_train_all[name] = tr_arr
    x_val_all[name] = va_arr
    x_test_all[name] = te_arr

    m_train_all[name] = tr_mask
    m_val_all[name] = va_mask
    m_test_all[name] = te_mask

    used_columns[name] = keep_cols

print("\nFinal module input dimensions:")
for name in MODULE_ORDER:
    print(f"{name}: {x_train_all[name].shape[1]}")

# =========================================================
# 7. Labels
# =========================================================
time_train = train_meta["time"].values.astype(np.float32)
event_train = train_meta["event"].values.astype(np.float32)
event3y_train = train_meta["event_3y"].values.astype(np.float32)
highrisk_train = train_meta["highrisk"].values.astype(np.float32)

time_val = val_meta["time"].values.astype(np.float32)
event_val = val_meta["event"].values.astype(np.float32)
event3y_val = val_meta["event_3y"].values.astype(np.float32)
highrisk_val = val_meta["highrisk"].values.astype(np.float32)

time_test = test_meta["time"].values.astype(np.float32)
event_test = test_meta["event"].values.astype(np.float32)
event3y_test = test_meta["event_3y"].values.astype(np.float32)
highrisk_test = test_meta["highrisk"].values.astype(np.float32)

def get_pos_weight(arr):
    pos = arr.sum()
    neg = len(arr) - pos
    if pos <= 0:
        return 1.0
    return float(neg / (pos + 1e-8))

pos_weight_event3y = get_pos_weight(event3y_train)
pos_weight_highrisk = get_pos_weight(highrisk_train)

print(f"\nTrain pos_weight event_3y : {pos_weight_event3y:.4f}")
print(f"Train pos_weight highrisk : {pos_weight_highrisk:.4f}")

# =========================================================
# 8. Tensor helpers
# =========================================================
def to_torch_dict(x_dict, m_dict):
    x_t = {
        k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
        for k, v in x_dict.items()
    }
    m_t = {
        k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
        for k, v in m_dict.items()
    }
    return x_t, m_t

# =========================================================
# 9. Train one mode
# =========================================================
def run_one_mode(mode):
    print("\n" + "=" * 70)
    print(f"Running mode: {mode}")
    print("=" * 70)

    set_seed(SEED)

    # hard ablation first
    train_x_mode, train_m_mode = apply_module_ablation(x_train_all, m_train_all, mode)
    val_x_mode, val_m_mode = apply_module_ablation(x_val_all, m_val_all, mode)
    test_x_mode, test_m_mode = apply_module_ablation(x_test_all, m_test_all, mode)

    # torch tensors
    x_train_t, m_train_t_base = to_torch_dict(train_x_mode, train_m_mode)
    x_val_t, m_val_t = to_torch_dict(val_x_mode, val_m_mode)
    x_test_t, m_test_t = to_torch_dict(test_x_mode, test_m_mode)

    time_train_t = torch.tensor(time_train, dtype=torch.float32, device=DEVICE)
    event_train_t = torch.tensor(event_train, dtype=torch.float32, device=DEVICE)
    event3y_train_t = torch.tensor(event3y_train, dtype=torch.float32, device=DEVICE)
    highrisk_train_t = torch.tensor(highrisk_train, dtype=torch.float32, device=DEVICE)

    # model
    input_dims = {k: train_x_mode[k].shape[1] for k in MODULE_ORDER}
    model = ModularFrameworkV2(
        input_dims=input_dims,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    bce_event3y = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_event3y, dtype=torch.float32, device=DEVICE)
    )
    bce_highrisk = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_highrisk, dtype=torch.float32, device=DEVICE)
    )

    best_val_cindex = -np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # random module masking ONLY during training
        m_train_t = random_module_masking(m_train_t_base, drop_prob=MODULE_MASK_PROB)

        risk_train, event3y_logits_train, highrisk_logits_train, _ = model(x_train_t, m_train_t)

        loss_surv = cox_ph_loss(risk_train, time_train_t, event_train_t)
        loss_event3y = bce_event3y(event3y_logits_train, event3y_train_t)
        loss_highrisk = bce_highrisk(highrisk_logits_train, highrisk_train_t)

        loss = (
            LAMBDA_SURV * loss_surv
            + LAMBDA_EVENT3Y * loss_event3y
            + LAMBDA_HIGHRISK * loss_highrisk
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            risk_val, event3y_logits_val, highrisk_logits_val, weights_val = model(x_val_t, m_val_t)
            risk_val_np = risk_val.detach().cpu().numpy()

        val_cindex = concordance_index(
            time_val,
            -risk_val_np,
            event_val
        )

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Surv {loss_surv.item():.4f} | "
                f"3y {loss_event3y.item():.4f} | "
                f"HR {loss_highrisk.item():.4f} | "
                f"Val C-index {val_cindex:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # load best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        risk_test, event3y_logits_test, highrisk_logits_test, weights_test = model(x_test_t, m_test_t)

    risk_test_np = risk_test.detach().cpu().numpy()

    test_cindex = concordance_index(
        time_test,
        -risk_test_np,
        event_test
    )

    event3y_acc = binary_accuracy_from_logits(
        event3y_logits_test,
        torch.tensor(event3y_test, dtype=torch.float32, device=DEVICE)
    )
    highrisk_acc = binary_accuracy_from_logits(
        highrisk_logits_test,
        torch.tensor(highrisk_test, dtype=torch.float32, device=DEVICE)
    )

    weights_test_np = weights_test.detach().cpu().numpy()
    mean_weights = weights_test_np.mean(axis=0)

    result = {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_cindex": float(best_val_cindex),
        "test_cindex": float(test_cindex),
        "test_event3y_acc": float(event3y_acc),
        "test_highrisk_acc": float(highrisk_acc),
        "mean_weight_baseline": float(mean_weights[0]),
        "mean_weight_structure": float(mean_weights[1]),
        "mean_weight_state": float(mean_weights[2]),
        "mean_weight_dynamics": float(mean_weights[3]),
    }

    print("\nResult:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    return result

# =========================================================
# 10. Run all modes
# =========================================================
all_results = []
for mode in RUN_MODES:
    res = run_one_mode(mode)
    all_results.append(res)

results_df = pd.DataFrame(all_results)

mode_order_map = {
    "full": 0,
    "drop_baseline": 1,
    "drop_structure": 2,
    "drop_state": 3,
    "drop_dynamics": 4,
}
results_df["mode_order"] = results_df["mode"].map(mode_order_map)
results_df = results_df.sort_values("mode_order").drop(columns=["mode_order"])

save_path = os.path.join(RESULT_DIR, "module_ablation_results_v2.csv")
results_df.to_csv(save_path, index=False)

print("\n" + "=" * 70)
print("Final ablation summary (v2)")
print("=" * 70)
print(results_df)
print(f"\nSaved results to: {save_path}")