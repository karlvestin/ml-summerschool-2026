"""
Offline training script for the ESS 500kW cooling skid leak-LOCATION model.

The dataset label was changed from leak flow RATE (regression) to leak
LOCATION (10-class classification): 'No leak' plus nine
<section> <component> <side> classes (e.g. 'Spk LWU Return'). This script
trains a single mlpack DecisionTree classifier on the same skid sensors and the
same DecisionTree hyperparameters as the prior rate model.

HONESTY NOTE — read before trusting any accuracy number:
  The P&ID (M02430) has ONE process-water-in port (TT-102) and ONE out port
  (TT-110); every section/component branch merges OFF-skid, upstream of both.
  So (cooling-skid-physics-advisor):
    • component axis (LWU/Window/Antenna)  -> topologically UNOBSERVABLE
    • section axis   (HBL/Spk/MBL)         -> topologically UNOBSERVABLE
    • side axis      (Supply/Return)       -> thermal signal ~0.01-0.05 C,
                                              3-20x below PT100 tolerance
  On top of that, EACH leak location in this dataset is exactly ONE contiguous
  drain event (verified). 30 s-spaced rows inside one event are strongly
  autocorrelated, so a row-level split leaks the answer across train/test.

  We therefore report TWO regimes:
    (A) ROW-LEVEL stratified split  -> the optimistic, inflated number.
    (B) LEAVE-ONE-EVENT-OUT (LOEO)  -> the honest number. Holding a location's
        only event out means the model has never seen that class, so exact
        location recall is 0 by construction. What LOEO *does* measure is which
        axes survive: leak detection (observable) and Supply/Return (weak).

Feature change signed off this task by cooling-skid-physics-advisor +
feature-importance-validator:
  • dLevel/dt is now the windowed slope of level_corrected (thermally detrended)
    instead of raw LI-103 — exact, since a windowed lstsq slope is linear:
    slope(level_corrected) = slope(LI103) - beta*slope(TT102).
  • level_corrected (cumulative) and d_correctedLev_dt (rate) are KEPT SEPARATE
    (integral vs derivative; not redundant) — the proposed single-feature merge
    was rejected by both agents.

Run with (from the training_model/ folder — this is the single trainer for the
self-contained bench IOC; it writes the model artifacts the aSub loads next to it):
    conda run -n e3-mlpack python3 train_model.py
    conda run -n e3-mlpack python3 train_model.py --benchmark   # +DT vs RF
"""

import os
# Pin OpenMP to one thread BEFORE mlpack loads, for byte-reproducibility of the
# deployed artifact (OpenMP FP-reduction order can flip near-tie splits).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import xlrd
import numpy as np
import mlpack
import json
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
# Paths are resolved relative to THIS script's folder (training_model/) so the
# trainer works no matter which directory you launch it from, and writes the model
# files the bench IOC loads right beside it. HERE.parent is the repo root, where
# the shared data/ folder lives.
HERE       = Path(__file__).resolve().parent
DATA_PATH  = str(HERE.parent / "data" / "Dataset.xls")
MODEL_PATH = str(HERE / "leak_model.bin")
META_PATH  = str(HERE / "leak_model_meta.json")

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.20
# TEST_FRAC = 0.10 (remainder)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# DecisionTree hyperparameters (PRODUCTION model) — unchanged from the rate task.
DT_MAX_DEPTH  = 12
DT_MIN_LEAF   = 3
DT_MIN_GAIN   = 1e-7

# Random forest hyperparameters — ONLY for the --benchmark DT-vs-RF comparison.
RF_NUM_TREES     = 32
RF_MIN_LEAF_SIZE = 10
RF_MAX_DEPTH     = 8

# dLevel/dt windowed-slope config. Exported to metadata so the C++ aSub and the
# parity harness reproduce the exact same estimate.
DLDT_WINDOW   = 12     # samples in the slope window
VENT_STEP_MM  = 4.0    # |ΔLI103| above this between samples = N2 vent/top-up
GAP_S         = 300.0  # time gap (s) above this starts a new recording session/event

# Physics leak-DETECTION gate (the DEPLOYED leak/no-leak decision; the tree is only
# an advisory LOCATION hint). WHY detection was taken out of the tree: each leak
# location is ONE contiguous drain event, so the tree memorises that event's absolute
# operating-point fingerprint (temporal leakage) and constant-predicts the largest
# leak class on any held-out event — tree-based detection collapses. A real leak is
# instead CAUSAL and condition-invariant: a SUSTAINED drop of level_corrected, which
# generalises regardless of heat load (the tt102-slope guard rejects the heat-load
# confound). Full write-up: docs/EXERCISE_leak_gate.md §1 + meta "detection_rationale".
# Calibrated from the no-leak history; the C++ aSub reads these from metadata and
# achieves 0% no-leak false positives with every leak class detected.
DET_SLOPE_TRIP  = -0.015   # mm/s   fast-slope trip (~8 sigma of no-leak slope)
DET_CUM_TRIP    = -12.0    # mm     cumulative level_corrected drop over DET_CUM_WINDOW
DET_CUM_WINDOW  = 900.0    # s      trailing integrator window (15 min)
DET_TT_TRIP     = 0.01     # degC/s heat-load-transition guard (thermal-lag artifact)
DET_DEBOUNCE    = 3        # consecutive scans to latch / clear the detection

# ─── Column indices in the raw sheet ──────────────────────────────────────────
COL_TIME   = 0
COL_LI103  = 1   # expansion vessel level (mm)
COL_PSPEED = 2   # pump VFD speed (%)
COL_OSS101 = 3   # dissolved O2 — dropped (30-120 min lag, not used)
COL_PT101  = 4   # skid inlet pressure (barg) — dropped (~0.1% importance)
COL_TT110  = 5   # supply temp TO upstream section (°C)
COL_TT102  = 6   # return temp FROM upstream section (°C)
COL_LABEL  = 7   # leak LOCATION label (string; 'No leak' = no leak)

# ─── Load raw data ─────────────────────────────────────────────────────────────
print("Loading dataset...")
wb  = xlrd.open_workbook(DATA_PATH)
ws  = wb.sheet_by_index(0)
# Numeric sensor columns and the string label column are read separately: the
# label is now categorical text, so the sheet is no longer all-float.
num_cols = [COL_TIME, COL_LI103, COL_PSPEED, COL_OSS101, COL_PT101, COL_TT110, COL_TT102]
raw    = np.array([[ws.cell_value(i, c) for c in num_cols] for i in range(1, ws.nrows)],
                  dtype=np.float64)
labels = np.array([str(ws.cell_value(i, COL_LABEL)).strip() for i in range(1, ws.nrows)])
N = raw.shape[0]
print(f"  Loaded {N} samples, {len(num_cols)} numeric columns + 1 string label")

# Re-index the numeric block (columns were gathered in num_cols order).
time_ms = raw[:, 0]
li103   = raw[:, 1]
p_speed = raw[:, 2]
tt110   = raw[:, 5]
tt102   = raw[:, 6]

# ─── Feature Engineering ───────────────────────────────────────────────────────
# 1. ΔT = return − supply (heat-load context / thermal-detrend anchor).
delta_t = tt102 - tt110

# 2. level_corrected: remove thermal-expansion component from vessel level.
#    Fit β (mm/°C) on non-leak samples only (they span the full heat-load range),
#    dropping N2 top-up steps first. TT-102 drives expansion; mean-centre it.
nonleak_mask = (labels == "No leak")
step_ok      = (np.abs(np.diff(li103, prepend=li103[0])) < 1.0)
fit_mask     = nonleak_mask & step_ok
if fit_mask.sum() > 10:
    T_ref = float(tt102[fit_mask].mean())
    A = np.stack([tt102[fit_mask] - T_ref, np.ones(fit_mask.sum())], axis=1)
    beta = float(np.linalg.lstsq(A, li103[fit_mask], rcond=None)[0][0])
    assert 2.0 < beta < 25.0, f"β={beta:.3f} mm/°C outside physical range — check N2 top-up transients"
    print(f"  Thermal expansion β = {beta:.4f} mm/°C | T_ref = {T_ref:.2f} °C | fit n = {fit_mask.sum()}")
else:
    beta, T_ref = 0.0, 0.0
    print("  WARNING: too few baseline samples; thermal detrending disabled")

level_corrected = li103 - beta * (tt102 - T_ref)   # residual = leak-only level signal

# 3. d_correctedLev_dt: DLDT_WINDOW-sample windowed lstsq slope of level_corrected
#    (mm/s). Computed on the DETRENDED level (agent-endorsed): exact, since the
#    slope operator is linear. N2 vent/top-up jumps and cross-session gaps are
#    excluded so a window never mixes two drain events / recording sessions.
t_s     = time_ms / 1000.0
vent    = np.abs(np.diff(li103, prepend=li103[0])) > VENT_STEP_MM
session = np.zeros(N, dtype=int)                    # contiguous-session id
session[1:] = np.cumsum(np.diff(t_s) > GAP_S)
d_correctedLev_dt = np.zeros(N)
for i in range(N):
    sel = np.arange(max(0, i - DLDT_WINDOW + 1), i + 1)
    sel = sel[(session[sel] == session[i]) & (~vent[sel])]
    if len(sel) >= 3:
        tc  = t_s[sel] - t_s[sel].mean()
        den = float(np.dot(tc, tc))
        if den > 0.0:
            lc = level_corrected[sel] - level_corrected[sel].mean()
            d_correctedLev_dt[i] = float(np.dot(tc, lc) / den)

# 4. Speed / |ΔT| ratio: separates hydraulic overload from thermal overload.
speed_over_dt = p_speed / np.maximum(np.abs(delta_t), 0.5)

# ─── Feature matrix (6 features; order MUST match the C++ aSub) ────────────────
X = np.asfortranarray(np.column_stack([
    delta_t,            # 0: ΔT return−supply        (heat-load context / anchor)
    tt102,              # 1: return temperature       (thermal state)
    level_corrected,    # 2: detrended level          (cumulative leak depletion)
    d_correctedLev_dt,  # 3: windowed d(level_corr)/dt (instantaneous leak-rate signal)
    p_speed,            # 4: VFD pump speed            (hydraulic compensation)
    speed_over_dt,      # 5: speed/|ΔT|               (hydraulic vs thermal ratio)
]))
FEATURE_NAMES = [
    "delta_T", "TT102", "level_corrected",
    "d_correctedLev_dt", "P_speed", "speed_over_deltaT",
]
IDX_LEVELCORR, IDX_DLDT = 2, 3
print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

# ─── Label encoding (leak LOCATION strings → integer classes) ──────────────────
# Class 0 is 'No leak'; the nine leak locations follow in sorted order. Deterministic
# so the class index ↔ location map is stable across runs (C++/db/OPI depend on it).
NO_LEAK = "No leak"
leak_locations = sorted(set(labels) - {NO_LEAK})
class_names    = [NO_LEAK] + leak_locations
name_to_class  = {nm: i for i, nm in enumerate(class_names)}
class_to_name  = {i: nm for nm, i in name_to_class.items()}
n_classes      = len(class_names)
y_int          = np.array([name_to_class[v] for v in labels], dtype=np.uint64)

def class_side(i):
    """'Supply' / 'Return' / '' (No leak) — the one weakly-observable axis."""
    nm = class_to_name[i]
    if nm.endswith("Supply"): return "Supply"
    if nm.endswith("Return"): return "Return"
    return ""

print(f"  {n_classes} leak-location classes:")
for i in range(n_classes):
    count = int((y_int == i).sum())
    kind  = "no-leak" if i == 0 else "leak   "
    print(f"    class {i:2d} → {class_to_name[i]:<20} [{kind}]  "
          f"({count} samples, {100*count/N:.1f}%)")

# ─── Contiguous drain-event ids (one event per leak location, verified) ────────
# An event = a maximal run of one label with no >GAP_S time gap inside it.
event_id = np.zeros(N, dtype=int)
for i in range(1, N):
    event_id[i] = event_id[i-1] + (labels[i] != labels[i-1] or (t_s[i] - t_s[i-1]) > GAP_S)
n_events = int(event_id[-1]) + 1
print(f"  {n_events} contiguous events (one per class — each leak location drained once)")

# ─── Min-max normalisation ─────────────────────────────────────────────────────
## STUDENT INPUT

# ─── Metrics helpers (imbalance-aware; 89.5% no-leak makes accuracy misleading) ─
def confusion(y_true, y_pred, k):
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[int(t), int(p)] += 1
    return M

def class_report(M):
    """Per-class recall/precision/F1 + balanced accuracy + macro-F1 (present classes)."""
    k = M.shape[0]
    support = M.sum(axis=1)
    recall  = np.array([M[c, c] / support[c] if support[c] else np.nan for c in range(k)])
    colsum  = M.sum(axis=0)
    prec    = np.array([M[c, c] / colsum[c] if colsum[c] else np.nan for c in range(k)])
    f1      = np.array([2 * p * r / (p + r) if (p and r and p + r > 0) else 0.0
                        for p, r in zip(np.nan_to_num(prec), np.nan_to_num(recall))])
    present = support > 0
    return recall, prec, f1, float(np.nanmean(recall[present])), float(np.mean(f1[present]))

def fit_dt(Xtr, ytr):
## STUDENT INPUT


def dt_predict(m, X):
## STUDENT INPUT


# ─── Regime A: ROW-LEVEL stratified split (OPTIMISTIC — the deployed artifact) ──
# Stratify so every class keeps its proportion in all three splits. NOTE: because
# each location is one autocorrelated event, these numbers are inflated — the LOEO
# block below is the honest verdict. The DEPLOYED model is trained here.
def stratified_split(y, fracs, rng):
    tr, va, te = [], [], []
    f_tr, f_va = fracs
    for c in np.unique(y):
        m = rng.permutation(np.where(y == c)[0])
        a, b = int(len(m) * f_tr), int(len(m) * (f_tr + f_va))
        tr += m[:a].tolist(); va += m[a:b].tolist(); te += m[b:].tolist()
    return np.array(tr), np.array(va), np.array(te)

_rng = np.random.RandomState(RANDOM_SEED)
idx_train, idx_val, idx_test = stratified_split(y_int, (TRAIN_FRAC, VAL_FRAC), _rng)
X_train = np.asfortranarray(X_norm[idx_train]); y_train = y_int[idx_train]
X_val   = np.asfortranarray(X_norm[idx_val]);   y_val   = y_int[idx_val]
X_test  = np.asfortranarray(X_norm[idx_test]);  y_test  = y_int[idx_test]
for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
    assert arr.flags["F_CONTIGUOUS"], f"{name} is not F-contiguous (armadillo layout)"
print(f"\n  [Regime A] Stratified row split: train={len(X_train)}, "
      f"val={len(X_val)}, test={len(X_test)}  (armadillo layout verified)")

print("\nTraining mlpack DecisionTree classifier (deployed model)...")
model = mlpack.decision_tree(
    training=X_train, labels=y_train,
    maximum_depth=DT_MAX_DEPTH, minimum_leaf_size=DT_MIN_LEAF,
    minimum_gain_split=DT_MIN_GAIN, print_training_accuracy=True, verbose=True,
)["output_model"]

def evaluate(X, y_true_int, split_name):
    y_pred = dt_predict(model, X)
    acc = float(np.mean(y_pred == y_true_int))
    _, _, _, bal_acc, macro_f1 = class_report(confusion(y_true_int, y_pred, n_classes))
    print(f"  [{split_name}]  Acc={acc*100:6.2f}%  BalAcc={bal_acc*100:6.2f}%  "
          f"MacroF1={macro_f1*100:6.2f}%")
    return dict(acc=acc, bal_acc=bal_acc, macro_f1=macro_f1)

print("\nRegime A evaluation — OPTIMISTIC (row-level split; inflated by 30 s autocorrelation):")
m_train = evaluate(X_train, y_train, "Train")
m_val   = evaluate(X_val,   y_val,   "Val  ")
m_test  = evaluate(X_test,  y_test,  "Test ")

# Per-class + confusion on the row-level test split.
test_pred = dt_predict(model, X_test)
M = confusion(y_test, test_pred, n_classes)
recall, prec, f1, bal_acc, macro_f1 = class_report(M)
pc_names, pc_test_acc, pc_test_count, pc_train_count = [], [], [], []
print("\n  Per-class test metrics (recall = per-class accuracy):")
for c in range(n_classes):
    cnt = int(M[c].sum())
    r_c = None if cnt == 0 else float(recall[c])
    pc_names.append(class_to_name[c]); pc_test_acc.append(r_c)
    pc_test_count.append(cnt); pc_train_count.append(int((y_train == c).sum()))
    shown = f"recall={r_c*100:6.2f}%  F1={f1[c]*100:6.2f}%" if r_c is not None else "n/a"
    print(f"    class {c:2d} → {class_to_name[c]:<20} {shown}  (test n={cnt})")

print("\n  Confusion matrix (rows=true, cols=pred; class index):")
print("        " + "".join(f"{c:5d}" for c in range(n_classes)))
for c in range(n_classes):
    print(f"    {c:2d} |" + "".join(f"{M[c, p]:5d}" for p in range(n_classes)))

# ─── Regime B: LEAVE-ONE-EVENT-OUT (HONEST) ───────────────────────────────────
# Each leak location is ONE event, so holding it out removes its class from
# training entirely: exact-location recall is 0 by construction. LOEO instead
# measures which axes generalise to an unseen drain event:
#   detection = fraction of the held-out event flagged as SOME leak   [observable]
#   side      = of leak-flagged rows, fraction with matching Supply/Return [weak]
#   modal_pred= the class the unseen event is MOST often mapped to    [what it fingerprints]
# No-leak is split by TIME (first 80% train / last 20% test) so folds stay honest.
print("\n" + "─" * 68)
print("Regime B — LEAVE-ONE-EVENT-OUT honest validation (the real verdict)")
print("─" * 68)

nl_rows = np.where(y_int == 0)[0]
nl_rows = nl_rows[np.argsort(t_s[nl_rows])]
cut     = int(len(nl_rows) * 0.80)
nl_tr, nl_te = nl_rows[:cut], nl_rows[cut:]

leak_events = []   # (event_id, class_idx) for each of the nine leak events
for e in range(n_events):
    rows_e = np.where(event_id == e)[0]
    c_e    = int(y_int[rows_e[0]])
    if c_e != 0:
        leak_events.append((e, c_e))

loeo = []
print(f"  {'held-out event':<22} {'detect':>7} {'side':>7} {'→ mostly predicted as':<22}")
for e, c_e in leak_events:
    rows_e   = np.where(event_id == e)[0]
    other_lk = np.where((y_int != 0) & (event_id != e))[0]
    tr_idx   = np.concatenate([nl_tr, other_lk])
    m        = fit_dt(X_norm[tr_idx], y_int[tr_idx])
    pred_e   = dt_predict(m, X_norm[rows_e])
    detect   = float(np.mean(pred_e != 0))
    leak_hat = pred_e[pred_e != 0]
    side_e   = class_side(c_e)
    side_acc = (float(np.mean([class_side(int(p)) == side_e for p in leak_hat]))
                if len(leak_hat) else 0.0)
    modal    = int(np.bincount(pred_e, minlength=n_classes).argmax())
    loeo.append(dict(event=e, cls=c_e, name=class_to_name[c_e], detect=detect,
                     side=side_e, side_acc=side_acc, modal=modal,
                     modal_name=class_to_name[modal], location_recall=0.0))
    print(f"  {class_to_name[c_e]:<22} {detect*100:6.1f}% {side_acc*100:6.1f}%  "
          f"{class_to_name[modal]:<22} (unseen ⇒ exact-loc recall 0)")

# No-leak false-positive rate: train on all leaks + nl_tr, test on held-out nl_te.
m_nl   = fit_dt(X_norm[np.concatenate([nl_tr, np.where(y_int != 0)[0]])],
                y_int[np.concatenate([nl_tr, np.where(y_int != 0)[0]])])
pred_nl = dt_predict(m_nl, X_norm[nl_te])
fp_rate = float(np.mean(pred_nl != 0))

loeo_detect = float(np.mean([r["detect"]   for r in loeo]))
loeo_side   = float(np.mean([r["side_acc"] for r in loeo]))
print("─" * 68)
print(f"  LOEO means over 9 unseen leak events:")
print(f"    leak DETECTION recall (any-leak vs no-leak) = {loeo_detect*100:5.1f}%  [observable axis]")
print(f"    SUPPLY/RETURN accuracy (of detected rows)   = {loeo_side*100:5.1f}%  [weak thermal axis, ~chance≈50%]")
print(f"    exact-LOCATION recall                       =   0.0%  [unseen class — one event per location]")
print(f"    no-leak false-positive rate (held-out)      = {fp_rate*100:5.1f}%")
print("  VERDICT: leak/no-leak generalises; exact location does NOT (instrumentation")
print("  gap — needs per-branch flow/pressure sensors, not a better model or features).")

# ─── Verification gate: level features dominate (physics sign-off) ─────────────
# The monotonicity gate from the rate task is dropped: location is an UNORDERED
# target, so 'higher level ⇒ less leak' has no meaning. We keep the permutation-
# importance gate: the level pair (level_corrected + d_correctedLev_dt) must carry
# the bulk of the signal, since depletion is the only real leak observable.
print("\n" + "─" * 68)
print("Verification gate — level-feature dominance")
print("─" * 68)
base_acc = float(np.mean(dt_predict(model, X_test) == y_test))
perm_rng = np.random.RandomState(RANDOM_SEED)

def perm_drop(cols, repeats=5):
    drops = []
    for _ in range(repeats):
        Xp = X_test.copy()
        for j in cols:
            Xp[:, j] = Xp[perm_rng.permutation(len(Xp)), j]
        drops.append(base_acc - float(np.mean(dt_predict(model, Xp) == y_test)))
    return max(np.mean(drops), 0.0)

importance = np.array([perm_drop([j]) for j in range(X_test.shape[1])])
imp_frac = importance / importance.sum() if importance.sum() > 0 else importance
print("  Permutation importance (fraction of total):")
for j, nm in enumerate(FEATURE_NAMES):
    print(f"        {nm:<20} {imp_frac[j]*100:5.1f}%")
level_imp  = float(imp_frac[IDX_LEVELCORR] + imp_frac[IDX_DLDT])
gate_level = level_imp >= 0.40
print(f"      level_corrected + d_correctedLev_dt = {level_imp*100:5.1f}%  "
      f"(need ≥40%) → {'PASS' if gate_level else 'FAIL'}")

# Stratified k-fold CV (row-level; matches Regime A, so also optimistic).
def stratified_folds(y, k, rng):
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        for i, s in enumerate(rng.permutation(np.where(y == c)[0])):
            folds[i % k].append(s)
    return [np.array(f) for f in folds]

K_STRAT = 5
sfolds  = stratified_folds(y_int, K_STRAT, np.random.RandomState(RANDOM_SEED))
cv_bal, cv_f1 = [], []
for i in range(K_STRAT):
    te = sfolds[i]; tr = np.concatenate([sfolds[j] for j in range(K_STRAT) if j != i])
    m  = fit_dt(X_norm[tr], y_int[tr])
    _, _, _, b, f = class_report(confusion(y_int[te], dt_predict(m, X_norm[te]), n_classes))
    cv_bal.append(b); cv_f1.append(f)
cv_bal, cv_f1 = np.array(cv_bal), np.array(cv_f1)
print(f"  Stratified {K_STRAT}-fold CV (row-level, optimistic): "
      f"BalAcc={cv_bal.mean()*100:.2f}%±{cv_bal.std()*100:.2f}%  "
      f"MacroF1={cv_f1.mean()*100:.2f}%±{cv_f1.std()*100:.2f}%")
print("─" * 68)

# ─── Save model + metadata ────────────────────────────────────────────────────
print(f"\nSaving model → {MODEL_PATH}")
with open(MODEL_PATH, "wb") as f:
    f.write(model.__getstate__())

meta = {
    "task":          "leak_location_classification",
    "feature_names": FEATURE_NAMES,
    "feat_min":      feat_min.tolist(),
    "feat_max":      feat_max.tolist(),
    "feat_range":    feat_range.tolist(),
    "beta_thermal":  float(beta),
    "t_ref":         float(T_ref),
    "n_classes":     n_classes,
    "class_to_label": {str(i): class_to_name[i] for i in range(n_classes)},  # index → location string
    "label_to_class": {class_to_name[i]: i for i in range(n_classes)},
    "class_side":     {str(i): class_side(i) for i in range(n_classes)},
    "model_type": "decision_tree",
    "hyperparams": {
        "max_depth":      DT_MAX_DEPTH,
        "min_leaf_size":  DT_MIN_LEAF,
        "min_gain_split": DT_MIN_GAIN,
    },
    "d_level_dt": {                      # C++ aSub must reproduce this exactly
        "window_samples": DLDT_WINDOW,
        "vent_step_mm":   VENT_STEP_MM,
        "gap_seconds":    GAP_S,
        "slope_on":       "level_corrected",
        "method":         "windowed_lstsq_slope",
    },
    "detection": {                       # DEPLOYED leak/no-leak gate (read by the aSub)
        "method": "sustained_level_drop_gate",
        "_comment": ("Deployed leak DETECTION is gated on a sustained drop of "
                     "level_corrected, NOT the tree class (which fingerprints one-off "
                     "drain events). Calibrated from the no-leak history."),
        "slope_trip_mm_s":      DET_SLOPE_TRIP,
        "cum_drop_trip_mm":     DET_CUM_TRIP,
        "cum_window_s":         DET_CUM_WINDOW,
        "tt102_slope_trip_c_s": DET_TT_TRIP,
        "debounce_scans":       DET_DEBOUNCE,
        "calibration": {
            "no_leak_slope_sigma_mm_s": float(d_correctedLev_dt[nonleak_mask].std()),
            "no_leak_worst_slope_mm_s": float(d_correctedLev_dt[nonleak_mask].min()),
            "smallest_leak_class":      "HBL LWU Return",
        },
    },
    "verification_gates": {
        "level_importance": level_imp,
        "level_pass":       bool(gate_level),
        "cv_balacc_mean":   float(cv_bal.mean()),
        "cv_balacc_std":    float(cv_bal.std()),
        "cv_macrof1_mean":  float(cv_f1.mean()),
    },
    "metrics_rowlevel_optimistic": {     # Regime A — inflated by within-event autocorrelation
        "train": m_train, "val": m_val, "test": m_test,
        "test_balanced_accuracy": bal_acc,
        "test_macro_f1":          macro_f1,
    },
    "honest_leave_one_event_out": {      # Regime B — the trustworthy verdict
        "note": ("Each leak location is one contiguous drain event; held out whole, "
                 "so exact-location recall is 0 by construction. Only leak detection "
                 "and (weakly) Supply/Return generalise — location is an instrumentation "
                 "gap, not a modelling one."),
        "leak_detection_recall": loeo_detect,
        "supply_return_accuracy": loeo_side,
        "exact_location_recall":  0.0,
        "noleak_false_positive_rate": fp_rate,
        "per_event": [
            {"location": r["name"], "detect": r["detect"], "side": r["side"],
             "side_acc": r["side_acc"], "mapped_to": r["modal_name"]} for r in loeo
        ],
    },
    "observability_caveat": (
        "P&ID M02430: one supply + one return port; all section/component branches "
        "merge off-skid. Component and section axes are topologically unobservable; "
        "Supply/Return is ~0.01-0.05C, below PT100 tolerance. A high row-level score "
        "is an autocorrelation artifact, NOT physical localisation."
    ),
    "detection_rationale": (
        "Why leak/no-leak is a physics gate, NOT the tree: (1) each leak location is "
        "exactly one contiguous drain event, so the tree splits on that event's "
        "absolute operating-point fingerprint (temporal leakage), not the causal "
        "leak signal; (2) on a held-out event (leave-one-event-out / Regime B) the "
        "tree constant-predicts the largest leak class, so tree-based detection "
        "collapses and exact-location recall is 0 — a high row-level (Regime A) score "
        "is an autocorrelation artifact; (3) a genuine leak is causal and "
        "condition-invariant — escaping coolant drops level_corrected in a sustained "
        "way — which the gate detects and which generalises (LOEO detection ~88%) "
        "regardless of the day's heat load; (4) the heat-load confound (thermal "
        "expansion / thermal-lag residual drops) is rejected by the gate's "
        "temperature-transient guard, encoded as physics rather than learned splits. "
        "The tree is kept only as an advisory location hint."
    ),
    "split": {
        "n_total": int(N), "n_train": int(len(X_train)),
        "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "n_events": n_events, "stratified": True,
    },
    "per_class": {
        "labels":      pc_names,          # location strings, class-index order
        "test_acc":    pc_test_acc,       # per-class recall on the row-level test split
        "test_count":  pc_test_count,
        "train_count": pc_train_count,
    },
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved → {META_PATH}")

# ─── Optional: DecisionTree vs RandomForest benchmark ─────────────────────────
if "--benchmark" in sys.argv:
    print("\n" + "═" * 68)
    print("DecisionTree vs RandomForest benchmark (row-level; non-destructive)")
    print("═" * 68)
    def rf_predict(Xtr, ytr, Xte):
        m = mlpack.random_forest(training=np.asfortranarray(Xtr), labels=ytr,
                                 num_trees=RF_NUM_TREES, minimum_leaf_size=RF_MIN_LEAF_SIZE,
                                 maximum_depth=RF_MAX_DEPTH, seed=RANDOM_SEED)["output_model"]
        return np.asarray(mlpack.random_forest(input_model=m, test=np.asfortranarray(Xte))
                          ["predictions"]).astype(int).ravel()
    print("    model          BalAcc    MacroF1")
    for name, pred in [("DecisionTree", dt_predict(fit_dt(X_train, y_train), X_test)),
                       ("RandomForest", rf_predict(X_train, y_train, X_test))]:
        _, _, _, b, f = class_report(confusion(y_test, pred, n_classes))
        print(f"    {name:<13}  {b*100:6.2f}%   {f*100:6.2f}%")
    print("═" * 68)

print("\nDone.")
