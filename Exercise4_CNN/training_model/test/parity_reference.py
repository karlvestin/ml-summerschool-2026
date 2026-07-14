"""
Round-trip parity reference for the leak-LOCATION prediction IOC.

Loads the DEPLOYED model artifact (leak_model.bin + leak_model_meta.json) and the
raw dataset, then for a spread of real rows computes the class the LIVE IOC should
predict. Emits a JSON manifest that parity_drive.py replays against a running IOC.

Why d_correctedLev_dt = 0 for the "IOC-expected" column:
  The IOC derives d_correctedLev_dt from a 12-sample (DLDT_WINDOW) windowed
  least-squares slope of level_corrected, which a static row injection cannot
  reproduce. The driver holds each row's values for >= DLDT_WINDOW+1 scans, so the
  whole window fills with one constant level_corrected (constant LI103 AND TT102)
  -> the mean-centred deviations are all zero -> the slope == 0 exactly. We mirror
  that here by predicting with d_correctedLev_dt = 0. This makes class parity
  deterministic and exercises every other feature, the normalisation, and the tree.

We also report, per row, the TRAIN-PIPELINE prediction (train_model.py's native
windowed slope of level_corrected) and the dataset location label, for context.

Run:  python3 parity_reference.py   (from the training_model/ dir)
"""

import json
import numpy as np
import xlrd
import mlpack
from pathlib import Path

# Paths are relative to THIS script's folder (training_model/test/). The deployed
# model artifact lives in ../training/, the dataset in the repo-root data/, and the
# manifest is written here beside parity_drive.py.
HERE       = Path(__file__).resolve().parent
DATA_PATH  = str(HERE.parent.parent / "data" / "Dataset.xls")
MODEL_PATH = str(HERE.parent / "training" / "leak_model.bin")
META_PATH  = str(HERE.parent / "training" / "leak_model_meta.json")
OUT_PATH   = str(HERE / "parity_manifest.json")

# Column layout (must match train_model.py).
COL_TIME, COL_LI103, COL_PSPEED, COL_OSS101 = 0, 1, 2, 3
COL_PT101, COL_TT110, COL_TT102, COL_LABEL  = 4, 5, 6, 7

# Sensor-envelope bounds for choosing cleanly-injectable rows. parity_drive.py now
# drives the simulated path via the $(P)<tag>:SimRaw sources (which don't clamp),
# so these are no longer ao DRVL/DRVH limits — they are the training envelope.
# Keeping a row inside them avoids the aSub's out-of-range normalisation clamp, so
# the IOC and this reference agree exactly.
SIM_BOUNDS = {  # PV-suffix -> (lo, hi) training-envelope bounds
    "TT-102:MeasValue": (25.0, 33.0),
    "TT-110:MeasValue": (24.0, 30.0),
    "LI-103:MeasValue": (200.0, 545.0),
    "P-101:Speed":      (88.0, 96.0),
}

# ── load metadata + model ────────────────────────────────────────────────────
with open(META_PATH) as f:
    meta = json.load(f)
feat_min   = np.array(meta["feat_min"])
feat_range = np.array(meta["feat_range"])
beta       = meta["beta_thermal"]
t_ref      = meta["t_ref"]
n_features = len(meta["feature_names"])                       # 6
class_to_label = {int(k): str(v) for k, v in meta["class_to_label"].items()}  # index -> location
label_to_class = {v: int(k) for k, v in class_to_label.items()}
n_classes  = len(class_to_label)

seed = mlpack.decision_tree(
    training=np.asfortranarray(np.zeros((2, n_features))),
    labels=np.array([0, 1], dtype=np.uint64))
model = seed["output_model"]
with open(MODEL_PATH, "rb") as f:
    model.__setstate__(f.read())

# ── load raw data + engineer features (exactly like training) ─────────────────
wb  = xlrd.open_workbook(DATA_PATH)
ws  = wb.sheet_by_index(0)
num_cols = [COL_TIME, COL_LI103, COL_PSPEED, COL_OSS101, COL_PT101, COL_TT110, COL_TT102]
raw    = np.array([[ws.cell_value(i, c) for c in num_cols] for i in range(1, ws.nrows)],
                  dtype=np.float64)
labels = np.array([str(ws.cell_value(i, COL_LABEL)).strip() for i in range(1, ws.nrows)])

time_ms = raw[:, 0]
li103   = raw[:, 1]
p_speed = raw[:, 2]
tt110   = raw[:, 5]
tt102   = raw[:, 6]

# Windowed slope config, straight from metadata so this reproduces exactly what
# train_model.py and the C++ aSub compute.
DLDT_WINDOW  = meta["d_level_dt"]["window_samples"]          # 12
VENT_STEP_MM = meta["d_level_dt"]["vent_step_mm"]            # 4.0
GAP_S        = meta["d_level_dt"].get("gap_seconds", 300.0)  # session split

delta_t         = tt102 - tt110
level_corrected = li103 - beta * (tt102 - t_ref)
speed_over_dt   = p_speed / np.maximum(np.abs(delta_t), 0.5)

# 12-sample windowed lstsq slope of level_corrected (mm/s), vent- and gap-filtered,
# identical to train_model.py. Used only for the informational "train_pred" column.
t_s     = time_ms / 1000.0
vent    = np.abs(np.diff(li103, prepend=li103[0])) > VENT_STEP_MM
session = np.zeros(len(li103), dtype=int)
session[1:] = np.cumsum(np.diff(t_s) > GAP_S)
d_lev_dt = np.zeros(len(li103))
for i in range(len(li103)):
    sel = np.arange(max(0, i - DLDT_WINDOW + 1), i + 1)
    sel = sel[(session[sel] == session[i]) & (~vent[sel])]
    if len(sel) >= 3:
        tc  = t_s[sel] - t_s[sel].mean()
        den = float(np.dot(tc, tc))
        if den > 0.0:
            lc = level_corrected[sel] - level_corrected[sel].mean()
            d_lev_dt[i] = float(np.dot(tc, lc) / den)


def feature_row(i, d_level_dt):
    """6-feature vector for row i with an explicit d_correctedLev_dt value."""
    return np.array([delta_t[i], tt102[i], level_corrected[i],
                     d_level_dt, p_speed[i], speed_over_dt[i]])


def predict(vecs):
    """Classify already-engineered (un-normalised) feature rows."""
    X = np.asfortranarray((np.atleast_2d(vecs) - feat_min) / feat_range)
    out = mlpack.decision_tree(input_model=model, test=X)
    pred  = np.asarray(out["predictions"]).ravel().astype(int)
    probs = np.asarray(out["probabilities"])
    conf  = probs[np.arange(len(pred)), pred]
    return pred, conf


def in_bounds(i):
    return (SIM_BOUNDS["TT-102:MeasValue"][0] <= tt102[i] <= SIM_BOUNDS["TT-102:MeasValue"][1] and
            SIM_BOUNDS["TT-110:MeasValue"][0] <= tt110[i] <= SIM_BOUNDS["TT-110:MeasValue"][1] and
            SIM_BOUNDS["LI-103:MeasValue"][0] <= li103[i] <= SIM_BOUNDS["LI-103:MeasValue"][1] and
            SIM_BOUNDS["P-101:Speed"][0]      <= p_speed[i] <= SIM_BOUNDS["P-101:Speed"][1])


# ── per-class in-bounds availability report ──────────────────────────────────
print("Per-class injectable (within sim DRVL/DRVH) row availability:")
class_rows = {}
for c in range(n_classes):
    rows_c = np.where(labels == class_to_label[c])[0]
    inb    = [int(i) for i in rows_c if in_bounds(i)]
    class_rows[c] = inb
    print(f"  class {c:2d} ({class_to_label[c]:<20}): "
          f"{len(rows_c):4d} rows, {len(inb):4d} injectable")

# ── select one representative injectable row per class ───────────────────────
# Prefer the row with the smallest |windowed slope| so the train-pipeline and IOC
# (d=0) predictions coincide -> a clean apples-to-apples row.
selected = []
for c in range(n_classes):
    inb = class_rows[c]
    if not inb:
        continue
    i = min(inb, key=lambda k: abs(d_lev_dt[k]))
    selected.append(i)


def rounded_sensors(i):
    return {
        "TT-102:MeasValue": round(float(tt102[i]), 3),
        "TT-110:MeasValue": round(float(tt110[i]), 3),
        "LI-103:MeasValue": round(float(li103[i]), 2),
        "P-101:Speed":      round(float(p_speed[i]), 3),
    }


def ioc_vec_from_sensors(s):
    """6-feature vector the IOC builds from these sensors, with d = 0."""
    t102, t110 = s["TT-102:MeasValue"], s["TT-110:MeasValue"]
    li, spd    = s["LI-103:MeasValue"], s["P-101:Speed"]
    dT = t102 - t110
    return np.array([dT, t102, li - beta * (t102 - t_ref), 0.0, spd,
                     spd / max(abs(dT), 0.5)])


sensors_list = [rounded_sensors(i) for i in selected]
train_vecs   = [feature_row(i, d_lev_dt[i]) for i in selected]      # windowed slope
ioc_vecs     = [ioc_vec_from_sensors(s) for s in sensors_list]      # rounded, d->0
train_pred, _      = predict(train_vecs)
ioc_pred, ioc_conf = predict(ioc_vecs)

manifest = []
print("\nSelected parity rows (IOC-expected uses d_correctedLev_dt = 0):")
print(f"  {'row':>5} {'label':<20} {'train':>5} {'IOC':>4} {'conf':>6}  {'IOC location':<20}")
for k, i in enumerate(selected):
    exp_cls = int(ioc_pred[k])
    rec = {
        "row": int(i),
        "sensors": sensors_list[k],
        "true_label":       str(labels[i]),
        "true_class":       int(label_to_class[str(labels[i])]),
        "train_pred":       int(train_pred[k]),
        "expected_class":   exp_cls,          # raw (ungated) tree class -> aSub VALF
        "expected_location": class_to_label[exp_cls],
        "expected_detect":  0,                 # static d=0 injection never trips the
                                               # physics gate -> LeakDetected stays 0
        "expected_conf":    round(float(ioc_conf[k]), 4),
        "fwd_d_level_dt":   round(float(d_lev_dt[i]), 4),
    }
    manifest.append(rec)
    print(f"  {i:5d} {labels[i]:<20} {train_pred[k]:5d} {ioc_pred[k]:4d} "
          f"{ioc_conf[k]:6.3f}  {class_to_label[exp_cls]:<20}")

with open(OUT_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote {len(manifest)} rows -> {OUT_PATH}")

agree_train = sum(1 for r in manifest if r["train_pred"] == r["expected_class"])
agree_label = sum(1 for r in manifest if r["expected_class"] == r["true_class"])
print(f"train-pred == IOC-expected : {agree_train}/{len(manifest)} "
      f"(differences are the documented windowed-slope vs d=0 shift)")
print(f"IOC-expected == true label : {agree_label}/{len(manifest)}")
