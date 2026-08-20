#!/usr/bin/env python3
"""
=============================================================================
training_sim.py — data simulator + stats publisher for the leak-model bench
                  (LEAK-LOCATION CLASSIFICATION version)
=============================================================================

Companion to the training-bench IOC (st.cmd + leak_train.db). It does two
things, both over plain Channel Access (it shells out to `caput`/`caget`, so it
needs NO Python EPICS client library — only what the e3-mlpack env already has):

  1. STATISTICS — on start, and whenever the OPI sets $(P)Stats:Reload, it reads
     leak_model_meta.json and publishes the model's performance into $(P)Stats:*
     — the row-level (optimistic) accuracy / balanced-acc / macro-F1 AND the
     honest leave-one-event-out numbers (detection / supply-return / exact
     location), plus per-class accuracy, hyper-params and split sizes.

  2. SIMULATION — while $(P)Sim:Run is On, every cycle it replays one real row of
     the leak-LOCATION class picked in $(P)Sim:Scenario, writes the five sensor
     values to the $(P)<tag>:SimRaw PVs, forces one inference, then scores the
     model's ADVISORY tree class ($(P)LeakML:Predict.VALF) against the true class:
       Match       — exact location correct (tree class == true class)
       DetectMatch — tree leak vs no-leak correct
     (The DEPLOYED leak/no-leak comes from a real-time physics gate, not the tree,
     and can't be shown by time-compressed replay — flip $(P)Sim:Enable to Live
     PVs and restore SCAN=5 second to watch it on the real skid. This bench
     illustrates the tree's advisory location only.)
     $(P)Sim:Mode picks sequential vs shuffled row order; $(P)Sim:Noise scales
     added Gaussian jitter (in per-class std-dev units). There is no synthetic
     mode: a location target has no continuous axis to interpolate along.

Run (from this directory, e3-mlpack env active):
    python3 training_sim.py
    python3 training_sim.py --selftest   # load data + stats offline, no IOC
Stop with Ctrl-C.
=============================================================================
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import xlrd

# ─── Paths / config ────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))   # training_model/test/
DATA_PATH  = os.environ.get("DATASET", os.path.join(HERE, "..", "..", "data", "Dataset.xls"))
MODEL_DIR  = os.environ.get("LEAK_MODEL_DIR", os.path.join(HERE, "..", "training"))  # train_model.py writes the model there
META_PATH  = os.path.join(MODEL_DIR, "leak_model_meta.json")
PREFIX     = os.environ.get("P", "CWM-CWS02:WtrC-")

PUSH_PERIOD = 5.0    # target seconds per sample (so the aSub's slope dt ≈ 5 s)
PROC_SETTLE = 0.8    # wait after forcing inference for the CP readback to settle
CA_TIMEOUT  = 3.0    # seconds for each caget/caput

DRY = "--selftest" in sys.argv   # offline: stub CA, verify data + stats parsing

# Column layout of Dataset.xls (mirrors model/train_model.py). Label is a STRING.
COL_TIME, COL_LI103, COL_PSPEED, COL_OSS101, COL_PT101, COL_TT110, COL_TT102, COL_LABEL = range(8)

# Sensor order pushed to the :SimRaw PVs, with the tags used in the PV names.
SENSOR_TAGS = ["PT-101", "TT-102", "TT-110", "LI-103", "P-101"]   # P-101 -> :Speed


# ─── Channel Access helpers (subprocess-based) ─────────────────────────────────
def caput(pv, value):
    if DRY:
        print(f"    caput {pv} = {value}")
        return
    try:
        subprocess.run(["caput", "-t", "-w", str(CA_TIMEOUT), pv, str(value)],
                       check=False, capture_output=True, timeout=CA_TIMEOUT + 2)
    except subprocess.TimeoutExpired:
        print(f"  ! caput timeout on {pv}", file=sys.stderr)


def caput_array(pv, values):
    if DRY:
        print(f"    caput -a {pv} [{len(values)}] = {list(values)}")
        return
    args = ["caput", "-a", "-t", "-w", str(CA_TIMEOUT), pv, str(len(values))]
    args += [str(v) for v in values]
    try:
        subprocess.run(args, check=False, capture_output=True, timeout=CA_TIMEOUT + 2)
    except subprocess.TimeoutExpired:
        print(f"  ! caput timeout on {pv}", file=sys.stderr)


def caget_many(pvs):
    """Return {pv: float} for a list of PVs (numeric, terse). NaN on failure."""
    if DRY:
        return {pv: float("nan") for pv in pvs}
    try:
        out = subprocess.run(["caget", "-t", "-n", "-w", str(CA_TIMEOUT), *pvs],
                             check=False, capture_output=True, timeout=CA_TIMEOUT + 2,
                             text=True).stdout.split()
    except subprocess.TimeoutExpired:
        return {pv: float("nan") for pv in pvs}
    vals = {}
    for pv, tok in zip(pvs, out):
        try:
            vals[pv] = float(tok)
        except ValueError:
            vals[pv] = float("nan")
    return vals


# ─── Metadata (class map) ──────────────────────────────────────────────────────
def load_class_map():
    with open(META_PATH) as f:
        meta = json.load(f)
    # class index (int) -> location string, and its inverse.
    class_to_label = {int(k): str(v) for k, v in meta["class_to_label"].items()}
    label_to_class = {v: k for k, v in class_to_label.items()}
    return class_to_label, label_to_class


# ─── Dataset → per-class pools and statistics ──────────────────────────────────
def load_dataset(label_to_class):
    ws = xlrd.open_workbook(DATA_PATH).sheet_by_index(0)
    # Numeric sensors are read as float; the label column is a location STRING.
    nrows = ws.nrows
    sensors = np.array([[ws.cell_value(i, c) for c in
                         (COL_PT101, COL_TT102, COL_TT110, COL_LI103, COL_PSPEED)]
                        for i in range(1, nrows)], dtype=np.float64)
    labels = np.array([str(ws.cell_value(i, COL_LABEL)).strip() for i in range(1, nrows)])
    y = np.array([label_to_class.get(l, -1) for l in labels], dtype=int)

    pools, std = {}, {}
    for c in sorted(set(y)):
        if c < 0:
            continue
        rows = sensors[y == c]
        pools[c] = rows
        std[c]   = rows.std(axis=0)
    print(f"  Loaded {len(sensors)} rows, {len(pools)} classes from {os.path.basename(DATA_PATH)}")
    return pools, std


# ─── Statistics: meta JSON → $(P)Stats:* ───────────────────────────────────────
def publish_stats():
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  ! cannot read {META_PATH}: {e}", file=sys.stderr)
        return

    # Row-level (OPTIMISTIC) accuracy / balanced-acc / macro-F1.
    m = meta.get("metrics_rowlevel_optimistic", {})
    def rl(split, key):
        sub = m.get(split, {})
        return sub.get(key, 0.0) if isinstance(sub, dict) else 0.0

    caput(f"{PREFIX}Stats:TestAcc",    100.0 * rl("test",  "acc"))
    caput(f"{PREFIX}Stats:ValAcc",     100.0 * rl("val",   "acc"))
    caput(f"{PREFIX}Stats:TrainAcc",   100.0 * rl("train", "acc"))
    caput(f"{PREFIX}Stats:TestBalAcc", 100.0 * rl("test",  "bal_acc"))
    caput(f"{PREFIX}Stats:TestF1",     100.0 * rl("test",  "macro_f1"))
    caput(f"{PREFIX}Stats:ValF1",      100.0 * rl("val",   "macro_f1"))
    caput(f"{PREFIX}Stats:TrainF1",    100.0 * rl("train", "macro_f1"))

    # HONEST leave-one-event-out — the trustworthy verdict.
    lo = meta.get("honest_leave_one_event_out", {})
    caput(f"{PREFIX}Stats:LOEODetect", 100.0 * lo.get("leak_detection_recall", 0.0))
    caput(f"{PREFIX}Stats:LOEOSide",   100.0 * lo.get("supply_return_accuracy", 0.0))
    caput(f"{PREFIX}Stats:LOEOExact",  100.0 * lo.get("exact_location_recall", 0.0))

    # Physics gate.
    gates = meta.get("verification_gates", {})
    caput(f"{PREFIX}Stats:LevelGate", 1 if gates.get("level_pass") else 0)

    sp = meta.get("split", {})
    caput(f"{PREFIX}Stats:NTotal",   int(sp.get("n_total", 0)))
    caput(f"{PREFIX}Stats:NTrain",   int(sp.get("n_train", 0)))
    caput(f"{PREFIX}Stats:NVal",     int(sp.get("n_val", 0)))
    caput(f"{PREFIX}Stats:NTest",    int(sp.get("n_test", 0)))
    caput(f"{PREFIX}Stats:NEvents",  int(sp.get("n_events", 0)))
    caput(f"{PREFIX}Stats:NClasses", int(meta.get("n_classes", 0)))

    hp = meta.get("hyperparams", {})
    caput(f"{PREFIX}Stats:MaxDepth", int(hp.get("max_depth", 0)))
    caput(f"{PREFIX}Stats:MinLeaf",  int(hp.get("min_leaf_size", 0)))

    pc = meta.get("per_class", {})
    labels = pc.get("labels", [])                     # location strings now
    accs   = [0.0 if a is None else 100.0 * a for a in pc.get("test_acc", [])]
    counts = pc.get("test_count", [])
    if labels:
        caput_array(f"{PREFIX}Stats:ClassLabel",     labels)
        caput_array(f"{PREFIX}Stats:ClassTestAcc",   accs)
        caput_array(f"{PREFIX}Stats:ClassTestCount", counts)
    print(f"  Published stats: test_acc={100*rl('test','acc'):.2f}%  "
          f"test_balacc={100*rl('test','bal_acc'):.2f}%  "
          f"test_f1={100*rl('test','macro_f1'):.2f}%  |  "
          f"LOEO detect={100*lo.get('leak_detection_recall',0):.1f}%  "
          f"side={100*lo.get('supply_return_accuracy',0):.1f}%  "
          f"exact={100*lo.get('exact_location_recall',0):.1f}%")


# ─── Sample generation (replay only) ───────────────────────────────────────────
def replay_sample(pools, cls, cursors, shuffled, noise, std):
    """Next real row for the chosen class index (sequential or shuffled)."""
    rows = pools.get(cls)
    if rows is None or len(rows) == 0:
        return None
    if shuffled:
        i = np.random.randint(len(rows))
    else:
        i = cursors.get(cls, 0) % len(rows)
        cursors[cls] = i + 1
    sample = rows[i].copy()
    if noise > 0:
        sample = sample + noise * std[cls] * np.random.randn(5)
    return sample


def push_sensors(sample):
    """Write the five sensor values to their :SimRaw PVs."""
    for tag, val in zip(SENSOR_TAGS, sample):
        caput(f"{PREFIX}{tag}:SimRaw", round(float(val), 4))


# ─── Main loop ─────────────────────────────────────────────────────────────────
CTRL_PVS = [f"{PREFIX}Sim:Run", f"{PREFIX}Sim:Mode", f"{PREFIX}Sim:Scenario",
            f"{PREFIX}Sim:Noise", f"{PREFIX}Stats:Reload", f"{PREFIX}Sim:Reset"]


def reset_scoreboard():
    for pv, v in ((f"{PREFIX}Sim:NSamples", 0), (f"{PREFIX}Sim:RunAcc", 0),
                  (f"{PREFIX}Sim:RunDetectAcc", 0), (f"{PREFIX}Sim:Match", 0),
                  (f"{PREFIX}Sim:DetectMatch", 0)):
        caput(pv, v)


def selftest(pools, class_to_label):
    """Offline sanity check: exercise data + stats paths without an IOC."""
    print("\n[selftest] publish_stats():")
    publish_stats()
    print("\n[selftest] replay one row of each class (with 0.3 noise):")
    cursors = {}
    for c in sorted(pools):
        s = replay_sample(pools, c, cursors, False, 0.3, {c: pools[c].std(axis=0)})
        print(f"    class {c} {class_to_label.get(c,'?'):<20} "
              f"PT={s[0]:.2f} TT102={s[1]:.2f} TT110={s[2]:.2f} LI={s[3]:.1f} P={s[4]:.2f}")
    print("\n[selftest] OK — no crashes on string labels / new meta schema.")


def main():
    print("=" * 64)
    print("  Cooling-skid leak model — training bench simulator (classification)")
    print("=" * 64)
    class_to_label, label_to_class = load_class_map()
    pools, std = load_dataset(label_to_class)

    if DRY:
        selftest(pools, class_to_label)
        return

    publish_stats()
    print("  Waiting for $(P)Sim:Run = Running ...  (Ctrl-C to quit)")

    cursors = {}
    n = hits = det_hits = 0

    while True:
        c = caget_many(CTRL_PVS)
        if c.get(f"{PREFIX}Stats:Reload", 0) >= 1:
            publish_stats()
            caput(f"{PREFIX}Stats:Reload", 0)

        if c.get(f"{PREFIX}Sim:Reset", 0) >= 1:
            n = hits = det_hits = 0
            reset_scoreboard()
            caput(f"{PREFIX}Sim:Reset", 0)
            print("  scoreboard reset", flush=True)

        if c.get(f"{PREFIX}Sim:Run", 0) < 1:
            time.sleep(1.0)
            continue

        shuffled = int(c.get(f"{PREFIX}Sim:Mode", 0) or 0) == 1
        noise    = max(0.0, c.get(f"{PREFIX}Sim:Noise", 0.0) or 0.0)
        true_cls = int(c.get(f"{PREFIX}Sim:Scenario", 0) or 0)
        true_cls = min(max(true_cls, 0), len(class_to_label) - 1)
        t0       = time.time()

        sample = replay_sample(pools, true_cls, cursors, shuffled, noise, std)
        if sample is None:
            print(f"  (no rows for class {true_cls}; skipping)")
            time.sleep(PUSH_PERIOD)
            continue

        push_sensors(sample)
        caput(f"{PREFIX}Sim:TrueClass", true_cls)

        # Force ONE inference on exactly these sensors. The bench sets the aSub SCAN
        # to Passive (see st.cmd), so this is the only thing that processes it — the
        # prediction is deterministically aligned to the pushed sample.
        caput(f"{PREFIX}LeakML:Predict.PROC", 1)
        time.sleep(PROC_SETTLE)
        # Score the ADVISORY tree class (Predict.VALF), NOT the deployed ClassIdx:
        # the latter is gated by the physics detector, a real-time sustained-drop
        # rule that can't be exercised by time-compressed replay. VALF is the raw
        # tree location this bench is meant to illustrate.
        pv = f"{PREFIX}LeakML:Predict.VALF"
        pred = caget_many([pv]).get(pv, float("nan"))
        if not np.isnan(pred):
            pred_cls = int(round(pred))
            n += 1
            hit     = (pred_cls == true_cls)                       # exact location (tree)
            det_hit = ((pred_cls != 0) == (true_cls != 0))         # tree leak vs no-leak
            hits     += int(hit)
            det_hits += int(det_hit)
            caput(f"{PREFIX}Sim:Match",         1 if hit else 0)
            caput(f"{PREFIX}Sim:DetectMatch",   1 if det_hit else 0)
            caput(f"{PREFIX}Sim:NSamples",      n)
            caput(f"{PREFIX}Sim:RunAcc",        round(100.0 * hits / n, 2))
            caput(f"{PREFIX}Sim:RunDetectAcc",  round(100.0 * det_hits / n, 2))
            print(f"  #{n:4d}  true={true_cls} [{class_to_label.get(true_cls,'?')}]  "
                  f"pred={pred_cls} [{class_to_label.get(pred_cls,'?')}]  "
                  f"{'HIT ' if hit else 'miss'}  detect={'ok' if det_hit else 'X'}  "
                  f"loc-acc={100*hits/n:5.1f}%  det-acc={100*det_hits/n:5.1f}%", flush=True)

        dt = time.time() - t0
        if dt < PUSH_PERIOD:
            time.sleep(PUSH_PERIOD - dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
