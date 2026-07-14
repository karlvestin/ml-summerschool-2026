# Bonus Exercise — Build the Leak-Detection Gate

**Optional. Attempt it during the training if you have time.** The classification
IOC (the mlpack DecisionTree that predicts a leak *location*) is complete and runs
without this exercise. What is **removed** from the code you were given is the
**leak-DETECTION gate**: the real-time rule that actually decides *leak vs no-leak*.
Your job is to build it back.

A full reference answer for every file below lives in **`solutions/`** — try it
yourself first.

---

## 1. Why detection is a gate, not the tree (read this first)

**In short** (this summary is mirrored verbatim in `leak_model_meta.json` →
`detection_rationale`, `train_model.py`, `leak_predict.cpp`, and both READMEs):

1. Each leak location is exactly **one** contiguous drain event, so the tree splits
   on that event's **absolute operating-point fingerprint** (temporal leakage), not
   the causal leak signal.
2. On a held-out event (leave-one-event-out / Regime B) the tree
   **constant-predicts the largest leak class** — tree-based detection collapses and
   exact-location recall is 0. The high row-level (Regime A) score is an
   autocorrelation artifact.
3. A genuine leak is **causal and condition-invariant** — escaping coolant drops
   `level_corrected` in a sustained way — which the gate detects and which
   generalises (LOEO detection ≈ 88%) regardless of the day's heat load.
4. The heat-load confound (thermal expansion / thermal-lag residual "drops") is
   rejected by the gate's **temperature-transient guard** — physics, not learned
   splits. The tree is kept only as an **advisory location hint**.

The rest of this section explains each point.

Each leak location in `data/Dataset.xls` appears as exactly **one** contiguous
drain event. A decision tree splits on **absolute** feature values, so it just
memorises the operating-point *fingerprint* of that single event (temporal
leakage). On a held-out event (leave-one-event-out, Regime B in `train_model.py`)
the tree **constant-predicts the largest leak class** — it never learned the causal
leak signal. So the tree is kept only as an **advisory location hint**; it must
**not** decide leak/no-leak.

A real leak has a **causal, condition-invariant** signature: coolant is escaping,
so the expansion-vessel level **drops in a sustained way**. That is what you detect
— with real-time signal processing on the level, not with the tree. It generalises
(LOEO detection ≈ 88%) precisely because it rests on physics that holds regardless
of the day's heat load.

**The hard part — telling a leak from a heat-load change.** A rising heat load also
moves the level (thermal expansion). `level_corrected` already subtracts
`β·(TT102 − t_ref)`, but that correction is steady-state; during a fast temperature
ramp, thermal lag leaves a residual fake "drop". So the gate must also reject fast
temperature transients. That guard is the core ML challenge, encoded as physics.

---

## 2. What you build (four files)

| File | Task |
|------|------|
| `training_model/leak_predict.cpp` | The gate math + debounce/latch, in the `leakPredict` aSub (the main task) |
| `training_model/train_model.py` | Calibrate the gate thresholds from the no-leak history and emit them |
| `training_model/leak_model_meta.json` | The `"detection"` object the aSub reads (produced by the trainer) |
| `training_model/leak_predict.db` | The three readback records that surface the gate outputs |

The OPI (`OPI/leak_predict.bob`) already has the widgets for `LeakDetected`,
`LevelRate` and `CumDrop5min`; they read blank until you create those PVs, then
come alive — that is your live feedback.

---

## 3. The gate, precisely

Everything you need is already computed in `leakPredict` (`leak_predict.cpp`) and
handed to you at the marked `STUDENT BONUS EXERCISE` block:

- `dLevelDt` — windowed least-squares slope of `level_corrected` (mm/s). Negative = draining.
- `dTt102Dt` — slope of `TT-102` over the same window (°C/s). The thermal guard.
- `ventInWindow` — a big raw-LI103 jump (an N₂ top-up) landed in the window; the fit is untrustworthy.

You must additionally compute one signal and then the decision:

**(a) `cumDrop`** — the trailing-window cumulative drop of `level_corrected`:
`mean(newest few non-vent samples) − mean(oldest few non-vent samples inside the
window)`. Negative = draining. This is the clean discriminator (no-leak stays within
a few mm; every real leak drops ≥ ~16 mm over the window).

**(b) the raw per-scan verdict:**
```
rawLeak = (cumDrop  <= cum_drop_trip_mm)      &&   // sustained loss of water AND
          (dLevelDt <= slope_trip_mm_s)       &&   // dropping fast enough AND
          (fabs(dTt102Dt) <= tt102_slope_trip_c_s) // temperature NOT lurching (else heat-load change)
```

**(c) robustness:**
- If `ventInWindow`, **hold** the current latched state (don't trust a vented window).
- **Debounce:** require `debounce_scans` consecutive agreeing scans before you flip
  the latched `detected` state on or off (kills single-sample glitches).

**(d) outputs** (already wired to the record's `VAL*`): gate `VALA` (location only when
`detected`), set `VALC = detected`, `VALE = cumDrop`. `VALD = dLevelDt` is already set.

Add the gate **state** to `LeakModelCtx` (the thresholds, the `goodCount`/`badCount`
debounce counters, and the latched `detected`) and **read the thresholds** from the
metadata `"detection"` object in `loadMeta()` (fall back to compiled defaults).

---

## 4. Calibrating the thresholds (`train_model.py` → meta)

Derive the numbers from the data instead of hard-coding blindly:

- `no_leak_sigma = d_correctedLev_dt[nonleak_mask].std()`  (≈ 0.0019 mm/s here).
- **`slope_trip_mm_s`** — several σ below 0. ~8σ (≈ **−0.015 mm/s**) puts the trip far
  out in the no-leak noise tail while still catching the weakest real leak
  (HBL LWU Return ≈ −0.028 mm/s). See the σ table in `solutions/leak_predict.cpp`.
- **`cum_drop_trip_mm`** — in the gap between the worst no-leak drift over the window
  (≈ −5 mm) and the weakest real-leak drop (≈ −25 mm): **−12 mm** over
  **`cum_window_s` = 900 s** (15 min).
- **`tt102_slope_trip_c_s`** — the divider between steady operation and a deliberate
  heat-load ramp: **0.01 °C/s** (≈ 0.6 °C/min).
- **`debounce_scans`** — **3**.

Write these into a `"detection"` object in the metadata dict so the aSub reads them.

---

## 5. Acceptance criteria

1. **Builds & runs:** `make install`, then the IOC starts and `LeakDetected` /
   `LevelRate` / `CumDrop5min` appear.
2. **Calibration target:** **0% no-leak false positives** and **every leak class
   detected** on the dataset (the honest LOEO detection recall stays ~88%).
3. **Tree parity still passes:** `python3 parity_reference.py` then
   `python3 parity_drive.py` against a running IOC — `VALF` matches the model and
   (now that it exists) `LeakDetected` stays **0** under static injection.
4. **Heat-load rejection:** during a `-5/-10/-15` heat-load ramp (fast `TT-102`),
   the gate does **not** fire.

---

## 6. How to test

```bash
conda activate e3-mlpack
cd training_model
python3 train_model.py      # re-emits leak_model.bin + meta (now with "detection")
make install                # rebuild the aSub with your gate
# run the IOC yourself, then in another shell:
python3 parity_reference.py
python3 parity_drive.py
```

Detection needs real wall-clock time to integrate the 15-min window, so watch it on
**live** data (`iocsh st.live.cmd`) via `LeakML:LevelRate` / `CumDrop5min` /
`LeakDetected` — time-compressed replay can't drive it.

---

## 7. Reference solution

`solutions/leak_predict.cpp`, `solutions/train_model.py`,
`solutions/leak_model_meta.json`, `solutions/leak_predict.db`. Diff your work
against these. Instructors: keep `solutions/` out of the student hand-out.
