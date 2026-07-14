# Leak-Model Training Bench

A self-contained cooling-skid leak-**location** IOC and teaching bench: this one
folder **builds** the `leakPredict` aSub (`src/`), **trains** the model
(`training/train_model.py`), and **runs** them on *simulated* sensor data — with no
dependency on any other folder. It is organised as:

- `db/` — the EPICS databases (`.db`) and the `.dbd`
- `src/` — the C++ aSub (`leak_predict.cpp`)
- `training/` — the offline trainer plus the `leak_model.bin` / `leak_model_meta.json` it writes
- `test/` — the parity harness and the replay simulator (`training_sim.py`)

`Makefile` and `st.cmd` sit at the `training_model/` root. Students can:

1. **See the model's statistics** — the row-level (optimistic) accuracy /
   balanced-accuracy / macro-F1 **and** the honest leave-one-event-out numbers
   (leak-detection recall, supply/return accuracy, exact-location recall),
   per-class accuracy, hyper-parameters and dataset split — on the training OPI.
2. **Test the model live** — replay real rows of a chosen leak **location** and
   watch the model's predicted location, scored on two axes against ground truth:
   **exact-location match** and **leak/no-leak detection match**.

> **Why two axes?** Exact leak location is not physically observable from these
> lumped skid sensors (all branches merge off-skid; see the leave-one-event-out
> check in `train_model.py`, Step 9). Only leak **detection** generalises.
> The bench makes that visible: on replayed data the exact-location match looks
> high, but detection is the number to trust.

## Files

| File | Role |
|------|------|
| `src/leak_predict.cpp` · `db/leak_predict.dbd` · `Makefile` | **Build sources** for the `leakpredict` aSub module — `make install` (from `training_model/`) compiles + installs them |
| `db/leak_predict.db` | The inference record (`$(P)LeakML:*`), installed as a module template |
| `st.cmd` | The single IOC startup — hosts both sensor streams; `$(P)Sim:Enable` switches the model between the replayed `:SimRaw` sources and the live gateway |
| `db/leak_sensors_sim.db` | The 5 `:SimRaw` sim sources + the 4 `$(P)<tag>:ModelIn` sim/live routing records |
| `db/leak_train.db` | Controls (incl. `$(P)Sim:Enable`), class-match scoreboard, and stats PVs |
| `test/training_sim.py` | Replays dataset rows into `:SimRaw` + publishes model stats over Channel Access |
| `training/train_model.py` | The **single** offline trainer. Trains the mlpack `DecisionTree` and writes the two files below into `training/` — the model the IOC loads |
| `training/leak_model.bin` · `training/leak_model_meta.json` | **Generated** by `train_model.py`: the serialised tree the aSub loads, plus normalisation ranges, the class↔location map, the `detection`-gate thresholds, and the stats the bench screen reads |
| `test/parity_reference.py` · `test/parity_drive.py` · `test/parity_manifest.json` | Parity harness — proves the C++ aSub reproduces the Python model's class exactly |

> **⚠️ The leak-DETECTION gate is a STUDENT BONUS EXERCISE — currently stubbed out.**
> The gate logic, its thresholds, and the `LeakDetected` / `LevelRate` / `CumDrop5min`
> records have been removed so students can build them; in the scaffold `LeakDetected`
> is always "No leak". Build it per **`docs/EXERCISE_leak_gate.md`** (reference answer
> in **`solutions/`**). The rest of this README describes the *intended* gate.

> **Deployed leak/no-leak is a real-time physics gate, not the tree.** This bench
> scores the model's **advisory tree location** (`LeakML:Predict.VALF`) on replayed
> rows — the exact-location axis. The trustworthy leak **detection** (`LeakDetected`)
> comes from a sustained level-drop gate that needs wall-clock time to integrate, so
> it can't be driven by time-compressed replay. To watch detection fire on the real
> plant, flip **`$(P)Sim:Enable`** to *Live PVs* and restore real-time scanning
> (`caput $(P)LeakML:Predict.SCAN "5 second"`), then watch
> `LeakML:LevelRate` / `LeakML:CumDrop5min` / `LeakML:LeakDetected`.
>
> **Why detection was taken out of the tree** (full write-up in
> [`docs/EXERCISE_leak_gate.md`](../docs/EXERCISE_leak_gate.md) §1; also carried in
> `leak_model_meta.json` → `detection_rationale`):
> 1. Each leak location is exactly **one** contiguous drain event, so the tree splits
>    on that event's **absolute operating-point fingerprint** (temporal leakage), not
>    the causal leak signal.
> 2. On a held-out event (leave-one-event-out / Regime B) the tree
>    **constant-predicts the largest leak class** — tree-based detection collapses and
>    exact-location recall is 0. The high row-level (Regime A) score is an
>    autocorrelation artifact, not real localisation.
> 3. A genuine leak is **causal and condition-invariant** — escaping coolant drops
>    `level_corrected` in a sustained way — which the gate detects and which
>    generalises (LOEO detection ≈ 88%) regardless of the day's heat load.
> 4. The heat-load confound (thermal expansion / thermal-lag residual "drops") is
>    rejected by the gate's **temperature-transient guard** — physics, not learned
>    splits. So the tree is kept only as an **advisory location hint**.

> The IOC is **fully self-contained**: `training/train_model.py` writes
> `leak_model.bin` + `leak_model_meta.json` into `training/`, `st.cmd` points
> `LEAK_MODEL_DIR` there, and `training_sim.py` reads that same
> `leak_model_meta.json`. Retrain, restart the IOC, and press **Reload stats** on
> the OPI to refresh the numbers. The aSub loads the `.bin` and reads its
> `detection`-gate thresholds from the meta — this is the production model
> end-to-end, no other folder required.

## How the sim/live switch works

Both sensor streams exist at once, as distinct PVs, and the model reads one
switchable input per sensor:

- **Simulated stream** — the `…<tag>:SimRaw` sources that `training_sim.py`
  writes (it adds any noise before the write).
- **Live stream** — the real skid PVs `…<tag>:MeasValue` / `:Speed`. Not defined
  locally, so they resolve on the ESS read-only gateway.
- **Switch** — `…<tag>:ModelIn` (`ai`, one per model sensor) is in EPICS
  **simulation mode** on `…Sim:Enable`: *Simulated* → it reads its `SIOL`
  (the `:SimRaw` replay); *Live PVs* → it reads its `INP` (the live gateway PV).

The `leakPredict` aSub samples the four `…<tag>:ModelIn` records, so switching
`…Sim:Enable` re-points the model between bench replay and the live plant without
any PV-name collision. The sim path is simply
`training_sim.py → :SimRaw → :ModelIn → aSub`.

## Run it

```bash
conda activate e3-mlpack
cd training_model

# one-time: build/install the leakPredict aSub module (the Makefile lives here;
# sources are under src/ and db/).
make install

# train the model — writes leak_model.bin + leak_model_meta.json into training/
( cd training && python3 train_model.py )

# terminal 1 — the IOC, from training_model/. Boots in bench mode
# (Sim:Enable = Simulated); the model reads the replayed :SimRaw sources.
iocsh st.cmd
#   ...to point the SAME model at the REAL skid instead of replayed data,
#   flip the switch (no restart, no terminal 2) and restore real-time scanning:
#   caput CWM-CWS02:WtrC-Sim:Enable "Live PVs"
#   caput CWM-CWS02:WtrC-LeakML:Predict.SCAN "5 second"

# terminal 2 — the data replayer + stats publisher (bench mode only), from test/
cd test && python3 training_sim.py     # offline check first: python3 training_sim.py --selftest

# then open OPI/leak_training.bob in Phoebus
```

On the OPI: pick a **Replay scenario** (a leak location), choose **sequential** or
**shuffled** row order, turn **Stream** to *Running*, and watch the predicted
**Location** vs the **TRUE** location while the session **exact-location** and
**detection** match-rates accumulate. Because each location was drained only once,
watch how a replayed leak is often mapped to a *different* location of similar flow
signature while **detection** stays correct — the observability gap, live.

## Prove the C++ matches the Python model (parity)

With the IOC running (bench mode), from `test/`:

```bash
cd test
python3 parity_reference.py   # regenerate parity_manifest.json from the deployed model
python3 parity_drive.py       # drive the sim path and compare the IOC vs the model
```

`parity_drive.py` sets `Sim:Enable = Simulated`, writes each row to the `:SimRaw`
sources, and PROCs the aSub enough times to fill the level-slope window so
`d_correctedLev_dt == 0` — matching `parity_reference.py` exactly.
