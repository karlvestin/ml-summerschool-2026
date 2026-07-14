# Reference solution — leak-detection gate bonus exercise

**Instructor answer key. Do not include in the student hand-out.**

These are verbatim copies of the working files *before* the leak-DETECTION gate was
removed to make [`docs/EXERCISE_leak_gate.md`](../docs/EXERCISE_leak_gate.md). They
are the complete, deployed physics gate — the reference students diff against.

| File here | Replaces (student scaffold) | What the gate parts are |
|-----------|------------------------------|-------------------------|
| `leak_predict.cpp` | `training_model/leak_predict.cpp` | `cumDrop` integrator, `rawLeak` gate, vent-hold + debounce/latch, gate state in `LeakModelCtx`, `"detection"` threshold loading in `loadMeta()` |
| `train_model.py` | `training_model/train_model.py` | `DET_*` threshold constants + the `"detection"` metadata block (with calibration stats) |
| `leak_model_meta.json` | `training_model/leak_model_meta.json` | the `"detection"` object the aSub reads |
| `leak_predict.db` | `training_model/leak_predict.db` | the `LeakDetected` (bi), `LevelRate` / `CumDrop5min` (ai) records + their fanout links |
| `parity_drive.py` | `training_model/parity_drive.py` | the *original* driver (asserts `LeakDetected` exists and stays 0). The scaffold version tolerates the PV being absent so tree-parity runs before the bonus is done. |

To restore the full working IOC, copy these back over `training_model/` and rebuild
(`make install`). The OPI (`OPI/*.bob`) was left untouched — it already references
the detection PVs.
