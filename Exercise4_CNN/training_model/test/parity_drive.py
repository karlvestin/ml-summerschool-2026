"""
Replay the parity manifest against a RUNNING leak-prediction IOC and check that
the live cpp aSub's TREE inference matches the deployed model (parity_reference.py).

NOTE: leak/no-leak is a physics gate (a sustained level drop), not the tree. This
parity check targets the TREE, so it reads the RAW (ungated) tree class on the
aSub's VALF field — which is what the reference manifest's expected_class holds.

The $(P)LeakML:LeakDetected PV is part of the leak-DETECTION gate, which is a
STUDENT BONUS EXERCISE (docs/EXERCISE_leak_gate.md). If it is not built yet the PV
won't exist; this driver then SKIPS the detection check and verifies the tree class
only. When it does exist, static row injection holds the level constant so the gate
never fires -> LeakDetected must stay 0.

This drives the SIMULATED path of the single switchable IOC (st.cmd):
  * $(P)Sim:Enable is set to Simulated (On), so each $(P)<tag>:ModelIn routing
    record reads its :SimRaw replay instead of the live gateway.
  * We write the raw sensor values to the $(P)<tag>:SimRaw sources; those flow
    SimRaw -> ModelIn -> the aSub.

Protocol per row:
  1. caput the 4 raw sensor values to their :SimRaw PVs (PT-101 is not a model input).
  2. Let the SimRaw -> ModelIn chain settle, then PROC the aSub N_FILL
     (>= DLDT_WINDOW+1) times so the whole 12-sample slope window refills with one
     constant level -> the cpp computes d_correctedLev_dt == 0 exactly, matching the
     reference. (The bench keeps LeakML:Predict.SCAN = Passive, so we PROC it
     ourselves rather than relying on the autonomous 5 s scan.)
  3. caget Predict.VALF (raw tree class); if present, LeakDetected; Confidence.

PASS criterion: VALF == expected_class AND (LeakDetected absent OR == expected_detect (0)).
Confidence is reported within a tolerance (informational).

Prereq: launch the IOC yourself first, e.g.
    conda activate e3-mlpack && cd training_model && iocsh st.cmd
Then:
    conda run -n e3-mlpack python3 parity_drive.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PREFIX     = "CWM-CWS02:WtrC-"
HERE       = Path(__file__).resolve().parent
MANIFEST   = str(HERE / "parity_manifest.json")
CA_TIMEOUT = "3"

# Slope-window fill: PROC the aSub this many times per row (>= DLDT_WINDOW+1 = 13)
# so the 12-sample level-slope window refills with one constant level -> slope 0.
N_FILL     = 14
SETTLE_S   = 1.0      # let SimRaw -> ModelIn propagate before the first PROC
FILL_GAP_S = 0.3      # spacing between PROCs (distinct timestamps; lets CA settle)

ENABLE_PV = PREFIX + "Sim:Enable"            # bo: Simulated (On) routes ModelIn <- :SimRaw
PROC_PV   = PREFIX + "LeakML:Predict.PROC"   # force one aSub process
CLASS_PV  = PREFIX + "LeakML:Predict.VALF"   # raw (ungated) tree class
DETECT_PV = PREFIX + "LeakML:LeakDetected"
CONF_PV   = PREFIX + "LeakML:Confidence"


def caput(pv, val):
    subprocess.run(["caput", "-w", CA_TIMEOUT, pv, str(val)],
                   check=True, capture_output=True, text=True)


def caget(pv):
    r = subprocess.run(["caget", "-t", "-w", CA_TIMEOUT, pv],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"caget {pv} failed: {r.stderr.strip() or 'no value'}")
    return r.stdout.strip()


def caget_opt(pv):
    """Like caget but returns None if the PV can't be read (e.g. the bonus
    detection gate isn't built yet, so LeakDetected doesn't exist)."""
    try:
        return caget(pv)
    except Exception:
        return None


def simraw_pv(suffix):
    """Manifest sensor suffix (e.g. 'TT-102:MeasValue', 'P-101:Speed') -> the
    :SimRaw source PV that feeds the simulated path for that sensor's tag."""
    tag = suffix.split(":", 1)[0]
    return f"{PREFIX}{tag}:SimRaw"


def main():
    with open(MANIFEST) as f:
        rows = json.load(f)

    # Connectivity check.
    try:
        caget(CLASS_PV)
    except Exception as e:
        print(f"ERROR: cannot reach the IOC ({e}).")
        print("Start it first:  conda activate e3-mlpack && cd training_model && iocsh st.cmd")
        sys.exit(2)

    # Route the model inputs to the simulated stream for the whole run.
    caput(ENABLE_PV, 1)   # Simulated -> ModelIn reads the :SimRaw replay

    print(f"Driving {len(rows)} rows via the sim path "
          f"(Sim:Enable=On, {N_FILL} PROCs/row). Comparing IOC vs deployed model.\n")
    print(f"  {'row':>5} {'exp_cls':>7} {'got_cls':>7} {'exp_det':>7} {'got_det':>7} "
          f"{'exp_cf':>6} {'got_cf':>6}  result")

    npass = 0
    for r in rows:
        # Push this row's sensors to their :SimRaw sources (sim path).
        for suffix, val in r["sensors"].items():
            caput(simraw_pv(suffix), val)
        # Let SimRaw -> ModelIn settle, then refill the slope window.
        time.sleep(SETTLE_S)
        for _ in range(N_FILL):
            caput(PROC_PV, 1)
            time.sleep(FILL_GAP_S)

        got_cls = int(round(float(caget(CLASS_PV))))
        det_raw = caget_opt(DETECT_PV)                # None if the bonus gate PV isn't built
        got_det = int(round(float(det_raw))) if det_raw is not None else None
        got_conf = float(caget(CONF_PV))

        cls_ok  = got_cls == r["expected_class"]
        det_ok  = got_det is None or got_det == r["expected_detect"]
        conf_ok = abs(got_conf - r["expected_conf"]) <= 0.02
        ok = cls_ok and det_ok
        npass += ok
        flag = "PASS" if ok else "FAIL"
        if got_det is None:
            flag += " (LeakDetected n/a — bonus gate not built)"
        elif not conf_ok:
            flag += " (conf differs)"
        det_str = f"{got_det:7d}" if got_det is not None else f"{'n/a':>7}"
        print(f"  {r['row']:5d} {r['expected_class']:7d} {got_cls:7d} "
              f"{r['expected_detect']:7d} {det_str} "
              f"{r['expected_conf']:6.3f} {got_conf:6.3f}  {flag}")

    print(f"\nParity: {npass}/{len(rows)} rows match the deployed model.")
    sys.exit(0 if npass == len(rows) else 1)


if __name__ == "__main__":
    main()
