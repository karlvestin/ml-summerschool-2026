"""
Replay the parity manifest against a RUNNING leak-prediction IOC and check that
the live cpp aSub's TREE inference matches the deployed model (parity_reference.py).

NOTE: leak/no-leak is now a physics gate (a sustained level drop), not the tree.
Static row injection holds the level constant, so the gate never fires -> the gated
outputs (ClassIdx / LeakDetected / Status) are all 'No leak'. We therefore verify
the RAW (ungated) tree class on the aSub's VALF field, which is what the reference
manifest's expected_class holds, and confirm LeakDetected stays 0.

Protocol per row:
  1. caput the 4 raw sensor values (PT-101 is no longer a model input).
  2. Dwell >= DLDT_WINDOW+1 IOC scans (SCAN = 5 s) so the whole 12-sample slope
     window fills with one constant level -> the cpp computes d_correctedLev_dt == 0
     exactly, matching the reference.
  3. caget Predict.VALF (raw tree class) / LeakDetected / Confidence, compare.

PASS criterion: VALF == expected_class AND LeakDetected == expected_detect (0).
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
DWELL_S    = 70.0     # >= DLDT_WINDOW+1 (13) scans at SCAN = 5 s, so the 12-sample
                      # slope window fully refills with the new constant level
CA_TIMEOUT = "3"

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


def main():
    with open(MANIFEST) as f:
        rows = json.load(f)

    # Connectivity check.
    try:
        caget(CLASS_PV)
    except Exception as e:
        print(f"ERROR: cannot reach the IOC ({e}).")
        print("Start it first:  conda activate e3-mlpack && cd model && iocsh st.cmd")
        sys.exit(2)

    print(f"Driving {len(rows)} rows, {DWELL_S:.0f}s dwell each "
          f"(~{len(rows)*DWELL_S/60:.1f} min). Comparing IOC vs deployed model.\n")
    print(f"  {'row':>5} {'exp_cls':>7} {'got_cls':>7} {'exp_det':>7} {'got_det':>7} "
          f"{'exp_cf':>6} {'got_cf':>6}  result")

    npass = 0
    for r in rows:
        for suffix, val in r["sensors"].items():
            caput(PREFIX + suffix, val)
        time.sleep(DWELL_S)

        got_cls = int(round(float(caget(CLASS_PV))))
        got_det = int(round(float(caget(DETECT_PV))))
        got_conf = float(caget(CONF_PV))

        cls_ok  = got_cls == r["expected_class"]
        det_ok  = got_det == r["expected_detect"]
        conf_ok = abs(got_conf - r["expected_conf"]) <= 0.02
        ok = cls_ok and det_ok
        npass += ok
        flag = "PASS" if ok else "FAIL"
        if not conf_ok:
            flag += " (conf differs)"
        print(f"  {r['row']:5d} {r['expected_class']:7d} {got_cls:7d} "
              f"{r['expected_detect']:7d} {got_det:7d} "
              f"{r['expected_conf']:6.3f} {got_conf:6.3f}  {flag}")

    print(f"\nParity: {npass}/{len(rows)} rows match the deployed model.")
    sys.exit(0 if npass == len(rows) else 1)


if __name__ == "__main__":
    main()
