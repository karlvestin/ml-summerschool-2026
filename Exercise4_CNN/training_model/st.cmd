# ============================================================================
#  st.cmd — cooling-skid leak-prediction IOC (single, switchable sim/live bench)
#
#  ONE IOC that hosts BOTH sensor streams at once and lets the model's input be
#  switched at runtime with the $(P)Sim:Enable bo:
#      Sim:Enable = Simulated (On, default)  -> the model reads the replayed
#          :SimRaw sensors that training_sim.py drives; the bench scores predictions
#          against ground truth and the parity harness runs.
#      Sim:Enable = Live PVs  (Off)          -> the model reads the REAL skid PVs
#          ($(P)<tag>:MeasValue / :Speed) from the ESS read-only gateway, for
#          production monitoring of the deployed physics gate on the live plant.
#  The switch lives in the per-sensor $(P)<tag>:ModelIn routing records
#  (db/leak_sensors_sim.db); the simulated and live PVs coexist as distinct names,
#  so nothing served here collides with the gateway.
#
#  Layout: db/ (databases + dbd), src/ (leak_predict.cpp), training/ (train_model.py
#  + the leak_model.bin/meta it writes), test/ (parity + training_sim.py).
#
#  Prerequisites (run FROM training_model/ unless noted):
#      make install                                              # build+install the leakpredict module
#      conda run -n e3-mlpack python3 training/train_model.py    # writes leak_model.bin + meta into training/
#
#  Run (with the e3-mlpack conda env active), FROM training_model/:
#      iocsh st.cmd
#  Bench (default Sim:Enable = Simulated): in a second terminal start the replayer
#  from test/, then open OPI/leak_training.bob in Phoebus:
#      cd test && conda run -n e3-mlpack python3 training_sim.py
#  Live monitoring: point the model at the real skid by flipping the switch AND
#  restoring real-time scanning (the physics gate needs it to integrate the
#  level trend); the gateway must be reachable, and do NOT run training_sim.py:
#      caput $(P)Sim:Enable "Live PVs"
#      caput $(P)LeakML:Predict.SCAN "5 second"
# ============================================================================

require leakpredict, 1.0.0

# Point CA/PVA clients at the ESS read-only gateway so the LIVE skid PVs
# ($(P)<tag>:MeasValue / :Speed — deliberately not defined locally) resolve
# there. The simulated SimRaw / ModelIn PVs are served locally.
epicsEnvSet("EPICS_CA_ADDR_LIST",  "idmz-ro-epics-gw-tn.esss.lu.se")
epicsEnvSet("EPICS_PVA_ADDR_LIST", "idmz-ro-epics-gw-tn.esss.lu.se")

# PV prefix — shared by the sensor PVs, the model PVs, and the bench PVs.
epicsEnvSet("P", "CWM-CWS02:WtrC-")

# Where the model files live. training/train_model.py writes leak_model.bin +
# leak_model_meta.json into training/, so the IOC is self-contained — run the
# trainer before starting the IOC. The aSub loads the .bin and reads its
# "detection" gate thresholds from the meta. LEAK_MODEL_DIR defaults to training/
# (run the IOC from training_model/).
epicsEnvSet("LEAK_MODEL_DIR", "$(LEAK_MODEL_DIR=training)")

# Inference engine + operator readbacks (the installed module template). This
# provides the leakPredict aSub — which reads the four $(P)<tag>:ModelIn switch
# records — and the $(P)LeakML:* PVs.
dbLoadRecords("leak_predict.db", "P=$(P)")

# Sensors: the simulated :SimRaw sources (written by training_sim.py) PLUS the
# $(P)<tag>:ModelIn routing records that switch each model input between the
# :SimRaw replay and the live gateway PV via $(P)Sim:Enable. BENCH_TOP locates the
# local dbs (they live in db/); defaults to db/ for a run from training_model/,
# override if you start the IOC from elsewhere.
epicsEnvSet("BENCH_TOP", "$(BENCH_TOP=db)")
dbLoadRecords("$(BENCH_TOP)/leak_sensors_sim.db", "P=$(P)")
# Controls (incl. the $(P)Sim:Enable switch), ground-truth comparison, and stats.
dbLoadRecords("$(BENCH_TOP)/leak_train.db", "P=$(P)")

iocInit()

# Bench default (Sim:Enable = Simulated): drive inference on demand. training_sim.py
# PROCs $(P)LeakML:Predict once per simulated sample (and the parity harness PROCs
# it to fill the slope window), so each prediction is deterministically aligned to
# its input and the slope window sees a consistent step. Disable the autonomous 5 s
# scan, which would otherwise interleave stale samples and muddy the scoreboard.
# NOTE: to MONITOR THE LIVE PLANT (Sim:Enable = Live PVs), restore real-time
# scanning — the physics gate needs it to integrate the level trend:
#     caput $(P)LeakML:Predict.SCAN "5 second"
dbpf("$(P)LeakML:Predict.SCAN", "Passive")
