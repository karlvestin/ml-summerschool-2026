// =============================================================================
//  leak_predict.cpp — EPICS aSub subroutine: live cooling-skid leak inference
// =============================================================================
//
//  Wraps the offline-trained mlpack DecisionTree (see model/train_model.py) so
//  the ESS water-cooling-skid IOC predicts the upstream-section leak LOCATION
//  class from live sensor PVs (a 10-class problem: 'No leak' + 9 locations).
//
//  HONESTY NOTE: per the physics review, exact leak LOCATION is not physically
//  observable from these lumped skid sensors (all branches merge off-skid) — see
//  train_model.py's leave-one-event-out validation. Only leak DETECTION (any-leak
//  vs no-leak) generalises. The class index is still emitted, but the record also
//  exposes a plain leak-detected flag (VALC) as the trustworthy output.
//
//  PIPELINE (must mirror train_model.py EXACTLY, or inference is garbage):
//    raw sensors ──▶ 6 engineered features ──▶ min-max normalise ──▶ tree
//                ──▶ class index (decoded to a location string by the .db mbbi)
//
//  The 6 features, in the order the model was trained on:
//    0 delta_T            = TT102 - TT110              (return − supply temp)
//    1 TT102              = return temperature
//    2 level_corrected    = LI103 - beta*(TT102-t_ref) (thermal-detrended level)
//    3 d_correctedLev_dt  = d(level_corrected)/dt      (detrended-level rate, mm/s)
//    4 P_speed            = pump VFD speed
//    5 speed_over_deltaT  = P_speed / max(|delta_T|, 0.5)
//
//  (Raw PT101 and raw LI103 are NOT model inputs: PT101 carried no leak signal
//   ~0.1% permutation importance) and LI103 enters only via level_corrected.)
//
//  Two of these are STATEFUL across processings:
//    * level_corrected needs `beta` and `t_ref` (thermal params, from metadata).
//    * d_correctedLev_dt is a 12-sample WINDOWED least-squares slope of
//      level_corrected vs time (mm/s), N2-vent-filtered — NOT a single-sample
//      finite difference. Vent detection still uses the RAW LI103 jump (a physical
//      N2 top-up), but the slope is fit on the thermally-detrended level, exactly
//      as train_model.py does. The window is a fixed number of scans; at the
//      IOC SCAN period a 12-sample window spans ~1 min (the offline data was
//      ~30 s-spaced, but the fit uses real timestamps so units stay mm/s). Samples
//      where the level jumps > vent_step_mm still occupy a window slot but are
//      excluded from the fit. The slope is 0 until at least 3 non-vent samples are
//      in the window. Window size and vent step come from metadata (12, 4.0 mm).
//
//  Model + normalisation parameters are loaded ONCE at iocInit:
//    * leak_model.bin       — the serialised mlpack tree (cereal binary).
//    * leak_model_meta.json — feat_min/feat_range, beta_thermal, t_ref,
//                             d_level_dt config, n_classes.
//  Both are read from the directory named by the LEAK_MODEL_DIR env var
//  (default ".").  A load failure is reported loudly; the record then alarms
//  INVALID rather than emitting silent garbage predictions.
//
//  aSub wiring (see leak_predict.db):
//    INPA=TT102  INPB=TT110  INPC=LI103  INPD=P_speed              (all DOUBLE)
//    VALA=class index  VALB=confidence  VALC=leak-detected (0/1)     (all DOUBLE)
// =============================================================================
//
// -----------------------------------------------------------------------------
//  A PRIMER FOR NEWCOMERS (read this before the code below)
//
//  WHAT IS AN "aSub"?  EPICS is the accelerator's control software; it publishes
//  every sensor as a named value called a PV (Process Variable). An "aSub" record
//  is a PV whose value is computed by a C++ function we write. EPICS calls our
//  function on a schedule, hands us the input PVs (named prec->a, prec->b, ...),
//  and publishes whatever we write to the output slots (prec->vala, prec->valb...).
//
//  Two functions do all the work here, and EPICS calls them at two different times:
//    * leakPredictInit(...)  — runs ONCE when the IOC boots. Loads the trained
//                              model + settings from disk into memory.
//    * leakPredict(...)      — runs EVERY scan (e.g. every few seconds). Reads the
//                              4 live sensors, builds features, and predicts.

//    ctx              -> our own bundle ("context") holding the model + its state.
//
//  THE BIG IDEA OF THIS FILE:  the decision tree learned to *label* leak events
//  but, on live data, it can't truly locate a leak (the sensors don't carry that
//  information). So we DO NOT trust the tree to decide "leak or not". Instead the
//  intended design decides leak/no-leak from a simple, physics-based rule — "is the
//  water level steadily dropping?" — keeping the tree's answer only as a hint.
//
//  WHY detection was taken out of the tree (summary; full write-up in
//  docs/EXERCISE_leak_gate.md §1, and in leak_model_meta.json "detection_rationale"):
//    1. Each leak location is exactly ONE contiguous drain event, so the tree splits
//       on that event's ABSOLUTE operating-point fingerprint (temporal leakage), not
//       the causal leak signal.
//    2. On a held-out event (leave-one-event-out) the tree constant-predicts the
//       largest leak class — tree-based detection collapses; the high row-level score
//       is an autocorrelation artifact, not real localisation.
//    3. A real leak is CAUSAL and condition-invariant — escaping coolant drops
//       level_corrected in a sustained way — so the physics gate generalises
//       (leave-one-event-out detection ~88%) regardless of the day's heat load, and
//       its temperature-transient guard rejects the heat-load confound.
//
//  >>> STUDENT BONUS EXERCISE <<<  That physics leak-DETECTION gate is NOT
//  implemented in this file — building it is an optional exercise. See
//  docs/EXERCISE_leak_gate.md; the reference answer is solutions/leak_predict.cpp.
//  Until it is built, `detected` is always false (the record reports "No leak").
// -----------------------------------------------------------------------------

#include <mlpack.hpp>          // the machine-learning library: gives us DecisionTree, matrices, data::Load

// cereal ships its own copy of rapidjson; CEREAL_RAPIDJSON_NAMESPACE resolves to
// the namespace it was built with (default `rapidjson`). Referencing it through
// the macro keeps us correct no matter how cereal was configured.
#include <cereal/external/rapidjson/document.h>        // a JSON reader (to parse leak_model_meta.json)
#include <cereal/external/rapidjson/istreamwrapper.h>  // lets that JSON reader read straight from a file stream from disk
namespace rj = CEREAL_RAPIDJSON_NAMESPACE;             // short alias "rj" for that JSON library's namespace

#include <fstream>       // reading files from disk (std::ifstream)
#include <string>        // the std::string text type
#include <cstdlib>       // std::getenv (read an environment variable), std::strtoul( "string to unsigned long")
#include <cmath>         // std::fabs (absolute value)
#include <algorithm>     // std::max / std::min

// EPICS headers (C linkage) come after the C++ ones so their macros don't leak
// into mlpack/armadillo template code.
#include <aSubRecord.h>       // the definition of the aSubRecord struct (our "prec")
#include <registryFunction.h> // let us register our functions by name so the .db can find them
#include <dbDefs.h>           // common EPICS database constants/macros
#include <recGbl.h>           // recGblSetSevr — raise an alarm on a record
#include <alarm.h>            // alarm severity/status codes (INVALID_ALARM, READ_ALARM, ...)
#include <errlog.h>           // errlogPrintf — print a message to the IOC console/log
#include <epicsTime.h>        // epicsTimeStamp + time helpers (for the level-slope timing)
#include <epicsExport.h>      // epicsRegisterFunction — export our functions to EPICS

using namespace mlpack;       // so we can write DecisionTree<> instead of mlpack::DecisionTree<>

// -----------------------------------------------------------------------------
//  Model wrapper — MUST match what mlpack's `decision_tree` Python binding
//  serialises: a DecisionTree<> followed by a DatasetInfo, in that order. This
//  is a byte-for-byte copy of mlpack::DecisionTreeModel in the 4.7.0 header
//  mlpack/methods/decision_tree/decision_tree_model.hpp (the type the binding's
//  PARAM_MODEL_OUT / __getstate__ actually writes). Loading into a bare
//  DecisionTree<> would read past the tree and crash.
// -----------------------------------------------------------------------------
namespace {   // anonymous namespace: everything inside is PRIVATE to this file only
// the sizes have to be constexpr constants known before the program runs
constexpr int    NUM_FEATURES   = 6;     // the model always takes exactly 6 features (see the list up top)
constexpr size_t MAX_CLASSES    = 64;    // metadata has 10; this is headroom
constexpr double MIN_ABS_DT_C   = 0.5;   // |delta_T| floor for speed/|delta_T| so the speed_over_deltaT feature can't divide by ~0.

// d_level_dt windowed-slope defaults (overridden by metadata if present).
constexpr size_t DEF_WIN_SAMPLES = 12;   // number of samples as window span for the LSQ slope
constexpr size_t MAX_WIN_SAMPLES = 64;   // compile-time cap on the slope window
constexpr size_t MAX_RING        = 512;  // history buffer for sensor samples so the code can look back in time (>= (cum_window_s / 5 s) which is 180 samples)
constexpr double DEF_VENT_STEP_MM = 4.0; // |dLevel| above this = N2 vent event

// ─── Physics leak-DETECTION gate: STUDENT BONUS EXERCISE ─────────────────────
// The deployed leak/no-leak decision is meant to come from a real-time physics
// gate (a SUSTAINED drop of level_corrected), NOT from the tree — which only
// fingerprints one-off drain events and cannot detect on live data. That gate,
// its calibrated thresholds (slope / cumulative-drop / thermal-guard / debounce)
// and the metadata that feeds them are intentionally REMOVED here for students to
// build. See docs/EXERCISE_leak_gate.md; reference: solutions/leak_predict.cpp.

// A tiny wrapper type that must have the EXACT same shape the Python side wrote,
// so loading the file byte-for-byte reconstructs the tree correctly.
class DecisionTreeModel
{
 public:
  DecisionTree<>     tree;   // the trained decision tree itself (the thing that classifies)
  data::DatasetInfo  info;   // bookkeeping mlpack stores next to the tree (column types etc.)

  // serialize() is how the cereal library reads/writes this object to a file.
  // The SAME function does both directions; `ar(...)` streams each member in order.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(tree));    // read/write the tree   (NVP = "named value pair", tag "tree")
    ar(CEREAL_NVP(info));    // read/write the info   (must come AFTER tree, the same order Python used)
  }
};

// Per-record private state, allocated once in the init routine and stashed in
// prec->dpvt so the process routine can reach it (and so several aSub instances
// could coexist without sharing state).
// One slot of the d_correctedLev_dt window in the ring buffer: a detrended-level reading, when it
// was taken, and whether it was flagged as an N2 vent jump (excluded from the fit).

/*
This block defines struct Sample — the record type for one remembered snapshot of a past scan. To understand why it exists, you have to know that two of the model's features can't be computed from the current reading alone; they need history.

Why history is needed at all?

Two things the code computes look backwards in time:

- d_correctedLev_dt — the slope of the water level, i.e. how fast it's changing. A slope is "change ÷ time," so you need several past (level, time) points to fit a line through.
- cumDrop — how much the level fell over the last 15 minutes. Again, you must compare now against then.

But leakPredict runs once per scan and then returns. Any ordinary local variable inside it vanishes when it returns. So to remember earlier scans, the code keeps an array of Samples (the "ring buffer," Sample ring[MAX_RING]) that survives between calls. Sample is the shape of one entry in that history.
These are default member initializers. When the ring[MAX_RING] array is created at startup, every Sample in it starts as clean zeros instead of leftover garbage from memory. That matters because in the first few seconds the buffer isn't full yet — the code walks ringCount valid entries, but well-defined defaults mean even an untouched slot is harmless rather than a random value that could poison a fit.

Putting it together

So the lifecycle is:

1. Startup: LeakModelCtx (containing Sample ring[512]) is allocated once and pinned to prec->dpvt.
2. Each scan: build a fresh Sample from this moment ({tsec, levelCorr, tt102, ventNow}), drop it into the next ring slot.
3. Read back: the slope fit and cumulative-drop loops walk the recent Samples, using t and level for the math, tt102 for the temperature guard, and skipping any with vent == true.

*/
struct Sample
{
  double t     = 0.0;      // seconds since the first scan (epoch)
  double level = 0.0;      // level_corrected = LI103 - beta*(TT102-t_ref) (mm)
  double tt102 = 0.0;      // inlet temp at this scan (for the dTT102/dt guard)
  bool   vent  = false;    // true = raw LI103 jumped > vent_step_mm vs prior scan
};

// LeakModelCtx = the whole "memory" of this record: the loaded model, the
// normalisation numbers, and all the running state we carry from one scan to the
// next. One of these is created at startup and handed back to us every scan.
struct LeakModelCtx
{
  DecisionTreeModel model;                      // the loaded tree + info (from leak_model.bin/xml)
  double            featMin[NUM_FEATURES];      // per-feature minimum used to normalise (from meta JSON)
  double            featRange[NUM_FEATURES];    // per-feature (max-min) used to normalise (from meta JSON)
  double            beta = 0.0;                 // thermal-expansion coefficient for level_corrected
  double            tRef = 0.0;                 // t_ref for level_corrected
  size_t            nClasses = 0;               // class index -> location string
                                                // lives in the .db mbbi, not here

  // d_correctedLev_dt: 12-sample windowed least-squares slope of the detrended
  // level (level_corrected) vs time; vent detection uses raw LI103 jumps.
  size_t            win = DEF_WIN_SAMPLES;      // 12 active slope window length (<= cap)
  double            ventStepMm = DEF_VENT_STEP_MM;  // 4 mm level jump above this = an N2 top-up, not a leak
  Sample            ring[MAX_RING];             // circular history buffer
  size_t            ringCount = 0;              // valid slots in the ring buffer(grows to MAX_RING)
  size_t            ringHead = 0;               // next write position
  bool              haveEpoch = false;          // first-scan time captured yet?
  epicsTimeStamp    epoch{};                    // t=0 reference for the slope fit
  bool              havePrev = false;           // prior level seen (for vent step)
  double            prevLevel = 0.0;            // last LI103 (vent-step compare)
  double            lastSlope = 0.0;            // held d_correctedLev_dt (vent windows)

  // (Physics leak-detection gate STATE — thresholds + latch counters + the
  //  latched `detected` flag — is a STUDENT BONUS EXERCISE; add the fields here.
  //  See docs/EXERCISE_leak_gate.md; reference: solutions/leak_predict.cpp.)
};

// Read feat_min/feat_range, beta_thermal and class_to_label from the JSON that
// train_model.py writes. Returns false (with a logged reason) on any problem.
// (Returning false = "give up cleanly" so the IOC alarms instead of guessing.)
bool loadMeta(const std::string& path, LeakModelCtx* ctx)
{
  std::ifstream ifs(path);           // try to open the JSON metadata file for reading
  if (!ifs)                          // if it didn't open (missing / no permission)...
  {
    errlogPrintf("leakPredict: cannot open metadata '%s'\n", path.c_str());  // ...log why...
    return false;                    // ...and report failure to the caller
  }

  rj::IStreamWrapper isw(ifs);       // wrap the file so the JSON parser can read from it
  rj::Document d;                    // an empty JSON document to fill
  d.ParseStream(isw);                // parse the whole file into d
  if (d.HasParseError() || !d.IsObject())   // if the JSON is broken or isn't a { ... } object...
  {
    errlogPrintf("leakPredict: malformed JSON in '%s'\n", path.c_str());     // ...complain...
    return false;                    // ...and fail
  }

  // Confirm the two normalisation arrays exist, are arrays, and have length 6.
  if (!d.HasMember("feat_min")   || !d["feat_min"].IsArray()   ||
      !d.HasMember("feat_range") || !d["feat_range"].IsArray() ||
      d["feat_min"].Size()   != NUM_FEATURES ||
      d["feat_range"].Size() != NUM_FEATURES)
  {
    errlogPrintf("leakPredict: feat_min/feat_range missing or not length %d\n",
                 NUM_FEATURES);
    return false;                    // wrong shape -> normalisation would be wrong -> fail
  }
  for (int i = 0; i < NUM_FEATURES; ++i)   // copy each of the 6 min/range values into ctx
  {
    ctx->featMin[i]   = d["feat_min"][i].GetDouble();    // per-feature minimum
    ctx->featRange[i] = d["feat_range"][i].GetDouble();  // per-feature (max-min)
    if (ctx->featRange[i] == 0.0)            // guard against /0 on a flat feature
      ctx->featRange[i] = 1.0;               // a range of 0 would divide by zero; use 1 instead
  }

  // Thermal params for level_corrected; default to 0 (= no correction) if absent.
  ctx->beta = d.HasMember("beta_thermal") ? d["beta_thermal"].GetDouble() : 0.0; 
  ctx->tRef = d.HasMember("t_ref")        ? d["t_ref"].GetDouble()        : 0.0;

  // d_level_dt windowed-slope config (object with window_samples / vent_step_mm);
  // fall back to the compiled defaults if the section or a field is absent.
  if (d.HasMember("d_level_dt") && d["d_level_dt"].IsObject())   // only if the JSON has this sub-object...
  {
    const rj::Value& dl = d["d_level_dt"];                       // shorthand handle to it to avoid re-doing the key lookup every access and the & makes it a zero-copy view — and a plain copy wouldn't even compile
    if (dl.HasMember("window_samples") && dl["window_samples"].IsNumber())
      ctx->win = static_cast<size_t>(dl["window_samples"].GetUint());   // how many scans in the slope window (the cast bridges unsigned int → size_t)
    if (dl.HasMember("vent_step_mm") && dl["vent_step_mm"].IsNumber())
      ctx->ventStepMm = dl["vent_step_mm"].GetDouble();          // level jump that counts as an N2 vent
  }
  if (ctx->win < 3)               ctx->win = DEF_WIN_SAMPLES;  // need >=3 to fit
  if (ctx->win > MAX_WIN_SAMPLES) ctx->win = MAX_WIN_SAMPLES;  // slope-window cap

  // (Physics leak-detection thresholds would be read from the metadata
  //  "detection" object here — STUDENT BONUS EXERCISE. See docs/EXERCISE_leak_gate.md.)

  // Class count: labels are location STRINGS (index -> location name is done
  // by the .db mbbi, not here), so we only need how many classes the tree has.
  // Prefer the explicit n_classes field; fall back to counting class_to_label.
  if (d.HasMember("n_classes") && d["n_classes"].IsNumber())   // best case: JSON states it directly
  {
    ctx->nClasses = static_cast<size_t>(d["n_classes"].GetUint());
  }
  else if (d.HasMember("class_to_label") && d["class_to_label"].IsObject())  // else count the label map
  {
    size_t maxIdx = 0;                                          // track the largest class index seen
    for (auto it = d["class_to_label"].MemberBegin();           // loop over every "index": "name" pair
         it != d["class_to_label"].MemberEnd(); ++it)
      maxIdx = std::max(maxIdx,                                 // keep the biggest index...
                        static_cast<size_t>(std::strtoul(it->name.GetString(),
                                                         nullptr, 10)));  // (turn the key text into a number)
    ctx->nClasses = maxIdx + 1;                                 // classes are 0..maxIdx, so count = maxIdx+1
  }
  else
  {
    errlogPrintf("leakPredict: n_classes / class_to_label missing\n");  // neither present -> can't proceed
    return false;
  }
  if (ctx->nClasses == 0 || ctx->nClasses > MAX_CLASSES)        // sanity-check the count is 1..64
  {
    errlogPrintf("leakPredict: nClasses %zu out of range (1..%zu)\n",
                 ctx->nClasses, MAX_CLASSES);
    return false;
  }
  return true;                                                  // everything parsed OK
}

// -----------------------------------------------------------------------------
//  INAM routine — runs once at iocInit. Loads the model + metadata.
// -----------------------------------------------------------------------------
long leakPredictInit(aSubRecord* prec)   // EPICS calls this ONCE at startup; prec = our record
{
  const char* dir = std::getenv("LEAK_MODEL_DIR");             // where are the model files? (env var)
  const std::string base = (dir && dir[0]) ? dir : ".";        // use it if set, else "." (current dir)
  // Model file: prefer a human-readable XML tree (the student training bench
  // writes one so the model can be opened and inspected) if it exists; otherwise
  // fall back to the compact cereal BINARY the production pipeline writes. Both
  // hold the same DecisionTreeModel under the cereal NVP "model", so data::Load
  // reads either — the serialisation format is inferred from the file extension.
  const std::string xmlPath   = base + "/leak_model.xml";      // candidate 1: readable XML model
  const std::string binPath   = base + "/leak_model.bin";      // candidate 2: compact binary model
  const std::string modelPath = std::ifstream(xmlPath).good() ? xmlPath : binPath;  // pick XML if it opens, else BIN
  const std::string metaPath  = base + "/leak_model_meta.json";// the normalisation + settings file

  auto* ctx = new (std::nothrow) LeakModelCtx();               // allocate our state bundle (nothrow = return null on failure)
  if (!ctx)                                                    // if the machine is out of memory...
  {
    errlogPrintf("leakPredict: out of memory allocating context\n");
    return -1;                                                 // ...bail out (nonzero return = failure)
  }

  if (!loadMeta(metaPath, ctx))                                // read the JSON metadata into ctx
  {
    delete ctx;                                                // parsing failed: free what we allocated
    return -1;                       // leave dpvt null -> process alarms INVALID
  }

  try                                                          // model loading can throw; catch it so we fail cleanly
  {
    // fatal=true: data::Load throws on failure so we never run on a half-model.
    // NOTE: this is the named-model Load(file, name, obj, fatal) overload, which
    // is [[deprecated]] in mlpack 4.7.0 and slated for removal in mlpack 5.0. It
    // is the correct call for 4.x: internally it deserialises through the
    // hard-coded cereal NVP "model" (core/data/load_model.hpp), matching what the
    // Python decision_tree binding writes via __getstate__(). On a 5.0 upgrade,
    // switch to data::Load(file, obj, opts) with DataOptions.
    if (!data::Load(modelPath.c_str(), "model", ctx->model, true))   // read the tree from disk into ctx->model (model is the default tag name for mlpack models)
    {
      errlogPrintf("leakPredict: data::Load returned false for '%s'\n",
                   modelPath.c_str());
      delete ctx;                                              // load failed: clean up...
      return -1;                                               // ...and report failure
    }
  }
  catch (const std::exception& e)                              // if Load threw an error object...
  {
    errlogPrintf("leakPredict: failed to load model '%s': %s\n",
                 modelPath.c_str(), e.what());                 // ...log its message...
    delete ctx;                                                // ...free memory...
    return -1;                                                 // ...and fail
  }

  prec->dpvt = ctx;                                            // stash ctx on the record so every scan can reach it
  errlogPrintf("leakPredict: loaded model '%s' (%zu classes, beta=%.4f, "
               "t_ref=%.4f, dLevel window=%zu, vent=%.2f mm)\n",
               modelPath.c_str(), ctx->nClasses, ctx->beta, ctx->tRef,
               ctx->win, ctx->ventStepMm);                     // announce success + key settings to the IOC log
  return 0;                                                    // 0 = success; EPICS will now call leakPredict each scan
}

// -----------------------------------------------------------------------------
//  SNAM routine — runs every scan. Builds the 6 features, normalises (clamped),
//  runs the tree for an advisory location, and DECIDES leak/no-leak from a
//  physics gate (sustained level drop), not from the tree. Writes gated class
//  index, advisory confidence, leak-detected flag, and rate diagnostics.
// -----------------------------------------------------------------------------
long leakPredict(aSubRecord* prec)   // EPICS calls this EVERY scan; prec = our record
{
  auto* ctx = static_cast<LeakModelCtx*>(prec->dpvt);          // get back the state we stashed at init
  if (!ctx)                                                    // if init failed, dpvt is null...
  {
    // Init failed (model/metadata could not be loaded): refuse to guess.
    recGblSetSevr(prec, READ_ALARM, INVALID_ALARM);            // ...raise an INVALID alarm on the PV...
    return -1;                                                 // ...and do nothing else
  }


  // STUDENT INPUT
  // Read the 4 live sensor values EPICS handed us. 

  // STUDENT INPUT
  // Write the time stamp procedure using epicsTimeStamp class and epicsTimeGetCurrent and epicsTimeDiffInSeconds class methods
  
  
  // A big sudden jump in RAW level means an N2 top-up (vent), not a leak. Flag it
  // so it can be excluded from the slope fit below.
  const bool ventNow = ctx->havePrev &&
                       std::fabs(li103 - ctx->prevLevel) > ctx->ventStepMm;

  // Push this scan into the circular history buffer (holds up to MAX_RING scans,
  // enough to span cum_window_s at the 5 s cadence).
  ctx->ring[ctx->ringHead] = { tsec, levelCorr, tt102, ventNow };   // write this sample into the next slot
  ctx->ringHead = (ctx->ringHead + 1) % MAX_RING;             // advance the write position, wrapping around
  if (ctx->ringCount < MAX_RING) ctx->ringCount++;            // grow the valid-count until the buffer is full

  // Helper: fetch the i-th most recent sample (0 = newest). It walks BACKWARDS
  // from the write head, wrapping around the circular buffer.
  auto recent = [&](size_t i) -> const Sample& {
    return ctx->ring[(ctx->ringHead + MAX_RING - 1 - i) % MAX_RING];
  };

  // d_correctedLev_dt: LSQ slope of level_corrected over the last `win` slots
  // (vent samples excluded). dTT102/dt is fit on the same window for the
  // heat-load-transition guard. If a vent lands anywhere in the window the whole
  // fit is untrustworthy, so we HOLD the last valid slope (mirrors train_model.py:
  // vent detected on raw LI103, slope fit on the detrended level).
  const size_t wlook = std::min(ctx->win, ctx->ringCount);    // look back over up to `win` samples (fewer early on)
  bool   ventInWindow = false;                                // did a vent land inside this window?
  double dLevelDt     = ctx->lastSlope;                       // start from the last good slope (used if we can't refit)
  double dTt102Dt     = 0.0;                                  // temperature slope over the same window
  {
    // --- Pass 1: sums, to get the average time / level / temp (needed by least squares) ---
    double st = 0.0, slv = 0.0, stt = 0.0;                    // running sums of time, level, temp
    int n = 0;                                                // count of non-vent samples used
    for (size_t i = 0; i < wlook; ++i)                        // walk the window, newest to oldest
    {
      const Sample& s = recent(i);                            // the i-th most recent sample
      if (s.vent) { ventInWindow = true; continue; }          // skip vent samples (but note one occurred)
      st += s.t; slv += s.level; stt += s.tt102; ++n;         // accumulate sums for the good samples
    }
    if (n >= 3)                                               // need at least 3 points to trust a line fit
    {
      const double tm = st / n, lm = slv / n, ttm = stt / n;  // means of time, level, temp
      // --- Pass 2: least-squares slope = sum((t-tmean)*(y-ymean)) / sum((t-tmean)^2) ---
      double numL = 0.0, numT = 0.0, den = 0.0;               // numerators (level, temp) and shared denominator
      for (size_t i = 0; i < wlook; ++i)
      {
        const Sample& s = recent(i);
        if (s.vent) continue;                                 // again ignore vent samples
        const double tc = s.t - tm;                           // this point's time, centred on the mean
        numL += tc * (s.level - lm);                          // build the level-vs-time covariance
        numT += tc * (s.tt102 - ttm);                         // build the temp-vs-time covariance
        den  += tc * tc;                                      // build the time variance
      }
      if (den > 0.0) { dLevelDt = numL / den; dTt102Dt = numT / den; }  // slope = covariance / variance
    }
  }
  ctx->lastSlope = dLevelDt;                                  // remember this slope for windows that get vent-blocked

  // ─── cumDrop: STUDENT BONUS EXERCISE ────────────────────────────────────────
  // The trailing-window cumulative drop of level_corrected is the clean leak
  // discriminator (no-leak stays within a few mm; every leak drops >= ~16 mm).
  // Until you build it (docs/EXERCISE_leak_gate.md) it is fixed at 0.
  double cumDrop = 0.0;   // TODO(student): trailing-window drop of level_corrected (mm)

  ctx->prevLevel = li103;             // raw level, for the next vent-step compare
  ctx->havePrev  = true;              // we now have a previous level to compare against

  const double speedOverDt = pspeed / std::max(std::fabs(deltaT), MIN_ABS_DT_C);  // feature 5 (guarded against /0)

  // Assemble the 6 features in the EXACT order the model was trained on.
  const double feat[NUM_FEATURES] = {
    deltaT, tt102, levelCorr, dLevelDt, pspeed, speedOverDt
  };


  //STUDENT INPUT
  // Min-max normalise, CLAMPED to [0,1]; flag any out-of-envelope input so an
  // extrapolated (e.g. cold-idle) point cannot masquerade as a confident class.
  arma::vec x(NUM_FEATURES);                                  // the normalised feature vector we feed the tree
  bool ood = false;                                          // "out of distribution": did any feature leave the training range?
  for (int i = 0; i < NUM_FEATURES; ++i)
  {
    double u = (feat[i] - ctx->featMin[i]) / ctx->featRange[i];  // scale so training-min->0 and training-max->1
    if      (u < 0.0) { u = 0.0; ood = true; }               // below training range: clamp to 0 and flag it
    else if (u > 1.0) { u = 1.0; ood = true; }               // above training range: clamp to 1 and flag it
    x[i] = u;                                                // store the normalised value
  }

  //STUDENT INPUT

    // the tree's predicted class index (0..9)
    // the tree's confidence for each class
    // run the decision tree on the features
    // confidence of the chosen class (0..1)

  // ═══════════════════════════════════════════════════════════════════════════
  //  Physics leak-DETECTION gate — STUDENT BONUS EXERCISE (docs/EXERCISE_leak_gate.md)
  //  Decide leak/no-leak from a SUSTAINED level drop, NOT from the tree. Building
  //  blocks already available above:  dLevelDt (mm/s), cumDrop (mm), dTt102Dt
  //  (degC/s, thermal guard) and ventInWindow (N2-vent glitch). Steps to implement:
  //    1. rawLeak = (cumDrop <= cum_trip) && (dLevelDt <= slope_trip)
  //                 && (fabs(dTt102Dt) <= tt_slope_trip);   // reject heat-load ramps
  //    2. hold state through vent windows (ventInWindow); debounce N scans;
  //    3. latch a persistent ctx->detected.
  //  Reference answer: solutions/leak_predict.cpp.
  (void) dTt102Dt; (void) ventInWindow;   // provided for the exercise (unused until built)
  const bool detected = false;            // no gate yet -> record reports "No leak"
  // ═══════════════════════════════════════════════════════════════════════════

  //STUDENT INPUT

  // (GUIDANCE: Outputs VALA = GATED class index (No leak=0 unless a physical leak is
  // detected; the location is advisory-only). VALB = advisory-location
  // confidence. VALC = physics leak-detected (the trustworthy alarm). VALD/E =
  // level rate + 5-min drop diagnostics. VALF = raw (ungated) tree class hint.)
  
  // VALG = out-of-training-envelope flag.
  // VALA: location, but only if the gate fired
  // tell EPICS VALA holds 1 element
  // VALB: tree's confidence in that location
  // VALC: the trustworthy leak/no-leak flag
  // VALD: current level slope (mm/s) diagnostic
  // VALE: cumulative drop over the window (mm)
  // VALF: raw tree hint, even when not detected
  // VALG: 1 if inputs left the training envelope

  return 0;                                                  // 0 = success; EPICS publishes the outputs we just set
}

} // anonymous namespace

// Register both routines so the .dbd `function()` entries resolve at iocInit.
// (This is what lets the .db/.dbd files find our C++ functions by name.)
extern "C" {                              // "C" linkage so EPICS (a C program) can see these symbols
epicsRegisterFunction(leakPredictInit);   // make "leakPredictInit" callable from the database
epicsRegisterFunction(leakPredict);       // make "leakPredict" callable from the database
}
