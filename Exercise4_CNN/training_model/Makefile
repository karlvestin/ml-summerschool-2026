# ============================================================================
#  E3 require module Makefile for the cooling-skid leak-prediction IOC support.
#
#  Builds a loadable module (library + dbd) named `leakpredict` that provides
#  the leakPredict / leakPredictInit aSub subroutines, plus the databases.
#  Load it from an IOC with:  require leakpredict
#
#  Build (with the e3 / e3-mlpack conda env active so EPICS_BASE, the require
#  tools, and CONDA_PREFIX are set):
#      make install
#
#  mlpack (header-only), armadillo, cereal and its rapidjson all come from the
#  active conda environment ($(CONDA_PREFIX)).
# ============================================================================
# MODULE must be set BEFORE the include and with `override`: the current
# driver.makefile resets `MODULE=` unconditionally and computes its PRJ symbol
# with immediate (`:=`) expansion while it is being included, so a plain
# assignment placed after the include (or even before it) is wiped. `override`
# makes our value win over driver.makefile's reset and any recursive sub-make.
override MODULE := leakpredict

include $(E3_REQUIRE_TOOLS)/driver.makefile

LIBVERSION = 1.0.0

# --- sources, dbd, databases (paths are relative to the module root; driver.makefile
#     builds a vpath from each entry's directory, so the src/ and db/ prefixes are
#     required now that the files live in subdirectories) ---
SOURCES   += src/leak_predict.cpp
DBDS      += db/leak_predict.dbd
TEMPLATES += db/leak_predict.db

# --- mlpack / armadillo / cereal toolchain from the conda env ---
# mlpack 4.x is header-only; it links against armadillo (which pulls BLAS/LAPACK)
# and uses OpenMP. -rpath bakes the conda lib dir in so the IOC finds them.
# armadillo is a third-party/system shared lib, so it goes in USR_SYS_LIBS
# (USR_LIBS is for EPICS-built modules and triggers .dep resolution).
USR_INCLUDES += -I$(CONDA_PREFIX)/include
USR_CXXFLAGS += -std=c++17 -fopenmp
USR_LDFLAGS  += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib -fopenmp
USR_SYS_LIBS += armadillo
