# iEDA — Build & Run Notes

Base: OSCC-Project/iEDA. This tree integrates the additional tool sources directly
into the standard iEDA layout; there is no separate overlay to track.

## Toolchain (no sudo — userspace conda)
micromamba env `ieda-build` (at /home/lxq/AiEDA/micromamba/envs) provides gcc-10,
cmake, ninja, boost, glog, gflags, eigen, gtest, tbb, gmp, metis, libhwloc,
cairo, tk(tcl), libunwind, rust, pybind11. Eigen exposed via
`$CONDA_PREFIX/include/Eigen` symlink -> eigen3/Eigen.

## Build iEDA (shared libs — conda has no static glog/gflags)
    scripts/integration/ieda_build.sh clean   # configure + build target iEDA
Key flags: -DBUILD_STATIC_LIB=OFF, conda CC/CXX, LIBRARY_PATH=$CONDA_PREFIX/lib.
Output: bin/iEDA (run with LD_LIBRARY_PATH=$CONDA_PREFIX/lib).

## Run gcd flows (netlist -> GDS)
- sky130:    scripts/integration/run_sky130.sh   -> result/final_design.gds2
- ics55:     PDK_DIR=/home/lxq/AiEDA/ics55-pdk scripts/integration/run_ics55.sh
- nangate45: scripts/design/nangate45_gcd/run_iEDA.sh  (foundry: Foundary/nangate45)

## Metrics
- STA:     result/to/{drv,hold}/sta/gcd.rpt (slack / TNS / freq)
- Power:   iPW run_power -> result/sta/gcd.pwr
- DRC:     iRT DRC + iDRC signoff -> result/drc/detail.drc
- IR-drop: report_ir_drop needs PDN power-source (bump) geometry — see PDN config

## Status
sky130 gcd netlist->GDS + STA/power/DRC: DONE. ics55 gcd netlist->GDS: DONE (DRC 0).
TODO: ics55 power/DRC report scripts; sky130 IR-drop PDN; nangate45 gcd DRC closure.
