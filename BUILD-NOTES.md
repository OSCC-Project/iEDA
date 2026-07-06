# iEDA + ecc-tools Integration — Build & Run Notes

Base: original OSCC-Project/iEDA (retain all iEDA content; ecc-tools wins on
code-level conflicts). ecc-tools main is fetched into ref `refs/remotes/ecc/main`
(and the alternative "ecc as base" merge is preserved on branch `wip/ecc-based-merge`).

## Toolchain (no sudo — userspace conda)
micromamba env `ecc` at /home/lxq/AiEDA/micromamba/envs/ecc provides gcc-10,
cmake, ninja, boost, glog, gflags, eigen, gtest, tbb, gmp, metis, libhwloc,
cairo, tk(tcl), libunwind, rust, pybind11. Eigen exposed via
`$CONDA_PREFIX/include/Eigen` symlink -> eigen3/Eigen.

## Build iEDA (shared libs — conda has no static glog/gflags)
    scripts/ecc_integration/ieda_build.sh clean   # configure+build target iEDA
Key flags: -DBUILD_STATIC_LIB=OFF, conda CC/CXX, LIBRARY_PATH=$CONDA_PREFIX/lib.
Output: bin/iEDA (run with LD_LIBRARY_PATH=$CONDA_PREFIX/lib).

## Run gcd flows (netlist -> GDS)
- sky130: scripts/ecc_integration/run_sky130.sh  -> result/final_design.gds2
- ics55:  PDK from openecos-projects/icsprout55-pdk (extract liberty releases into
  IP/STD_cell/.../H7C{L,R,H}/liberty; symlink *_ieda.lef -> *_ecos.lef);
  PDK_DIR=/home/lxq/AiEDA/ics55-pdk scripts/ecc_integration/run_ics55.sh

## Metrics (sky130 verified)
- STA:   result/to/{drv,hold}/sta/gcd.rpt (slack/TNS/freq)
- Power: iPW run_power -> result/sta/gcd.pwr (Total 1.96 mW)
- DRC:   iRT DRC -> result/drc/detail.drc (0 violations)
- IR-drop: report_ir_drop needs PDN power-source (bump) geometry — pending PDN config

## Status
sky130 gcd netlist->GDS + STA/power/DRC: DONE. ics55 gcd netlist->GDS: DONE.
TODO: ics55 power/DRC report scripts; sky130 IR-drop PDN; iRCX port; per-tool ecc merge.
