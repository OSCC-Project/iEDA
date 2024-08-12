set -e

export WORKSPACE=$(cd "$(dirname "$0")";pwd)

# (fixed) iEDA setting
export CONFIG_DIR=$WORKSPACE/iEDA_config
export FOUNDRY_DIR=$WORKSPACE/../../foundry/sky130
export RESULT_DIR=$WORKSPACE/result
export TCL_SCRIPT_DIR=$WORKSPACE/script

# design files
export DESIGN_TOP=gcd
export NETLIST_FILE=$WORKSPACE/result/verilog/gcd.v
export SDC_FILE=$FOUNDRY_DIR/sdc/gcd.sdc
export SPEF_FILE=$FOUNDRY_DIR/spef/gcd.spef

# floorplan setting
## gcd
export DIE_AREA="0.0    0.0   149.96   150.128"
export CORE_AREA="9.996 10.08 139.964  140.048"

# system variables
PATH=$WORKSPACE/../../../bin:$PATH

./iEDA -script "${TCL_SCRIPT_DIR}/iFP_script/run_iFP.tcl"
sed -i 's/\( [^+ ]*\) + NET  +/\1 + NET\1 +/' ${RESULT_DIR}/iFP_result.def

TCL_SCRIPTS="iNO_script/run_iNO_fix_fanout.tcl
iPL_script/run_iPL.tcl
iCTS_script/run_iCTS.tcl
iCTS_script/run_iCTS_STA.tcl
iTO_script/run_iTO_drv.tcl
iTO_script/run_iTO_drv_STA.tcl
iTO_script/run_iTO_hold.tcl
iTO_script/run_iTO_hold_STA.tcl
iPL_script/run_iPL_legalization.tcl
iRT_script/run_iRT.tcl
iRT_script/run_iRT_DRC.tcl
iPL_script/run_iPL_filler.tcl
DB_script/run_def_to_gds_text.tcl"

for SCRIPT in $TCL_SCRIPTS; do
    echo ">>> $ iEDA -script ${TCL_SCRIPT_DIR}/${SCRIPT}"
    ./iEDA -script "${TCL_SCRIPT_DIR}/${SCRIPT}"
done
