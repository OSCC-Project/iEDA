set -e

export WORKSPACE=$(cd "$(dirname "$0")";pwd)

# (fixed) iEDA setting
# export RESULT_DIR=$WORKSPACE/result
export FOUNDRY_DIR=$WORKSPACE/../../foundry/ihp130
export IEDA_CONFIG_DIR=$WORKSPACE/iEDA_config
export IEDA_TCL_SCRIPT_DIR=$WORKSPACE/script

# design files
export TOP_NAME=gcd
export NETLIST_FILE=$WORKSPACE/result/verilog/gcd.v
export SDC_FILE=$WORKSPACE/gcd.sdc
# export SPEF_FILE=$FOUNDRY_DIR/spef/gcd.spef

# floorplan setting
## gcd
export DIE_AREA="0.0  0.0 150 150"
export CORE_AREA="20 20 130 130"

TCL_SCRIPTS="iFP_script/run_iFP.tcl
iNO_script/run_iNO_fix_fanout.tcl
iPL_script/run_iPL.tcl
iCTS_script/run_iCTS.tcl
iPL_script/run_iPL_legalization.tcl
iRT_script/run_iRT.tcl
iPL_script/run_iPL_filler.tcl
DB_script/run_def_to_gds_text.tcl"

for SCRIPT in $TCL_SCRIPTS; do
    echo ">>> $ iEDA -script ${IEDA_TCL_SCRIPT_DIR}/${SCRIPT}"
    ./iEDA -script "${IEDA_TCL_SCRIPT_DIR}/${SCRIPT}"
done
