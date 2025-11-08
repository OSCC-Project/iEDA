#!/bin/bash
set -e

TOP_NAME="gcd"
CLK_PORT_NAME="clk"
export USE_FIXED_BBOX=False
export CORE_UTIL=0.2
# export DIE_AREA="0.0  0.0 150 150"
# export CORE_AREA="20 20 130 130"
echo "Running iEDA Netlist-to-GDS flow for design: $TOP_NAME (clock port: $CLK_PORT_NAME)"

# Ensure PDK_DIR is provided in the environment (must be exported); abort if missing.
if [ -z "${PDK_DIR:-}" ]; then
    echo "Error: PDK_DIR is not set. Please export PDK_DIR before running this script."
    exit 1
fi
export TECH_LEF="${PDK_DIR}/prtech/techLEF/N551P6M_ieda.lef"
export LEF_STDCELL="${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR_ieda.lef \
${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL_ieda.lef"
export LIB_STDCELL="${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib \
${PDK_DIR}/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_ss_rcworst_1p08_125_nldm.lib"
export TAPCELL="FILLTAPH7R"
export TAP_DISTANCE=58
export ENDCAP="FILLTAPH7R"

export WORKSPACE=$(cd "$(dirname "$0")";pwd)

# (fixed) iEDA setting
export RESULT_DIR=$WORKSPACE/result
export IEDA_CONFIG_DIR=$WORKSPACE/iEDA_config
export IEDA_TCL_SCRIPT_DIR=$WORKSPACE/script
export TCL_SCRIPT_DIR=$WORKSPACE/script
export DEF_DIR=$WORKSPACE/result
export SDC_FILE=$WORKSPACE/default.sdc

export IEDA_BINARY="${IEDA_BINARY:-$WORKSPACE/../../../bin/iEDA}"

if [ ! -x "$IEDA_BINARY" ]; then
    echo "Warning: IEDA binary '$IEDA_BINARY' not found or not executable."
fi

# Set configuration by design name
export TOP_NAME="$TOP_NAME"
export CLK_PORT_NAME="$CLK_PORT_NAME"
export NETLIST_FILE="$WORKSPACE/result/verilog/${TOP_NAME}_nl.v"
export SDC_FILE="$WORKSPACE/default.sdc"

if [ ! -f "$NETLIST_FILE" ]; then
    echo "Error: Netlist file '$NETLIST_FILE' does not exist."
    exit 1
fi


IEDA_TCL_SCRIPTS="iFP_script/run_iFP.tcl
iNO_script/run_iNO_fix_fanout.tcl
iPL_script/run_iPL.tcl
iCTS_script/run_iCTS.tcl
iPL_script/run_iPL_legalization.tcl
iRT_script/run_iRT.tcl
iPL_script/run_iPL_filler.tcl
DB_script/run_def_to_gds_text.tcl
"

for SCRIPT in $IEDA_TCL_SCRIPTS; do
    echo ">>> Running step: $STEP_NAME"
    echo ">>> $ iEDA -script ${IEDA_TCL_SCRIPT_DIR}/${SCRIPT}"
    $IEDA_BINARY -script "${IEDA_TCL_SCRIPT_DIR}/${SCRIPT}"
done
