#!/bin/bash
set -e

# Check if a design name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <design_name>"
    echo "Supported designs: gcd, aes, picorv32a"
    exit 1
fi

DESIGN_NAME="$1"

# Support only specific designs
case "$DESIGN_NAME" in
    "gcd"|"aes_cipher_top"|"picorv32a")
        echo "Running iEDA flow for design: $DESIGN_NAME"
        ;;
    *)
        echo "Error: Unsupported design '$DESIGN_NAME'"
        echo "Supported designs: gcd, aes, picorv32a"
        exit 1
        ;;
esac

export WORKSPACE=$(cd "$(dirname "$0")";pwd)

# (fixed) iEDA setting
export RESULT_DIR=$WORKSPACE/result
export FOUNDRY_DIR=$WORKSPACE/../../foundry/ihp130
export IEDA_CONFIG_DIR=$WORKSPACE/iEDA_config
export IEDA_TCL_SCRIPT_DIR=$WORKSPACE/script
export TCL_SCRIPT_DIR=$WORKSPACE/script
export DEF_DIR=$WORKSPACE/result

# Set configuration by design name
export TOP_NAME="$DESIGN_NAME"
export NETLIST_FILE="$WORKSPACE/result/verilog/${DESIGN_NAME}_nl.v"
export SDC_FILE="$WORKSPACE/default.sdc"

if [ ! -f "$NETLIST_FILE" ]; then
    echo "Error: Netlist file '$NETLIST_FILE' does not exist."
    exit 1
fi

case "$DESIGN_NAME" in
    "gcd")
        export DIE_AREA="0.0  0.0 150 150"
        export CORE_AREA="20 20 130 130"
        ;;
    "aes")
        export DIE_AREA="0 0 547.1137448407128 547.1137448407128"
        export CORE_AREA="10 10 537.1137448407128 537.1137448407128"
        ;;
    "picorv32a")
        export DIE_AREA="0 0 629.3543698046318 629.3543698046318"
        export CORE_AREA="10 10 619.3543698046318 619.3543698046318"
        ;;
esac

echo "Design configuration:"
echo "  TOP_NAME: $TOP_NAME"
echo "  NETLIST_FILE: $NETLIST_FILE"
echo "  SDC_FILE: $SDC_FILE"
echo "  DIE_AREA: $DIE_AREA"
echo "  CORE_AREA: $CORE_AREA"
echo ""

export INPUT_DEF="$WORKSPACE/result/iFP_result.def"
export INPUT_VERILOG="$WORKSPACE/result/verilog/${DESIGN_NAME}.v"
export OUTPUT_DEF="$WORKSPACE/result/${DESIGN_NAME}_nl.def"
export USE_VERILOG="true"

TCL_SCRIPTS="iFP_script/run_iFP.tcl
iEVAL_script/run_iEVAL.tcl
iNO_script/run_iNO_fix_fanout.tcl
iPL_script/run_iPL.tcl
iCTS_script/run_iCTS.tcl
iPL_script/run_iPL_legalization.tcl
iRT_script/run_iRT.tcl
iPL_script/run_iPL_filler.tcl
DB_script/run_def_to_gds_text.tcl"

for SCRIPT in $TCL_SCRIPTS; do
    echo ">>> $ iEDA -script ${IEDA_TCL_SCRIPT_DIR}/${SCRIPT}"
    iEDA -script "${IEDA_TCL_SCRIPT_DIR}/${SCRIPT}"
done
