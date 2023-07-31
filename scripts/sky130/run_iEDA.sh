set -e

WORKSPACE="$(cd "$(dirname "$0")";pwd)"
LOG_DIR="${WORKSPACE}/log"
TCL_SCRIPT_DIR="${WORKSPACE}/script"
TCL_SCRIPTS="iFP_script/run_iFP.tcl
iNO_script/run_iNO_fix_fanout.tcl
iPL_script/run_iPL.tcl
iPL_script/run_iPL_eval.tcl
iCTS_script/run_iCTS.tcl
iCTS_script/run_iCTS_eval.tcl
iCTS_script/run_iCTS_STA.tcl
iTO_script/run_iTO_drv.tcl
iTO_script/run_iTO_drv_STA.tcl
iTO_script/run_iTO_hold.tcl
iTO_script/run_iTO_hold_STA.tcl
iPL_script/run_iPL_legalization.tcl
iPL_script/run_iPL_legalization_eval.tcl
iRT_script/run_iRT.tcl
iRT_script/run_iRT_eval.tcl
iRT_script/run_iRT_STA.tcl
iRT_script/run_iRT_DRC.tcl
iPL_script/run_iPL_filler.tcl
DB_script/run_def_to_gds_text.tcl"

mkdir -p $LOG_DIR

for SCRIPT in $TCL_SCRIPTS; do
    time ./iEDA -script "${TCL_SCRIPT_DIR}/${SCRIPT}" 2>&1 | tee "${LOG_DIR}/$(basename $SCRIPT .tcl).log"
done
