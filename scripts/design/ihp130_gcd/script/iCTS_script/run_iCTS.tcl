#===========================================================

#===========================================================
set RESULT_DIR          "./result"

# input files
set INPUT_DEF           "$RESULT_DIR/iPL_result.def"

# output files
set TOOL_REPORT_DIR     "$RESULT_DIR/cts"
set OUTPUT_DEF          "$RESULT_DIR/iCTS_result.def"
set OUTPUT_VERILOG      "$RESULT_DIR/iCTS_result.v"
set DESIGN_STAT_TEXT    "$RESULT_DIR/report/cts_stat.rpt"
set DESIGN_STAT_JSON    "$RESULT_DIR/report/cts_stat.json"
set TOOL_METRICS_JSON   "$RESULT_DIR/metric/iCTS_metrics.json"

# script path
set IEDA_CONFIG_DIR     "$::env(IEDA_CONFIG_DIR)"
set IEDA_TCL_SCRIPT_DIR "$::env(IEDA_TCL_SCRIPT_DIR)"

#===========================================================
#   override variables from env
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/env_var_setup.tcl

#===========================================================
##   init flow config
#===========================================================
flow_init -config $IEDA_CONFIG_DIR/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config $IEDA_CONFIG_DIR/db_default_config.json -output_dir_path $RESULT_DIR

#===========================================================
##   reset data path
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_path_setting.tcl

#===========================================================
##   reset lib
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_init_lib.tcl

#===========================================================
##   reset sdc
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_init_sdc.tcl

#===========================================================
##   read lef
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_init_lef.tcl

#===========================================================
##   read def
#===========================================================
def_init -path $INPUT_DEF

#===========================================================
##   run CTS
#===========================================================
run_cts -config $IEDA_CONFIG_DIR/cts_default_config.json -work_dir $TOOL_REPORT_DIR

#===========================================================
##   def & netlist
#===========================================================
def_save -path $OUTPUT_DEF

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path $OUTPUT_VERILOG -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path $DESIGN_STAT_TEXT
feature_summary -path $DESIGN_STAT_JSON -step CTS

feature_tool -path $TOOL_METRICS_JSON -step CTS

#===========================================================
##   report CTS
#===========================================================
cts_report -path $RESULT_DIR/cts

#===========================================================
##   Exit 
#===========================================================
flow_exit
