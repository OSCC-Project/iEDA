#===========================================================

#===========================================================
set RESULT_DIR          "./result"

# input variables
if { [info exists ::env(USE_VERILOG)] && [string tolower $::env(USE_VERILOG)] == "true" } {
    set USE_VERILOG         true
    set TOP_NAME            "$::env(TOP_NAME)"
    set INPUT_VERILOG       "$::env(INPUT_VERILOG)"
} else {
    set USE_VERILOG         false
    set INPUT_DEF           "$RESULT_DIR/iFP_result.def"
}

# output files
set OUTPUT_DEF          "$RESULT_DIR/iNO_fix_fanout_result.def"
set OUTPUT_VERILOG      "$RESULT_DIR/iNO_fix_fanout_result.v"
set DESIGN_STAT_TEXT    "$RESULT_DIR/report/fix_fanout_db.rpt"
set DESIGN_STAT_JSON    "$RESULT_DIR/report/fix_fanout_db.json"
# set TOOL_METRICS_JSON   "$RESULT_DIR/metric/iNO_metrics.json"

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
##   read verilog/def
#===========================================================
if { $USE_VERILOG } {
    verilog_init -path $INPUT_VERILOG -top $TOP_NAME
} else {
    def_init -path $INPUT_DEF
}

#===========================================================
##   run TO to fix fanout
#===========================================================
run_no_fixfanout -config $IEDA_CONFIG_DIR/no_default_config_fixfanout.json

#===========================================================
##   save def 
#===========================================================
def_save -path $OUTPUT_DEF

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path $OUTPUT_VERILOG -exclude_cell_names {}

#===========================================================
##   report db summary and metrics
#===========================================================
report_db -path $DESIGN_STAT_TEXT
feature_summary -step fixFanout -path $DESIGN_STAT_JSON

# Disable because the STA result is not correct
# feature_tool -step fixFanout -path $TOOL_METRICS_JSON

#===========================================================
##   Exit 
#===========================================================
flow_exit
