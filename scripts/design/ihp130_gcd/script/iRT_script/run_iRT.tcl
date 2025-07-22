#===========================================================

#===========================================================
set RESULT_DIR          "./result"

# inputs
set INPUT_DEF           "$RESULT_DIR/iPL_lg_result.def"
set NUM_THREADS         64

# output files
set OUTPUT_DEF          "$RESULT_DIR/iRT_result.def"
set OUTPUT_VERILOG      "$RESULT_DIR/iRT_result.v"
set DESIGN_STAT_TEXT    "$RESULT_DIR/report/routing_stat.rpt"
set DESIGN_STAT_JSON    "$RESULT_DIR/report/routing_stat.json"
set TOOL_METRICS_JSON   "$RESULT_DIR/metric/iRT_routing_metrics.json"
set TOOL_REPORT_DIR     "$RESULT_DIR/report/rt/"

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
##   run Router
#===========================================================
init_rt -temp_directory_path $TOOL_REPORT_DIR \
        -bottom_routing_layer "Metal2" \
        -top_routing_layer "Metal5" \
        -thread_number $NUM_THREADS \
        -output_inter_result 0 \
        -enable_notification 0 \
        -enable_timing 0 \
        -enable_fast_mode 0

run_rt

destroy_rt

# report_timing -stage "dr"
feature_tool -path $TOOL_METRICS_JSON -step route

#===========================================================
##   save def & netlist
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
feature_summary -path $DESIGN_STAT_JSON -step route

#===========================================================
##   Exit 
#===========================================================
flow_exit
