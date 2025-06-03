#===========================================================

#===========================================================
set RESULT_DIR          "./result"

# input files
set INPUT_DEF           "$RESULT_DIR/iCTS_result.def"

# output files
set OUTPUT_DEF          "$RESULT_DIR/iPL_lg_result.def"
set OUTPUT_VERILOG      "$RESULT_DIR/iPL_lg_result.v"
set DESIGN_STAT_TEXT    "$RESULT_DIR/report/legalization_stat.rpt"
set DESIGN_STAT_JSON    "$RESULT_DIR/report/legalization_stat.json"

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
##   run Placer
#===========================================================
run_incremental_flow -config $IEDA_CONFIG_DIR/pl_default_config.json

#===========================================================
##   save def 
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
feature_summary -path $DESIGN_STAT_JSON -step legalization

#===========================================================
##   Exit 
#===========================================================
flow_exit
