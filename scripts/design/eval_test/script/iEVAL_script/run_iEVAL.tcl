#===========================================================

#===========================================================
set RESULT_DIR          "./result"

if { [info exists ::env(USE_VERILOG)] && [string tolower $::env(USE_VERILOG)] == "true" } {
    set USE_VERILOG    true
    set TOP_NAME            "$::env(TOP_NAME)"
    set INPUT_VERILOG       "$::env(INPUT_VERILOG)"
} else {
    set USE_VERILOG         false
    set INPUT_DEF           "$::env(INPUT_DEF)"
}

# Create result directory if it doesn't exist
file mkdir $RESULT_DIR

# input files
# set INPUT_DEF           "$RESULT_DIR/iNO_fix_fanout_result.def"
set INPUT_DEF           "$::env(INPUT_DEF)"

# script path
set IEDA_CONFIG_DIR     "$::env(IEDA_CONFIG_DIR)"
set IEDA_TCL_SCRIPT_DIR "$::env(IEDA_TCL_SCRIPT_DIR)"

#===========================================================
#   override variables from env setup
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/env_var_setup.tcl

#===========================================================
##   init flow config
#===========================================================
# flow_init -config $IEDA_CONFIG_DIR/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config $IEDA_CONFIG_DIR/db_default_config.json -output_dir_path $RESULT_DIR

#===========================================================
##   reset data path
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_path_setting.tcl

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
##   run Evaluation
#===========================================================
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_sdc.tcl

run_timing_eval -eval_output_path $RESULT_DIR -routing_type FLUTE
run_wirelength_eval -eval_output_path $RESULT_DIR
run_density_eval -eval_output_path $RESULT_DIR -grid_size 2000 -stage placement

#===========================================================
##   Exit
#===========================================================
# flow_exiw
