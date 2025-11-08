#===========================================================
set RESULT_DIR          "./result"

# input variables
if { [info exists ::env(USE_VERILOG)] && [string tolower $::env(USE_VERILOG)] == "true" } {
    set USE_VERILOG         true
    set TOP_NAME            "$::env(TOP_NAME)"
    set INPUT_VERILOG       "$::env(INPUT_VERILOG)"
} else {
    set USE_VERILOG         false
}

# output files
set TOOL_REPORT_DIR     "$RESULT_DIR/sta/"

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
##   run STA
#===========================================================
run_sta -output $TOOL_REPORT_DIR

#===========================================================
##   Exit 
#===========================================================
flow_exit
