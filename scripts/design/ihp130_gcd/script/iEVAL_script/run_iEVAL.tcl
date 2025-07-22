#===========================================================

#===========================================================
set RESULT_DIR          "./result"

if { [info exists ::env(USE_VERILOG)] && [string tolower $::env(USE_VERILOG)] == "true" } {
    set USE_VERILOG              true
    set TOP_NAME                 "$::env(TOP_NAME)"
    set EVAL_INPUT_VERILOG       "$::env(EVAL_INPUT_VERILOG)"
} else {
    set USE_VERILOG              false
    set EVAL_INPUT_DEF           "$::env(EVAL_INPUT_DEF)"
}

# Create result directory if it doesn't exist
file mkdir $RESULT_DIR

# script path
set IEDA_CONFIG_DIR     "$::env(IEDA_CONFIG_DIR)"
set IEDA_TCL_SCRIPT_DIR "$::env(IEDA_TCL_SCRIPT_DIR)"

#===========================================================
#   override variables from env setup
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/env_var_setup.tcl

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
set_design_workspace $RESULT_DIR/eval_sta/$::env(STEP_NAME)
read_liberty $LIB_PATH

if { $USE_VERILOG } {
    read_netlist $EVAL_INPUT_VERILOG
} else {
    # Collect all LEF files from both tech and standard cell directories
    set LEF_FILES [glob -nocomplain $TECH_LEF_PATH/*.lef]
    foreach file [glob -nocomplain $LEF_PATH/*.lef] {
        lappend LEF_FILES $file
    }

    if {[llength $LEF_FILES] == 0} {
        puts stderr "Error: No LEF files found in $TECH_LEF_PATH or $LEF_PATH"
        flow_exit
    }
    read_lef_def -lef $LEF_FILES -def $EVAL_INPUT_DEF
}

link_design $TOP_NAME
read_sdc  $SDC_FILE

report_timing -json -max_path 5
report_power -json -toggle 0.1

#===========================================================
##   Exit
#===========================================================
flow_exit
