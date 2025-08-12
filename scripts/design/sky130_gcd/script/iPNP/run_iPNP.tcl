#===========================================================
##   init flow config
#===========================================================
flow_init -config $::env(CONFIG_DIR)/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config $::env(CONFIG_DIR)/db_default_config.json -output_dir_path $::env(RESULT_DIR)

#===========================================================
##   reset data path
#===========================================================
source $::env(TCL_SCRIPT_DIR)/DB_script/db_path_setting.tcl

#===========================================================
##   reset lib
#===========================================================
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib.tcl

#===========================================================
##   reset sdc
#===========================================================
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_sdc.tcl

#===========================================================
##   read lef
#===========================================================
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lef.tcl

#===========================================================
##   read def
#===========================================================
set DEFAULT_INPUT_DEF "$::env(RESULT_DIR)/iFP_result.def"
def_init -path [expr {[info exists ::env(INPUT_DEF)]? $::env(INPUT_DEF) : $DEFAULT_INPUT_DEF}]

#===========================================================
##   run PNP
#===========================================================
run_pnp -config $::env(CONFIG_DIR)/pnp_default_config.json

#===========================================================
##   def & netlist
#===========================================================
set DEFAULT_OUTPUT_DEF "$::env(RESULT_DIR)/iPNP_result.def"
def_save -path [expr {[info exists ::env(OUTPUT_DEF)] ? $::env(OUTPUT_DEF) : $DEFAULT_OUTPUT_DEF}]

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path $::env(RESULT_DIR)/iPNP_result.v -exclude_cell_names {}

#===========================================================
##   Exit 
#===========================================================
flow_exit
