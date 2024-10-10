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
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib_drv.tcl

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
set DEFAULT_INPUT_DEF "$::env(RESULT_DIR)/iCTS_result.def"
def_init -path [expr {[info exists ::env(INPUT_DEF)]? $::env(INPUT_DEF) : $DEFAULT_INPUT_DEF}]

#===========================================================
##   run TO to fix_drv，opt_hold, opt_setup
#===========================================================
run_to_drv -config $::env(CONFIG_DIR)/to_default_config_drv.json
feature_tool -path $::env(RESULT_DIR)/feature/ito_optDrv.json -step optDrv

#===========================================================
##   save def 
#===========================================================
set DEFAULT_OUTPUT_DEF "$::env(RESULT_DIR)/iTO_drv_result.def"
def_save -path [expr {[info exists ::env(OUTPUT_DEF)] ? $::env(OUTPUT_DEF) : $DEFAULT_OUTPUT_DEF}]

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path $::env(RESULT_DIR)/iTO_drv_result.v -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path "$::env(RESULT_DIR)/report/drv_db.rpt"

feature_summary -path $::env(RESULT_DIR)/feature/summary_ito_optDrv.json -step optDrv

#===========================================================
##   Exit 
#===========================================================
flow_exit
