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
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib_setup.tcl

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
def_init -path $::env(RESULT_DIR)/iTO_hold_result.def

#===========================================================
##   run TO to  opt_setup
#===========================================================
run_to_setup -config $::env(CONFIG_DIR)/to_default_config_setup.json

feature_tool -path $::env(RESULT_DIR)/feature/ito_optsetup.json -step optSetup


#===========================================================
##   save def 
#===========================================================
def_save -path $::env(RESULT_DIR)/iTO_setup_result.def

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path $::env(RESULT_DIR)/iTO_setup_result.v -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path "$::env(RESULT_DIR)/report/setup_db.rpt"
feature_summary -path $::env(RESULT_DIR)/feature/summary_ito_optsetup.json -step optSetup
#===========================================================
##   Exit 
#===========================================================
flow_exit
