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
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib_fixfanout.tcl

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
def_init -path $::env(RESULT_DIR)/iFP_result.def

#===========================================================
##   run TO to fix fanout
#===========================================================
run_no_fixfanout -config $::env(CONFIG_DIR)/no_default_config_fixfanout.json
feature_tool -path $::env(RESULT_DIR)/feature/ino_opt.json -step fixFanout

#===========================================================
##   save def 
#===========================================================
def_save -path $::env(RESULT_DIR)/iTO_fix_fanout_result.def

#===========================================================
##   save netlist 
#===========================================================
netlist_save -path $::env(RESULT_DIR)/iTO_fix_fanout_result.v -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path "$::env(RESULT_DIR)/report/fixfanout_db.rpt"
feature_summary -path $::env(RESULT_DIR)/feature/summary_fixFanout.json -step fixFanout

#===========================================================
##   Exit 
#===========================================================
flow_exit
