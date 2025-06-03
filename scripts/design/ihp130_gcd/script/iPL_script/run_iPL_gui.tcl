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
##   read lef
#===========================================================
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lef.tcl

#===========================================================
##   read def
#===========================================================
#def_init -path $::env(RESULT_DIR)/iTO_fix_fanout_result.def

#===========================================================
##   run Placer
#===========================================================
#run_placer -config $::env(CONFIG_DIR)/pl_default_config.json

#===========================================================
##   run gui
#===========================================================
def_init -path $::env(RESULT_DIR)/iPL_result.def
gui_start -type global_place
gui_show_pl -dir $::env(RESULT_DIR)/pl/gui/


