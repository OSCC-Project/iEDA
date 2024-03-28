#===========================================================
##   init flow config
#===========================================================
flow_init -config $::env(CONFIG_DIR)/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config $::env(CONFIG_DIR)/db_default_config.json

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
def_init -path $::env(RESULT_DIR)/iPL_filler_result.def

#===========================================================
##   save json 
##   Full layer information instance:(-discard li/mcon/nwell/pwell/met/via)
##   use (-discard null) to choose all layer
#===========================================================
json_save -path $::env(RESULT_DIR)/final_design.json -discard li

#===========================================================
##   Exit 
#=======================                                 ====================================
flow_exit
