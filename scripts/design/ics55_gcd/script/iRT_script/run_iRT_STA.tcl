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
def_init -path $::env(RESULT_DIR)/iRT_result.def

#===========================================================
##   run STA
#===========================================================
init_sta -output $::env(RESULT_DIR)/rt/sta/

init_rt -temp_directory_path "$::env(RESULT_DIR)/rt/" \
        -bottom_routing_layer "met1" \
        -top_routing_layer "met4" \
        -enable_timing 1

# run_rt -flow vr
run_rt

report_timing -stage "dr"

destroy_rt

#===========================================================
##   Exit
#===========================================================
flow_exit
