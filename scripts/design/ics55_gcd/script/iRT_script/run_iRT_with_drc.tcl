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
def_init -path $::env(RESULT_DIR)/iPL_lg_result.def

set temp_folder_path $::env(RESULT_DIR)/rt/

init_drc_api

init_rt -temp_directory_path $temp_folder_path \
        -log_level 0 \
        -thread_number 8 \
        -bottom_routing_layer "" \
        -top_routing_layer "" \
        -ra_initial_penalty 100 \
        -ra_penalty_drop_rate 0.8 \
        -ra_outer_max_iter_num 10 \
        -ra_inner_max_iter_num 10

run_rt -flow "dr"

destroy_rt

destroy_drc_api


#===========================================================
##   save def & netlist
#===========================================================
def_save -path $::env(RESULT_DIR)/iRT_result.def

#===========================================================
##   save netlist
#===========================================================
netlist_save -path $::env(RESULT_DIR)/iRT_result.v -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path "$::env(RESULT_DIR)/report/rt_db.rpt"

#===========================================================
##   Exit
#===========================================================
flow_exit
