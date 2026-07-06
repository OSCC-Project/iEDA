flow_init -config $::env(IEDA_CONFIG_DIR)/flow_config.json
db_init -config $::env(IEDA_CONFIG_DIR)/db_default_config.json -output_dir_path $::env(RESULT_DIR)
source $::env(TCL_SCRIPT_DIR)/DB_script/db_path_setting.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lef.tcl
def_init -path $::env(RESULT_DIR)/iRT_result.def
run_drc -config $::env(IEDA_CONFIG_DIR)/drc_default_config.json -path $::env(RESULT_DIR)/report/drc.rpt
save_drc -path $::env(RESULT_DIR)/drc/detail.json
flow_exit
