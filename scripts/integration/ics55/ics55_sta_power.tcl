flow_init -config $::env(IEDA_CONFIG_DIR)/flow_config.json
db_init -config $::env(IEDA_CONFIG_DIR)/db_default_config.json -output_dir_path $::env(RESULT_DIR)
source $::env(TCL_SCRIPT_DIR)/DB_script/db_path_setting.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_sdc.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lef.tcl
def_init -path $::env(RESULT_DIR)/iRT_result.def
init_sta -output $::env(RESULT_DIR)/sta/
report_sta -output $::env(RESULT_DIR)/sta/timing.log
run_power -output $::env(RESULT_DIR)/sta/
flow_exit
