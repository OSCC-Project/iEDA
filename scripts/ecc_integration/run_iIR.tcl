flow_init -config $::env(CONFIG_DIR)/flow_config.json
db_init -config $::env(CONFIG_DIR)/db_default_config.json -output_dir_path $::env(RESULT_DIR)
source $::env(TCL_SCRIPT_DIR)/DB_script/db_path_setting.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lib.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_sdc.tcl
source $::env(TCL_SCRIPT_DIR)/DB_script/db_init_lef.tcl
def_init -path $::env(RESULT_DIR)/iPL_filler_result.def
run_power -output $::env(RESULT_DIR)/sta/
report_ir_drop -net_name {VDD VSS}
flow_exit
