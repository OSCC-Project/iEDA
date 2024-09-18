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
set DEFAULT_INPUT_DEF "$::env(RESULT_DIR)/iRT_result.def"
def_init -path [expr {[info exists ::env(INPUT_DEF)]? $::env(INPUT_DEF) : $DEFAULT_INPUT_DEF}]

#===========================================================
##   run Filler
#===========================================================
run_filler -config $::env(CONFIG_DIR)/pl_default_config.json

#===========================================================
##   save def 
#===========================================================
set DEFAULT_OUTPUT_DEF "$::env(RESULT_DIR)/iPL_filler_result.def"
def_save -path [expr {[info exists ::env(OUTPUT_DEF)] ? $::env(OUTPUT_DEF) : $DEFAULT_OUTPUT_DEF}]

#===========================================================
##   save netlist 
#===========================================================
# netlist_save -path $::env(RESULT_DIR)/iPL_filler_result.v -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path "$::env(RESULT_DIR)/report/filler_db.rpt"

feature_summary -path $::env(RESULT_DIR)/feature/summary_ipl_filler.json -step filler

#===========================================================
##   Exit 
#===========================================================
flow_exit
