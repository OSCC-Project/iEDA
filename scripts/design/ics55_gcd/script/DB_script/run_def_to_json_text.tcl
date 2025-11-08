#===========================================================

#===========================================================
set RESULT_DIR          "./result"
# override by "$::env(RESULT_DIR)" if exist

# input variables
set INPUT_DEF           "$RESULT_DIR/iPL_filler_result.def"

# output files
set LAYOUT_JSON_FILE     "$RESULT_DIR/final_design.json"

# script path
set IEDA_CONFIG_DIR     "$::env(IEDA_CONFIG_DIR)"
set IEDA_TCL_SCRIPT_DIR "$::env(IEDA_TCL_SCRIPT_DIR)"

#===========================================================
#   override variables from env
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/env_var_setup.tcl

#===========================================================
##   init flow config
#===========================================================
flow_init -config $IEDA_CONFIG_DIR/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config $IEDA_CONFIG_DIR/db_default_config.json -output_dir_path $RESULT_DIR

#===========================================================
##   reset data path
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_path_setting.tcl

#===========================================================
##   read lef
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/db_init_lef.tcl

#===========================================================
##   read def
#===========================================================
def_init -path $INPUT_DEF

#===========================================================
##   save json 
##   Full layer information instance:(-discard li/mcon/nwell/pwell/met/via)
##   use (-discard null) to choose all layer
#===========================================================
json_save -path $LAYOUT_JSON_FILE -discard null

#===========================================================
##   Exit 
#===========================================================
flow_exit
