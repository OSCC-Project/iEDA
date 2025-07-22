#===========================================================
#   override variables from env
#===========================================================
set INPUT_DEF           "$::env(INPUT_DEF)"
set LAYOUT_JSON_FILE    "$::env(LAYOUT_JSON_FILE)"
set RESULT_DIR          "$::env(RESULT_DIR)"

set IEDA_CONFIG_DIR     "$::env(IEDA_CONFIG_DIR)"
set IEDA_TCL_SCRIPT_DIR "$::env(IEDA_TCL_SCRIPT_DIR)"

if { $INPUT_DEF == "" } {
  set INPUT_DEF "$RESULT_DIR/iPL_filler_result.def"
}

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
json_save -path $LAYOUT_JSON_FILE -discard li

#===========================================================
##   Exit 
#===========================================================
flow_exit
