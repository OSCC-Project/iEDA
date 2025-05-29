#===========================================================

#===========================================================
set RESULT_DIR          "./result"
# override by "$::env(RESULT_DIR)" if exist

# input files
set NETLIST_FILE        "$::env(NETLIST_FILE)"

# input variables
set TOP_NAME            "$::env(TOP_NAME)"
set DIE_AREA            "$::env(DIE_AREA)"
set CORE_AREA           "$::env(CORE_AREA)"

# output files
set OUTPUT_DEF          "$RESULT_DIR/iFP_result.def"
set DESIGN_STAT_TEXT    "$RESULT_DIR/report/floorplan_stat.rpt"
set DESIGN_STAT_JSON    "$RESULT_DIR/report/floorplan_stat.json"
# override by :
# "$::env(OUTPUT_DEF)" 
# "$::env(DESIGN_STAT_TEXT)" 
# "$::env(DESIGN_STAT_JSON)" 
# if exist

# script path
set IEDA_CONFIG_DIR     "$::env(IEDA_CONFIG_DIR)"
set IEDA_TCL_SCRIPT_DIR "$::env(IEDA_TCL_SCRIPT_DIR)"

#===========================================================
#   override variables from env
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/DB_script/env_var_setup.tcl

#===========================================================
##   init flow config
#===========================================================Standard Cell Rows
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
##   read verilog
#===========================================================
verilog_init -path $NETLIST_FILE -top $TOP_NAME

#===========================================================
##   init floorplan
#===========================================================
set PLACE_SITE CoreSite 
set IO_SITE sg13g2_ioSite
set CORNER_SITE sg13g2_ioSite

init_floorplan \
   -die_area $DIE_AREA \
   -core_area $CORE_AREA \
   -core_site $PLACE_SITE \
   -io_site $IO_SITE \
   -corner_site $CORNER_SITE

source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/create_tracks.tcl

#===========================================================
##   Place IO Port
#===========================================================
auto_place_pins -layer Metal5 -width 1000 -height 1000

#===========================================================
##   Tap Cell
#===========================================================
# There are no Endcap and Welltie cells in this PDK, so
# `cut_rows` has to be called from the tapcell script.
# But yet iFP has no cut_rows support.
# tapcell \
#    -tapcell sky130_fd_sc_hs__tap_1 \
#    -distance 14 \
#    -endcap sky130_fd_sc_hs__fill_1

#===========================================================
##   PDN 
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/pdn.tcl 

#===========================================================
##   set clock net
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/set_clocknet.tcl

#===========================================================
##   save def 
#===========================================================
def_save -path $OUTPUT_DEF

#===========================================================
##   report db summary
#===========================================================
report_db -path $DESIGN_STAT_TEXT
feature_summary -step floorplan -path $DESIGN_STAT_JSON

#===========================================================
##   Exit 
#===========================================================
flow_exit
