#===========================================================

#===========================================================
set RESULT_DIR          "./result"
# override by "$::env(RESULT_DIR)" if exist

# input files
set NETLIST_FILE        "$::env(NETLIST_FILE)"

# input variables
set TOP_NAME            "$::env(TOP_NAME)"
set CLK_PORT_NAME       "$::env(CLK_PORT_NAME)"
set USE_FIXED_BBOX      "$::env(USE_FIXED_BBOX)"
puts "iFP: USE_FIXED_BBOX $USE_FIXED_BBOX"
if { $USE_FIXED_BBOX == "False" } {
   set CORE_UTIL        "$::env(CORE_UTIL)"
} else {
   set DIE_BBOX         "$::env(DIE_BBOX)"
   set CORE_BBOX        "$::env(CORE_BBOX)"
}
set TAPCELL             "$::env(TAPCELL)"
set TAP_DISTANCE        "$::env(TAP_DISTANCE)"
set ENDCAP              "$::env(ENDCAP)"

# output files
set OUTPUT_DEF          "$RESULT_DIR/iFP_result.def"
set OUTPUT_VERILOG      "$RESULT_DIR/iFP_result.v"
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
##   read verilog
#===========================================================
verilog_init -path $NETLIST_FILE -top $TOP_NAME

#===========================================================
##   init floorplan
#===========================================================
set PLACE_SITE core7 
set IO_SITE core7
set CORNER_SITE core7

if { $USE_FIXED_BBOX == "False" } {
   puts "iFP: init with core_util $CORE_UTIL"
   init_floorplan \
      -core_util $CORE_UTIL \
      -core_site $PLACE_SITE \
      -io_site $IO_SITE \
      -corner_site $CORNER_SITE
} else {
   puts "iFP: init with fixed area die $DIE_BBOX, and core $CORE_BBOX"
   init_floorplan \
      -die_area $DIE_BBOX \
      -core_area $CORE_BBOX \
      -core_site $PLACE_SITE \
      -io_site $IO_SITE \
      -corner_site $CORNER_SITE
}

source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/create_tracks.tcl

#===========================================================
##   Place IO Port
#===========================================================
# -sides "left right top bottom", src/interface/tcl/tcl_ifp/tcl_io.cpp:30
auto_place_pins -layer MET3 -width 300 -height 600

#===========================================================
##   Tap Cell
#===========================================================
tapcell \
   -tapcell $TAPCELL \
   -distance $TAP_DISTANCE \
   -endcap $ENDCAP

#===========================================================
##   PDN 
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/global_net.tcl
source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/add_stripe.tcl

#===========================================================
##   set clock net
#===========================================================
source $IEDA_TCL_SCRIPT_DIR/iFP_script/module/set_clocknet.tcl

#===========================================================
##   save def 
#===========================================================
def_save -path $OUTPUT_DEF
netlist_save -path $OUTPUT_VERILOG -exclude_cell_names {}

#===========================================================
##   report db summary
#===========================================================
report_db -path $DESIGN_STAT_TEXT
feature_summary -step floorplan -path $DESIGN_STAT_JSON

#===========================================================
##   Exit 
#===========================================================
flow_exit
