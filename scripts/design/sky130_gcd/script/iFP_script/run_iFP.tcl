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
##   read verilog
#===========================================================
verilog_init -path $::env(NETLIST_FILE) -top $::env(DESIGN_TOP)

#===========================================================
##   init floorplan
##   gcd & & APU & uart
#===========================================================
set DIE_AREA $::env(DIE_AREA)
set CORE_AREA $::env(CORE_AREA)
# if { $DESIGN_TOP == "gcd" } {
#     set DIE_AREA "0.0    0.0   149.96   150.128"
#     set CORE_AREA "9.996 10.08 139.964  140.048"
# } elseif { $DESIGN_TOP == "APU" } {
#     set DIE_AREA "0.0    0.0   500   500"
#     set CORE_AREA "10.0  10.0  490   490"
# } else {
#     set DIE_AREA "0.0    0.0   149.96   150.128"
#     set CORE_AREA "9.996 10.08 139.964  140.048"
# }

#===========================================================
##   init floorplan
##   aes_cipher_top
#===========================================================
#set DIE_AREA "0.0    0.0   1100   1100"
#set CORE_AREA "10.0 10.0 1090.0  1090.0"

set PLACE_SITE unit 
set IO_SITE unit
set CORNER_SITE unit

init_floorplan \
   -die_area $DIE_AREA \
   -core_area $CORE_AREA \
   -core_site $PLACE_SITE \
   -io_site $IO_SITE \
   -corner_site $CORNER_SITE

source $::env(TCL_SCRIPT_DIR)/iFP_script/module/create_tracks.tcl

#===========================================================
##   Place IO Port
#===========================================================
auto_place_pins -layer met5 -width 2000 -height 2000

#===========================================================
##   Tap Cell
#===========================================================
tapcell \
   -tapcell sky130_fd_sc_hs__tap_1 \
   -distance 14 \
   -endcap sky130_fd_sc_hs__fill_1

#===========================================================
##   PDN 
#===========================================================
#source $::env(TCL_SCRIPT_DIR)/iFP_script/module/pdn.tcl 

#===========================================================
##   set clock net
#===========================================================
source $::env(TCL_SCRIPT_DIR)/iFP_script/module/set_clocknet.tcl

#===========================================================
##   save def 
#===========================================================
set DEFAULT_OUTPUT_DEF "$::env(RESULT_DIR)/iFP_result.def"
def_save -path [expr {[info exists ::env(OUTPUT_DEF)] ? $::env(OUTPUT_DEF) : $DEFAULT_OUTPUT_DEF}]

#===========================================================
##   report db summary
#===========================================================
report_db -path "$::env(RESULT_DIR)/report/fp_db.rpt"

# run_power -output $::env(RESULT_PATH)/sta/

#===========================================================
##   Exit 
#===========================================================
flow_exit
