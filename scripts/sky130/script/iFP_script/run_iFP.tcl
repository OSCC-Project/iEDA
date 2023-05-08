#===========================================================
##   init flow config
#===========================================================
flow_init -config ./iEDA_config/flow_config.json

#===========================================================
##   read db config
#===========================================================
db_init -config ./iEDA_config/db_default_config.json

#===========================================================
##   reset data path
#===========================================================
source ./script/DB_script/db_path_setting.tcl

#===========================================================
##   read lef
#===========================================================
source ./script/DB_script/db_init_lef.tcl

#===========================================================
##   read verilog
#===========================================================
verilog_init -path $VERILOG_PATH -top gcd

#===========================================================
##   read def
#===========================================================
#def_init -path $PRE_RESULT_PATH/$DESIGN.def

#===========================================================
##   init floorplan
##   gcd & uart
#===========================================================
set DIE_AREA "0.0    0.0   279.96   280.128"
set CORE_AREA "9.996 10.08 269.964  270.048"

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

source ./script/iFP_script/module/gern_tracks.tcl

#===========================================================
##   Place IO
#===========================================================
#source $PROJ_PATH/scripts/$DESIGN/iFP_script/asic_top_0627.io.tcl

#===========================================================
##   Place Macro
#===========================================================
#source $PROJ_PATH/scripts/$DESIGN/iFP_script/macro_place.tcl

#===========================================================
##   Place IO Port
#===========================================================
#source $PROJ_PATH/scripts/$DESIGN/iFP_script/place_pad_new.tcl
auto_place_pins -layer met5 -width 2000 -height 2000

#===========================================================
##   Add IO Filler
#===========================================================
#placeIoFiller \
#   -filler_types "PFILL50W PFILL20W PFILL10W PFILL5W PFILL2W PFILL01W PFILL001W" \
#   -prefix IOFIL

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
source ./script/iFP_script/module/user_pg.tcl 
#source ./script/iFP_script/module/place_powerPad.tcl
source ./script/iFP_script/module/addPowerStripe.tcl
#source ./script/iFP_script/module/connect_power_io.tcl 

#===========================================================
##   set clock net
#===========================================================
source ./script/iFP_script/module/set_clocknet.tcl

#===========================================================
##   remove pg net
#===========================================================
#source $PROJ_PATH/scripts/$DESIGN/iFP_script/clear_blockage.tcl

#===========================================================
##   Save def 
#===========================================================
def_save -path ./result/iFP_result.def

#===========================================================
##   report 
#===========================================================
report_db -path "./result/report/fp_db.rpt"

#===========================================================
##   Exit 
#===========================================================
flow_exit
