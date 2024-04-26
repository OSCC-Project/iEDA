set DB_LEF_PATH "$::env(FOUNDRY_DIR)/lef"
set DB_LIB_PATH "$::env(FOUNDRY_DIR)/lib"

#===========================================================
##   set HD or HS
#===========================================================
set CELL_TYPE "HS"

#===========================================================
##   set tech lef path
#===========================================================
set TECH_LEF_HS_PATH "$DB_LEF_PATH/sky130_fd_sc_hs.tlef"
set TECH_LEF_HD_PATH "$DB_LEF_PATH/sky130_fd_sc_hd.tlef"

if { $CELL_TYPE == "HS" } {
    set TECH_LEF_PATH $TECH_LEF_HS_PATH
} else {
    set TECH_LEF_PATH $TECH_LEF_HD_PATH
}
#===========================================================
##   set lef path
#===========================================================                             
set LEF_HS_PATH "$DB_LEF_PATH/sky130_fd_sc_hs_merged.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_10um.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_1um.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_20um.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_5um.lef \
            $DB_LEF_PATH/sky130_ef_io__connect_vcchib_vccd_and_vswitch_vddio_slice_20um.lef \
            $DB_LEF_PATH/sky130_ef_io__corner_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__disconnect_vccd_slice_5um.lef \
            $DB_LEF_PATH/sky130_ef_io__disconnect_vdda_slice_5um.lef \
            $DB_LEF_PATH/sky130_ef_io__gpiov2_pad_wrapped.lef \
            $DB_LEF_PATH/sky130_ef_io__vccd_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vccd_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vdda_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vdda_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vddio_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vddio_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssa_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssa_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssd_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssd_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssio_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssio_lvc_pad.lef \
            $DB_LEF_PATH/sky130_fd_io__top_xres4v2.lef \
            $DB_LEF_PATH/sky130io_fill.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_128x256_8.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_44x64_8.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_64x256_8.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_80x64_8.lef"

set LEF_HD_PATH "$DB_LEF_PATH/sky130_fd_sc_hd_merged.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_10um.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_1um.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_20um.lef \
            $DB_LEF_PATH/sky130_ef_io__com_bus_slice_5um.lef \
            $DB_LEF_PATH/sky130_ef_io__connect_vcchib_vccd_and_vswitch_vddio_slice_20um.lef \
            $DB_LEF_PATH/sky130_ef_io__corner_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__disconnect_vccd_slice_5um.lef \
            $DB_LEF_PATH/sky130_ef_io__disconnect_vdda_slice_5um.lef \
            $DB_LEF_PATH/sky130_ef_io__gpiov2_pad_wrapped.lef \
            $DB_LEF_PATH/sky130_ef_io__vccd_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vccd_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vdda_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vdda_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vddio_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vddio_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssa_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssa_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssd_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssd_lvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssio_hvc_pad.lef \
            $DB_LEF_PATH/sky130_ef_io__vssio_lvc_pad.lef \
            $DB_LEF_PATH/sky130_fd_io__top_xres4v2.lef \
            $DB_LEF_PATH/sky130io_fill.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_128x256_8.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_44x64_8.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_64x256_8.lef \
            $DB_LEF_PATH/sky130_sram_1rw1r_80x64_8.lef"

if { $CELL_TYPE == "HS" } {
    set LEF_PATH ${LEF_HS_PATH}
} else {
    set LEF_PATH ${LEF_HD_PATH}
}

#===========================================================
##   set common lib path
#===========================================================
set LIB_HS_PATH "$DB_LIB_PATH/sky130_fd_sc_hs__tt_025C_1v80.lib \
      $DB_LIB_PATH/sky130_dummy_io.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"

set LIB_HD_PATH "$DB_LIB_PATH/sky130_fd_sc_hd__tt_025C_1v80.lib \
      $DB_LIB_PATH/sky130_dummy_io.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      $DB_LIB_PATH/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"

if { $CELL_TYPE == "HS" } {
    set LIB_PATH ${LIB_HS_PATH}
} else {
    set LIB_PATH ${LIB_HD_PATH}
}

#===========================================================
##   set fix fanout lib path
#===========================================================
set LIB_PATH_FIXFANOUT ${LIB_PATH}

#===========================================================
##   set drv lib path
#===========================================================
set LIB_PATH_DRV ${LIB_PATH}

#===========================================================
##   set hold lib path
#===========================================================
set LIB_PATH_HOLD ${LIB_PATH}

#===========================================================
##   set setup lib path
#===========================================================
set LIB_PATH_SETUP ${LIB_PATH}

#===========================================================
##   set sdc path
#===========================================================
set SDC_PATH "$::env(SDC_FILE)"

#===========================================================
##   set spef path
#===========================================================
if {[info exists ::env(SPEF_FILE)]} {
    set SPEF_PATH $::env(SPEF_FILE)
}
