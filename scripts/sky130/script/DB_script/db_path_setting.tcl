#===========================================================
##   set tech lef path
#===========================================================
set TECH_LEF_PATH "./lef/sky130_fd_sc_hs.tlef"
#===========================================================
##   set lef path
#===========================================================                             
set LEF_PATH "./lef/sky130_ef_io__com_bus_slice_10um.lef \
              ./lef/sky130_ef_io__corner_pad.lef \
              ./lef/sky130_ef_io__vccd_lvc_pad.lef \
              ./lef/sky130_ef_io__vssa_hvc_pad.lef \
              ./lef/sky130_ef_io__vssio_lvc_pad.lef \
              ./lef/sky130_sram_1rw1r_80x64_8.lef \
              ./lef/sky130_ef_io__com_bus_slice_1um.lef \
              ./lef/sky130_ef_io__disconnect_vccd_slice_5um.lef \
              ./lef/sky130_ef_io__vdda_hvc_pad.lef \
              ./lef/sky130_ef_io__vssa_lvc_pad.lef \
              ./lef/sky130_fd_io__top_xres4v2.lef \
              ./lef/sky130io_fill.lef \
              ./lef/sky130_ef_io__com_bus_slice_20um.lef \
              ./lef/sky130_ef_io__disconnect_vdda_slice_5um.lef \
              ./lef/sky130_ef_io__vdda_lvc_pad.lef \
              ./lef/sky130_ef_io__vssd_hvc_pad.lef \
              ./lef/sky130_fd_sc_hd_merged.lef \
              ./lef/sky130_sram_1rw1r_128x256_8.lef \
              ./lef/sky130_ef_io__com_bus_slice_5um.lef \
              ./lef/sky130_ef_io__gpiov2_pad_wrapped.lef \
              ./lef/sky130_ef_io__vddio_hvc_pad.lef \
              ./lef/sky130_ef_io__vssd_lvc_pad.lef \
              ./lef/sky130_sram_1rw1r_44x64_8.lef \
              ./lef/sky130_ef_io__connect_vcchib_vccd_and_vswitch_vddio_slice_20um.lef \
              ./lef/sky130_ef_io__vccd_hvc_pad.lef \
              ./lef/sky130_ef_io__vddio_lvc_pad.lef \
              ./lef/sky130_ef_io__vssio_hvc_pad.lef \
              ./lef/sky130_fd_sc_hs_merged.lef \
              ./lef/sky130_sram_1rw1r_64x256_8.lef"
#===========================================================
##   set common lib path
#===========================================================
set LIB_PATH "./lib/sky130_dummy_io.lib \
      ./lib/sky130_fd_sc_hs__tt_025C_1v80.lib \
      ./lib/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_fd_sc_hd__tt_025C_1v80.lib \
      ./lib/sky130_fd_sc_hs__tt_100C_1v80.lib \
      ./lib/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"
#===========================================================
##   set fix fanout lib path
#===========================================================
set LIB_PATH_FIXFANOUT "./lib/sky130_dummy_io.lib \
      ./lib/sky130_fd_sc_hs__tt_025C_1v80.lib \
      ./lib/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_fd_sc_hd__tt_025C_1v80.lib \
      ./lib/sky130_fd_sc_hs__tt_100C_1v80.lib \
      ./lib/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"
#===========================================================
##   set drv lib path
#===========================================================
set LIB_PATH_DRV "./lib/sky130_dummy_io.lib \
      ./lib/sky130_fd_sc_hs__tt_025C_1v80.lib \
      ./lib/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_fd_sc_hd__tt_025C_1v80.lib \
      ./lib/sky130_fd_sc_hs__tt_100C_1v80.lib \
      ./lib/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"
#===========================================================
##   set hold lib path
#===========================================================
set LIB_PATH_HOLD "./lib/sky130_dummy_io.lib \
      ./lib/sky130_fd_sc_hs__tt_025C_1v80.lib \
      ./lib/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_fd_sc_hd__tt_025C_1v80.lib \
      ./lib/sky130_fd_sc_hs__tt_100C_1v80.lib \
      ./lib/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"
#===========================================================
##   set setup lib path
#===========================================================
set LIB_PATH_SETUP "./lib/sky130_dummy_io.lib \
      ./lib/sky130_fd_sc_hs__tt_025C_1v80.lib \
      ./lib/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib \
      ./lib/sky130_fd_sc_hd__tt_025C_1v80.lib \
      ./lib/sky130_fd_sc_hs__tt_100C_1v80.lib \
      ./lib/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib \
      ./lib/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib"
#===========================================================
##   set sdc path
#===========================================================
set SDC_PATH "./sdc/gcd.sdc"
#set SDC_PATH "./sdc/uart.sdc"
#set SDC_PATH "./sdc/aes_cipher_top.sdc"
#===========================================================
##   set spef path
#===========================================================
#set SPEF_PATH "./spef/xxx.spef"
