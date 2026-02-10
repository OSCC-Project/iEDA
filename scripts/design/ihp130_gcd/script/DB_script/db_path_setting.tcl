set SDC_FILE "$::env(SDC_FILE)"

if {[info exists ::env(SPEF_FILE)]} {
    set SPEF_PATH $::env(SPEF_FILE)
}

#===========================================================
##   set HD or HS
#===========================================================
# set CELL_TYPE "HS"

#===========================================================
##   set tech lef path
#===========================================================
set TECH_LEF_PATH "$::env(FOUNDRY_DIR)/ihp-sg13g2/libs.ref/sg13g2_stdcell/lef/sg13g2_tech.lef"

#===========================================================
##   set lef path
#===========================================================                             
set LEF_PATH "$::env(FOUNDRY_DIR)/ihp-sg13g2/libs.ref/sg13g2_stdcell/lef/sg13g2_stdcell.lef \
              $::env(FOUNDRY_DIR)/ihp-sg13g2/libs.ref/sg13g2_io/lef/sg13g2_io.lef"

#===========================================================
##   set common lib path (using patched libs for iEDA compatibility)
#===========================================================
set PATCHED_LIB_DIR "[file dirname $::env(TCL_SCRIPT_DIR)]/lib_patched"
set LIB_PATH "$PATCHED_LIB_DIR/sg13g2_stdcell_typ_1p20V_25C.lib \
              $PATCHED_LIB_DIR/sg13g2_io_typ_1p2V_3p3V_25C.lib"

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
