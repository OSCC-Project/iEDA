#!/bin/python3
import os
import re
import sys
import argparse  # argument parsing
import time

#===========================================================
##   run floorplan
#===========================================================
os.system('./iEDA -script ./script/iFP_script/run_iFP.tcl')

#===========================================================
##   run NO -- fix fanout
#===========================================================
os.system('./iEDA -script ./script/iNO_script/run_iNO_fix_fanout.tcl')

#===========================================================
##   run Placer
#===========================================================
os.system('./iEDA -script ./script/iPL_script/run_iPL.tcl')
os.system('./iEDA -script ./script/iPL_script/run_iPL_eval.tcl')

# ===========================================================
#   run CTS
# ===========================================================
os.system('./iEDA -script ./script/iCTS_script/run_iCTS.tcl')
os.system('./iEDA -script ./script/iCTS_script/run_iCTS_eval.tcl')
os.system('./iEDA -script ./script/iCTS_script/run_iCTS_STA.tcl')

#===========================================================
##   run TO -- fix_drv
#===========================================================
os.system('./iEDA -script ./script/iTO_script/run_iTO_drv.tcl')
os.system('./iEDA -script ./script/iTO_script/run_iTO_drv_STA.tcl')

#===========================================================
#   run TO -- opt_hold
#===========================================================
os.system('./iEDA -script ./script/iTO_script/run_iTO_hold.tcl')
os.system('./iEDA -script ./script/iTO_script/run_iTO_hold_STA.tcl')

# ===========================================================
# #   run TO -- opt_setup
# ===========================================================
# os.system('./iEDA -script ./script/iTO_script/run_iTO_setup.tcl')

#===========================================================
#   run PL Incremental Flow
#===========================================================
os.system('./iEDA -script ./script/iPL_script/run_iPL_legalization.tcl')
os.system('./iEDA -script ./script/iPL_script/run_iPL_legalization_eval.tcl')

# ===========================================================
# #   run Router
##./iEDA -script ./script/iRT_script/run_iGR.tcl
##./iEDA -script ./script/iRT_script/run_iRT2.0.tcl
# ===========================================================
os.system('./iEDA -script ./script/iRT_script/run_iRT.tcl')
os.system('./iEDA -script ./script/iRT_script/run_iRT_eval.tcl')
os.system('./iEDA -script ./script/iRT_script/run_iRT_STA.tcl')
os.system('./iEDA -script ./script/iRT_script/run_iRT_DRC.tcl')

#===========================================================
##   run DRC --- report
#===========================================================
# os.system('./iEDA -script ./script/iDRC_script/run_iDRC.tcl')

#===========================================================
##   run Filler
#===========================================================
os.system('./iEDA -script ./script/iPL_script/run_iPL_filler.tcl')

#===========================================================
##   run ECO
#===========================================================

#===========================================================
##   run PV
#===========================================================

#===========================================================
##   run def to gdsii
#===========================================================
os.system('./iEDA -script ./script/DB_script/run_def_to_gds_text.tcl')

#===========================================================
##   run STA
#===========================================================
#os.system('./iEDA -script ./script/iSTA_script/run_iSTA.tcl')

#===========================================================
##   run GUI
# # ./iEDA -script ./script/iGUI_script/run_iGUI.tcl
# # ./iEDA_gui -script ./script/iGUI_script/run_iGUI.tcl
#===========================================================

#===========================================================
##   run DB
# os.system('./iEDA -script ./script/DB_script/run_db.tcl')
# os.system('./iEDA -script ./script/DB_script/run_def_to_gds_text.tcl')
##./iEDA -script ./script/DB_script/run_db_checknet.tcl
#===========================================================