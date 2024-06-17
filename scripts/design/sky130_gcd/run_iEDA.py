#!/bin/python3
from config import *
import os

pwd = os.getcwd()
foundry_dir = f'{pwd}/../../foundry/sky130'

env_vars = {
    'DESIGN_TOP'    : 'gcd',
    'NETLIST_FILE'  : f'{pwd}/result/verilog/gcd.v',
    'SDC_FILE'      : f'{foundry_dir}/sdc/gcd.sdc',
    'SPEF_FILE'     : f'{foundry_dir}/spef/gcd.spef',
    'DIE_AREA'      : '0.0   0.0   149.96   150.128',
    'CORE_AREA'     : '9.996 10.08 139.964  140.048',
    'FOUNDRY_DIR'   : foundry_dir,
    'CONFIG_DIR'    : f'{pwd}/iEDA_config',
    'RESULT_DIR'    : f'{pwd}/result',
    'TCL_SCRIPT_DIR': f'{pwd}/script',
}

def main():
    set_environment_variables(env_vars)

    #===========================================================
    ##   run floorplan
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iFP_script/run_iFP.tcl')

    #===========================================================
    ##   run NO -- fix fanout
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iNO_script/run_iNO_fix_fanout.tcl')

    #===========================================================
    ##   run Placer
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iPL_script/run_iPL.tcl')
    # execute_shell_command('./iEDA -script ./script/iPL_script/run_iPL_eval.tcl')

    # ===========================================================
    #   run CTS
    # ===========================================================
    execute_shell_command('./iEDA -script ./script/iCTS_script/run_iCTS.tcl')
    # execute_shell_command('./iEDA -script ./script/iCTS_script/run_iCTS_eval.tcl')
    execute_shell_command('./iEDA -script ./script/iCTS_script/run_iCTS_STA.tcl')

    #===========================================================
    ##   run TO -- fix_drv
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iTO_script/run_iTO_drv.tcl')
    execute_shell_command('./iEDA -script ./script/iTO_script/run_iTO_drv_STA.tcl')

    #===========================================================
    #   run TO -- opt_hold
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iTO_script/run_iTO_hold.tcl')
    execute_shell_command('./iEDA -script ./script/iTO_script/run_iTO_hold_STA.tcl')

    # ===========================================================
    # #   run TO -- opt_setup
    # ===========================================================
    # execute_shell_command('./iEDA -script ./script/iTO_script/run_iTO_setup.tcl')

    #===========================================================
    #   run PL Incremental Flow
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iPL_script/run_iPL_legalization.tcl')
    # execute_shell_command('./iEDA -script ./script/iPL_script/run_iPL_legalization_eval.tcl')

    # ===========================================================
    # #   run Router
    execute_shell_command('./iEDA -script ./script/iRT_script/run_iRT.tcl')
    # execute_shell_command('./iEDA -script ./script/iRT_script/run_iRT_eval.tcl')
    # execute_shell_command('./iEDA -script ./script/iRT_script/run_iRT_STA.tcl')
    execute_shell_command('./iEDA -script ./script/iRT_script/run_iRT_DRC.tcl')

    #===========================================================
    ##   run DRC --- report
    #===========================================================
    # execute_shell_command('./iEDA -script ./script/iDRC_script/run_iDRC.tcl')

    #===========================================================
    ##   run Filler
    #===========================================================
    execute_shell_command('./iEDA -script ./script/iPL_script/run_iPL_filler.tcl')

    #===========================================================
    ##   run ECO
    #===========================================================

    #===========================================================
    ##   run PV
    #===========================================================

    #===========================================================
    ##   run def to gdsii
    #===========================================================
    execute_shell_command('./iEDA -script ./script/DB_script/run_def_to_gds_text.tcl')

if __name__ == "__main__":
    main()
