
# (fixed) iEDA setting
set ::env(CONFIG_DIR) $::env(WORKSPACE)/iEDA_config
set ::env(FOUNDRY_DIR) $::env(WORKSPACE)/../../foundry/sky130
set ::env(RESULT_DIR) $::env(WORKSPACE)/result
set ::env(TCL_SCRIPT_DIR) $::env(WORKSPACE)/script

# design files
set ::env(DESIGN_TOP) gcd
set ::env(NETLIST_FILE) $::env(WORKSPACE)/result/verilog/gcd.v
set ::env(SDC_FILE) $::env(FOUNDRY_DIR)/sdc/gcd.sdc
set ::env(SPEF_FILE) $::env(FOUNDRY_DIR)/spef/gcd.spef

# floorplan setting
## gcd
set ::env(DIE_AREA) "0.0    0.0   149.96   150.128"
set ::env(CORE_AREA) "9.996 10.08 139.964  140.048"

puts "execuate iFP script start"
exec -ignorestderr $::env(iEDA) -script "$::env(TCL_SCRIPT_DIR)/iFP_script/run_iFP.tcl" >@ stdout
puts "execuate iFP script finished"

set pattern {s/\( [^+ ]*\) + NET  +/\1 + NET\1 +/}
exec sed -i $pattern $::env(RESULT_DIR)/iFP_result.def

set TCL_SCRIPTS "iNO_script/run_iNO_fix_fanout.tcl \
iPL_script/run_iPL.tcl \
iCTS_script/run_iCTS.tcl \
iCTS_script/run_iCTS_STA.tcl \
iTO_script/run_iTO_drv.tcl \
iTO_script/run_iTO_drv_STA.tcl \
iTO_script/run_iTO_hold.tcl \
iTO_script/run_iTO_hold_STA.tcl \
iPL_script/run_iPL_legalization.tcl \
iRT_script/run_iRT.tcl \
iRT_script/run_iRT_DRC.tcl \
iPL_script/run_iPL_filler.tcl \
DB_script/run_def_to_gds_text.tcl"

foreach SCRIPT $TCL_SCRIPTS {
    puts "execuate script $::env(TCL_SCRIPT_DIR)/${SCRIPT} start"
    exec -ignorestderr $::env(iEDA) -script "$::env(TCL_SCRIPT_DIR)/${SCRIPT}" >@ stdout
    puts "execuate script $::env(TCL_SCRIPT_DIR)/${SCRIPT} finished"
}
