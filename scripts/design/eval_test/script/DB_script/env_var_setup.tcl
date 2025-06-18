
if { [info exists ::env(NUM_THREADS)] } {
  set NUM_THREADS   "$::env(NUM_THREADS)"
}

if { [info exists ::env(RESULT_DIR)] } {
  set RESULT_DIR    "$::env(RESULT_DIR)"
}

if { [info exists ::env(INPUT_DEF)] } {
  set INPUT_DEF     "$::env(INPUT_DEF)"
}

if { [info exists ::env(OUTPUT_DEF)] } {
  set OUTPUT_DEF    "$::env(OUTPUT_DEF)"
}

if { [info exists ::env(OUTPUT_VERILOG)] } {
  set OUTPUT_VERILOG    "$::env(OUTPUT_VERILOG)"
}

if { [info exists ::env(DESIGN_STAT_TEXT)] } {
  set DESIGN_STAT_TEXT "$::env(DESIGN_STAT_TEXT)"
}

if { [info exists ::env(DESIGN_STAT_JSON)] } {
  set DESIGN_STAT_JSON "$::env(DESIGN_STAT_JSON)"
}

if { [info exists ::env(TOOL_METRICS_JSON)] } {
  set TOOL_METRICS_JSON "$::env(TOOL_METRICS_JSON)"
}

if { [info exists ::env(TOOL_REPORT_DIR)] } {
  set TOOL_REPORT_DIR "$::env(TOOL_REPORT_DIR)"
}

if {[info exists ::env(SPEF_FILE)]} {
    set SPEF_PATH $::env(SPEF_FILE)
}

if {[info exists ::env(SDC_FILE)]} {
    set SDC_PATH $::env(SDC_FILE)
}
