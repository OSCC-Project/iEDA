
set env_vars {
  NUM_THREADS
  RESULT_DIR
  INPUT_DEF
  OUTPUT_DEF
  OUTPUT_VERILOG
  DESIGN_STAT_TEXT
  DESIGN_STAT_JSON
  TOOL_METRICS_JSON
  TOOL_REPORT_DIR
  SPEF_PATH
  SDC_FILE
  GDS_FILE
  LAYOUT_JSON_FILE
}

foreach var $env_vars {
    if { [info exists ::env($var)] } {
        set $var $::env($var)
    }
}
