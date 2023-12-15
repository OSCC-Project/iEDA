set work_dir /home/taosimin/T28

set_design_workspace /home/longshuaiying/verilog_parser_rust/T28_SYN

# read_netlist /home/taosimin/T28/tapout/asic_top_1220.v
read_netlist /home/taosimin/T28/ieda_1208/asic_top_1208.syn.v

set LIB_FILES      "\
      $work_dir/ccslib/tcbn28hpcplusbwp30p140hvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140lvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140mblvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140mbssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140opphvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140opplvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140oppssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140oppuhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140oppulvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140ssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140uhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp30p140ulvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140hvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140lvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140mbhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140mblvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140mbssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140opphvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140opplvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140oppssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140oppuhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140oppulvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140ssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140uhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp35p140ulvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140ehvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140hvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140lvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140mbhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140mbssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140oppehvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140opphvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140opplvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140oppssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140oppuhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140ssg0p81v125c_ccs.lib \
      $work_dir/ccslib/tcbn28hpcplusbwp40p140uhvtssg0p81v125c_ccs.lib \
      $work_dir/ccslib/ts5n28hpcplvta256x32m4fw_130a_ssg0p81v125c.lib \
      $work_dir/ccslib/ts5n28hpcplvta64x128m2fw_130a_ssg0p81v125c.lib \
      $work_dir/ccslib/tphn28hpcpgv18ssg0p81v1p62v125c.lib \
      $work_dir/ccslib/PLLTS28HPMLAINT_SS_0P81_125C.lib"


read_liberty $LIB_FILES

link_design asic_top

read_sdc  /home/taosimin/T28/ieda_1204/asic_top_SYN_MAX.sdc 
#read_spef /home/taosimin/T28/spef/asic_top.rcworst.125c.spef 

report_timing
