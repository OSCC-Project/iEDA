set design_name s1238

set work_dir "/data/yexinyu/iEDA/src/operation/iSTA/source/data"

set_design_workspace /data/yexinyu/t28_model/dataset/IEDA_report/$design_name

read_netlist $work_dir/$design_name/$design_name.v


set LIB_DIR /home/taosimin/T28/lib
set LIB_FILES "$LIB_DIR/tcbn28hpcplusbwp30p140hvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140lvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140mblvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140mbssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140opphvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140opplvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140oppssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140oppuhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140oppulvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140ssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140uhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp30p140ulvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140hvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140lvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140mbhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140mblvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140mbssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140opphvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140opplvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140oppssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140oppuhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140oppulvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140ssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140uhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp35p140ulvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140ehvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140hvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140lvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140mbhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140mbssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140oppehvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140opphvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140opplvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140oppssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140oppuhvtssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140ssg0p81v125c.lib \
      $LIB_DIR/tcbn28hpcplusbwp40p140uhvtssg0p81v125c.lib \
      $LIB_DIR/ts5n28hpcplvta256x32m4fw_130a_ssg0p81v125c.lib \
      $LIB_DIR/ts5n28hpcplvta64x128m2fw_130a_ssg0p81v125c.lib \
      $LIB_DIR/tphn28hpcpgv18ssg0p81v1p62v125c.lib \
      $LIB_DIR/PLLTS28HPMLAINT_SS_0P81_125C.lib"

read_liberty $LIB_FILES

link_design $design_name

read_sdc  $work_dir/$design_name/$design_name.sdc
read_spef $work_dir/$design_name/${design_name}.spef.rcworst.0c

report_timing

#report_power
