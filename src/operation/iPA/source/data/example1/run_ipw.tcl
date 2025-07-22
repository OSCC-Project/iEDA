
set work_dir "../src/operation/iSTA/source/data/example1"

set_design_workspace $work_dir/rpt


read_netlist $work_dir/example1.v

set LIB_FILES $work_dir/example1_slow.lib

read_liberty $LIB_FILES

link_design top

read_sdc  $work_dir/example1.sdc
read_spef $work_dir/example1.spef

report_timing

report_power

