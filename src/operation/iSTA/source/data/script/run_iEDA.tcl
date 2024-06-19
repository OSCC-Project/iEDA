set design_name s1238

set work_dir "/data/yexinyu/iEDA/src/operation/iSTA/source/data"

set_design_workspace /data/yexinyu/t28_model/dataset/IEDA_report/$design_name

read_netlist $work_dir/$design_name/$design_name.v


set LIB_DIR /home/taosimin/T28/lib
set LIB_FILES ""

read_liberty $LIB_FILES

link_design $design_name

read_sdc  $work_dir/$design_name/$design_name.sdc
read_spef $work_dir/$design_name/${design_name}.spef.rcworst.0c

report_timing

#report_power
