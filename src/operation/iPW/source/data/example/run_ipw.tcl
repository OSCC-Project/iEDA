
set work_dir ../src/operation/iPW/source/data/example
set_design_workspace $work_dir/rpt

read_netlist $work_dir/aes_cipher_top.v

set LIB_FILES  $work_dir/sky130_fd_sc_hd__tt_025C_1v80.lib

read_liberty $LIB_FILES

link_design aes_cipher_top

read_sdc  $work_dir/aes_cipher_top.sdc
read_spef $work_dir/aes_cipher_top.spef

report_timing

report_power