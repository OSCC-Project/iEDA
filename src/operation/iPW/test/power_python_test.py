import ipower_cpp

work_dir = "/home/taosimin/iEDA/src/operation/iPW/source/data/example1"

ipower_cpp.set_design_workspace(work_dir + "/rpt")
ipower_cpp.read_netlist(work_dir + "/aes_cipher_top.v")
ipower_cpp.read_liberty(work_dir + "/sky130_fd_sc_hd__tt_025C_1v80.lib")
ipower_cpp.link_design("aes_cipher_top")
ipower_cpp.read_sdc(work_dir + "/aes_cipher_top.sdc")
ipower_cpp.read_spef(work_dir + "/aes_cipher_top.spef")
ipower_cpp.report_timing()

ipower_cpp.report_power()
