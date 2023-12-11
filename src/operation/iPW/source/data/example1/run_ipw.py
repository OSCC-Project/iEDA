import ipower_cpp

work_dir = "../src/operation/iSTA/source/data/example1"

ipower_cpp.set_design_workspace(work_dir + "/rpt")
ipower_cpp.read_netlist(work_dir + "/example1.v")
ipower_cpp.read_liberty([work_dir + "/example1_slow.lib"])
ipower_cpp.link_design("top")
ipower_cpp.read_sdc(work_dir + "/example1.sdc")
ipower_cpp.read_spef(work_dir + "/example1.spef")
ipower_cpp.report_timing()

ipower_cpp.report_power()
