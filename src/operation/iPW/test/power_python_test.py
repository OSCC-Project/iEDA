
import ipower_cpp

import ieda_py

ieda_py.set_design_workspace(
    "/home/shaozheqing/iEDA/src/operation/iPW/source/data/example1/rpt")
ieda_py.read_netlist(
    "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/example1.v")
ieda_py.read_liberty(
    "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/example1_slow.lib")
ieda_py.link_design("top")
ieda_py.read_sdc(
    "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/example1.sdc")
ieda_py.read_spef(
    "/home/taosimin/iEDA/src/operation/iSTA/source/data/example1/example1.spef")

ipower_cpp.report_power_cpp()
