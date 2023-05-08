/**
 * @file main.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The main function of sta.
 * @version 0.1
 * @date 2020-11-23
 */
#include <gflags/gflags.h>
// #include <gperftools/heap-profiler.h>
// #include <gperftools/profiler.h>

#include <iostream>

#include "absl/strings/match.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "liberty/Liberty.hh"
#include "log/Log.hh"
#include "netlist/Netlist.hh"
#include "sdc-cmd/Cmd.hh"
#include "sta/Sta.hh"
#include "sta/StaAnalyze.hh"
#include "sta/StaApplySdc.hh"
#include "sta/StaBuildGraph.hh"
#include "sta/StaBuildRCTree.hh"
#include "sta/StaClockPropagation.hh"
#include "sta/StaDataPropagation.hh"
#include "sta/StaDelayPropagation.hh"
#include "sta/StaDump.hh"
#include "sta/StaGraph.hh"
#include "sta/StaSlewPropagation.hh"
#include "tcl/UserShell.hh"
#include "usage/usage.hh"

#define TCL_USERSHELL

#ifdef TCL_USERSHELL
#include "sdc-cmd/Cmd.hh"
#include "shell-cmd/ShellCmd.hh"
#endif

DEFINE_string(confPath, "test.conf", "program configure file.");

using namespace ista;

#ifdef TCL_USERSHELL
int registerCommands() {
  registerTclCmd(CmdSetDesignWorkSpace, "set_design_workspace");
  registerTclCmd(CmdReadVerilog, "read_netlist");
  registerTclCmd(CmdReadLiberty, "read_liberty");
  registerTclCmd(CmdLinkDesign, "link_design");
  registerTclCmd(CmdReadSpef, "read_spef");
  registerTclCmd(CmdReadSdc, "read_sdc");
  registerTclCmd(CmdReportTiming, "report_timing");
  registerTclCmd(CmdReportConstraint, "report_constraint");

  return EXIT_SUCCESS;
}
#endif

using ieda::Stats;

int main(int argc, char** argv) {
  Log::init(argv);

  std::string hello_info =
      "\033[49;32m***************************\n"
      "  _  _____  _____   ___  \n"
      " (_)/  ___||_   _| / _ \\ \n"
      "  _ \\ `--.   | |  / /_\\ \\\n"
      " | | `--. \\  | |  |  _  |\n"
      " | |/\\__/ /  | |  | | | |\n"
      " |_|\\____/   \\_/  \\_| |_/\n"
      "***************************\n"
      "WELCOME TO iSTA TCL-shell interface. \n";

  // get an UserShell (singleton) instance
  auto shell = ieda::UserShell::getShell();
  shell->displayHello(hello_info);

  // set call back for register commands
  shell->set_init_func(registerCommands);

  // get Tcl file path from main args
  char* tcl_file_path = nullptr;
  if (argc == 2) {
    tcl_file_path = argv[1];
  } else {
    shell->displayHelp();
  }

  // tcl_file_path = "/home/smtao/90bak/iEDA/src/iSTA/run_ista.tcl";

  // start a user shell
  shell->userMain(tcl_file_path);

  Log::end();

  return 0;
}
