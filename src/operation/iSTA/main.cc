// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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
#include <string>

#include "absl/strings/match.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "include/Version.hh"
#include "include/cxxopts.hpp"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Netlist.hh"
#include "sdc-cmd/Cmd.hh"
#include "shell-cmd/ShellCmd.hh"
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
#include "time/Time.hh"

DEFINE_string(confPath, "test.conf", "program configure file.");

using namespace ista;

int registerCommands() {
  registerTclCmd(CmdSetDesignWorkSpace, "set_design_workspace");
  registerTclCmd(CmdReadVerilog, "read_netlist");
  registerTclCmd(CmdReadLefDef, "read_lef_def");
  registerTclCmd(CmdReadLiberty, "read_liberty");
  registerTclCmd(CmdLinkDesign, "link_design");
  registerTclCmd(CmdReadSpef, "read_spef");
  registerTclCmd(CmdReadSdc, "read_sdc");
  registerTclCmd(CmdReportTiming, "report_timing");
  registerTclCmd(CmdReportConstraint, "report_constraint");
  registerTclCmd(CmdDefToVerilog, "def_to_verilog");
  registerTclCmd(CmdVerilogToDef, "verilog_to_def");
  registerTclCmd(CmdDumpGraphData, "dump_graph_data");

  return EXIT_SUCCESS;
}

using ieda::Stats;
using ieda::Time;

int main(int argc, char** argv) {
  Log::init(argv);
  // Start the timer
  Time::start();

  // for debug
  // Log::setVerboseLogLevel("Arnoldi*", 1);

  // for (int i = 0; i < argc; ++i) {
  //   std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
  // }

  std::string hello_info =
      "\033[49;32m***************************\n"
      "  _  _____  _____   ___  \n"
      " (_)/  ___||_   _| / _ \\ \n"
      "  _ \\ `--.   | |  / /_\\ \\\n"
      " | | `--. \\  | |  |  _  |\n"
      " | |/\\__/ /  | |  | | | |\n"
      " |_|\\____/   \\_/  \\_| |_/\n"
      "***************************\n"
      "WELCOME TO iSTA TCL-shell interface. \e[0m";

  // get an UserShell (singleton) instance
  auto shell = ieda::UserShell::getShell();

  // set call back for register commands
  shell->set_init_func(registerCommands);

  // if no args, run interactively
  cxxopts::Options options("iSTA", "iSTA command line help.");
  options.add_options()("v,version", "Print Git Version")(
      "h,help", "iSTA command usage.")("script", "Tcl script file",
                                       cxxopts::value<std::string>());

  try {
    options.parse_positional("script");

    auto argv_parse_result = options.parse(argc, argv);

    if (argv_parse_result.count("help")) {
      options.custom_help("script [OPTIONS...]");
      std::cout << std::endl;
      std::cout << options.help() << std::endl;
      return 0;
    }

    shell->displayHello(hello_info);

    if (argv_parse_result.count("version")) {
      std::cout << "\033[49;32mGit Version: " << GIT_VERSION << "\033[0m"
                << std::endl;
    }

    if (argc == 1) {
      shell->displayHelp();
      shell->userMain(argc, argv);
    } else if (argc == 2) {
      shell->userMain(argv[1]);
    } else {
      if (argv_parse_result.count("script")) {
        // discard the first arg from main()
        // pass the rest of the args to Tcl interpreter
        auto tcl_argc = argc - 1;
        auto tcl_argv = argv + 1;
        shell->userMain(tcl_argc, tcl_argv);
      }
    }
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Error parsing options: " << e.what() << std::endl;
    return 1;
  }

  Log::end();

  return 0;
}
