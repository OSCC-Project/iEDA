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
 * @brief The main file of ipower.
 * @version 0.1
 * @date 2023-05-05
 */
#include <signal.h>

#include <cstdlib>
#include <fstream>

#include "api/Power.hh"
#include "api/TimingEngine.hh"
#include "include/PwrConfig.hh"
#include "include/Version.hh"
#include "include/cxxopts.hpp"
#include "log/Log.hh"
#include "ops/build_graph/PwrBuildGraph.hh"
#include "shell-cmd/PowerShellCmd.hh"
#include "shell-cmd/ShellCmd.hh"
#include "tcl/UserShell.hh"
#include "usage/usage.hh"

using namespace ipower;
using namespace ista;
using ieda::Stats;

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
  registerTclCmd(CmdSetPwrDesignWorkSpace, "set_pwr_design_workspace");
  registerTclCmd(CmdReadVcd, "read_vcd");
  registerTclCmd(CmdReportPower, "report_power");

  registerTclCmd(CmdReadPGSpef, "read_pg_spef");
  registerTclCmd(CmdReportIRDrop, "report_ir_drop");

  return EXIT_SUCCESS;
}

void sig_handler(int sig) {
  std::time_t t = std::time(nullptr);

  std::cout << "received signal " << sig << std::endl;

  std::cout << "ipower finished at "
            << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")
            << std::endl;

  abort();
}

int main(int argc, char** argv) {
  Log::init(argv);
  // Start the timer
  Time::start();

  std::string hello_info =
      "\033[49;32m********************************\n"
      "  _   _______  ____      ____     \n"
      " (_) |_   __ \\|_  _|    |_  _|  \n"
      " __    | |__) | \\ \\  /\\  / /   \n"
      "[  |   |  ___/   \\ \\/  \\/ / \n "
      "| |  _| |_       \\  /\\  /    \n"
      "[___]|_____|       \\/  \\/  \n"
      "********************************\n"

      "WELCOME TO iPower TCL-shell interface. \e[0m";

  // get an UserShell (singleton) instance
  auto* shell = ieda::UserShell::getShell();
  shell->displayHello(hello_info);

  // set call back for register commands
  shell->set_init_func(registerCommands);

  // if no args, run interactively
  cxxopts::Options options("iPW", "iPW command line help.");
  options.add_options()("v,version", "Print Git Version")(
      "h,help", "iPW command usage.")("script", "Tcl script file",
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
