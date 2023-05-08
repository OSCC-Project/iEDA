/**
 * @file main.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The main file of ipower.
 * @version 0.1
 * @date 2023-05-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <signal.h>

#include <cstdlib>
#include <fstream>

#include "VCDFileParser.hpp"
#include "api/Power.hh"
#include "api/TimingEngine.hh"
#include "include/PwrConfig.hh"
#include "log/Log.hh"
#include "ops/build_graph/PwrBuildGraph.hh"
#include "ops/read_vcd/VCDParserWrapper.hh"
#include "tcl/UserShell.hh"
#include "usage/usage.hh"

#define TCL_USERSHELL

#ifdef TCL_USERSHELL
#include "shell-cmd/PowerShellCmd.hh"
#include "shell-cmd/ShellCmd.hh"
#endif

using namespace ipower;
using namespace ista;
using ieda::Stats;

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
  registerTclCmd(CmdReportConstraint, "report_constraint");
  registerTclCmd(CmdReadVcd, "read_vcd");
  registerTclCmd(CmdReportPower, "report_power");

  return EXIT_SUCCESS;
}
#endif

#ifndef TEST_SHELL
#define TEST_SHELL 1
#define TEST_1 0
#define TEST_2 0
#else
#define TEST_SHELL 1
#endif

void sig_handler(int sig) {
  std::time_t t = std::time(nullptr);

  std::cout << "received signal " << sig << std::endl;

  std::cout << "ipower finished at "
            << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")
            << std::endl;

  abort();
}

int main(int argc, char** argv) {
  // signal(SIGINT, sig_handler);
  // signal(SIGILL, sig_handler);
  // signal(SIGABRT, sig_handler);
  // signal(SIGFPE, sig_handler);
  // signal(SIGSEGV, sig_handler);
  // signal(SIGTERM, sig_handler);
  // signal(SIGQUIT, sig_handler);
  // signal(SIGKILL, sig_handler);

#if TEST_SHELL

  Log::init(argv);

  std::string hello_info =
      "\033[49;32m********************************\n"
      "  _   _______  ____      ____     \n"
      " (_) |_   __ \\|_  _|    |_  _|  \n"
      " __    | |__) | \\ \\  /\\  / /   \n"
      "[  |   |  ___/   \\ \\/  \\/ / \n "
      "| |  _| |_       \\  /\\  /    \n"
      "[___]|_____|       \\/  \\/  \n"
      "********************************\n"

      "WELCOME TO iPower TCL-shell interface. \n";

  // get an UserShell (singleton) instance
  auto* shell = ieda::UserShell::getShell();
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

  // start a user shell
  shell->userMain(tcl_file_path);

  Log::end();

#endif


  Log::end();

  return 0;
}
