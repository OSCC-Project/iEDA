// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file UserShell.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-13
 */

#include "UserShell.hh"

// read line
#include <readline/history.h>
#include <readline/readline.h>
// args
#include <getopt.h>

#include <iostream>

#include "CommandList.hh"
#include "log/Log.hh"
#include "sta/Sta.hh"
#include "tcl/ScriptEngine.hh"
// commands implements
// #include "sdc-cmd/Cmd.hh"
#include "shell-cmd/ShellCmd.hh"

namespace ista {

UserShell* UserShell::getShell() {
  static UserShell shell;
  return &shell;
}

int UserShell::userMain(int argc, char** argv) {
  // initShellLog(argv);
  initCommandList();
  if (argc > 1) {
    return parseOptionArgs(argc, argv);
  }

  displayHello();
  execCMD();
  exitWithMessage("exit program");
  return 1;
}

int UserShell::parseOptionArgs(int argc, char** argv_const) {
  const static struct option table[] = {
      {"file", required_argument, NULL, 'f'},
      {"help", no_argument, NULL, 'h'},
  };

  char** argv = const_cast<char**>(argv_const);
  int option = 0;

  while ((option = getopt_long(argc, argv, "-f:h", table, NULL)) != EOF) {
    switch (option) {
      case 'f':
        execFile(optarg);
        break;
      default:
        displayHelp();
        break;
    }
  }
  return 0;
}

// make it library/archive?
void UserShell::initCommandList() {
  auto cmd_ptr = std::make_unique<CmdReadVerilog>("read_verilog");
  TclCmds::addTclCmd(std::move(cmd_ptr));

  auto cmd_ptr1 = std::make_unique<CmdReadLiberty>("read_liberty");
  TclCmds::addTclCmd(std::move(cmd_ptr1));

  auto cmd_ptr2 = std::make_unique<CmdLinkDesign>("link_design");
  TclCmds::addTclCmd(std::move(cmd_ptr2));

  auto cmd_ptr3 = std::make_unique<CmdReadSpef>("read_spef");
  TclCmds::addTclCmd(std::move(cmd_ptr3));

  auto cmd_ptr4 = std::make_unique<CmdReadSdc>("read_sdc");
  TclCmds::addTclCmd(std::move(cmd_ptr4));

  auto cmd_ptr5 = std::make_unique<CmdReportTiming>("report_timing");
  TclCmds::addTclCmd(std::move(cmd_ptr5));

  auto cmd_ptr6 = std::make_unique<CmdReportConstraint>("report_constraint");
  TclCmds::addTclCmd(std::move(cmd_ptr6));
}

void UserShell::initShellLog(const char** argv) {
  // Log::init(argv);
}

/**
 * @brief Get the Input command from user shell
 *
 * @param buffer
 */
void getInputCMD(char*& buffer) {
  if (buffer) {
    free(buffer);
    buffer = nullptr;
  }

  buffer = readline("\033[49;34m$>\033[0m");

  if (buffer && *buffer) {
    add_history(buffer);
  }
}

int UserShell::execCMD() {
  while (1) {
    static char* cmd_buffer = nullptr;
    getInputCMD(cmd_buffer);

    if (Str::equal(cmd_buffer, ":q")) {
      break;
    } else if (!isCommandValid(cmd_buffer)) {
      std::cerr << "\033[49;31minvalid command\033[0m" << std::endl;
      continue;
    } else {
      ScriptEngine::getOrCreateInstance()->evalString(cmd_buffer);
      std::cerr << cmd_buffer << std::endl;
    }
  }
  return 1;
}

/**
 * @brief execute TCL script file
 *
 * @param file_path
 * @return unsigned:
 */
int UserShell::execFile(const char* file_path) {
  return ScriptEngine::getOrCreateInstance()->evalScriptFile(file_path);
}

void UserShell::displayHello() {
  std::cerr << "\033[49;32m***************************\n";
  std::cerr << "  _  _____  _____   ___  \n";
  std::cerr << " (_)/  ___||_   _| / _ \\ \n";
  std::cerr << "  _ \\ `--.   | |  / /_\\ \\\n";
  std::cerr << " | | `--. \\  | |  |  _  |\n";
  std::cerr << " | |/\\__/ /  | |  | | | |\n";
  std::cerr << " |_|\\____/   \\_/  \\_| |_/\n";
  std::cerr << "***************************\n";
  std::cerr << "| WELCOME TO iSTA TCL-shell interface. Enter \033[1m:q\033[0m "
               "\033[49;32mto quit. |\033[0m\n\n";
}

void UserShell::displayHelp() {
  std::cerr << "\033[49;32m"
               "*************************************\n"
            << "\033[1m"
               "Usage:"
               "\033[0m\n\033[49;32m"
            << " iSTA [--help/h] [--file/f file_path]\n"
            << "\033[1m"
               "Options:"
               "\033[0m\n\033[49;32m"
            << " --help\t display usage of iSTA\n"
            << " --file\t execute tcl script file\n"
            << "\033[1m"
               "TCL command:"
               "\033[0m\n\033[49;32m"
            << " Input TCL command after '$>'\n"
            << " Input ':q' to quit TCL-shell\n"
            << "*************************************"
               "\033[0m\n";
  exitWithMessage("see ya");
}

void UserShell::exitWithMessage(const char* message) {
  Sta* ista = Sta::getOrCreateSta();
  Sta::destroySta();
  std::cerr << "\033[49;32m| " << message << " |\033[0m\n" << std::endl;
  // Log::end();
  exit(0);
}

}  // namespace ista