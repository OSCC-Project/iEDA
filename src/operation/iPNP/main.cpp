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
 * @file main.cpp
 * @author Jianrong Su
 * @brief The main function of iPNP. function: Launch the tcl console and interact with the user. Refer to iSTA/main.cc.
 * @version 1.0
 * @date 2025-07-01
 */

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include "PNP.hh"
#include "PNPConfig.hh"
#include "PNPShellCmd.hh"
#include "cxxopts.hpp"
#include "ipnp_api.hh"
#include "log/Log.hh"
#include "tcl/UserShell.hh"

using namespace idb;
using namespace ipnp;

int registerCommands()
{
  registerTclCmd(ipnp::CmdRunPnp, "run_pnp");
  registerTclCmd(ipnp::CmdAddVIA1, "add_via1");

  return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
  Log::init(argv, "/home/sujianrong/iEDA/src/operation/iPNP/example/result/log_info/");

  std::string hello_info
      = "\033[49;32m***************************\n"
        "    _ ____  _   ______  \n"
        "   (_) __ \\/ | / / __ \\ \n"
        "  / / /_/ /  |/ / /_/ / \n"
        " / / ____/ /|  / ____/  \n"
        "/_/_/   /_/ |_/_/       \n"
        "                       \n"
        "***************************\n"
        "WELCOME TO iPNP interface. \e[0m";

  std::cout << hello_info << std::endl;

  auto shell = ieda::UserShell::getShell();

  shell->set_init_func(registerCommands);

  cxxopts::Options options("iPNP", "iPNP command line help.");
  options.add_options()("c,config", "JSON configuration file", cxxopts::value<std::string>())(
      "o,output", "Output DEF file path", cxxopts::value<std::string>())("i,interactive", "Run in interactive TCL shell mode",
                                                                         cxxopts::value<bool>()->default_value("false"))(
      "s,script", "TCL script file to run", cxxopts::value<std::string>())("v,version", "Print version")("h,help", "Print help");

  try {
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    if (result.count("version")) {
      std::cout << "iPNP Version 1.0" << std::endl;
      return 0;
    }

    // Check if running in interactive mode
    bool interactive_mode = result["interactive"].as<bool>();

    // Check if running script file
    bool script_mode = result.count("script");
    std::string script_file;
    if (script_mode) {
      script_file = result["script"].as<std::string>();
      if (!std::filesystem::exists(script_file)) {
        LOG_ERROR << "Script file does not exist: " << script_file << std::endl;
        return 1;
      }
    }

    if (interactive_mode) {
      // Run interactive TCL shell
      shell->userMain(argc, argv);
    } else if (script_mode) {
      // Run TCL script file
      auto tcl_argc = argc - 1;
      auto tcl_argv = argv + 1;
      shell->userMain(tcl_argc, tcl_argv);
    } else {
      // Get configuration file path
      std::string config_file_path;
      if (result.count("config")) {
        config_file_path = result["config"].as<std::string>();
      } else {
        config_file_path = "../src/operation/iPNP/example/pnp_config.json";
        LOG_INFO << "Using default configuration file: " << config_file_path << std::endl;
      }

      // Check if configuration file exists
      if (!std::filesystem::exists(config_file_path)) {
        LOG_ERROR << "Configuration file does not exist: " << config_file_path << std::endl;
        return 1;
      }

      LOG_INFO << "Running iPNP with configuration: " << config_file_path;

      std::string start_info
          = "\033[49;32m"
            "    _ ____  _   ______     ______________    ____  ______\n"
            "   (_) __ \\/ | / / __ \\   / ___/_  __/   |  / __ \\/_  __/\n"
            "  / / /_/ /  |/ / /_/ /   \\__ \\ / / / /| | / /_/ / / /   \n"
            " / / ____/ /|  / ____/   ___/ // / / ___ |/ _, _/ / /    \n"
            "/_/_/   /_/ |_/_/       /____//_/ /_/  |_/_/ |_| /_/     \n"
            "                                                         \n"
            "\e[0m";

      std::cout << start_info << std::endl;

      ipnp::PNPApi::run_pnp(config_file_path);

      std::string finish_info
          = "\033[49;32m"
            "    _ ____  _   ______     ___________   ___________ __  __\n"
            "   (_) __ \\/ | / / __ \\   / ____/  _/ | / /  _/ ___// / / /\n"
            "  / / /_/ /  |/ / /_/ /  / /_   / //  |/ // / \\__ \\/ /_/ / \n"
            " / / ____/ /|  / ____/  / __/ _/ // /|  // / ___/ / __  /  \n"
            "/_/_/   /_/ |_/_/      /_/   /___/_/ |_/___//____/_/ /_/   \n"
            "                                                           \n"
            "\e[0m";

      std::cout << finish_info << std::endl;
    }

  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Error parsing options: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  Log::end();

  return 0;
}
