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
 * @file CmdRunPnp.cc
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-30
 */

#include "PNPShellCmd.hh"
#include "log/Log.hh"
#include "iPNP.hh"
#include "iPNPApi.hh"
#include <filesystem>
#include <iostream>

namespace ipnp {

  CmdRunPnp::CmdRunPnp(const char* cmd_name) : TclCmd(cmd_name) {

    auto* config_option = new TclStringOption("-config", 0, "");
    addOption(config_option);

  }

  unsigned CmdRunPnp::check() {

    TclOption* config_option = getOptionOrArg("-config");

    if (config_option) {
      auto* config_file = config_option->getStringVal();
      if (!std::filesystem::exists(config_file)) {
        LOG_ERROR << "Configuration file not found: " << config_file;
        return 0;
      }
    }
    return 1; // check success
  }

  unsigned CmdRunPnp::exec() {
    if (!check()) {
      return 0;
    }

    TclOption* config_option = getOptionOrArg("-config");

    if (config_option) {
      auto* config_file = config_option->getStringVal();

      LOG_INFO << "Running iPNP with configuration: " << config_file;

      std::string start_info =
        "\033[49;32m"
        "    _ ____  _   ______     ______________    ____  ______\n"
        "   (_) __ \\/ | / / __ \\   / ___/_  __/   |  / __ \\/_  __/\n"
        "  / / /_/ /  |/ / /_/ /   \\__ \\ / / / /| | / /_/ / / /   \n"
        " / / ____/ /|  / ____/   ___/ // / / ___ |/ _, _/ / /    \n"
        "/_/_/   /_/ |_/_/       /____//_/ /_/  |_/_/ |_| /_/     \n"
        "                                                         \n"
        "\e[0m";

      std::cout << start_info << std::endl;

      // 直接调用静态方法
      ipnp::PNPApi::run_pnp(config_file);

      std::string finish_info =
        "\033[49;32m"
        "    _ ____  _   ______     ___________   ___________ __  __\n"
        "   (_) __ \\/ | / / __ \\   / ____/  _/ | / /  _/ ___// / / /\n"
        "  / / /_/ /  |/ / /_/ /  / /_   / //  |/ // / \\__ \\/ /_/ / \n"
        " / / ____/ /|  / ____/  / __/ _/ // /|  // / ___/ / __  /  \n"
        "/_/_/   /_/ |_/_/      /_/   /___/_/ |_/___//____/_/ /_/   \n"
        "                                                           \n"
        "\e[0m";

      std::cout << finish_info << std::endl;
    } else {
      LOG_ERROR << "Configuration file is required for running iPNP.";
      return 0;
    }

    return 1;
  }

} // namespace ipnp 