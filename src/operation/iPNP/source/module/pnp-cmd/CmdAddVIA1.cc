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
 * @file CmdAddVIA1.cc
 * @author Jianrong Su
 * @brief Command to add VIA1 connections between M2 and M1 layers
 * @version 1.0
 * @date 2025-06-30
 */

#include "PNPShellCmd.hh"
#include "log/Log.hh"
#include "iPNP.hh"
#include "iPNPApi.hh"
#include <filesystem>

namespace ipnp {

  CmdAddVIA1::CmdAddVIA1(const char* cmd_name) : TclCmd(cmd_name) {
    auto* config_option = new TclStringOption("-config", 0, "");
    addOption(config_option);
  }

  unsigned CmdAddVIA1::check() {
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

  unsigned CmdAddVIA1::exec() {
    if (!check()) {
      return 0;
    }

    TclOption* config_option = getOptionOrArg("-config");
    
    if (config_option) {
      auto* config_file = config_option->getStringVal();

      if (iPNPApi::getInstance()) {
        LOG_ERROR << "An existing iPNP instance was found. It will be replaced.";
        delete iPNPApi::getInstance();
        iPNPApi::setInstance(nullptr);
      }

      LOG_INFO << "Initializing iPNP with configuration: " << config_file;
      auto* new_instance = new ipnp::iPNP(config_file);
      iPNPApi::setInstance(new_instance);
    }

    auto* ipnp = iPNPApi::getInstance();

    LOG_INFO << "Adding VIA1 connections between M2 and M1 layers...";

    ipnp->connect_M2_M1();

    std::string output_def_path;
    if (ipnp->get_config() && !ipnp->get_config()->get_output_def_path().empty()) {
      output_def_path = ipnp->get_config()->get_output_def_path();
      LOG_INFO << "DEF output path read from configuration file: " << output_def_path;
    } else {
      output_def_path = "./output.def";
      LOG_ERROR << "DEF output path not found in configuration file, using default path.";
    }

    ipnp->writeIdbToDef(output_def_path);

    LOG_INFO << "VIA1 connections added successfully.";

    return 1;
  }

} // namespace ipnp