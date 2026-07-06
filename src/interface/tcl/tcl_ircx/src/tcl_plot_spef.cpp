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
 * @file tcl_plot_spef.cpp
 * @author Yipei Xu (yipeix@163.com)
 * @brief
 * @version 0.1
 * @date 2026-06-01
 */
#include <cstdlib>
#include <limits>
#include <utility>

#include "RCXAPI.hh"
#include "log/Log.hh"
#include "tcl_ircx.h"

namespace tcl {
namespace {

constexpr const char* kSpefArg = "spef";
constexpr const char* kOutputDirArg = "output_dir";

auto getStringValue(TclOption* option) -> const char*
{
  if (option == nullptr || !option->is_set_val()) {
    return nullptr;
  }
  return option->getStringVal();
}

auto isOptionSet(TclOption* option) -> bool
{
  return option != nullptr && option->is_set_val();
}

auto parseInt(const char* text, int& value) -> bool
{
  if (text == nullptr || *text == '\0') {
    return false;
  }

  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < static_cast<long>(std::numeric_limits<int>::min())
      || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }

  value = static_cast<int>(parsed);
  return true;
}

auto setIntOption(TclOption* option, const char* option_name, int& value) -> bool
{
  if (!isOptionSet(option)) {
    return true;
  }
  if (parseInt(option->getStringVal(), value)) {
    return true;
  }

  LOG_ERROR << "plot_spef " << option_name << " requires an integer value.";
  return false;
}

}  // namespace

TclPlotSpef::TclPlotSpef(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(kSpefArg, 1, nullptr));
  addOption(new TclStringOption(kOutputDirArg, 1, nullptr));
  addOption(new TclStringOption("-net", 1, nullptr));
  addOption(new TclStringOption("-cores", 1, nullptr));
  addOption(new TclSwitchOption("-R"));
  addOption(new TclSwitchOption("-Cc"));
  addOption(new TclSwitchOption("-Cg"));
}

unsigned TclPlotSpef::check()
{
  if (getStringValue(getOptionOrArg(kSpefArg)) == nullptr || getStringValue(getOptionOrArg(kOutputDirArg)) == nullptr) {
    LOG_ERROR << "plot_spef requires spef and output directory arguments.";
    return 0;
  }
  return 1;
}

unsigned TclPlotSpef::exec()
{
  if (!check()) {
    return 0;
  }

  ircx::plot_spef::Config config;
  config.spef_file = getStringValue(getOptionOrArg(kSpefArg));
  config.output_dir = getStringValue(getOptionOrArg(kOutputDirArg));
  if (const char* net_name = getStringValue(getOptionOrArg("-net")); net_name != nullptr) {
    config.net_name = net_name;
  }
  if (!setIntOption(getOptionOrArg("-cores"), "-cores", config.cores)) {
    return 0;
  }
  config.output_resistance = isOptionSet(getOptionOrArg("-R"));
  config.output_coupling_cap = isOptionSet(getOptionOrArg("-Cc"));
  config.output_ground_cap = isOptionSet(getOptionOrArg("-Cg"));

  return RCX_API_INST.plot_spef(std::move(config)) ? 1U : 0U;
}

}  // namespace tcl
