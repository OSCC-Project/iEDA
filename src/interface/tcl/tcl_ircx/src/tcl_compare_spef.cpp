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
 * @file tcl_compare_spef.cpp
 * @author Yipei Xu (yipeix@163.com)
 * @brief
 * @version 0.1
 * @date 2026-05-28
 */
#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "RCXAPI.hh"
#include "log/Log.hh"
#include "tcl_ircx.h"

namespace tcl {
namespace {

constexpr const char* kTestArg = "test";
constexpr const char* kReferenceArg = "reference";
constexpr const char* kCcapRelArg = "ccap_rel";

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

auto parseDouble(const char* text, double& value) -> bool
{
  if (text == nullptr || text[0] == '\0') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  value = std::strtod(text, &end);
  return errno == 0 && end != text && *end == '\0';
}

auto parseInt(const char* text, int& value) -> bool
{
  if (text == nullptr || text[0] == '\0') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }

  value = static_cast<int>(parsed);
  return true;
}

auto splitWords(const char* text) -> std::vector<std::string>
{
  std::vector<std::string> words;
  if (text == nullptr) {
    return words;
  }

  std::istringstream iss(text);
  std::string word;
  while (iss >> word) {
    words.push_back(std::move(word));
  }
  return words;
}

auto setDoubleOption(TclOption* option, const char* option_name, double& value) -> bool
{
  if (!isOptionSet(option)) {
    return true;
  }
  if (parseDouble(option->getStringVal(), value)) {
    return true;
  }

  LOG_ERROR << "compare_spef " << option_name << " requires a numeric value.";
  return false;
}

auto parseNamedDouble(const char* text, const char* option_name, double& value) -> bool
{
  if (parseDouble(text, value)) {
    return true;
  }

  LOG_ERROR << "compare_spef " << option_name << " requires a numeric value.";
  return false;
}

auto setIntOption(TclOption* option, const char* option_name, int& value) -> bool
{
  if (!isOptionSet(option)) {
    return true;
  }
  if (parseInt(option->getStringVal(), value)) {
    return true;
  }

  LOG_ERROR << "compare_spef " << option_name << " requires an integer value.";
  return false;
}

auto setStringOption(TclOption* option, std::string& value) -> void
{
  const char* option_value = getStringValue(option);
  if (option_value != nullptr) {
    value = option_value;
  }
}

auto setCcapOption(TclOption* ccap_option, TclOption* ccap_rel_arg, ircx::compare_spef::Config& config) -> bool
{
  const bool ccap_is_set = isOptionSet(ccap_option);
  const char* ccap_rel_value = getStringValue(ccap_rel_arg);
  if (!ccap_is_set) {
    if (ccap_rel_value != nullptr) {
      LOG_ERROR << "compare_spef unexpected argument: " << ccap_rel_value;
      return false;
    }
    return true;
  }

  const auto ccap_values = splitWords(ccap_option->getStringVal());
  if (ccap_values.size() == 2 && ccap_rel_value == nullptr) {
    return parseNamedDouble(ccap_values[0].c_str(), "-ccap", config.ccap_abs_threshold)
           && parseNamedDouble(ccap_values[1].c_str(), "-ccap", config.ccap_rel_threshold);
  }
  if (ccap_values.size() == 1 && ccap_rel_value != nullptr) {
    return parseNamedDouble(ccap_values[0].c_str(), "-ccap", config.ccap_abs_threshold)
           && parseNamedDouble(ccap_rel_value, "-ccap", config.ccap_rel_threshold);
  }

  LOG_ERROR << "compare_spef -ccap requires two numeric values.";
  return false;
}

}  // namespace

TclCompareSpef::TclCompareSpef(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(kTestArg, 1, nullptr));
  addOption(new TclStringOption(kReferenceArg, 1, nullptr));
  addOption(new TclStringOption(kCcapRelArg, 1, nullptr));

  addOption(new TclStringOption("-cores", 0, nullptr));
  addOption(new TclStringOption("-tcap", 0, nullptr));
  addOption(new TclStringOption("-ccap", 0, nullptr));
  addOption(new TclStringOption("-res", 0, nullptr));
  addOption(new TclStringOption("-corner", 0, nullptr));
  addOption(new TclStringOption("-match", 0, nullptr));
  addOption(new TclStringOption("-net", 0, nullptr));
  addOption(new TclStringOption("-from_pin", 0, nullptr));
  addOption(new TclStringOption("-to_pin", 0, nullptr));
  addOption(new TclStringOption("-net_config", 0, nullptr));
  addOption(new TclStringOption("-timeout", 0, nullptr));
  addOption(new TclStringOption("-delay", 0, nullptr));
  addOption(new TclStringOption("-output_dir", 0, nullptr));

  addOption(new TclSwitchOption("-r"));
  addOption(new TclSwitchOption("-c"));
  addOption(new TclSwitchOption("-d"));
  addOption(new TclSwitchOption("-delay_pin_load"));
}

unsigned TclCompareSpef::check()
{
  if (getStringValue(getOptionOrArg(kTestArg)) == nullptr || getStringValue(getOptionOrArg(kReferenceArg)) == nullptr) {
    LOG_ERROR << "compare_spef requires test and reference arguments.";
    return 0;
  }

  const bool ccap_is_set = isOptionSet(getOptionOrArg("-ccap"));
  if (!ccap_is_set && getStringValue(getOptionOrArg(kCcapRelArg)) != nullptr) {
    LOG_ERROR << "compare_spef unexpected argument: " << getOptionOrArg(kCcapRelArg)->getStringVal();
    return 0;
  }

  return 1;
}

unsigned TclCompareSpef::exec()
{
  if (!check()) {
    return 0;
  }

  ircx::compare_spef::Config config;
  config.test_file = getStringValue(getOptionOrArg(kTestArg));
  config.reference_file = getStringValue(getOptionOrArg(kReferenceArg));

  setStringOption(getOptionOrArg("-corner"), config.corner);
  setStringOption(getOptionOrArg("-match"), config.match_mode);
  setStringOption(getOptionOrArg("-net"), config.net_name);
  setStringOption(getOptionOrArg("-from_pin"), config.from_pin);
  setStringOption(getOptionOrArg("-to_pin"), config.to_pin);
  setStringOption(getOptionOrArg("-net_config"), config.net_config_file);
  setStringOption(getOptionOrArg("-output_dir"), config.output_dir);

  config.compare_resistance = isOptionSet(getOptionOrArg("-r"));
  config.compare_capacitance = isOptionSet(getOptionOrArg("-c"));
  config.compare_delay = isOptionSet(getOptionOrArg("-d"));
  config.delay_pin_load = isOptionSet(getOptionOrArg("-delay_pin_load"));

  if (!setIntOption(getOptionOrArg("-cores"), "-cores", config.cores)
      || !setDoubleOption(getOptionOrArg("-tcap"), "-tcap", config.tcap_threshold)
      || !setCcapOption(getOptionOrArg("-ccap"), getOptionOrArg(kCcapRelArg), config)
      || !setDoubleOption(getOptionOrArg("-res"), "-res", config.res_threshold)
      || !setIntOption(getOptionOrArg("-timeout"), "-timeout", config.timeout_seconds)
      || !setDoubleOption(getOptionOrArg("-delay"), "-delay", config.delay_threshold)) {
    return 0;
  }

  return RCX_API_INST.compare_spef(std::move(config)) ? 1U : 0U;
}

}  // namespace tcl
