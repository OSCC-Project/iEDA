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
#include "config/CompareSpefConfig.hh"

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "StringUtils.hh"
#include "log/Log.hh"

namespace ircx {
namespace compare_spef {

auto NetConfigReader::read(Config& config) const -> bool
{
  if (config.net_config_file.empty()) {
    return true;
  }

  std::ifstream ifs(config.net_config_file);
  if (!ifs.is_open()) {
    LOG_ERROR << "compare_spef failed: cannot open -net_config file " << config.net_config_file;
    return false;
  }

  std::string raw_line;
  while (std::getline(ifs, raw_line)) {
    addLine(config, raw_line);
  }
  return true;
}

void NetConfigReader::addLine(Config& config, std::string_view raw_line) const
{
  constexpr std::string_view net_prefix = "NET:";
  constexpr std::string_view from_prefix = "FROM_PIN:";
  constexpr std::string_view to_prefix = "TO_PIN:";
  constexpr std::string_view from_to_prefix = "FROM_TO_PINS:";

  const Str line = string::trim(raw_line);
  if (line.empty() || string::starts_with(line, "//") || string::starts_with(line, "**")) {
    return;
  }

  if (string::starts_with(line, net_prefix)) {
    const Str value = string::trim(line.substr(net_prefix.size()));
    if (!value.empty()) {
      config.net_names.emplace_back(value);
    }
    return;
  }
  if (string::starts_with(line, from_prefix)) {
    const Str value = string::trim(line.substr(from_prefix.size()));
    if (!value.empty()) {
      config.from_pins.emplace_back(value);
    }
    return;
  }
  if (string::starts_with(line, to_prefix)) {
    const Str value = string::trim(line.substr(to_prefix.size()));
    if (!value.empty()) {
      config.to_pins.emplace_back(value);
    }
    return;
  }
  if (string::starts_with(line, from_to_prefix)) {
    std::istringstream iss(string::trim(line.substr(from_to_prefix.size())));
    std::string from_pin;
    std::string to_pin;
    iss >> from_pin >> to_pin;
    if (!from_pin.empty() && !to_pin.empty()) {
      config.from_to_pins.emplace_back(std::move(from_pin), std::move(to_pin));
    }
  }
}

auto ConfigValidator::validate(const Config& config) const -> bool
{
  if (config.test_file.empty() || config.reference_file.empty()) {
    LOG_ERROR << "compare_spef requires test and reference SPEF files.";
    return false;
  }
  if (config.match_mode != "name") {
    LOG_ERROR << "compare_spef currently supports only -match name for SPEF comparison.";
    return false;
  }
  if (!config.corner.empty()) {
    LOG_ERROR << "compare_spef -corner is valid only for GPD comparison, which is not supported yet.";
    return false;
  }
  if (config.compare_delay) {
    LOG_ERROR << "compare_spef -d/-delay Elmore delay comparison is not implemented yet.";
    return false;
  }
  if (!config.net_name.empty() && (!config.from_pin.empty() || !config.to_pin.empty())) {
    LOG_ERROR << "compare_spef -net cannot be used with -from_pin or -to_pin.";
    return false;
  }
  if (config.tcap_threshold < 0.0 || config.ccap_abs_threshold < 0.0 || config.ccap_rel_threshold < 0.0 || config.res_threshold < 0.0) {
    LOG_ERROR << "compare_spef thresholds must be non-negative.";
    return false;
  }
  if (config.cores <= 0) {
    LOG_ERROR << "compare_spef -cores must be a positive integer.";
    return false;
  }

  return true;
}

}  // namespace compare_spef
}  // namespace ircx
