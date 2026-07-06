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
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ircx {
namespace compare_spef {

struct Config
{
  std::string test_file;
  std::string reference_file;
  std::string output_dir = ".";

  int cores = 1;
  double tcap_threshold = 3.0;
  double ccap_abs_threshold = 0.3;
  double ccap_rel_threshold = 0.1;
  double res_threshold = 50.0;

  bool compare_capacitance = false;
  bool compare_resistance = false;
  bool compare_delay = false;
  bool delay_pin_load = false;

  std::string corner;
  std::string match_mode = "name";
  std::string net_name;
  std::string from_pin;
  std::string to_pin;
  std::string net_config_file;
  int timeout_seconds = 5400;
  double delay_threshold = 1.0;

  std::vector<std::string> net_names;
  std::vector<std::string> from_pins;
  std::vector<std::string> to_pins;
  std::vector<std::pair<std::string, std::string>> from_to_pins;
};

class NetConfigReader
{
 public:
  auto read(Config& config) const -> bool;

 private:
  void addLine(Config& config, std::string_view raw_line) const;
};

class ConfigValidator
{
 public:
  auto validate(const Config& config) const -> bool;
};

}  // namespace compare_spef
}  // namespace ircx
