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

namespace ircx::plot_spef {

struct Config
{
  std::string spef_file;
  std::string output_dir;
  std::string net_name;
  int dbu = 1000;
  int cores = 1;
  bool output_resistance = false;
  bool output_coupling_cap = false;
  bool output_ground_cap = false;

  auto hasNetFilter() const -> bool { return !net_name.empty(); }
  auto hasOutputFilter() const -> bool { return output_resistance || output_coupling_cap || output_ground_cap; }
  auto plotResistance() const -> bool { return !hasOutputFilter() || output_resistance; }
  auto plotCouplingCap() const -> bool { return !hasOutputFilter() || output_coupling_cap; }
  auto plotGroundCap() const -> bool { return !hasOutputFilter() || output_ground_cap; }
};

class ConfigValidator
{
 public:
  auto validate(const Config& config) const -> bool;
};

}  // namespace ircx::plot_spef
