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
 * @file FastStaLibertyModel.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief CTS fast STA Liberty arc and lookup-table data.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace icts {

enum class FastStaTransition
{
  kRise,
  kFall
};

enum class FastStaLibertyTableKind
{
  kCellDelay,
  kOutputSlew,
  kInternalPower
};

enum class FastStaLibertyAxisKind
{
  kInputSlew,
  kOutputLoad,
  kUnknown
};

struct FastStaLibertyAxis
{
  FastStaLibertyAxisKind kind = FastStaLibertyAxisKind::kUnknown;
  std::vector<double> values;
};

struct FastStaLibertyTable
{
  FastStaLibertyTableKind kind = FastStaLibertyTableKind::kCellDelay;
  FastStaTransition transition = FastStaTransition::kRise;
  std::vector<FastStaLibertyAxis> axes;
  std::vector<double> values;

  [[nodiscard]] auto empty() const -> bool { return values.empty(); }
  [[nodiscard]] auto lookup(double input_slew_ns, double output_load_pf) const -> std::optional<double>;
};

struct FastStaLibertyArc
{
  std::string from_port;
  std::string to_port;
  bool negative_unate = false;
  std::vector<FastStaLibertyTable> delay_tables;
  std::vector<FastStaLibertyTable> slew_tables;
  std::vector<FastStaLibertyTable> internal_power_tables;
};

struct FastStaLibertyCell
{
  std::string cell_master;
  std::string input_port;
  std::string output_port;
  double input_cap_pf = 0.0;
  double output_cap_limit_pf = 0.0;
  double input_slew_limit_ns = 0.0;
  double input_threshold_rise = 0.5;
  double input_threshold_fall = 0.5;
  double output_threshold_rise = 0.5;
  double output_threshold_fall = 0.5;
  double slew_lower_threshold_rise = 0.3;
  double slew_lower_threshold_fall = 0.3;
  double slew_upper_threshold_rise = 0.7;
  double slew_upper_threshold_fall = 0.7;
  double slew_derate_from_library = 1.0;
  double area_um2 = 0.0;
  double voltage_v = 0.0;
  double leakage_power_w = 0.0;
  FastStaLibertyArc timing_arc;
};

}  // namespace icts
