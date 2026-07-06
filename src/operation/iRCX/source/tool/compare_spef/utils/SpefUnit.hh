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

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

namespace ircx {
namespace compare_spef {
namespace spef_unit {

inline auto uppercase(std::string text) -> std::string
{
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return text;
}

inline auto capacitanceScaleToFf(const std::string& unit_text) -> double
{
  double multiplier = 1.0;
  std::string unit;
  std::istringstream iss(unit_text);
  iss >> multiplier >> unit;
  unit = uppercase(unit);

  if (unit == "F") {
    return multiplier * 1.0e15;
  }
  if (unit == "NF") {
    return multiplier * 1.0e6;
  }
  if (unit == "PF") {
    return multiplier * 1.0e3;
  }
  if (unit == "FF") {
    return multiplier;
  }
  if (unit == "AF") {
    return multiplier * 1.0e-3;
  }
  return 1.0;
}

inline auto resistanceScaleToOhm(const std::string& unit_text) -> double
{
  double multiplier = 1.0;
  std::string unit;
  std::istringstream iss(unit_text);
  iss >> multiplier >> unit;
  unit = uppercase(unit);

  if (unit == "OHM" || unit == "OHMS") {
    return multiplier;
  }
  if (unit == "KOHM" || unit == "KOHMS") {
    return multiplier * 1.0e3;
  }
  if (unit == "MOHM" || unit == "MOHMS") {
    return multiplier * 1.0e6;
  }
  return 1.0;
}

}  // namespace spef_unit
}  // namespace compare_spef
}  // namespace ircx
