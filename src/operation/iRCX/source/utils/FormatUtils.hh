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
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace ircx {
namespace format {

inline auto fixed(double value, int precision = 3, std::string_view non_finite = "NA") -> std::string
{
  if (!std::isfinite(value)) {
    return std::string{non_finite};
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

inline auto percent(double value, int precision = 3, std::string_view non_finite = "NA") -> std::string
{
  if (!std::isfinite(value)) {
    return std::string{non_finite};
  }
  return fixed(value, precision, non_finite) + "%";
}

inline auto significant(double value, int digits = 3) -> std::string
{
  if (!std::isfinite(value)) {
    return "NA";
  }
  std::ostringstream oss;
  oss << std::setprecision(digits) << value;
  return oss.str();
}

inline auto unit_symbol(std::string_view unit_name) -> std::string_view
{
  const auto space = unit_name.rfind(' ');
  if (space == std::string_view::npos || space + 1 >= unit_name.size()) {
    return unit_name;
  }
  return unit_name.substr(space + 1);
}

inline auto with_unit(double value, std::string_view unit_name, int digits = 3) -> std::string
{
  std::string result = significant(value, digits);
  if (!unit_name.empty()) {
    result += " ";
    result += unit_name;
  }
  return result;
}

inline auto escape_xml(std::string_view text) -> std::string
{
  std::string escaped;
  escaped.reserve(text.size());
  for (char ch : text) {
    switch (ch) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

}  // namespace format
}  // namespace ircx
