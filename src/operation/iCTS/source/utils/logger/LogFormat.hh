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
 * @file LogFormat.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-16
 * @brief Lightweight title + ASCII-table formatter for iCTS logs.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace icts::logformat {

using TableRow = std::vector<std::string>;
using TableRows = std::vector<TableRow>;

namespace detail {

inline auto PadRight(const std::string& value, std::size_t width) -> std::string
{
  if (value.size() >= width) {
    return value;
  }
  return value + std::string(width - value.size(), ' ');
}

inline auto BuildBorder(const std::vector<std::size_t>& widths) -> std::string
{
  std::string border = "+";
  for (const std::size_t width : widths) {
    border += std::string(width + 2U, '-');
    border += "+";
  }
  return border;
}

inline auto BuildRowLine(const TableRow& row, const std::vector<std::size_t>& widths) -> std::string
{
  std::ostringstream oss;
  oss << "|";
  for (std::size_t i = 0; i < widths.size(); ++i) {
    const std::string cell = i < row.size() ? row[i] : std::string{};
    oss << " " << PadRight(cell, widths[i]) << " |";
  }
  return oss.str();
}

}  // namespace detail

inline auto MakeTitle(const std::string& title, char fill = '=', std::size_t width = 72U) -> std::string
{
  if (title.empty()) {
    return std::string(width, fill);
  }

  std::string centered = " " + title + " ";
  if (centered.size() >= width) {
    return centered;
  }

  const std::size_t left = (width - centered.size()) / 2U;
  const std::size_t right = width - centered.size() - left;
  return std::string(left, fill) + centered + std::string(right, fill);
}

inline auto FormatFixed(double value, int precision = 4) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ios::fixed, std::ios::floatfield);
  stream << std::setprecision(precision) << value;
  return stream.str();
}

inline auto FormatScientific(double value, int precision = 3) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ios::scientific, std::ios::floatfield);
  stream << std::setprecision(precision) << value;
  return stream.str();
}

inline auto FormatFixedOrScientific(double value, int fixed_precision = 4, int scientific_precision = 3, double scientific_lower = 1e-3,
                                    double scientific_upper = 1e4) -> std::string
{
  const double abs_value = std::abs(value);
  const bool use_fixed = abs_value == 0.0 || (abs_value >= scientific_lower && abs_value < scientific_upper);
  return use_fixed ? FormatFixed(value, fixed_precision) : FormatScientific(value, scientific_precision);
}

inline auto FormatWithUnit(double value, std::string_view unit, int fixed_precision = 4, int scientific_precision = 3,
                           double scientific_lower = 1e-3, double scientific_upper = 1e4) -> std::string
{
  std::string formatted_value = FormatFixedOrScientific(value, fixed_precision, scientific_precision, scientific_lower, scientific_upper);
  if (unit.empty()) {
    return formatted_value;
  }
  return formatted_value + " " + std::string(unit);
}

inline auto FormatBool(bool value) -> std::string
{
  return value ? "true" : "false";
}

inline auto FormatPercent(double ratio, int precision = 2) -> std::string
{
  return FormatFixed(ratio * 100.0, precision) + " %";
}

inline auto JoinStrings(const std::vector<std::string>& values, std::string_view separator = ", ") -> std::string
{
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) {
      stream << separator;
    }
    stream << values.at(index);
  }
  return stream.str();
}

inline auto JoinUnsigned(const std::vector<unsigned>& values, std::string_view separator = ", ") -> std::string
{
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) {
      stream << separator;
    }
    stream << values.at(index);
  }
  return stream.str();
}

inline auto FormatEngineering(double value, std::string_view unit, int precision = 3) -> std::string
{
  struct Prefix
  {
    int exponent;
    std::string_view prefix;
  };

  constexpr std::array<Prefix, 9> prefixes = {{
      {12, "T"},
      {9, "G"},
      {6, "M"},
      {3, "k"},
      {0, ""},
      {-3, "m"},
      {-6, "u"},
      {-9, "n"},
      {-12, "p"},
  }};

  if (value == 0.0) {
    return FormatFixed(0.0, precision) + (unit.empty() ? "" : " " + std::string(unit));
  }

  const double abs_value = std::abs(value);
  if (abs_value < std::pow(10.0, static_cast<double>(prefixes.back().exponent))
      || abs_value >= std::pow(10.0, static_cast<double>(prefixes.front().exponent + 3))) {
    return FormatWithUnit(value, unit, precision, precision);
  }

  Prefix chosen = prefixes.back();
  for (const auto& prefix : prefixes) {
    if (abs_value >= std::pow(10.0, static_cast<double>(prefix.exponent))) {
      chosen = prefix;
      break;
    }
  }

  const double scaled_value = value / std::pow(10.0, static_cast<double>(chosen.exponent));
  const std::string engineering_unit = std::string(chosen.prefix) + std::string(unit);
  return FormatFixed(scaled_value, precision) + (engineering_unit.empty() ? "" : " " + engineering_unit);
}

inline auto FormatPowerW(double power_w, int precision = 3) -> std::string
{
  return FormatEngineering(power_w, "W", precision);
}

inline auto MakeTable(const std::vector<std::string>& headers, const TableRows& rows) -> std::string
{
  std::size_t column_count = headers.size();
  for (const auto& row : rows) {
    column_count = std::max(column_count, row.size());
  }
  if (column_count == 0U) {
    return "";
  }

  std::vector<std::size_t> widths(column_count, 0U);
  for (std::size_t i = 0; i < headers.size(); ++i) {
    widths[i] = std::max(widths[i], headers[i].size());
  }
  for (const auto& row : rows) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      widths[i] = std::max(widths[i], row[i].size());
    }
  }

  const std::string border = detail::BuildBorder(widths);
  std::ostringstream oss;
  oss << border << '\n';
  if (!headers.empty()) {
    oss << detail::BuildRowLine(headers, widths) << '\n';
    oss << border << '\n';
  }
  for (const auto& row : rows) {
    oss << detail::BuildRowLine(row, widths) << '\n';
  }
  oss << border;
  return oss.str();
}

inline auto MakeTitledTable(const std::string& title, const std::vector<std::string>& headers, const TableRows& rows, char fill = '=')
    -> std::string
{
  const std::string table = MakeTable(headers, rows);
  if (table.empty()) {
    return MakeTitle(title, fill);
  }
  return MakeTitle(title, fill) + '\n' + table;
}

inline auto MakeKeyValueTable(const std::string& title, const std::vector<std::pair<std::string, std::string>>& fields, char fill = '=')
    -> std::string
{
  TableRows rows;
  rows.reserve(fields.size());
  for (const auto& [key, value] : fields) {
    rows.push_back({key, value});
  }
  return MakeTitledTable(title, {"Field", "Value"}, rows, fill);
}

inline auto MakeStageMarker(std::string_view module, std::string_view stage, std::string_view state) -> std::string
{
  std::ostringstream oss;
  oss << "[" << module << "][" << state << "] " << stage;
  return oss.str();
}

}  // namespace icts::logformat
