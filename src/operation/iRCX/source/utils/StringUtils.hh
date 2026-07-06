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

#include <cerrno>
#include <charconv>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include "Types.hh"
#include "log/Log.hh"

namespace ircx {
namespace string {

inline auto trim(std::string_view value) -> Str
{
  const auto first = value.find_first_not_of(" \t\n\r\f\v");
  if (first == std::string_view::npos) {
    return "";
  }

  const auto last = value.find_last_not_of(" \t\n\r\f\v");
  return Str(value.substr(first, last - first + 1));
}

inline auto trim_view(std::string_view value) -> std::string_view
{
  const auto first = value.find_first_not_of(" \t\n\r\f\v");
  if (first == std::string_view::npos) {
    return {};
  }

  const auto last = value.find_last_not_of(" \t\n\r\f\v");
  return value.substr(first, last - first + 1);
}

inline auto starts_with(std::string_view value, std::string_view prefix) -> bool
{
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

inline auto take_token(std::string_view& value) -> std::string_view
{
  value = trim_view(value);
  if (value.empty()) {
    return {};
  }

  const auto end = value.find_first_of(" \t\n\r\f\v");
  if (end == std::string_view::npos) {
    const auto token = value;
    value = {};
    return token;
  }

  const auto token = value.substr(0, end);
  value = value.substr(end + 1);
  return token;
}

inline auto contains(std::string_view value, std::string_view pattern) -> bool
{
  return value.find(pattern) != std::string_view::npos;
}

inline auto to_lower(Str value) -> Str
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

inline auto identifier(std::string_view value, std::string_view fallback = "id") -> Str
{
  Str text;
  text.reserve(value.size());
  for (char ch : value) {
    if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '$') {
      text.push_back(ch);
    } else {
      text.push_back('_');
    }
  }
  if (text.empty()) {
    text = fallback.empty() ? "id" : Str(fallback);
  }
  if (std::isdigit(static_cast<unsigned char>(text.front()))) {
    text.insert(text.begin(), 'n');
  }
  return text;
}

inline auto after_prefix(std::string_view value, std::string_view prefix) -> std::optional<std::string_view>
{
  if (!starts_with(value, prefix)) {
    return std::nullopt;
  }
  return value.substr(prefix.size());
}

template <typename T>
inline auto parse_number(std::string_view value) -> std::optional<T>
{
  value = trim_view(value);
  if (value.empty()) {
    return std::nullopt;
  }

  if constexpr (std::is_integral_v<T>) {
    T number{};
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto [ptr, error] = std::from_chars(begin, end, number);
    if (error != std::errc{} || ptr != end) {
      return std::nullopt;
    }
    return number;
  } else if constexpr (std::is_floating_point_v<T>) {
    const Str number_text(value);
    char* parse_end = nullptr;
    errno = 0;

    const auto number = std::strtod(number_text.c_str(), &parse_end);
    if (errno == ERANGE || parse_end != number_text.c_str() + number_text.size()) {
      return std::nullopt;
    }
    return static_cast<T>(number);
  } else {
    static_assert(std::is_arithmetic_v<T>, "parse_number only supports arithmetic types");
    return std::nullopt;
  }
}

template <typename T>
inline auto parse_after_prefix(std::string_view value, std::string_view prefix) -> std::optional<T>
{
  const auto token_value = after_prefix(value, prefix);
  if (!token_value.has_value()) {
    return std::nullopt;
  }
  return parse_number<T>(*token_value);
}

inline auto parse_int_after_prefix(std::string_view value, std::string_view prefix) -> std::optional<int>
{
  return parse_after_prefix<int>(value, prefix);
}

inline auto parse_double_after_prefix(std::string_view value, std::string_view prefix) -> std::optional<double>
{
  return parse_after_prefix<double>(value, prefix);
}

template <typename T = int>
inline auto parse_prefixed_index(std::string_view value, char prefix = '*') -> std::optional<T>
{
  static_assert(std::is_integral_v<T>, "parse_prefixed_index requires an integral result type");
  if (value.size() < 2 || value.front() != prefix || !std::isdigit(static_cast<unsigned char>(value[1]))) {
    return std::nullopt;
  }
  return parse_number<T>(value.substr(1));
}

inline auto require_non_empty(std::string_view value, std::string_view field_name) -> bool
{
  if (!value.empty()) {
    return true;
  }

  LOG_ERROR << "RCX field is empty: " << field_name;
  return false;
}

inline auto escape_spef_name(Str name) -> Str
{
  if (name.find('.') == Str::npos) {
    return name;
  }

  Str escaped_name;
  escaped_name.reserve(name.size());
  for (Size idx = 0; idx < name.size(); ++idx) {
    const char current_char = name[idx];
    const bool needs_escape =
        current_char == '.' || current_char == '[' || current_char == ']';
    if (needs_escape && (idx == 0 || name[idx - 1] != '\\')) {
      escaped_name.push_back('\\');
    }
    escaped_name.push_back(current_char);
  }

  return escaped_name;
}

}  // namespace string
}  // namespace ircx
