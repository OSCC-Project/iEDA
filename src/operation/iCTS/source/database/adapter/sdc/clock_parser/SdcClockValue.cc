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
 * @file SdcClockValue.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief SDC clock value and expression helper implementation.
 */

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "SdcClockParser.hh"
#include "SdcClockReader.hh"

namespace icts::sdc_reader {

auto Trim(const std::string& text) -> std::string
{
  const auto first = std::ranges::find_if_not(text, [](unsigned char ch) -> bool { return std::isspace(ch) != 0; });
  if (first == text.end()) {
    return {};
  }
  const auto last
      = std::ranges::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) -> bool { return std::isspace(ch) != 0; }).base();
  return std::string(first, last);
}

auto IsOption(const std::string& text) -> bool
{
  return text.size() > 1U && text.front() == '-';
}

auto JoinStrings(const std::vector<std::string>& values) -> std::string
{
  std::string result;
  for (const auto& value : values) {
    if (!result.empty()) {
      result += ' ';
    }
    result += value;
  }
  return result;
}

auto SplitListText(const std::string& text) -> std::vector<std::string>
{
  std::vector<std::string> values;
  std::istringstream stream(text);
  std::string value;
  while (stream >> value) {
    values.push_back(value);
  }
  if (values.empty() && !text.empty()) {
    values.push_back(text);
  }
  return values;
}

namespace {

auto ToLower(std::string text) -> std::string
{
  std::ranges::transform(text, text.begin(), [](unsigned char ch) -> char { return static_cast<char>(std::tolower(ch)); });
  return text;
}

auto HasOnlyTrailingSpaces(std::istringstream& stream) -> bool
{
  if (stream.eof()) {
    return true;
  }
  stream >> std::ws;
  return stream.eof();
}

}  // namespace

auto ValueToString(const SdcValue& value) -> std::string
{
  if (!value.objects.empty()) {
    std::vector<std::string> patterns;
    patterns.reserve(value.objects.size());
    std::ranges::transform(value.objects, std::back_inserter(patterns),
                           [](const SdcObjectRef& object) -> std::string { return object.pattern; });
    return JoinStrings(patterns);
  }
  return JoinStrings(value.strings);
}

auto MakeStringValue(std::string text) -> SdcValue
{
  SdcValue value;
  value.strings.push_back(std::move(text));
  return value;
}

auto MakeObjectValue(SdcObjectKind kind, const std::vector<std::string>& patterns) -> SdcValue
{
  SdcValue value;
  value.objects.reserve(patterns.size());
  for (const auto& pattern : patterns) {
    if (pattern.empty()) {
      continue;
    }
    value.objects.emplace_back(SdcObjectRef{kind, pattern, true});
  }
  return value;
}

auto AppendRefsFromValue(std::vector<SdcObjectRef>& refs, const SdcValue& value, SdcObjectKind default_kind) -> void
{
  if (!value.objects.empty()) {
    refs.insert(refs.end(), value.objects.begin(), value.objects.end());
    return;
  }
  for (const auto& text : value.strings) {
    for (const auto& item : SplitListText(text)) {
      refs.emplace_back(SdcObjectRef{default_kind, item, false});
    }
  }
}

auto ParseDoubleValue(const std::string& text, double& value) -> bool
{
  const auto clean_text = Trim(text);
  if (clean_text.empty()) {
    return false;
  }
  std::istringstream stream(clean_text);
  stream >> value;
  return !stream.fail() && HasOnlyTrailingSpaces(stream);
}

auto ParseIntValue(const std::string& text, int& value) -> bool
{
  const auto clean_text = Trim(text);
  if (clean_text.empty()) {
    return false;
  }
  long parsed = 0;
  std::istringstream stream(clean_text);
  stream >> parsed;
  if (stream.fail() || !HasOnlyTrailingSpaces(stream)) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

auto TimeUnitToNs(const std::string& unit) -> double
{
  const auto normalized = ToLower(Trim(unit));
  const std::map<std::string, double> scale_by_unit = {
      {"fs", 0.000001}, {"ps", 0.001}, {"ns", 1.0}, {"us", 1000.0}, {"ms", 1000000.0}, {"s", 1000000000.0},
  };
  if (const auto iter = scale_by_unit.find(normalized); iter != scale_by_unit.end()) {
    return iter->second;
  }

  std::size_t suffix_pos = normalized.size();
  while (suffix_pos > 0U && std::isalpha(static_cast<unsigned char>(normalized[suffix_pos - 1U])) != 0) {
    --suffix_pos;
  }
  if (suffix_pos == normalized.size() || suffix_pos == 0U) {
    return 1.0;
  }
  const auto suffix = normalized.substr(suffix_pos);
  const auto suffix_iter = scale_by_unit.find(suffix);
  if (suffix_iter == scale_by_unit.end()) {
    return 1.0;
  }
  double numeric_scale = 0.0;
  if (!ParseDoubleValue(normalized.substr(0U, suffix_pos), numeric_scale)) {
    return 1.0;
  }
  return numeric_scale * suffix_iter->second;
}

ArithmeticParser::ArithmeticParser(std::string expression) : _expression(std::move(expression))
{
}

auto ArithmeticParser::parse(double& value) -> bool
{
  _pos = 0U;
  std::vector<double> values;
  std::vector<char> operators;
  bool expect_value = true;

  while (true) {
    skipSpaces();
    if (_pos >= _expression.size()) {
      break;
    }
    if (matchWord("double")) {
      _pos += 6U;
      continue;
    }

    const char token = _expression[_pos];
    if (token == '(') {
      operators.push_back(token);
      ++_pos;
      expect_value = true;
      continue;
    }
    if (token == ')') {
      if (!applyUntilOpen(values, operators)) {
        return false;
      }
      ++_pos;
      expect_value = false;
      continue;
    }
    if (isOperator(token)) {
      if (expect_value) {
        if (token == '+') {
          ++_pos;
          continue;
        }
        if (token == '-') {
          ++_pos;
          skipSpaces();
          if (_pos < _expression.size() && _expression[_pos] == '(') {
            values.push_back(0.0);
            while (!operators.empty() && operators.back() != '(' && precedence(operators.back()) >= precedence('-')) {
              if (!applyTopOperator(values, operators)) {
                return false;
              }
            }
            operators.push_back('-');
            continue;
          }
          double number = 0.0;
          if (!parseNumber(number)) {
            return false;
          }
          values.push_back(-number);
          expect_value = false;
          continue;
        }
        return false;
      }
      while (!operators.empty() && operators.back() != '(' && precedence(operators.back()) >= precedence(token)) {
        if (!applyTopOperator(values, operators)) {
          return false;
        }
      }
      operators.push_back(token);
      ++_pos;
      expect_value = true;
      continue;
    }

    double number = 0.0;
    if (!parseNumber(number)) {
      return false;
    }
    values.push_back(number);
    expect_value = false;
  }

  if (expect_value || values.empty()) {
    return false;
  }
  while (!operators.empty()) {
    if (operators.back() == '(' || !applyTopOperator(values, operators)) {
      return false;
    }
  }
  if (values.size() != 1U) {
    return false;
  }
  value = values.back();
  return true;
}

auto ArithmeticParser::isOperator(char token) -> bool
{
  return token == '+' || token == '-' || token == '*' || token == '/';
}

auto ArithmeticParser::precedence(char token) -> int
{
  if (token == '*' || token == '/') {
    return 2;
  }
  return token == '+' || token == '-' ? 1 : 0;
}

auto ArithmeticParser::applyTopOperator(std::vector<double>& values, std::vector<char>& operators) -> bool
{
  if (values.size() < 2U || operators.empty()) {
    return false;
  }
  const auto op = operators.back();
  operators.pop_back();
  const auto rhs = values.back();
  values.pop_back();
  const auto lhs = values.back();
  values.pop_back();

  if (op == '+') {
    values.push_back(lhs + rhs);
  } else if (op == '-') {
    values.push_back(lhs - rhs);
  } else if (op == '*') {
    values.push_back(lhs * rhs);
  } else if (op == '/') {
    if (rhs == 0.0) {
      return false;
    }
    values.push_back(lhs / rhs);
  } else {
    return false;
  }
  return true;
}

auto ArithmeticParser::applyUntilOpen(std::vector<double>& values, std::vector<char>& operators) -> bool
{
  while (!operators.empty() && operators.back() != '(') {
    if (!applyTopOperator(values, operators)) {
      return false;
    }
  }
  if (operators.empty() || operators.back() != '(') {
    return false;
  }
  operators.pop_back();
  return true;
}

auto ArithmeticParser::parseNumber(double& value) -> bool
{
  skipSpaces();
  const auto start = _pos;
  bool has_digit = false;
  while (_pos < _expression.size()) {
    const auto ch = _expression[_pos];
    if (std::isdigit(static_cast<unsigned char>(ch)) != 0 || ch == '.') {
      has_digit = true;
      ++_pos;
      continue;
    }
    if ((ch == 'e' || ch == 'E') && _pos + 1U < _expression.size()) {
      ++_pos;
      if (_expression[_pos] == '+' || _expression[_pos] == '-') {
        ++_pos;
      }
      continue;
    }
    break;
  }
  if (!has_digit) {
    return false;
  }
  return ParseDoubleValue(_expression.substr(start, _pos - start), value);
}

auto ArithmeticParser::skipSpaces() -> void
{
  while (_pos < _expression.size() && std::isspace(static_cast<unsigned char>(_expression[_pos])) != 0) {
    ++_pos;
  }
}

auto ArithmeticParser::matchWord(std::string_view word) const -> bool
{
  if (_expression.size() - _pos < word.size()) {
    return false;
  }
  if (_expression.compare(_pos, word.size(), word) != 0) {
    return false;
  }
  const auto next_pos = _pos + word.size();
  return next_pos == _expression.size() || std::isalnum(static_cast<unsigned char>(_expression[next_pos])) == 0;
}

}  // namespace icts::sdc_reader
