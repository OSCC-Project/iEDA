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
 * @file SdcClockCommands.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief SDC clock command handlers.
 */

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SdcClockParser.hh"
#include "SdcClockReader.hh"

namespace icts::sdc_reader {

auto SdcSubsetEvaluator::evaluateSet(const std::vector<ParsedWord>& words) -> SdcValue
{
  if (words.size() < 2U) {
    _data.diagnostics.emplace_back("ignored_set_without_variable");
    return {};
  }
  const auto variable_name = words[1].text;
  if (words.size() == 2U) {
    if (const auto iter = _variables.find(variable_name); iter != _variables.end()) {
      return iter->second;
    }
    return {};
  }
  if (words.size() == 3U) {
    auto value = evaluateWord(words[2]);
    _variables[variable_name] = value;
    return value;
  }
  auto values = evaluateWords(words, 2U);
  std::vector<std::string> joined_values;
  joined_values.reserve(values.size());
  std::ranges::transform(values, std::back_inserter(joined_values),
                         [](const SdcValue& value) -> std::string { return ValueToString(value); });
  auto value = MakeStringValue(JoinStrings(joined_values));
  _variables[variable_name] = value;
  return value;
}

auto SdcSubsetEvaluator::evaluateSetPlain(const std::vector<ParsedWord>& words) -> SdcValue
{
  if (words.size() < 2U) {
    _data.diagnostics.emplace_back("ignored_set_without_variable");
    return {};
  }
  const auto variable_name = words[1].text;
  if (words.size() == 2U) {
    if (const auto iter = _variables.find(variable_name); iter != _variables.end()) {
      return iter->second;
    }
    return {};
  }
  if (words.size() == 3U) {
    auto value = evaluatePlainWord(words[2]);
    _variables[variable_name] = value;
    return value;
  }
  auto values = evaluatePlainWords(words, 2U);
  std::vector<std::string> joined_values;
  joined_values.reserve(values.size());
  std::ranges::transform(values, std::back_inserter(joined_values),
                         [](const SdcValue& value) -> std::string { return ValueToString(value); });
  auto value = MakeStringValue(JoinStrings(joined_values));
  _variables[variable_name] = value;
  return value;
}

auto SdcSubsetEvaluator::evaluateExpr(const std::vector<ParsedWord>& words) -> SdcValue
{
  std::string expression;
  for (std::size_t index = 1U; index < words.size(); ++index) {
    if (!expression.empty()) {
      expression += ' ';
    }
    expression += words[index].braced ? substituteVariablesToString(words[index].text) : ValueToString(evaluateWord(words[index]));
  }
  double expression_value = 0.0;
  if (!ArithmeticParser(expression).parse(expression_value)) {
    _data.diagnostics.emplace_back("unresolved_expr:" + expression);
    return MakeStringValue("0");
  }
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10) << expression_value;
  return MakeStringValue(stream.str());
}

auto SdcSubsetEvaluator::evaluateExprPlain(const std::vector<ParsedWord>& words) -> SdcValue
{
  std::string expression;
  for (std::size_t index = 1U; index < words.size(); ++index) {
    if (!expression.empty()) {
      expression += ' ';
    }
    expression += words[index].braced ? substituteVariablesToString(words[index].text) : ValueToString(evaluatePlainWord(words[index]));
  }
  double expression_value = 0.0;
  if (!ArithmeticParser(expression).parse(expression_value)) {
    _data.diagnostics.emplace_back("unresolved_expr:" + expression);
    return MakeStringValue("0");
  }
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10) << expression_value;
  return MakeStringValue(stream.str());
}

auto SdcSubsetEvaluator::evaluateSetUnits(const std::vector<SdcValue>& args) -> void
{
  for (std::size_t index = 0U; index < args.size(); ++index) {
    if (ValueToString(args[index]) == "-time" && index + 1U < args.size()) {
      _time_unit_ns = TimeUnitToNs(ValueToString(args[index + 1U]));
      ++index;
    }
  }
}

auto SdcSubsetEvaluator::evaluateCollection(SdcObjectKind kind, const std::vector<SdcValue>& args) -> SdcValue
{
  std::vector<std::string> patterns;
  for (std::size_t index = 0U; index < args.size(); ++index) {
    const auto text = ValueToString(args[index]);
    if (text.empty()) {
      continue;
    }
    if (IsOption(text)) {
      if ((text == "-of_objects" || text == "-filter") && index + 1U < args.size()) {
        ++index;
      }
      continue;
    }
    for (const auto& item : SplitListText(text)) {
      patterns.emplace_back(item);
    }
  }
  return MakeObjectValue(kind, patterns);
}

auto SdcSubsetEvaluator::evaluateGetClocks(const std::vector<SdcValue>& args) -> SdcValue
{
  if (args.empty()) {
    return evaluateAllClocks();
  }
  return evaluateCollection(SdcObjectKind::kClock, args);
}

auto SdcSubsetEvaluator::evaluateAllClocks() -> SdcValue
{
  std::vector<std::string> clock_names;
  clock_names.reserve(_clock_period_by_name.size());
  for (const auto& [clock_name, period] : _clock_period_by_name) {
    (void) period;
    clock_names.emplace_back(clock_name);
  }
  return MakeObjectValue(SdcObjectKind::kClock, clock_names);
}

auto SdcSubsetEvaluator::evaluateCreateClock(const std::vector<SdcValue>& args) -> SdcValue
{
  SdcClockDecl clock;
  clock.kind = SdcClockDecl::Kind::kPrimary;
  double period = 0.0;
  bool period_resolved = false;

  for (std::size_t index = 0U; index < args.size(); ++index) {
    const auto token = ValueToString(args[index]);
    if (token == "-name" && index + 1U < args.size()) {
      clock.clock_name = ValueToString(args[++index]);
    } else if (token == "-period" && index + 1U < args.size()) {
      period_resolved = ParseDoubleValue(ValueToString(args[++index]), period);
    } else if (token == "-waveform" && index + 1U < args.size()) {
      ++index;
    } else if (token == "-add") {
      continue;
    } else if (IsOption(token)) {
      if (index + 1U < args.size()) {
        ++index;
      }
    } else {
      AppendRefsFromValue(clock.targets, args[index], SdcObjectKind::kUnknown);
    }
  }

  clock.is_virtual = clock.targets.empty();
  if (clock.clock_name.empty() && !clock.targets.empty()) {
    clock.clock_name = clock.targets.front().pattern;
  }
  if (clock.clock_name.empty()) {
    _data.diagnostics.emplace_back("ignored_create_clock_without_name");
    return {};
  }
  clock.period_ns = period * _time_unit_ns;
  clock.period_resolved = period_resolved;
  _clock_period_by_name[clock.clock_name] = clock.period_ns;
  const auto clock_name = clock.clock_name;
  _data.clocks.push_back(std::move(clock));
  return MakeObjectValue(SdcObjectKind::kClock, {clock_name});
}

auto SdcSubsetEvaluator::evaluateCreateGeneratedClock(const std::vector<SdcValue>& args) -> SdcValue
{
  SdcClockDecl clock;
  clock.kind = SdcClockDecl::Kind::kGenerated;
  for (std::size_t index = 0U; index < args.size(); ++index) {
    const auto token = ValueToString(args[index]);
    if (token == "-name" && index + 1U < args.size()) {
      clock.clock_name = ValueToString(args[++index]);
    } else if (token == "-source" && index + 1U < args.size()) {
      AppendRefsFromValue(clock.generated_sources, args[++index], SdcObjectKind::kUnknown);
    } else if (token == "-master_clock" && index + 1U < args.size()) {
      clock.master_clock_name = ValueToString(args[++index]);
    } else if (token == "-divide_by" && index + 1U < args.size()) {
      (void) ParseIntValue(ValueToString(args[++index]), clock.divide_by);
    } else if (token == "-multiply_by" && index + 1U < args.size()) {
      (void) ParseIntValue(ValueToString(args[++index]), clock.multiply_by);
    } else if (token == "-invert") {
      clock.invert = true;
    } else if ((token == "-edges" || token == "-edge_shift") && index + 1U < args.size()) {
      ++index;
    } else if (IsOption(token)) {
      if (index + 1U < args.size()) {
        ++index;
      }
    } else {
      AppendRefsFromValue(clock.targets, args[index], SdcObjectKind::kUnknown);
    }
  }

  clock.is_virtual = clock.targets.empty();
  if (clock.clock_name.empty()) {
    _data.diagnostics.emplace_back("ignored_generated_clock_without_name");
    return {};
  }

  auto lookup_name = clock.master_clock_name;
  if (lookup_name.empty() && !clock.generated_sources.empty()) {
    lookup_name = clock.generated_sources.front().pattern;
  }
  if (const auto iter = _clock_period_by_name.find(lookup_name); iter != _clock_period_by_name.end()) {
    clock.period_ns = iter->second;
    clock.period_resolved = true;
    if (clock.divide_by > 1) {
      clock.period_ns *= clock.divide_by;
    }
    if (clock.multiply_by > 1) {
      clock.period_ns /= clock.multiply_by;
    }
  }
  _clock_period_by_name[clock.clock_name] = clock.period_ns;
  const auto clock_name = clock.clock_name;
  _data.clocks.push_back(std::move(clock));
  return MakeObjectValue(SdcObjectKind::kClock, {clock_name});
}

auto SdcSubsetEvaluator::evaluateSetCaseAnalysis(const std::vector<SdcValue>& args) -> void
{
  if (args.size() < 2U) {
    _data.diagnostics.emplace_back("ignored_set_case_analysis_without_object");
    return;
  }
  SdcCaseAnalysis analysis;
  (void) ParseIntValue(ValueToString(args.front()), analysis.value);
  for (std::size_t index = 1U; index < args.size(); ++index) {
    AppendRefsFromValue(analysis.objects, args[index], SdcObjectKind::kUnknown);
  }
  _data.case_analyses.push_back(std::move(analysis));
}

namespace {

auto ObjectKindToSourcePrefix(SdcObjectKind kind) -> const char*
{
  switch (kind) {
    case SdcObjectKind::kPort:
      return "port:";
    case SdcObjectKind::kPin:
      return "pin:";
    case SdcObjectKind::kNet:
      return "net:";
    case SdcObjectKind::kClock:
      return "clock:";
    case SdcObjectKind::kUnknown:
      return "";
  }
  return "";
}

}  // namespace

auto PrimarySourceExpression(const SdcClockDecl& clock) -> std::string
{
  const auto& refs = clock.targets.empty() ? clock.generated_sources : clock.targets;
  if (refs.empty()) {
    return {};
  }
  return std::string(ObjectKindToSourcePrefix(refs.front().kind)) + refs.front().pattern;
}

}  // namespace icts::sdc_reader
