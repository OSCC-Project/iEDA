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
 * @file SdcClockEvaluator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief SDC clock command evaluation dispatch.
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SdcClockParser.hh"
#include "SdcClockReader.hh"

namespace icts::sdc_reader {

auto SdcSubsetEvaluator::substituteVariablesToString(const std::string& text) -> std::string
{
  std::string result;
  for (std::size_t index = 0U; index < text.size();) {
    if (text[index] != '$') {
      result += text[index++];
      continue;
    }
    const auto [var_name, end_pos] = parseVariableName(text, index);
    if (var_name.empty()) {
      result += text[index++];
      continue;
    }
    if (const auto iter = _variables.find(var_name); iter != _variables.end()) {
      result += ValueToString(iter->second);
    }
    index = end_pos;
  }
  return result;
}

auto SdcSubsetEvaluator::evaluateWord(const ParsedWord& word) -> SdcValue
{
  if (word.braced) {
    return MakeStringValue(word.text);
  }
  const auto clean_text = Trim(word.text);
  if (!clean_text.empty() && clean_text.front() == '$') {
    const auto [var_name, end_pos] = parseVariableName(clean_text, 0U);
    if (!var_name.empty() && end_pos == clean_text.size()) {
      if (const auto iter = _variables.find(var_name); iter != _variables.end()) {
        return iter->second;
      }
      return {};
    }
  }
  if (clean_text.size() >= 2U && clean_text.front() == '[') {
    const auto close_pos = matchingBracketPos(clean_text, 0U);
    if (close_pos == clean_text.size() - 1U) {
      return evaluateCommandWithoutCommandSubstitution(expandBracketCommandsToString(clean_text.substr(1U, clean_text.size() - 2U)));
    }
  }

  const auto expanded = expandBracketCommandsToString(word.text);
  return MakeStringValue(substituteVariablesToString(expanded));
}

auto SdcSubsetEvaluator::evaluatePlainWord(const ParsedWord& word) -> SdcValue
{
  if (word.braced) {
    return MakeStringValue(word.text);
  }
  const auto clean_text = Trim(word.text);
  if (!clean_text.empty() && clean_text.front() == '$') {
    const auto [var_name, end_pos] = parseVariableName(clean_text, 0U);
    if (!var_name.empty() && end_pos == clean_text.size()) {
      if (const auto iter = _variables.find(var_name); iter != _variables.end()) {
        return iter->second;
      }
      return {};
    }
  }
  return MakeStringValue(substituteVariablesToString(word.text));
}

auto SdcSubsetEvaluator::evaluatePlainWords(const std::vector<ParsedWord>& words, std::size_t start_index) -> std::vector<SdcValue>
{
  std::vector<SdcValue> values;
  values.reserve(words.size() - std::min(start_index, words.size()));
  for (std::size_t index = start_index; index < words.size(); ++index) {
    values.push_back(evaluatePlainWord(words[index]));
  }
  return values;
}

auto SdcSubsetEvaluator::expandBracketCommandsToString(std::string text) -> std::string
{
  for (std::size_t iteration = 0U; iteration < kCommandSubstitutionLimit; ++iteration) {
    const auto [open_pos, close_pos] = findInnermostBracket(text);
    if (open_pos == std::string::npos || close_pos == std::string::npos) {
      return text;
    }
    const auto command = text.substr(open_pos + 1U, close_pos - open_pos - 1U);
    const auto command_value = evaluateCommandWithoutCommandSubstitution(command);
    text.replace(open_pos, close_pos - open_pos + 1U, ValueToString(command_value));
  }
  _data.diagnostics.emplace_back("command_substitution_depth_limit");
  return text;
}

auto SdcSubsetEvaluator::evaluateWords(const std::vector<ParsedWord>& words, std::size_t start_index) -> std::vector<SdcValue>
{
  std::vector<SdcValue> values;
  values.reserve(words.size() - std::min(start_index, words.size()));
  for (std::size_t index = start_index; index < words.size(); ++index) {
    values.push_back(evaluateWord(words[index]));
  }
  return values;
}

auto SdcSubsetEvaluator::evaluateCommand(const std::string& command) -> SdcValue
{
  const auto words = parseWords(command);
  if (words.empty()) {
    return {};
  }
  const auto command_name = ValueToString(evaluateWord(words.front()));
  if (command_name == "set") {
    return evaluateSet(words);
  }
  if (command_name == "expr") {
    return evaluateExpr(words);
  }
  if (command_name == "set_units") {
    evaluateSetUnits(evaluateWords(words, 1U));
    return {};
  }
  if (command_name == "get_ports") {
    return evaluateCollection(SdcObjectKind::kPort, evaluateWords(words, 1U));
  }
  if (command_name == "get_pins") {
    return evaluateCollection(SdcObjectKind::kPin, evaluateWords(words, 1U));
  }
  if (command_name == "get_nets") {
    return evaluateCollection(SdcObjectKind::kNet, evaluateWords(words, 1U));
  }
  if (command_name == "get_clocks") {
    return evaluateGetClocks(evaluateWords(words, 1U));
  }
  if (command_name == "all_clocks") {
    return evaluateAllClocks();
  }
  if (command_name == "create_clock") {
    return evaluateCreateClock(evaluateWords(words, 1U));
  }
  if (command_name == "create_generated_clock") {
    return evaluateCreateGeneratedClock(evaluateWords(words, 1U));
  }
  if (command_name == "set_case_analysis") {
    evaluateSetCaseAnalysis(evaluateWords(words, 1U));
    return {};
  }
  if (!command_name.empty()) {
    _data.diagnostics.emplace_back("ignored_sdc_command:" + command_name);
  }
  return {};
}

auto SdcSubsetEvaluator::evaluateCommandWithoutCommandSubstitution(const std::string& command) -> SdcValue
{
  const auto words = parseWords(command);
  if (words.empty()) {
    return {};
  }
  const auto command_name = ValueToString(evaluatePlainWord(words.front()));
  if (command_name == "set") {
    return evaluateSetPlain(words);
  }
  if (command_name == "expr") {
    return evaluateExprPlain(words);
  }
  if (command_name == "set_units") {
    evaluateSetUnits(evaluatePlainWords(words, 1U));
    return {};
  }
  if (command_name == "get_ports") {
    return evaluateCollection(SdcObjectKind::kPort, evaluatePlainWords(words, 1U));
  }
  if (command_name == "get_pins") {
    return evaluateCollection(SdcObjectKind::kPin, evaluatePlainWords(words, 1U));
  }
  if (command_name == "get_nets") {
    return evaluateCollection(SdcObjectKind::kNet, evaluatePlainWords(words, 1U));
  }
  if (command_name == "get_clocks") {
    return evaluateGetClocks(evaluatePlainWords(words, 1U));
  }
  if (command_name == "all_clocks") {
    return evaluateAllClocks();
  }
  if (command_name == "create_clock") {
    return evaluateCreateClock(evaluatePlainWords(words, 1U));
  }
  if (command_name == "create_generated_clock") {
    return evaluateCreateGeneratedClock(evaluatePlainWords(words, 1U));
  }
  if (command_name == "set_case_analysis") {
    evaluateSetCaseAnalysis(evaluatePlainWords(words, 1U));
    return {};
  }
  if (!command_name.empty()) {
    _data.diagnostics.emplace_back("ignored_sdc_command:" + command_name);
  }
  return {};
}

}  // namespace icts::sdc_reader
