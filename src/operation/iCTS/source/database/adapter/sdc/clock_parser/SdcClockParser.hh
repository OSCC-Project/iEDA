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
 * @file SdcClockParser.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief SDC clock subset parser declarations.
 */

#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SdcClockReader.hh"

namespace icts::sdc_reader {

inline constexpr std::size_t kCommandSubstitutionLimit = 1024U;

struct ParsedWord
{
  std::string text;
  bool braced = false;
};

struct SdcValue
{
  std::vector<std::string> strings;
  std::vector<SdcObjectRef> objects;
};

class ArithmeticParser
{
 public:
  explicit ArithmeticParser(std::string expression);

  auto parse(double& value) -> bool;

 private:
  static auto isOperator(char token) -> bool;
  static auto precedence(char token) -> int;
  static auto applyTopOperator(std::vector<double>& values, std::vector<char>& operators) -> bool;
  static auto applyUntilOpen(std::vector<double>& values, std::vector<char>& operators) -> bool;

  auto parseNumber(double& value) -> bool;
  auto skipSpaces() -> void;
  auto matchWord(std::string_view word) const -> bool;

  std::string _expression;
  std::size_t _pos = 0U;
};

class SdcSubsetEvaluator
{
 public:
  auto readFile(const std::string& sdc_path) -> SdcClockData;

 private:
  static auto splitCommands(const std::string& text) -> std::vector<std::string>;
  static auto parseWords(const std::string& command) -> std::vector<ParsedWord>;
  static auto parseBalanced(const std::string& text, std::size_t open_pos, char open_ch, char close_ch)
      -> std::pair<std::string, std::size_t>;
  static auto matchingBracketPos(const std::string& text, std::size_t open_pos) -> std::size_t;
  static auto findInnermostBracket(const std::string& text) -> std::pair<std::size_t, std::size_t>;
  static auto parseVariableName(const std::string& text, std::size_t dollar_pos) -> std::pair<std::string, std::size_t>;

  auto substituteVariablesToString(const std::string& text) -> std::string;
  auto evaluateWord(const ParsedWord& word) -> SdcValue;
  auto evaluatePlainWord(const ParsedWord& word) -> SdcValue;
  auto evaluatePlainWords(const std::vector<ParsedWord>& words, std::size_t start_index = 0U) -> std::vector<SdcValue>;
  auto expandBracketCommandsToString(std::string text) -> std::string;
  auto evaluateWords(const std::vector<ParsedWord>& words, std::size_t start_index = 0U) -> std::vector<SdcValue>;
  auto evaluateCommand(const std::string& command) -> SdcValue;
  auto evaluateCommandWithoutCommandSubstitution(const std::string& command) -> SdcValue;

  auto evaluateSet(const std::vector<ParsedWord>& words) -> SdcValue;
  auto evaluateSetPlain(const std::vector<ParsedWord>& words) -> SdcValue;
  auto evaluateExpr(const std::vector<ParsedWord>& words) -> SdcValue;
  auto evaluateExprPlain(const std::vector<ParsedWord>& words) -> SdcValue;
  auto evaluateSetUnits(const std::vector<SdcValue>& args) -> void;
  static auto evaluateCollection(SdcObjectKind kind, const std::vector<SdcValue>& args) -> SdcValue;
  auto evaluateGetClocks(const std::vector<SdcValue>& args) -> SdcValue;
  auto evaluateAllClocks() -> SdcValue;
  auto evaluateCreateClock(const std::vector<SdcValue>& args) -> SdcValue;
  auto evaluateCreateGeneratedClock(const std::vector<SdcValue>& args) -> SdcValue;
  auto evaluateSetCaseAnalysis(const std::vector<SdcValue>& args) -> void;

  double _time_unit_ns = 1.0;
  SdcClockData _data;
  std::unordered_map<std::string, SdcValue> _variables;
  std::map<std::string, double> _clock_period_by_name;
};

auto Trim(const std::string& text) -> std::string;
auto IsOption(const std::string& text) -> bool;
auto JoinStrings(const std::vector<std::string>& values) -> std::string;
auto SplitListText(const std::string& text) -> std::vector<std::string>;
auto ValueToString(const SdcValue& value) -> std::string;
auto MakeStringValue(std::string text) -> SdcValue;
auto MakeObjectValue(SdcObjectKind kind, const std::vector<std::string>& patterns) -> SdcValue;
auto AppendRefsFromValue(std::vector<SdcObjectRef>& refs, const SdcValue& value, SdcObjectKind default_kind) -> void;
auto ParseDoubleValue(const std::string& text, double& value) -> bool;
auto ParseIntValue(const std::string& text, int& value) -> bool;
auto TimeUnitToNs(const std::string& unit) -> double;
auto PrimarySourceExpression(const SdcClockDecl& clock) -> std::string;

}  // namespace icts::sdc_reader
