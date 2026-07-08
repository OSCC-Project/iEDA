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
 * @file SdcClockParser.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief SDC clock command lexer and parser helper implementation.
 */

#include "SdcClockParser.hh"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace icts::sdc_reader {

auto SdcSubsetEvaluator::readFile(const std::string& sdc_path) -> SdcClockData
{
  std::ifstream stream(sdc_path);
  if (!stream.is_open()) {
    _data.diagnostics.emplace_back("failed_to_open_sdc_file");
    return _data;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  for (const auto& command : splitCommands(buffer.str())) {
    (void) evaluateCommand(command);
  }
  return _data;
}

auto SdcSubsetEvaluator::splitCommands(const std::string& text) -> std::vector<std::string>
{
  std::vector<std::string> commands;
  std::string command;
  int brace_depth = 0;
  int bracket_depth = 0;
  bool in_quote = false;
  bool escaped = false;
  bool command_word_start = true;

  for (std::size_t index = 0U; index < text.size(); ++index) {
    const char ch = text[index];
    if (escaped) {
      command += ch;
      escaped = false;
      command_word_start = false;
      continue;
    }
    if (ch == '\\') {
      command += ch;
      escaped = true;
      continue;
    }
    if (!in_quote && brace_depth == 0 && bracket_depth == 0 && ch == '#' && command_word_start) {
      while (index < text.size() && text[index] != '\n') {
        ++index;
      }
      if (index < text.size()) {
        const auto clean_command = Trim(command);
        if (!clean_command.empty()) {
          commands.emplace_back(clean_command);
        }
        command.clear();
        command_word_start = true;
      }
      continue;
    }
    if (ch == '"' && brace_depth == 0) {
      in_quote = !in_quote;
      command += ch;
      command_word_start = false;
      continue;
    }
    if (!in_quote) {
      if (ch == '{') {
        ++brace_depth;
      } else if (ch == '}' && brace_depth > 0) {
        --brace_depth;
      } else if (ch == '[') {
        ++bracket_depth;
      } else if (ch == ']' && bracket_depth > 0) {
        --bracket_depth;
      }
    }
    if (!in_quote && brace_depth == 0 && bracket_depth == 0 && (ch == '\n' || ch == ';')) {
      const auto clean_command = Trim(command);
      if (!clean_command.empty()) {
        commands.emplace_back(clean_command);
      }
      command.clear();
      command_word_start = true;
      continue;
    }
    command += ch;
    command_word_start = command_word_start && std::isspace(static_cast<unsigned char>(ch)) != 0;
  }

  const auto clean_command = Trim(command);
  if (!clean_command.empty()) {
    commands.emplace_back(clean_command);
  }
  return commands;
}

auto SdcSubsetEvaluator::parseWords(const std::string& command) -> std::vector<ParsedWord>
{
  std::vector<ParsedWord> words;
  std::size_t index = 0U;
  while (index < command.size()) {
    while (index < command.size() && std::isspace(static_cast<unsigned char>(command[index])) != 0) {
      ++index;
    }
    if (index >= command.size()) {
      break;
    }
    if (command[index] == '{') {
      auto [word_text, end_pos] = parseBalanced(command, index, '{', '}');
      words.emplace_back(ParsedWord{std::move(word_text), true});
      index = end_pos;
      continue;
    }
    if (command[index] == '"') {
      ++index;
      std::string word_text;
      bool escaped = false;
      while (index < command.size()) {
        const char ch = command[index++];
        if (escaped) {
          word_text += ch;
          escaped = false;
          continue;
        }
        if (ch == '\\') {
          escaped = true;
          continue;
        }
        if (ch == '"') {
          break;
        }
        word_text += ch;
      }
      words.emplace_back(ParsedWord{std::move(word_text), false});
      continue;
    }

    std::string word_text;
    int bracket_depth = 0;
    int brace_depth = 0;
    bool escaped = false;
    while (index < command.size()) {
      const char ch = command[index];
      if (escaped) {
        word_text += ch;
        escaped = false;
        ++index;
        continue;
      }
      if (ch == '\\') {
        escaped = true;
        ++index;
        continue;
      }
      if (bracket_depth == 0 && brace_depth == 0 && std::isspace(static_cast<unsigned char>(ch)) != 0) {
        break;
      }
      if (ch == '[') {
        ++bracket_depth;
      } else if (ch == ']' && bracket_depth > 0) {
        --bracket_depth;
      } else if (ch == '{') {
        ++brace_depth;
      } else if (ch == '}' && brace_depth > 0) {
        --brace_depth;
      }
      word_text += ch;
      ++index;
    }
    words.emplace_back(ParsedWord{std::move(word_text), false});
  }
  return words;
}

auto SdcSubsetEvaluator::parseBalanced(const std::string& text, std::size_t open_pos, char open_ch, char close_ch)
    -> std::pair<std::string, std::size_t>
{
  std::string result;
  int depth = 0;
  bool escaped = false;
  for (std::size_t index = open_pos; index < text.size(); ++index) {
    const char ch = text[index];
    if (escaped) {
      result += ch;
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == open_ch) {
      if (depth > 0) {
        result += ch;
      }
      ++depth;
      continue;
    }
    if (ch == close_ch) {
      --depth;
      if (depth == 0) {
        return {result, index + 1U};
      }
      result += ch;
      continue;
    }
    result += ch;
  }
  return {result, text.size()};
}

auto SdcSubsetEvaluator::matchingBracketPos(const std::string& text, std::size_t open_pos) -> std::size_t
{
  int bracket_depth = 0;
  int brace_depth = 0;
  bool in_quote = false;
  bool escaped = false;
  for (std::size_t index = open_pos; index < text.size(); ++index) {
    const char ch = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"' && brace_depth == 0) {
      in_quote = !in_quote;
      continue;
    }
    if (!in_quote) {
      if (ch == '{') {
        ++brace_depth;
      } else if (ch == '}' && brace_depth > 0) {
        --brace_depth;
      } else if (ch == '[' && brace_depth == 0) {
        ++bracket_depth;
      } else if (ch == ']' && brace_depth == 0) {
        --bracket_depth;
        if (bracket_depth == 0) {
          return index;
        }
      }
    }
  }
  return std::string::npos;
}

auto SdcSubsetEvaluator::findInnermostBracket(const std::string& text) -> std::pair<std::size_t, std::size_t>
{
  std::vector<std::size_t> open_positions;
  int brace_depth = 0;
  bool in_quote = false;
  bool escaped = false;
  for (std::size_t index = 0U; index < text.size(); ++index) {
    const char ch = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"' && brace_depth == 0) {
      in_quote = !in_quote;
      continue;
    }
    if (in_quote) {
      continue;
    }
    if (ch == '{') {
      ++brace_depth;
      continue;
    }
    if (ch == '}' && brace_depth > 0) {
      --brace_depth;
      continue;
    }
    if (brace_depth != 0) {
      continue;
    }
    if (ch == '[') {
      open_positions.push_back(index);
    } else if (ch == ']' && !open_positions.empty()) {
      return {open_positions.back(), index};
    }
  }
  return {std::string::npos, std::string::npos};
}

auto SdcSubsetEvaluator::parseVariableName(const std::string& text, std::size_t dollar_pos) -> std::pair<std::string, std::size_t>
{
  if (dollar_pos + 1U >= text.size()) {
    return {{}, dollar_pos + 1U};
  }
  if (text[dollar_pos + 1U] == '{') {
    const auto close_pos = text.find('}', dollar_pos + 2U);
    if (close_pos == std::string::npos) {
      return {{}, dollar_pos + 1U};
    }
    return {text.substr(dollar_pos + 2U, close_pos - dollar_pos - 2U), close_pos + 1U};
  }
  std::size_t pos = dollar_pos + 1U;
  while (pos < text.size()) {
    const auto ch = static_cast<unsigned char>(text[pos]);
    if (std::isalnum(ch) == 0 && text[pos] != '_') {
      break;
    }
    ++pos;
  }
  if (pos == dollar_pos + 1U) {
    return {{}, dollar_pos + 1U};
  }
  return {text.substr(dollar_pos + 1U, pos - dollar_pos - 1U), pos};
}

}  // namespace icts::sdc_reader
