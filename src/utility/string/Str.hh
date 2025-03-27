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
 * @file Str.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The C-style char* class.
 * @version 0.1
 * @date 2020-11-22
 */

#pragma once

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ieda {

/**
 * @brief The class is for providing process API of the C-style string.
 *
 */
class Str
{
 public:
  static char* allocate();
  static char* copy(const char* str);
  static char* copy(std::initializer_list<const char*> str_list);
  static char* join(char* str1, const char* str2);
  static std::string join(const std::vector<std::string>& strs, const char* seperator);
  static char* printf(const char* format, ...);
  static void free(const char* str);
  static char* toUpper(const char* str);
  static char* toLower(const char* str);
  static double toDouble(const char* str);
  static float toFloat(const char* str);
  static unsigned toUnsigned(const char* str);
  static int toInt(const char* str);
  static int caseCmp(const char* lhs, const char* rhs);
  static int noCaseCmp(const char* lhs, const char* rhs);
  static bool equal(const char* lhs, const char* rhs);
  static bool noCaseEqual(const char* lhs, const char* rhs);
  static const char* trimmed(const char* str);
  static const char* trimmedWithSquareBracket(const char* str);
  static std::pair<std::string, std::string> splitTwoPart(const char* orig, const char* delimiter);
  static std::vector<std::string> split(const char* orig, const char* delimiter);
  static std::vector<int> splitInt(const char* orig, const char* delimiter);
  static std::vector<double> splitDouble(const char* orig, const char* delimiter);
  static std::string stripPrefix(std::string_view str, std::string_view prefix);
  static std::string stripSuffix(std::string_view str, std::string_view suffix);
  static bool startWith(const char* str, const char* prefix);
  static bool endWith(const char* str, const char* suffix);
  static std::string replace(const std::string& str, const char* replace_str, const char* new_str);
  static bool contain(const char* str, const char* sub_str);
  static std::vector<std::string> matchPattern(const char* str, std::string regex_pattern);
  static std::pair<std::string, std::optional<int>> matchBusName(const char* str);
  static std::pair<std::string, std::optional<std::pair<int, int>>> matchBusSliceName(const char* str);

  static std::string trimBackslash(std::string origin_str);
  static std::string trimEscape(std::string origin_str);
  static std::string addBackslash(std::string origin_str);
  static std::string addDoubleBackslash(std::string origin_str);
  static std::string concateBackSlashStr(std::string original_str);
};

}  // namespace ieda
