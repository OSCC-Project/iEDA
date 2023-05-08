/**
 * @file Str.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The C-style char* class.
 * @version 0.1
 * @date 2020-11-22
 *
 * @copyright Copyright (c) 2020
 *
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
  static std::string addBackslash(std::string origin_str);
};

}  // namespace ieda
