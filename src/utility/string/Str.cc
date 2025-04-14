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
 * @file Str.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of C-style char* class operation.
 * @version 0.1
 * @date 2020-11-22
 */

#include "Str.hh"

#include <stdarg.h>
#include <stdio.h>

#include <cstring>
#include <mutex>
#include <regex>

#include "MemoryPool.hh"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

namespace ieda {

constexpr unsigned max_char_size = 4096 * 2;

struct pool_char_tag
{
};  // for tag
using spl = SingletonPool<pool_char_tag,
                          max_char_size>;  // assume common char not exceed
                                           // max_char_size byte. If c-style
                                           // char exceed max_char_size, please
                                           // not use this.We use the memory
                                           // pool for allocate space, so we can
                                           // avoid memory fragmentation and
                                           // improve allocate speed.

/**
 * @brief Allocate a block to store data.
 *
 */
char* Str::allocate()
{
  char* buffer = static_cast<char*>(spl::malloc());
  memset(buffer, 0, max_char_size);
  return buffer;
}

/**
 * @brief Copy a C-sytle string.
 *
 * @param str
 * @return char*
 */
char* Str::copy(const char* str)
{
  if (!str) {
    return nullptr;
  }

  char* buffer = static_cast<char*>(spl::malloc());
  char* new_str = buffer;
  memset(buffer, 0, max_char_size);
  std::size_t len = strlen(str);
  assert(len < max_char_size);
  memcpy(buffer, str, len);

  return new_str;
}

/**
 * @brief Support variably char* copy.
 *
 * @param str_list
 * @return char*
 */
char* Str::copy(std::initializer_list<const char*> str_list)
{
  char* buffer = static_cast<char*>(spl::malloc());
  char* new_str = buffer;
  memset(buffer, 0, max_char_size);
  std::size_t total_len = 0;

  for (auto tmp : str_list) {
    if (!tmp) {
      continue;
    }

    std::size_t len = strlen(tmp);
    total_len += len;
    assert(total_len < max_char_size);
    memcpy(buffer, tmp, len);
    buffer += len;
  }

  return new_str;
}

/**
 * @brief Join the str2 to str1.
 *
 * @param str1
 * @param str2
 * @return char*
 */
char* Str::join(char* str1, const char* str2)
{
  if (!str1 || !str2) {
    return nullptr;
  }

  // need str1 is from memory pool;
  assert(spl::is_from((void*) str1));
  size_t len1 = strlen(str1);
  size_t len2 = strlen(str2);

  // assure the size is enough.
  assert(len1 + len2 < max_char_size);

  memcpy((void*) (str1 + len1), str2, len2);
  return str1;
}

/**
 * @brief Join the string in the vector to one string use the seperator.
 *
 * @param strs
 * @return std::string
 */
std::string Str::join(const std::vector<std::string>& strs, const char* seperator)
{
  return absl::StrJoin(strs, seperator);
}

/**
 * @brief The Str::printf use a static memory to save c style string.
 * The static memory will repeated use. We can not print the string exceed the
 * max_char_size. The memory will be back to system when program end.You don't
 * need to release the return c-string.And we need consume the c-string
 * immediately, We damage the previous c-string when sprintf next time.
 *
 * @param format The string format.
 * @param ... The data needed to be printed.
 * @return char* The return string, do not need to be release.
 */
char* Str::printf(const char* format, ...)
{
  // record the buffer begin position.
  static char* buffer = static_cast<char*>(spl::malloc());
  static std::mutex mt;
  static unsigned buffer_size = max_char_size;

  std::lock_guard<std::mutex> lk(mt);

  char* ret_val = buffer;

  va_list args;
  va_start(args, format);
  vsnprintf(buffer, buffer_size, format,
            args);  // windows to be sure whether use vsnprintf
  va_end(args);

  return ret_val;
}

/**
 * @brief Free the char*.
 *
 */
void Str::free(const char* str)
{
  if (!str) {
    return;
  }

  assert(spl::is_from((void*) str));
  spl::free((void*) (str));
}

/**
 * @brief Convert the letter of string to upper.This will be allocate
 * a new str.
 *
 * @param str The string to be upper.
 * @return char* The upper string.
 */
char* Str::toUpper(const char* str)
{
  if (!str) {
    return nullptr;
  }

  char* upper_str = static_cast<char*>(spl::malloc());
  char* ret_val = upper_str;

  while (*str != '\0') {
    *upper_str = toupper(*str);
    ++str;
    ++upper_str;
  }

  *upper_str = '\0';

  return ret_val;
}

/**
 * @brief Convert the letter of string to lower.This will be allocate
 * a new str.
 *
 * @param str The string to be upper.
 * @return char* The upper string.
 */
char* Str::toLower(const char* str)
{
  if (!str) {
    return nullptr;
  }

  char* lower_str = static_cast<char*>(spl::malloc());
  char* ret_val = lower_str;

  while (*str != '\0') {
    *lower_str = tolower(*str);
    ++str;
    ++lower_str;
  }

  *lower_str = '\0';

  return ret_val;
}

/**
 * @brief convert str to double.
 *
 * @param str The string to be convert to double.
 * @return double The double value.
 */
double Str::toDouble(const char* str)
{
  double d = std::strtod(str, nullptr);
  return d;
}

/**
 * @brief convert str to float.
 *
 * @param str
 * @return float
 */
float Str::toFloat(const char* str)
{
  auto f = static_cast<float>(std::strtof(str, nullptr));
  return f;
}

/**
 * @brief convert str to unsigned.
 *
 * @param str
 * @return unsigned
 */
unsigned Str::toUnsigned(const char* str)
{
  auto u = static_cast<unsigned>(std::atoi(str));
  return u;
}

/**
 * @brief convert str to int.
 *
 * @param str
 * @return int
 */
int Str::toInt(const char* str)
{
  int i = std::atoi(str);
  return i;
}

/**
 * @brief Cmp the two C-style string.
 *
 * @param lhs
 * @param rhs
 * @return  Negative value if lhs appears before rhs in lexicographical
 * order. Zero if lhs and rhs compare equal. Positive value if lhs appears
 * after rhs in lexicographical order.
 */
int Str::caseCmp(const char* lhs, const char* rhs)
{
  return std::strcmp(lhs, rhs);
}

/**
 * @brief Cmp the two C-style string ignore the upper/lower case.
 *
 * @param lhs
 * @param rhs
 * @return int Zero if lhs and rhs compare equal, else 1.
 */
int Str::noCaseCmp(const char* lhs, const char* rhs)
{
  std::string_view str1 = lhs;
  std::string_view str2 = rhs;

  bool isEqual
      = std::equal(str1.begin(), str1.end(), str2.begin(), str2.end(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });

  // return zero if equal.
  return !isEqual;
}

/**
 * @brief Judge whether the lhs C-style string equal to rhs C-style string.
 *
 * @param lhs
 * @param rhs
 * @return true lhs == rhs
 * @return false lhs != rhs
 */
bool Str::equal(const char* lhs, const char* rhs)
{
  return (0 == std::strcmp(lhs, rhs));
}

/**
 * @brief Judge whether the lhs C-style string equal to rhs C-style string in
 * nocase.
 *
 * @param lhs
 * @param rhs
 * @return true lhs == rhs
 * @return false lhs != rhs
 */
bool Str::noCaseEqual(const char* lhs, const char* rhs)
{
  return (0 == Str::noCaseCmp(lhs, rhs));
}

/**
 * @brief Trim the string, remove the escape(\)
 * character at the start and the whitespace at the end.
 *
 * @param str
 * @return char* May be origin str or the trimmed str, not allocate memory.
 */
const char* Str::trimmed(const char* str)
{
  if (!str) {
    return nullptr;
  }
  absl::string_view str_view(str);
  absl::ConsumePrefix(&str_view, "\\");
  absl::ConsumeSuffix(&str_view, " ");
  absl::ConsumeSuffix(&str_view, "\n");
  return Str::printf("%s", std::string(str_view).c_str());
}

/**
 * @brief Trim the string, remove the escape(\)
 * character with square brackets at the start and the whitespace at the end.(\\key[0]->key[0])
 *
 * @param str
 * @return char* May be origin str or the trimmed str, not allocate memory.
 */
const char* Str::trimmedWithSquareBracket(const char* str)
{
  if (!str) {
    return nullptr;
  }

  absl::string_view str_view(str);
  std::string str_(str);
  if (str_.find('[') != std::string::npos && str_.find(']') != std::string::npos) {
    absl::ConsumePrefix(&str_view, "\\");
    absl::ConsumeSuffix(&str_view, " ");
  }

  return Str::printf("%s", std::string(str_view).c_str());
}

/**
 * @brief Split the orig string to two part accord the delimiter like '\', or
 * ':'.
 *
 * @param orig
 * @param delimiter
 * @return std::pair<std::string, std::string>
 */
std::pair<std::string, std::string> Str::splitTwoPart(const char* orig, const char* delimiter)
{
  std::string tmp = orig;
  auto delimiter_pos = tmp.find_last_of(delimiter);
  if (delimiter_pos == std::string::npos) {
    return std::make_pair(tmp, "");
  }
  std::string first_str = tmp.substr(0, delimiter_pos);
  std::string second_str = tmp.substr(delimiter_pos + 1);

  return std::make_pair(first_str, second_str);
}

/**
 * @brief Split the orig string accord the delimiter like '\', or ':'.
 *
 * @param orig
 * @param delimiter
 * @return std::vector<std::string>
 */
std::vector<std::string> Str::split(const char* orig, const char* delimiter)
{
  auto copy_string = [](const char* src) {
    size_t len = strlen(src);
    char* dest = (char*) std::malloc((len + 1) * sizeof(char));
    std::strcpy(dest, src);
    return dest;
  };

  char* copy_str = copy_string(orig);
  std::vector<std::string> results;

  char* token = strtok(copy_str, delimiter);
  while (token) {
    results.push_back(token);
    token = strtok(nullptr, delimiter);
  }

  std::free(copy_str);

  return results;
}

/**
 * @brief Split the orig string to int vector accord the delimiter like '\', or
 * ':'.
 *
 * @param orig
 * @param delimiter
 * @return std::vector<int>
 */
std::vector<int> Str::splitInt(const char* orig, const char* delimiter)
{
  char* copy_str = Str::copy(orig);
  std::vector<int> results;

  char* token = strtok(copy_str, delimiter);
  while (token) {
    if (*token != '{') {
      results.push_back(atoi(token));
    }
    token = strtok(nullptr, delimiter);
  }

  Str::free(copy_str);

  return results;
}

/**
 * @brief Split the orig string to double vector accord the delimiter like '\',
 * or ':'.
 *
 * @param orig
 * @param delimiter
 * @return std::vector<double>
 */
std::vector<double> Str::splitDouble(const char* orig, const char* delimiter)
{
  char* copy_str = Str::copy(orig);
  std::vector<double> results;

  char* token = strtok(copy_str, delimiter);
  while (token) {
    results.push_back(strtod(token, nullptr));
    token = strtok(nullptr, delimiter);
  }

  Str::free(copy_str);

  return results;
}

/**
 * @brief Strip the string prefix.
 *
 * @param str
 * @return std::string
 */
std::string Str::stripPrefix(std::string_view str, std::string_view prefix)
{
  return std::string(absl::StripPrefix(str, prefix));
}

/**
 * @brief Strip the string suffix.
 *
 * @param str
 * @param suffix
 * @return std::string
 */
std::string Str::stripSuffix(std::string_view str, std::string_view suffix)
{
  return std::string(absl::StripSuffix(str, suffix));
}

/**
 * @brief Judge the string whether start with prefix.
 *
 * @param str
 * @param prefix
 * @return true
 * @return false
 */
bool Str::startWith(const char* str, const char* prefix)
{
  return absl::StartsWith(str, prefix);
}

/**
 * @brief Judge the string whether end with prefix.
 *
 * @param str
 * @param suffix
 * @return true
 * @return false
 */
bool Str::endWith(const char* str, const char* suffix)
{
  return absl::EndsWith(str, suffix);
}

/**
 * @brief Replace the replace str to the new str.
 *
 * @param str
 * @param replace_str
 * @param new_str
 * @return std::string
 */
std::string Str::replace(const std::string& str, const char* replace_str, const char* new_str)
{
  std::regex re(replace_str);
  return std::regex_replace(str, re, new_str);
}

/**
 * @brief Judge whether str contain sub str;
 *
 * @param str
 * @param sub_str
 * @return true
 * @return false
 */
bool Str::contain(const char* str, const char* sub_str)
{
  return absl::StrContains(str, sub_str);
}

/**
 * @brief The string regex pattern match, return the matched sub string.
 *
 * @param str
 * @param regex_pattern
 * @return std::vector<std::string>
 */
std::vector<std::string> Str::matchPattern(const char* str, std::string regex_pattern)
{
  std::vector<std::string> match_strs;
  const std::regex str_regex(regex_pattern);
  std::cmatch str_match;
  std::regex_match(str, str_match, str_regex);

  for (size_t i = 0; i < str_match.size(); ++i) {
    auto& sub_match = str_match[i];
    std::string sub_str = sub_match.str();
    match_strs.push_back(sub_str);
  }
  return match_strs;
}

/**
 * @brief Match the bus pattern.
 *
 * @param str
 * @return std::pair<std::string, std::optional<int>>
 */
std::pair<std::string, std::optional<int>> Str::matchBusName(const char* str)
{
  if (!Str::endWith(str, "]")) {
    return {str, std::nullopt};
  }

  char* copy_str = Str::copy(str);

  char* token = strtok(copy_str, "[");
  std::string base_name = token;
  if (token) {
    token = strtok(nullptr, "]");
    if (token) {
      int index = Str::toInt(token);
      return {base_name, index};
    }
  }

  Str::free(copy_str);

  return {str, std::nullopt};
}

/**
 * @brief Match the bus slice name, such as A[9:0].
 *
 * @param str
 * @return std::pair<std::string, std::optional<std::pair<int, int>>>
 */
std::pair<std::string, std::optional<std::pair<int, int>>> Str::matchBusSliceName(const char* str)
{
  if (!Str::endWith(str, "]")) {
    return {str, std::nullopt};
  }

  char* copy_str = Str::copy(str);

  char* token = strtok(copy_str, "[");
  std::string base_name = token;
  if (token) {
    token = strtok(nullptr, ":");
    if (token) {
      int index1 = Str::toInt(token);
      token = strtok(nullptr, "]");
      int index2 = Str::toInt(token);
      std::pair<int, int> bus_slice = {index1, index2};
      return {base_name, bus_slice};
    }
  }

  Str::free(copy_str);

  return {str, std::nullopt};
}

/**
 * @brief trim \[\] to [], and \\ to nothing
 *
 * @param origin_str
 * @return std::string
 */
std::string Str::trimBackslash(std::string origin_str)
{
  std::string replace_str = origin_str;
  if (ieda::Str::contain(replace_str.c_str(), R"(\[)") && ieda::Str::contain(replace_str.c_str(), R"(\])")) {
    std::string new_value_1 = replace(replace_str, R"(\\\[)", R"([)");
    std::string new_value_2 = replace(new_value_1, R"(\\\])", R"(])");

    replace_str = new_value_2;
  }

  replace_str = replace(replace_str, R"(\\)", "");

  return replace_str;
}

/**
 * @brief trim \
 *
 * @param origin_str
 * @return std::string
 */
std::string Str::trimEscape(std::string origin_str)
{
  std::string str = origin_str;
  str.erase(std::remove(str.begin(), str.end(), '\\'), str.end());
  return str;
}

/**
 * @brief change [] to \[\]
 *
 * @param origin_str
 * @return std::string
 */
std::string Str::addBackslash(std::string origin_str)
{
  if (ieda::Str::contain(origin_str.c_str(), "[") && ieda::Str::contain(origin_str.c_str(), "]")) {
    std::string new_value_1 = replace(origin_str, R"(\[)", R"(\[)");
    std::string new_value_2 = replace(new_value_1, R"(\])", R"(\])");

    return new_value_2;
  }

  return origin_str;
}

/**
 * @brief change [] to \\[\\]
 *
 * @param origin_str
 * @return std::string
 */
std::string Str::addDoubleBackslash(std::string origin_str)
{
  std::string new_value_2;

  if (ieda::Str::contain(origin_str.c_str(), "[") && ieda::Str::contain(origin_str.c_str(), "]")) {
    std::string new_value_1 = replace(origin_str, R"(\[)", R"(\[)");
    new_value_2 = replace(new_value_1, R"(\])", R"(\])");
  }

  if (ieda::Str::contain(new_value_2.c_str(), "[") && ieda::Str::contain(new_value_2.c_str(), "]")) {
    std::string new_value_3 = replace(new_value_2, R"(\[)", R"(\[)");
    std::string new_value_4 = replace(new_value_3, R"(\])", R"(\])");

    return new_value_4;
  }

  return origin_str;
}

/**
 * @brief conate backslash str, such as
 *   WEN & !( \
 *  (BWEN[0]) & \
 *  (BWEN[1]))
 * 
 * @param original_str 
 * @return std::string 
 */
std::string Str::concateBackSlashStr(std::string original_str) {
  auto it = std::remove_if(original_str.begin(), original_str.end(),  
  [&original_str](char ch) {  
      if (ch == '\\') {
          return true;   
      }  
      return false;  
  });

  original_str.erase(it, original_str.end());
  return original_str;
}

}  // namespace ieda  
