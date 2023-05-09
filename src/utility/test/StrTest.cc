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
#include "absl/strings/str_split.h"
#include "gmock/gmock.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"
#include "string/Str.hh"
#include "string/StrMap.hh"

using ieda::Str;
using ieda::StrMap;

namespace {

TEST(StrTest, copy1) {
  const char* test = "str test.";
  char* new_test = Str::copy(test);
  std::cout << new_test << "\n";
}

TEST(StrTest, copy2) {
  const char* test = "str ";
  const char* test1 = "test.";
  char* new_test = Str::copy({test, test1});
  const char* test2 = "test.";
  std::cout << Str::copy({new_test, test2}) << "\n";
  Str::free(new_test);
}

TEST(StrTest, copy3) {
  const char* test = "\0";
  char* new_test = Str::copy(test);
  std::cout << new_test << "\n";
  Str::free(new_test);
}

TEST(StrTest, copy4) {
  const char* test = nullptr;
  char* new_test = Str::copy(test);
  std::cout << new_test << "\n";
  Str::free(new_test);
}

TEST(StrTest, copy5) {
  const char* test = "str ";
  const char* test1 = nullptr;
  char* new_test = Str::copy({test, test1});
  const char* test2 = "test.";
  std::cout << Str::copy({new_test, test2}) << "\n";
  Str::free(new_test);
}

TEST(StrTest, release1) {
  const char* test = "\0";
  Str::free(test);
}

TEST(StrTest, release2) {
  const char* test = nullptr;
  Str::free(test);
}

TEST(StrTest, cmp) {
  const char* test1 = "test1";
  const char* test2 = "test2";

  int result = Str::caseCmp(test1, test2);

  EXPECT_EQ(result, -1);
}

TEST(StrTest, printf) {
  char* test = Str::printf("%s%d", "123", 4);
  EXPECT_STREQ(test, "1234");

  char* test1 = Str::printf("%s%d", "567", 8);
  EXPECT_STREQ(test1, "5678");
}

TEST(StrTest, toUpper) {
  const char* test1 = "test1";
  char* test2 = Str::toUpper(test1);

  EXPECT_STREQ(test2, "TEST1");
}

TEST(StrTest, toLower) {
  const char* test1 = "TEST1";
  char* test2 = Str::toLower(test1);

  EXPECT_STREQ(test2, "test1");
}

TEST(StrTest, toDouble) {
  const char* test1 = "1.2";
  double result = Str::toDouble(test1);

  EXPECT_EQ(result, 1.2);
}

TEST(StrTest, trimmed) {
  const char* test = "\\abc[1] ";
  const char* test1 = Str::trimmed(test);

  EXPECT_STREQ(test1, "abc[1]");
}

TEST(StrTest, nocase) {
  const char* test1 = "abc";
  const char* test2 = "ABC";

  EXPECT_TRUE(Str::noCaseEqual(test1, test2));
}

TEST(StrTest, split) {
  const char* test = "abc\\def\\gh";
  auto strs = Str::split(test, "\\");

  EXPECT_STREQ(strs[0].c_str(), "abc");
  EXPECT_STREQ(strs[1].c_str(), "def");
  EXPECT_STREQ(strs[2].c_str(), "gh");
}

TEST(StrTest, split1) {
  auto strs = Str::split("u0_rcg/u0_pll/CLK_OUT", "/");

  EXPECT_STREQ(strs[0].c_str(), "u0_rcg");
  EXPECT_STREQ(strs[1].c_str(), "u0_pll");
  EXPECT_STREQ(strs[2].c_str(), "CLK_OUT");
}

TEST(StrTest, join) {
  const char* test = "abc\\def\\gh";
  auto strs = Str::split(test, "\\");
  strs.pop_back();
  auto new_str = Str::join(strs, "\\");

  EXPECT_STREQ(new_str.c_str(), "abc\\def");
}

TEST(StrTest, str_map) {
  StrMap<int> test_str_map;
  test_str_map.insert("test", 1);
  auto found = test_str_map.find("test");

  EXPECT_TRUE(found != test_str_map.end());
}

TEST(StrTest, pattern_match1) {
  const char* str = "D[1]";
  std::string regex_pattern = "([A-Za-z]+)\\[(\\d+)\\]";
  auto ret_val = Str::matchPattern(str, regex_pattern);
  for (auto val : ret_val) {
    std::cout << val << "\n";
  }
}

TEST(StrTest, pattern_match2) {
  const char* str = "test";
  std::string regex_pattern = "([a-z]+)\\[(\\d+)\\]";
  auto ret_val = Str::matchPattern(str, regex_pattern);
  for (auto val : ret_val) {
    std::cout << val << "\n";
  }
}

TEST(StrTest, bus_match) {
  const char* str = "D[2]";

  auto [base_name, index] = Str::matchBusName(str);

  EXPECT_EQ(base_name, "D");
  EXPECT_EQ(*index, 2);
}

TEST(StrTest, bus_match1) {
  const char* str = "test";

  auto [base_name, index] = Str::matchBusName(str);

  EXPECT_EQ(base_name, "test");
}

TEST(StrTest, reg_replace) {
  const char* str = "abc\\[1\\]";
  std::string new_str = Str::replace(str, R"(\\)", "");

  EXPECT_EQ(new_str, "abc[1]");
}

}  // namespace
