// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>  // std::is_same

#include "cx/utility.hpp"  // cx::get, cx::make_pair, cx::pair, cx::tuple_size

#include "catch.hpp"

TEST_CASE("cx::pair<int, char> is compliant", "[cx::pair]")
{
  constexpr auto p1 = cx::pair<int, char>{};
  constexpr auto p2 = cx::pair<int, char>{10, 'a'};
  constexpr auto p3 = cx::make_pair(10, 'a');

  static_assert(std::is_same<decltype(p1), decltype(p2)>::value, "");
  static_assert(std::is_same<decltype(p1), decltype(p3)>::value, "");

  static_assert(cx::tuple_size<decltype(p1)>::value == 2, "");

  static_assert(p2 == p3, "");

  static_assert(cx::get<0>(p2) == 10, "");
  static_assert(cx::get<1>(p2) == 'a', "");
  static_assert(cx::get<int>(p2) == 10, "");
  static_assert(cx::get<char>(p2) == 'a', "");

  static_assert(cx::make_pair(10, 'a') != cx::make_pair(20, 'a'), "");
  static_assert(cx::make_pair(10, 'a') != cx::make_pair(10, 'b'), "");
  static_assert(cx::make_pair(10, 'a') != cx::make_pair(20, 'b'), "");
}
