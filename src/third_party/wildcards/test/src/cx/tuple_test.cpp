// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>  // std::is_same

#include "cx/tuple.hpp"  // cx::get, cx::make_tuple, cx::tuple, cx::tuple_size

#include "catch.hpp"

TEST_CASE("cx::tuple<> is compliant", "[cx::tuple]")
{
  constexpr auto t1 = cx::tuple<>{};
  constexpr auto t2 = cx::make_tuple();

  static_assert(std::is_same<decltype(t1), decltype(t2)>::value, "");

  static_assert(cx::tuple_size<decltype(t1)>::value == 0, "");

  static_assert(t1 == t2, "");
}

TEST_CASE("cx::tuple<int> is compliant", "[cx::tuple]")
{
  constexpr auto t1 = cx::tuple<int>{};
  constexpr auto t2 = cx::tuple<int>{10};
  constexpr auto t3 = cx::make_tuple(10);

  static_assert(std::is_same<decltype(t1), decltype(t2)>::value, "");
  static_assert(std::is_same<decltype(t1), decltype(t3)>::value, "");

  static_assert(cx::tuple_size<decltype(t1)>::value == 1, "");

  static_assert(t2 == t3, "");

  static_assert(cx::get<0>(t2) == 10, "");

  static_assert(cx::make_tuple(10) != cx::make_tuple(20), "");
}

TEST_CASE("cx::tuple<int, char> is compliant", "[cx::tuple]")
{
  constexpr auto t1 = cx::tuple<int, char>{};
  constexpr auto t2 = cx::tuple<int, char>{10, 'a'};
  constexpr auto t3 = cx::make_tuple(10, 'a');

  static_assert(std::is_same<decltype(t1), decltype(t2)>::value, "");
  static_assert(std::is_same<decltype(t1), decltype(t3)>::value, "");

  static_assert(cx::tuple_size<decltype(t1)>::value == 2, "");

  static_assert(t2 == t3, "");

  static_assert(cx::get<0>(t2) == 10, "");
  static_assert(cx::get<1>(t2) == 'a', "");

  static_assert(cx::make_tuple(10, 'a') != cx::make_tuple(20, 'a'), "");
  static_assert(cx::make_tuple(10, 'a') != cx::make_tuple(10, 'b'), "");
  static_assert(cx::make_tuple(10, 'a') != cx::make_tuple(20, 'b'), "");
}

TEST_CASE("cx::tuple<int, char, double> is compliant", "[cx::tuple]")
{
  constexpr auto t1 = cx::tuple<int, char, unsigned>{};
  constexpr auto t2 = cx::tuple<int, char, unsigned>{10, 'a', 5u};
  constexpr auto t3 = cx::make_tuple(10, 'a', 5u);

  static_assert(std::is_same<decltype(t1), decltype(t2)>::value, "");
  static_assert(std::is_same<decltype(t1), decltype(t3)>::value, "");

  static_assert(cx::tuple_size<decltype(t1)>::value == 3, "");

  static_assert(t2 == t3, "");

  static_assert(cx::get<0>(t2) == 10, "");
  static_assert(cx::get<1>(t2) == 'a', "");
  static_assert(cx::get<2>(t2) == 5u, "");

  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(20, 'a', 5u), "");
  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(10, 'b', 5u), "");
  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(10, 'a', 6u), "");
  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(20, 'b', 5u), "");
  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(20, 'a', 6u), "");
  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(10, 'b', 6u), "");
  static_assert(cx::make_tuple(10, 'a', 5u) != cx::make_tuple(20, 'b', 6u), "");
}
