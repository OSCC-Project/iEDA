// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "cx/array.hpp"  // cx::array, cx::get

#include "catch.hpp"

TEST_CASE("cx::array<int, 0> is compliant", "[cx::array]")
{
  constexpr auto a1 = cx::array<int, 0>{};

  static_assert(a1.size() == 0, "");

  static_assert(a1.empty(), "");

#if !defined(_MSC_VER) || _MSC_VER > 1900  // !VS2015
  static_assert(a1.begin() == a1.end(), "");
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
  REQUIRE(cx::array<int, 0>{} == cx::array<int, 0>{});
#else
  static_assert(cx::array<int, 0>{} == cx::array<int, 0>{}, "");
#endif
}

TEST_CASE("cx::array<int, 1> is compliant", "[cx::array]")
{
  constexpr auto a1 = cx::array<int, 1>{};
  constexpr auto a2 = cx::array<int, 1>{{10}};

  static_assert(a1.size() == 1, "");
  static_assert(a2.size() == 1, "");

  static_assert(!a1.empty(), "");
  static_assert(!a2.empty(), "");

  static_assert(a1.begin() != a1.end(), "");
  static_assert(a2.begin() != a2.end(), "");

  static_assert(a2[0] == 10, "");

  // static_assert(a2.at(0) == 10, "");

  static_assert(cx::get<0>(a2) == 10, "");

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
  REQUIRE(cx::array<int, 1>{{10}} == cx::array<int, 1>{{10}});
  REQUIRE(cx::array<int, 1>{{10}} != cx::array<int, 1>{{20}});
#else
  static_assert(cx::array<int, 1>{{10}} == cx::array<int, 1>{{10}}, "");
  static_assert(cx::array<int, 1>{{10}} != cx::array<int, 1>{{20}}, "");
#endif
}

TEST_CASE("cx::array<int, 2> is compliant", "[cx::array]")
{
  constexpr auto a1 = cx::array<int, 2>{};
  constexpr auto a2 = cx::array<int, 2>{{10}};
  constexpr auto a3 = cx::array<int, 2>{{10, 20}};

  static_assert(a1.size() == 2, "");
  static_assert(a2.size() == 2, "");
  static_assert(a3.size() == 2, "");

  static_assert(!a1.empty(), "");
  static_assert(!a2.empty(), "");
  static_assert(!a3.empty(), "");

  static_assert(a1.begin() != a1.end(), "");
  static_assert(a2.begin() != a2.end(), "");
  static_assert(a3.begin() != a3.end(), "");

  static_assert(a2[0] == 10, "");
  static_assert(a3[0] == 10, "");
  static_assert(a3[1] == 20, "");

  // static_assert(a2.at(0) == 10, "");
  // static_assert(a3.at(0) == 10, "");
  // static_assert(a3.at(1) == 20, "");

  static_assert(cx::get<0>(a2) == 10, "");
  static_assert(cx::get<0>(a3) == 10, "");
  static_assert(cx::get<1>(a3) == 20, "");

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
  REQUIRE(cx::array<int, 2>{{10, 20}} == cx::array<int, 2>{{10, 20}});
  REQUIRE(cx::array<int, 2>{{10, 20}} != cx::array<int, 2>{{20, 10}});
#else
  static_assert(cx::array<int, 2>{{10, 20}} == cx::array<int, 2>{{10, 20}}, "");
  static_assert(cx::array<int, 2>{{10, 20}} != cx::array<int, 2>{{20, 10}}, "");
#endif
}
