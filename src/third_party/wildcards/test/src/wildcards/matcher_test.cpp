// Copyright Tomas Zeman 2019.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "wildcards/matcher.hpp"  // wildcards::literals, wildcards::make_matcher
#include "cx/array.hpp"           // cx::array
#include "cx/string_view.hpp"     // cx::literals

#include "catch.hpp"

TEST_CASE("wildcards::matcher is compliant", "[wildcards::matcher]")
{
  using wildcards::make_matcher;

  SECTION(R"(matching "H?llo,*W*!")")
  {
    constexpr auto pattern = make_matcher("H?llo,*W*!");

    static_assert(pattern.matches("Hello, World!"), "");
  }

  SECTION(R"(matching "H_llo,%W%!")")
  {
    constexpr auto pattern = make_matcher("H_llo,%W%!", {'%', '_', '\\'});

    static_assert(pattern.matches("Hello, World!"), "");
  }

  SECTION(R"(matching "11*7?"_sv)")
  {
    struct equal_to
    {
      constexpr bool operator()(int n, char c) const
      {
        return n + 48 == c;
      }
    };

    using namespace cx::literals;

    constexpr auto pattern = make_matcher("11*7?"_sv, equal_to());

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(pattern.matches(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}));
#else
    static_assert(pattern.matches(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}), "");
#endif
  }

  SECTION(R"(matching "11%7_"_sv)")
  {
    struct equal_to
    {
      constexpr bool operator()(int n, char c) const
      {
        return n + 48 == c;
      }
    };

    using namespace cx::literals;

    constexpr auto pattern = make_matcher("11%7_"_sv, {'%', '_', '\\'}, equal_to());

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(pattern.matches(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}));
#else
    static_assert(pattern.matches(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}), "");
#endif
  }

  SECTION(R"(matching "H?llo,*W*!")")
  {
    using namespace wildcards::literals;

    constexpr auto pattern = "H?llo,*W*!"_wc;

    static_assert(pattern.matches("Hello, World!"), "");
  }

  SECTION(R"(matching u"H?llo,*W*!")")
  {
    using namespace wildcards::literals;

    constexpr auto pattern = u"H?llo,*W*!"_wc;

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(pattern.matches(u"Hello, World!"));
#else
    static_assert(pattern.matches(u"Hello, World!"), "");
#endif
  }

  SECTION(R"(matching U"H?llo,*W*!")")
  {
    using namespace wildcards::literals;

    constexpr auto pattern = U"H?llo,*W*!"_wc;

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(pattern.matches(U"Hello, World!"));
#else
    static_assert(pattern.matches(U"Hello, World!"), "");
#endif
  }

  SECTION(R"(matching L"H?llo,*W*!")")
  {
    using namespace wildcards::literals;

    constexpr auto pattern = L"H?llo,*W*!"_wc;

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(pattern.matches(L"Hello, World!"));
#else
    static_assert(pattern.matches(L"Hello, World!"), "");
#endif
  }
}
