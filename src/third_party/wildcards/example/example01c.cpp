// Copyright Tomas Zeman 2019.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "wildcards/matcher.hpp"

int main()
{
  using namespace wildcards::literals;

  constexpr auto pattern = "(CMakeLists.txt|*.[hc](pp|))"_wc;

  static_assert(pattern.matches("header.h"), "");
  static_assert(pattern.matches("header.hpp"), "");
  static_assert(pattern.matches("source.c"), "");
  static_assert(pattern.matches("source.cpp"), "");
  static_assert(pattern.matches("CMakeLists.txt"), "");

  static_assert(!pattern.matches("header.H"), "");
  static_assert(!pattern.matches("source.cc"), "");
  static_assert(!pattern.matches("object.o"), "");
  static_assert(!pattern.matches("CMakeLists.txt.bak"), "");

  return 0;
}
