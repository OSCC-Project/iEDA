// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "wildcards/match.hpp"

int main()
{
  constexpr char pattern[] = "(CMakeLists.txt|*.[hc](pp|))";

  static_assert(wildcards::match("header.h", pattern), "");
  static_assert(wildcards::match("header.hpp", pattern), "");
  static_assert(wildcards::match("source.c", pattern), "");
  static_assert(wildcards::match("source.cpp", pattern), "");
  static_assert(wildcards::match("CMakeLists.txt", pattern), "");

  static_assert(!wildcards::match("header.H", pattern), "");
  static_assert(!wildcards::match("source.cc", pattern), "");
  static_assert(!wildcards::match("object.o", pattern), "");
  static_assert(!wildcards::match("CMakeLists.txt.bak", pattern), "");

  return 0;
}
