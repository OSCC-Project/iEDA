// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "wildcards/match.hpp"  // wildcards::cards, wildcards::cards_type,
                                // wildcards::detail::alt_end, wildcards::detail::is_alt,
                                // wildcards::detail::is_set, wildcards::detail::match_set,
                                // wildcards::detail::set_end, wildcards::match
#include "cx/array.hpp"         // cx::array
#include "cx/iterator.hpp"      // cx::begin, cx::end
#include "cx/string_view.hpp"   // cx::literals

#include "catch.hpp"

TEST_CASE("wildcards::detail::is_set() is compliant", "[wildcards::detail::is_set]")
{
  using wildcards::cards;
  using wildcards::cards_type;
  using wildcards::detail::is_set;

  SECTION("checking sets without not")
  {
    constexpr char pattern1[] = "[a]";
    constexpr char pattern2[] = "[abc]";

    constexpr char pattern3[] = "[]]";
    constexpr char pattern4[] = "[]a]";
    constexpr char pattern5[] = "[]abc]";
    constexpr char pattern6[] = "[][]";
    constexpr char pattern7[] = "[][a]";
    constexpr char pattern8[] = "[][abc]";

    constexpr char pattern9[] = "[[]";
    constexpr char pattern10[] = "[[a]";
    constexpr char pattern11[] = "[[abc]";

    static_assert(is_set(cx::begin(pattern1), cx::end(pattern1)), "");
    static_assert(is_set(cx::begin(pattern2), cx::end(pattern2)), "");

    static_assert(is_set(cx::begin(pattern3), cx::end(pattern3)), "");
    static_assert(is_set(cx::begin(pattern4), cx::end(pattern4)), "");
    static_assert(is_set(cx::begin(pattern5), cx::end(pattern5)), "");
    static_assert(is_set(cx::begin(pattern6), cx::end(pattern6)), "");
    static_assert(is_set(cx::begin(pattern7), cx::end(pattern7)), "");
    static_assert(is_set(cx::begin(pattern8), cx::end(pattern8)), "");

    static_assert(is_set(cx::begin(pattern9), cx::end(pattern9)), "");
    static_assert(is_set(cx::begin(pattern10), cx::end(pattern10)), "");
    static_assert(is_set(cx::begin(pattern11), cx::end(pattern11)), "");
  }

  SECTION("checking sets with not")
  {
    constexpr char pattern1[] = "[!a]";
    constexpr char pattern2[] = "[!abc]";

    constexpr char pattern3[] = "[!]]";
    constexpr char pattern4[] = "[!]a]";
    constexpr char pattern5[] = "[!]abc]";
    constexpr char pattern6[] = "[!][]";
    constexpr char pattern7[] = "[!][a]";
    constexpr char pattern8[] = "[!][abc]";

    constexpr char pattern9[] = "[![]";
    constexpr char pattern10[] = "[![a]";
    constexpr char pattern11[] = "[![abc]";

    static_assert(is_set(cx::begin(pattern1), cx::end(pattern1)), "");
    static_assert(is_set(cx::begin(pattern2), cx::end(pattern2)), "");

    static_assert(is_set(cx::begin(pattern3), cx::end(pattern3)), "");
    static_assert(is_set(cx::begin(pattern4), cx::end(pattern4)), "");
    static_assert(is_set(cx::begin(pattern5), cx::end(pattern5)), "");
    static_assert(is_set(cx::begin(pattern6), cx::end(pattern6)), "");
    static_assert(is_set(cx::begin(pattern7), cx::end(pattern7)), "");
    static_assert(is_set(cx::begin(pattern8), cx::end(pattern8)), "");

    static_assert(is_set(cx::begin(pattern9), cx::end(pattern9)), "");
    static_assert(is_set(cx::begin(pattern10), cx::end(pattern10)), "");
    static_assert(is_set(cx::begin(pattern11), cx::end(pattern11)), "");
  }

  SECTION("checking non-sets")
  {
    constexpr char pattern1[] = "";
    constexpr char pattern2[] = "a";
    constexpr char pattern3[] = "!";

    constexpr char pattern4[] = "[";
    constexpr char pattern5[] = "[a";
    constexpr char pattern6[] = "[]";
    constexpr char pattern7[] = "[]a";

    constexpr char pattern8[] = "[!";
    constexpr char pattern9[] = "[!a";
    constexpr char pattern10[] = "[!]";
    constexpr char pattern11[] = "[!]a";

    static_assert(!is_set(cx::begin(pattern1), cx::end(pattern1)), "");
    static_assert(!is_set(cx::begin(pattern2), cx::end(pattern2)), "");
    static_assert(!is_set(cx::begin(pattern3), cx::end(pattern3)), "");

    static_assert(!is_set(cx::begin(pattern4), cx::end(pattern4)), "");
    static_assert(!is_set(cx::begin(pattern5), cx::end(pattern5)), "");
    static_assert(!is_set(cx::begin(pattern6), cx::end(pattern6)), "");
    static_assert(!is_set(cx::begin(pattern7), cx::end(pattern7)), "");

    static_assert(!is_set(cx::begin(pattern8), cx::end(pattern8)), "");
    static_assert(!is_set(cx::begin(pattern9), cx::end(pattern9)), "");
    static_assert(!is_set(cx::begin(pattern10), cx::end(pattern10)), "");
    static_assert(!is_set(cx::begin(pattern11), cx::end(pattern11)), "");
  }

  SECTION("checking sets using standard and/or extented cards")
  {
    constexpr char pattern[] = "[a]";

    static_assert(!is_set(cx::begin(pattern), cx::end(pattern), cards_type::standard), "");
    static_assert(is_set(cx::begin(pattern), cx::end(pattern), cards_type::extended), "");
  }
}

TEST_CASE("wildcards::detail::set_end() is compliant", "[wildcards::detail::set_end]")
{
  using wildcards::cards;
  using wildcards::cards_type;
  using wildcards::detail::set_end;

  SECTION("skipping sets without not")
  {
    constexpr char pattern1[] = "[a]";
    constexpr char pattern2[] = "[abc]";

    constexpr char pattern3[] = "[]]";
    constexpr char pattern4[] = "[]a]";
    constexpr char pattern5[] = "[]abc]";
    constexpr char pattern6[] = "[][]";
    constexpr char pattern7[] = "[][a]";
    constexpr char pattern8[] = "[][abc]";

    constexpr char pattern9[] = "[[]";
    constexpr char pattern10[] = "[[a]";
    constexpr char pattern11[] = "[[abc]";

    // NOTE: The '- 1' is just because of the null character.

    static_assert(set_end(cx::begin(pattern1), cx::end(pattern1)) == cx::end(pattern1) - 1, "");
    static_assert(set_end(cx::begin(pattern2), cx::end(pattern2)) == cx::end(pattern2) - 1, "");

    static_assert(set_end(cx::begin(pattern3), cx::end(pattern3)) == cx::end(pattern3) - 1, "");
    static_assert(set_end(cx::begin(pattern4), cx::end(pattern4)) == cx::end(pattern4) - 1, "");
    static_assert(set_end(cx::begin(pattern5), cx::end(pattern5)) == cx::end(pattern5) - 1, "");
    static_assert(set_end(cx::begin(pattern6), cx::end(pattern6)) == cx::end(pattern6) - 1, "");
    static_assert(set_end(cx::begin(pattern7), cx::end(pattern7)) == cx::end(pattern7) - 1, "");
    static_assert(set_end(cx::begin(pattern8), cx::end(pattern8)) == cx::end(pattern8) - 1, "");

    static_assert(set_end(cx::begin(pattern9), cx::end(pattern9)) == cx::end(pattern9) - 1, "");
    static_assert(set_end(cx::begin(pattern10), cx::end(pattern10)) == cx::end(pattern10) - 1, "");
    static_assert(set_end(cx::begin(pattern11), cx::end(pattern11)) == cx::end(pattern11) - 1, "");
  }

  SECTION("skipping sets with not")
  {
    constexpr char pattern1[] = "[!a]";
    constexpr char pattern2[] = "[!abc]";

    constexpr char pattern3[] = "[!]]";
    constexpr char pattern4[] = "[!]a]";
    constexpr char pattern5[] = "[!]abc]";
    constexpr char pattern6[] = "[!][]";
    constexpr char pattern7[] = "[!][a]";
    constexpr char pattern8[] = "[!][abc]";

    constexpr char pattern9[] = "[![]";
    constexpr char pattern10[] = "[![a]";
    constexpr char pattern11[] = "[![abc]";

    // NOTE: The '- 1' is just because of the null character.

    static_assert(set_end(cx::begin(pattern1), cx::end(pattern1)) == cx::end(pattern1) - 1, "");
    static_assert(set_end(cx::begin(pattern2), cx::end(pattern2)) == cx::end(pattern2) - 1, "");

    static_assert(set_end(cx::begin(pattern3), cx::end(pattern3)) == cx::end(pattern3) - 1, "");
    static_assert(set_end(cx::begin(pattern4), cx::end(pattern4)) == cx::end(pattern4) - 1, "");
    static_assert(set_end(cx::begin(pattern5), cx::end(pattern5)) == cx::end(pattern5) - 1, "");
    static_assert(set_end(cx::begin(pattern6), cx::end(pattern6)) == cx::end(pattern6) - 1, "");
    static_assert(set_end(cx::begin(pattern7), cx::end(pattern7)) == cx::end(pattern7) - 1, "");
    static_assert(set_end(cx::begin(pattern8), cx::end(pattern8)) == cx::end(pattern8) - 1, "");

    static_assert(set_end(cx::begin(pattern9), cx::end(pattern9)) == cx::end(pattern9) - 1, "");
    static_assert(set_end(cx::begin(pattern10), cx::end(pattern10)) == cx::end(pattern10) - 1, "");
    static_assert(set_end(cx::begin(pattern11), cx::end(pattern11)) == cx::end(pattern11) - 1, "");
  }

  SECTION("skipping non-sets")
  {
    const char pattern1[] = "";
    const char pattern2[] = "a";
    const char pattern3[] = "!";

    const char pattern4[] = "[";
    const char pattern5[] = "[a";
    const char pattern6[] = "[]";
    const char pattern7[] = "[]a";

    const char pattern8[] = "[!";
    const char pattern9[] = "[!a";
    const char pattern10[] = "[!]";
    const char pattern11[] = "[!]a";

    REQUIRE_THROWS(set_end(cx::begin(pattern1), cx::end(pattern1)));
    REQUIRE_THROWS(set_end(cx::begin(pattern2), cx::end(pattern2)));
    REQUIRE_THROWS(set_end(cx::begin(pattern3), cx::end(pattern3)));

    REQUIRE_THROWS(set_end(cx::begin(pattern4), cx::end(pattern4)));
    REQUIRE_THROWS(set_end(cx::begin(pattern5), cx::end(pattern5)));
    REQUIRE_THROWS(set_end(cx::begin(pattern6), cx::end(pattern6)));
    REQUIRE_THROWS(set_end(cx::begin(pattern7), cx::end(pattern7)));

    REQUIRE_THROWS(set_end(cx::begin(pattern8), cx::end(pattern8)));
    REQUIRE_THROWS(set_end(cx::begin(pattern9), cx::end(pattern9)));
    REQUIRE_THROWS(set_end(cx::begin(pattern10), cx::end(pattern10)));
    REQUIRE_THROWS(set_end(cx::begin(pattern11), cx::end(pattern11)));
  }

  SECTION("skipping sets using standard and/or extented cards")
  {
    const char pattern[] = "[a]";

    // NOTE: The '- 1' is just because of the null character.

    REQUIRE_THROWS(set_end(cx::begin(pattern), cx::end(pattern), cards_type::standard));
    REQUIRE(set_end(cx::begin(pattern), cx::end(pattern), cards_type::extended) ==
            cx::end(pattern) - 1);
  }
}

TEST_CASE("wildcards::detail::match_set() is compliant", "[wildcards::detail::match_set]")
{
  using wildcards::cards;
  using wildcards::cards_type;
  using wildcards::detail::match_set;

  SECTION(R"(matching "[a]")")
  {
    constexpr char pattern[] = "[a]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "a";
    constexpr char seq3[] = "b";

    static_assert(!match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[abc]")")
  {
    constexpr char pattern[] = "[abc]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "a";
    constexpr char seq3[] = "b";
    constexpr char seq4[] = "c";
    constexpr char seq5[] = "d";

    static_assert(!match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq4), cx::end(seq4), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq5), cx::end(seq5), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[]]")")
  {
    constexpr char pattern[] = "[]]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "]";
    constexpr char seq3[] = "a";

    static_assert(!match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[]abc]")")
  {
    constexpr char pattern[] = "[]abc]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "]";
    constexpr char seq3[] = "a";
    constexpr char seq4[] = "b";
    constexpr char seq5[] = "c";
    constexpr char seq6[] = "d";

    static_assert(!match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq4), cx::end(seq4), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq5), cx::end(seq5), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq6), cx::end(seq6), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[!a]")")
  {
    constexpr char pattern[] = "[!a]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "a";
    constexpr char seq3[] = "b";

    // We really need to exclude the null character here using '- 1'.
    // It is not necessary to do so for the other tests.
    static_assert(
        !match_set(cx::begin(seq1), cx::end(seq1) - 1, cx::begin(pattern), cx::end(pattern)), "");
    static_assert(!match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[!abc]")")
  {
    constexpr char pattern[] = "[!abc]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "a";
    constexpr char seq3[] = "b";
    constexpr char seq4[] = "c";
    constexpr char seq5[] = "d";

    // We really need to exclude the null character here using '- 1'.
    // It is not necessary to do so for the other tests.
    static_assert(
        !match_set(cx::begin(seq1), cx::end(seq1) - 1, cx::begin(pattern), cx::end(pattern)), "");
    static_assert(!match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq4), cx::end(seq4), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq5), cx::end(seq5), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[!]]")")
  {
    constexpr char pattern[] = "[!]]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "]";
    constexpr char seq3[] = "a";

    // We really need to exclude the null character here using '- 1'.
    // It is not necessary to do so for the other tests.
    static_assert(
        !match_set(cx::begin(seq1), cx::end(seq1) - 1, cx::begin(pattern), cx::end(pattern)), "");
    static_assert(!match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION(R"(matching "[!]abc]")")
  {
    constexpr char pattern[] = "[!]abc]";

    constexpr char seq1[] = "";
    constexpr char seq2[] = "]";
    constexpr char seq3[] = "a";
    constexpr char seq4[] = "b";
    constexpr char seq5[] = "c";
    constexpr char seq6[] = "d";

    // We really need to exclude the null character here using '- 1'.
    // It is not necessary to do so for the other tests.
    static_assert(
        !match_set(cx::begin(seq1), cx::end(seq1) - 1, cx::begin(pattern), cx::end(pattern)), "");
    static_assert(!match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq3), cx::end(seq3), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq4), cx::end(seq4), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(!match_set(cx::begin(seq5), cx::end(seq5), cx::begin(pattern), cx::end(pattern)),
                  "");
    static_assert(match_set(cx::begin(seq6), cx::end(seq6), cx::begin(pattern), cx::end(pattern)),
                  "");
  }

  SECTION("matching non-sets")
  {
    const char pattern1[] = "";
    const char pattern2[] = "a";
    const char pattern3[] = "!";

    const char pattern4[] = "[";
    const char pattern5[] = "[a";
    const char pattern6[] = "[]";
    const char pattern7[] = "[]a";

    const char pattern8[] = "[!";
    const char pattern9[] = "[!a";
    const char pattern10[] = "[!]";
    const char pattern11[] = "[!]a";

    const char seq1[] = "a";
    const char seq2[] = "b";

    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern1), cx::end(pattern1)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern2), cx::end(pattern2)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern3), cx::end(pattern3)));

    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern4), cx::end(pattern4)));
    REQUIRE(match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern5), cx::end(pattern5)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern5), cx::end(pattern5)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern6), cx::end(pattern6)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern6), cx::end(pattern6)));
    REQUIRE(match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern7), cx::end(pattern7)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern7), cx::end(pattern7)));

    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern8), cx::end(pattern8)));
    REQUIRE(!match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern9), cx::end(pattern9)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern9), cx::end(pattern9)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern10), cx::end(pattern10)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern10), cx::end(pattern10)));
    REQUIRE(!match_set(cx::begin(seq1), cx::end(seq1), cx::begin(pattern11), cx::end(pattern11)));
    REQUIRE_THROWS(
        match_set(cx::begin(seq2), cx::end(seq2), cx::begin(pattern11), cx::end(pattern11)));
  }

  SECTION("matching sets using standard and/or extented cards")
  {
    const char pattern[] = "[a]";

    const char seq[] = "a";

    REQUIRE_THROWS(!match_set(cx::begin(seq), cx::end(seq), cx::begin(pattern), cx::end(pattern),
                              cards_type::standard));
    REQUIRE(match_set(cx::begin(seq), cx::end(seq), cx::begin(pattern), cx::end(pattern),
                      cards_type::extended));
  }
}

TEST_CASE("wildcards::detail::is_alt() is compliant", "[wildcards::detail::is_alt]")
{
  using wildcards::cards;
  using wildcards::cards_type;
  using wildcards::detail::is_alt;

  SECTION("checking alternatives")
  {
    constexpr char pattern1[] = "()";
    constexpr char pattern2[] = "(a)";
    constexpr char pattern3[] = "(())";
    constexpr char pattern4[] = "((a))";
    constexpr char pattern5[] = "(a()a)";
    constexpr char pattern6[] = "(a(a)a)";

    constexpr char pattern7[] = "(\\()";
    constexpr char pattern8[] = "([(])";

    constexpr char pattern9[] = R"((a\*\(b+c\)/(sin\([abc]\)|[!abc])))";

    static_assert(is_alt(cx::begin(pattern1), cx::end(pattern1)), "");
    static_assert(is_alt(cx::begin(pattern2), cx::end(pattern2)), "");
    static_assert(is_alt(cx::begin(pattern3), cx::end(pattern3)), "");
    static_assert(is_alt(cx::begin(pattern4), cx::end(pattern4)), "");
    static_assert(is_alt(cx::begin(pattern5), cx::end(pattern5)), "");
    static_assert(is_alt(cx::begin(pattern6), cx::end(pattern6)), "");

    static_assert(is_alt(cx::begin(pattern7), cx::end(pattern7)), "");
    static_assert(is_alt(cx::begin(pattern8), cx::end(pattern8)), "");

    static_assert(is_alt(cx::begin(pattern9), cx::end(pattern9)), "");
  }

  SECTION("checking non-alternatives")
  {
    constexpr char pattern1[] = "";
    constexpr char pattern2[] = "a";
    constexpr char pattern3[] = "|";

    constexpr char pattern4[] = "(";
    constexpr char pattern5[] = "(a";
    constexpr char pattern6[] = "(|";
    constexpr char pattern7[] = "(()";
    constexpr char pattern8[] = "((a)";
    constexpr char pattern9[] = "(a()";
    constexpr char pattern10[] = "(a(a)";

    static_assert(!is_alt(cx::begin(pattern1), cx::end(pattern1)), "");
    static_assert(!is_alt(cx::begin(pattern2), cx::end(pattern2)), "");
    static_assert(!is_alt(cx::begin(pattern3), cx::end(pattern3)), "");

    static_assert(!is_alt(cx::begin(pattern4), cx::end(pattern4)), "");
    static_assert(!is_alt(cx::begin(pattern5), cx::end(pattern5)), "");
    static_assert(!is_alt(cx::begin(pattern6), cx::end(pattern6)), "");
    static_assert(!is_alt(cx::begin(pattern7), cx::end(pattern7)), "");
    static_assert(!is_alt(cx::begin(pattern8), cx::end(pattern8)), "");
    static_assert(!is_alt(cx::begin(pattern9), cx::end(pattern9)), "");
    static_assert(!is_alt(cx::begin(pattern10), cx::end(pattern10)), "");
  }

  SECTION("checking alternatives using standard and/or extented cards")
  {
    constexpr char pattern[] = "(a)";

    static_assert(!is_alt(cx::begin(pattern), cx::end(pattern), cards_type::standard), "");
    static_assert(is_alt(cx::begin(pattern), cx::end(pattern), cards_type::extended), "");
  }
}

TEST_CASE("wildcards::detail::alt_end() is compliant", "[wildcards::detail::alt_end]")
{
  using wildcards::cards;
  using wildcards::cards_type;
  using wildcards::detail::alt_end;

  SECTION("skipping alternatives")
  {
    constexpr char pattern1[] = "()";
    constexpr char pattern2[] = "(a)";
    constexpr char pattern3[] = "(())";
    constexpr char pattern4[] = "((a))";
    constexpr char pattern5[] = "(a()a)";
    constexpr char pattern6[] = "(a(a)a)";

    constexpr char pattern7[] = "(\\()";
    constexpr char pattern8[] = "([(])";

    constexpr char pattern9[] = R"((a\*\(b+c\)/(sin\([abc]\)|[!abc])))";

    // NOTE: The '- 1' is just because of the null character.

    static_assert(alt_end(cx::begin(pattern1), cx::end(pattern1)) == cx::end(pattern1) - 1, "");
    static_assert(alt_end(cx::begin(pattern2), cx::end(pattern2)) == cx::end(pattern2) - 1, "");
    static_assert(alt_end(cx::begin(pattern3), cx::end(pattern3)) == cx::end(pattern3) - 1, "");
    static_assert(alt_end(cx::begin(pattern4), cx::end(pattern4)) == cx::end(pattern4) - 1, "");
    static_assert(alt_end(cx::begin(pattern5), cx::end(pattern5)) == cx::end(pattern5) - 1, "");
    static_assert(alt_end(cx::begin(pattern6), cx::end(pattern6)) == cx::end(pattern6) - 1, "");

    static_assert(alt_end(cx::begin(pattern7), cx::end(pattern7)) == cx::end(pattern7) - 1, "");
    static_assert(alt_end(cx::begin(pattern8), cx::end(pattern8)) == cx::end(pattern8) - 1, "");

    static_assert(alt_end(cx::begin(pattern9), cx::end(pattern9)) == cx::end(pattern9) - 1, "");
  }

  SECTION("skipping non-alternatives")
  {
    char pattern1[] = "";
    char pattern2[] = "a";
    char pattern3[] = "|";

    char pattern4[] = "(";
    char pattern5[] = "(a";
    char pattern6[] = "(|";
    char pattern7[] = "(()";
    char pattern8[] = "((a)";
    char pattern9[] = "(a()";
    char pattern10[] = "(a(a)";

    REQUIRE_THROWS(alt_end(cx::begin(pattern1), cx::end(pattern1)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern2), cx::end(pattern2)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern3), cx::end(pattern3)));

    REQUIRE_THROWS(alt_end(cx::begin(pattern4), cx::end(pattern4)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern5), cx::end(pattern5)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern6), cx::end(pattern6)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern7), cx::end(pattern7)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern8), cx::end(pattern8)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern9), cx::end(pattern9)));
    REQUIRE_THROWS(alt_end(cx::begin(pattern10), cx::end(pattern10)));
  }

  SECTION("skipping alternatives using standard and/or extented cards")
  {
    char pattern[] = "(a)";

    // NOTE: The '- 1' is just because of the null character.

    REQUIRE_THROWS(alt_end(cx::begin(pattern), cx::end(pattern), cards_type::standard));
    REQUIRE(alt_end(cx::begin(pattern), cx::end(pattern), cards_type::extended) ==
            cx::end(pattern) - 1);
  }
}

TEST_CASE("wildcards::match() is compliant", "[wildcards::match]")
{
  using wildcards::match;

  SECTION(R"(matching "")")
  {
    constexpr char pattern1[] = "";
    constexpr char pattern2[] = R"(\)";

    static_assert(match("", pattern1), "");
    static_assert(match("", pattern2), "");

    static_assert(!match("Anything", pattern1), "");
    static_assert(!match("Anything", pattern2), "");
  }

  SECTION(R"(matching "A")")
  {
    constexpr char pattern1[] = "A";
    constexpr char pattern2[] = R"(A\)";
    constexpr char pattern3[] = R"(\A)";
    constexpr char pattern4[] = "[A]";
    constexpr char pattern5[] = "(A)";
    constexpr char pattern6[] = R"((\A))";
    constexpr char pattern7[] = R"(([A]))";

    static_assert(match("A", pattern1), "");
    static_assert(match("A", pattern2), "");
    static_assert(match("A", pattern3), "");
    static_assert(match("A", pattern4), "");
    static_assert(match("A", pattern5), "");
    static_assert(match("A", pattern6), "");
    static_assert(match("A", pattern7), "");

    static_assert(!match("", pattern1), "");
    static_assert(!match("", pattern2), "");
    static_assert(!match("", pattern3), "");
    static_assert(!match("", pattern4), "");
    static_assert(!match("", pattern5), "");
    static_assert(!match("", pattern6), "");
    static_assert(!match("", pattern7), "");

    static_assert(!match("a", pattern1), "");
    static_assert(!match("a", pattern2), "");
    static_assert(!match("a", pattern3), "");
    static_assert(!match("a", pattern4), "");
    static_assert(!match("a", pattern5), "");
    static_assert(!match("a", pattern6), "");
    static_assert(!match("a", pattern7), "");

    static_assert(!match("AA", pattern1), "");
    static_assert(!match("AA", pattern2), "");
    static_assert(!match("AA", pattern3), "");
    static_assert(!match("AA", pattern4), "");
    static_assert(!match("AA", pattern5), "");
    static_assert(!match("AA", pattern6), "");
    static_assert(!match("AA", pattern7), "");

    static_assert(!match("Something", pattern1), "");
    static_assert(!match("Something", pattern2), "");
    static_assert(!match("Something", pattern3), "");
    static_assert(!match("Something", pattern4), "");
    static_assert(!match("Something", pattern5), "");
    static_assert(!match("Something", pattern6), "");
    static_assert(!match("Something", pattern7), "");
  }

  SECTION(R"(matching "Hello!")")
  {
    constexpr char pattern1[] = "Hello!";
    constexpr char pattern2[] = R"(Hello!\)";
    constexpr char pattern3[] = R"(\H\e\l\l\o\!)";
    constexpr char pattern4[] = "[H][e][l][l][o]!";
    constexpr char pattern5[] = "(Hello!)";
    constexpr char pattern6[] = R"((\H\e\l\l\o\!))";
    constexpr char pattern7[] = R"(([H][e][l][l][o]!))";

    static_assert(match("Hello!", pattern1), "");
    static_assert(match("Hello!", pattern2), "");
    static_assert(match("Hello!", pattern3), "");
    static_assert(match("Hello!", pattern4), "");
    static_assert(match("Hello!", pattern5), "");
    static_assert(match("Hello!", pattern6), "");
    static_assert(match("Hello!", pattern7), "");

    static_assert(!match("", pattern1), "");
    static_assert(!match("", pattern2), "");
    static_assert(!match("", pattern3), "");
    static_assert(!match("", pattern4), "");
    static_assert(!match("", pattern5), "");
    static_assert(!match("", pattern6), "");
    static_assert(!match("", pattern7), "");

    static_assert(!match("Hallo!", pattern1), "");
    static_assert(!match("Hallo!", pattern2), "");
    static_assert(!match("Hallo!", pattern3), "");
    static_assert(!match("Hallo!", pattern4), "");
    static_assert(!match("Hallo!", pattern5), "");
    static_assert(!match("Hallo!", pattern6), "");
    static_assert(!match("Hallo!", pattern7), "");

    static_assert(!match("HHello!", pattern1), "");
    static_assert(!match("HHello!", pattern2), "");
    static_assert(!match("HHello!", pattern3), "");
    static_assert(!match("HHello!", pattern4), "");
    static_assert(!match("HHello!", pattern5), "");
    static_assert(!match("HHello!", pattern6), "");
    static_assert(!match("HHello!", pattern7), "");

    static_assert(!match("Hello!!", pattern1), "");
    static_assert(!match("Hello!!", pattern2), "");
    static_assert(!match("Hello!!", pattern3), "");
    static_assert(!match("Hello!!", pattern4), "");
    static_assert(!match("Hello!!", pattern5), "");
    static_assert(!match("Hello!!", pattern6), "");
    static_assert(!match("Hello!!", pattern7), "");

    static_assert(!match("Hello!Hello!", pattern1), "");
    static_assert(!match("Hello!Hello!", pattern2), "");
    static_assert(!match("Hello!Hello!", pattern3), "");
    static_assert(!match("Hello!Hello!", pattern4), "");
    static_assert(!match("Hello!Hello!", pattern5), "");
    static_assert(!match("Hello!Hello!", pattern6), "");
    static_assert(!match("Hello!Hello!", pattern7), "");
  }

  SECTION(R"(matching "*")")
  {
    constexpr char pattern1[] = "*";
    constexpr char pattern2[] = R"(*\)";
    constexpr char pattern3[] = R"(\*)";
    constexpr char pattern4[] = "[*]";
    constexpr char pattern5[] = "(*)";
    constexpr char pattern6[] = R"((\*))";
    constexpr char pattern7[] = "([*])";

    static_assert(match("", pattern1), "");
    static_assert(match("", pattern2), "");
    static_assert(!match("", pattern3), "");
    static_assert(!match("", pattern4), "");
    static_assert(match("", pattern5), "");
    static_assert(!match("", pattern6), "");
    static_assert(!match("", pattern7), "");

    static_assert(match("*", pattern1), "");
    static_assert(match("*", pattern2), "");
    static_assert(match("*", pattern3), "");
    static_assert(match("*", pattern4), "");
    // FIXME: Would work without the null character.
    // But in general, it is not a good idea to use '(*)'.
    // static_assert(match("*", pattern5), "");
    static_assert(match("*", pattern6), "");
    static_assert(match("*", pattern7), "");

    static_assert(match("Anything", pattern1), "");
    static_assert(match("Anything", pattern2), "");
    static_assert(!match("Anything", pattern3), "");
    static_assert(!match("Anything", pattern4), "");
    // FIXME: Would work without the null character.
    // But in general, it is not a good idea to use '(*)'.
    // static_assert(match("Anything", pattern5), "");
    static_assert(!match("Anything", pattern6), "");
    static_assert(!match("Anything", pattern7), "");
  }

  SECTION(R"(matching "?")")
  {
    constexpr char pattern1[] = "?";
    constexpr char pattern2[] = R"(?\)";
    constexpr char pattern3[] = R"(\?)";
    constexpr char pattern4[] = "[?]";
    constexpr char pattern5[] = "(?)";
    constexpr char pattern6[] = R"((\?))";
    constexpr char pattern7[] = R"([?])";

    static_assert(match("A", pattern1), "");
    static_assert(match("A", pattern2), "");
    static_assert(!match("A", pattern3), "");
    static_assert(!match("A", pattern4), "");
    static_assert(match("A", pattern5), "");
    static_assert(!match("A", pattern6), "");
    static_assert(!match("A", pattern7), "");

    static_assert(match("a", pattern1), "");
    static_assert(match("a", pattern2), "");
    static_assert(!match("a", pattern3), "");
    static_assert(!match("a", pattern4), "");
    static_assert(match("a", pattern5), "");
    static_assert(!match("a", pattern6), "");
    static_assert(!match("a", pattern7), "");

    static_assert(match("?", pattern1), "");
    static_assert(match("?", pattern2), "");
    static_assert(match("?", pattern3), "");
    static_assert(match("?", pattern4), "");
    static_assert(match("?", pattern5), "");
    static_assert(match("?", pattern6), "");
    static_assert(match("?", pattern7), "");

    static_assert(!match("", pattern1), "");
    static_assert(!match("", pattern2), "");
    static_assert(!match("", pattern3), "");
    static_assert(!match("", pattern4), "");
    static_assert(!match("", pattern5), "");
    static_assert(!match("", pattern6), "");
    static_assert(!match("", pattern7), "");

    static_assert(!match("Something", pattern1), "");
    static_assert(!match("Something", pattern2), "");
    static_assert(!match("Something", pattern3), "");
    static_assert(!match("Something", pattern4), "");
    static_assert(!match("Something", pattern5), "");
    static_assert(!match("Something", pattern6), "");
    static_assert(!match("Something", pattern7), "");
  }

  SECTION(R"(matching "H?llo,*W*!")")
  {
    constexpr char pattern[] = "H?llo,*W*!";

    static_assert(match("Hello, World!", pattern), "");
    static_assert(match("Hallo, World!", pattern), "");
    static_assert(match("Hallo,World!", pattern), "");
    static_assert(match("Hallo,WildCards!", pattern), "");
    static_assert(match("Hallo, crazy WildCards!", pattern), "");
    static_assert(match("Hallo, crazy WildCards! Still working?!", pattern), "");

    static_assert(!match("", pattern), "");
    static_assert(!match("Hllo, World!", pattern), "");
    static_assert(!match("Hello, World?", pattern), "");
    static_assert(!match("Hello, world!", pattern), "");
    static_assert(!match("Yes. Hello, World!", pattern), "");
    static_assert(!match("Hello, World!?", pattern), "");
  }

  SECTION(R"zzz(matching R"(\\\* *\? \*\\)")zzz")
  {
    constexpr char pattern1[] = R"(\\\* *\? \*\\)";
    constexpr char pattern2[] = R"([\][*] *[?] [*][\])";

    static_assert(match(R"(\* Hello? *\)", pattern1), "");
    static_assert(match(R"(\* Hello? *\)", pattern2), "");

    static_assert(match(R"(\* Hi? *\)", pattern1), "");
    static_assert(match(R"(\* Hi? *\)", pattern2), "");

    static_assert(match(R"(\* ? *\)", pattern1), "");
    static_assert(match(R"(\* ? *\)", pattern2), "");

    static_assert(!match(R"(\* Hello! *\)", pattern1), "");
    static_assert(!match(R"(\* Hello! *\)", pattern2), "");

    static_assert(!match(R"(* Hello? *\)", pattern1), "");
    static_assert(!match(R"(* Hello? *\)", pattern2), "");

    static_assert(!match(R"(\ Hello? *\)", pattern1), "");
    static_assert(!match(R"(\ Hello? *\)", pattern2), "");

    static_assert(!match(R"( Hello? *\)", pattern1), "");
    static_assert(!match(R"( Hello? *\)", pattern2), "");
  }

  SECTION(R"(matching u"H?llo,*W*!")")
  {
    constexpr char16_t pattern[] = u"H?llo,*W*!";

    static_assert(match(u"Hello, World!", pattern), "");
  }

  SECTION(R"(matching U"H?llo,*W*!")")
  {
    constexpr char32_t pattern[] = U"H?llo,*W*!";

    static_assert(match(U"Hello, World!", pattern), "");
  }

  SECTION(R"(matching L"H?llo,*W*!")")
  {
    constexpr wchar_t pattern[] = L"H?llo,*W*!";

    static_assert(match(L"Hello, World!", pattern), "");
  }

  SECTION(R"(matching "H_llo,%W%!")")
  {
    constexpr char pattern[] = "H_llo,%W%!";

    static_assert(match("Hello, World!", pattern, {'%', '_', '\\'}), "");
  }

  SECTION(R"(matching "H?llo,*W*!"_sv)")
  {
    using namespace cx::literals;

    constexpr auto pattern = "H?llo,*W*!"_sv;

    static_assert(match("Hello, World!"_sv, pattern), "");
  }

  SECTION(R"(matching u"H?llo,*W*!"_sv)")
  {
    using namespace cx::literals;

    constexpr auto pattern = u"H?llo,*W*!"_sv;

    static_assert(match(u"Hello, World!"_sv, pattern), "");
  }

  SECTION(R"(matching U"H?llo,*W*!"_sv)")
  {
    using namespace cx::literals;

    constexpr auto pattern = U"H?llo,*W*!"_sv;

    static_assert(match(U"Hello, World!"_sv, pattern), "");
  }

  SECTION(R"(matching L"H?llo,*W*!"_sv)")
  {
    using namespace cx::literals;

    constexpr auto pattern = L"H?llo,*W*!"_sv;

    static_assert(match(L"Hello, World!"_sv, pattern), "");
  }

  SECTION(R"(matching "H_llo,%W%!"_sv)")
  {
    using namespace cx::literals;

    constexpr auto pattern = "H_llo,%W%!"_sv;

    static_assert(match("Hello, World!"_sv, pattern, {'%', '_', '\\'}), "");
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

    constexpr auto pattern = "11*7?"_sv;

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(match(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}, pattern, equal_to()));
#else
    static_assert(match(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}, pattern, equal_to()), "");
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

    constexpr auto pattern = "11%7_"_sv;

#if defined(_MSC_VER) && _MSC_VER <= 1900  // VS2015
    REQUIRE(match(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}, pattern, {'%', '_', '\\'}, equal_to()));
#else
    static_assert(
        match(cx::array<int, 6>{{1, 1, 3, 5, 7, 9}}, pattern, {'%', '_', '\\'}, equal_to()), "");
#endif
  }

  SECTION("matching sets")
  {
    static_assert(match("aaa", "a[abc]a"), "");
    static_assert(!match("aaa", "a[bcd]a"), "");
    static_assert(!match("aaa", "a[a]]a"), "");
    static_assert(match("aa]a", "a[a]]a"), "");
    static_assert(match("aaa", "a[]abc]a"), "");
    static_assert(match("aaa", "a[[a]a"), "");
    static_assert(match("a[a", "a[[a]a"), "");
    static_assert(match("a]a", "a[]]a"), "");
    static_assert(!match("aa", "a[]a"), "");
    static_assert(match("a[]a", "a[]a"), "");

    static_assert(!match("aaa", "a[!a]a"), "");
    static_assert(match("aaa", "a[!b]a"), "");
    static_assert(!match("aaa", "a[b!b]a"), "");
    static_assert(match("a!a", "a[b!b]a"), "");
    static_assert(!match("a!a", "a[!]a"), "");
    static_assert(match("a[!]a", "a[!]a"), "");
  }

  SECTION("matching alternatives")
  {
    static_assert(match("aXb", "a(X|Y)b"), "");
    static_assert(match("aYb", "a(X|Y)b"), "");
    static_assert(!match("aZb", "a(X|Y)b"), "");
    static_assert(match("aXb", "(a(X|Y)b|c)"), "");
    static_assert(!match("a", "a|b"), "");
    static_assert(match("a|b", "a|b"), "");
    static_assert(match("(aa", "(a(a|b)"), "");
    static_assert(!match("a(a", "(a(a|b)"), "");
    static_assert(match("a(a", "(a[(]a|b)"), "");
    static_assert(match("aa", "a()a"), "");
    static_assert(match("", "(abc|)"), "");
  }
}
