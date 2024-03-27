// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>  // std::is_same

#include "cx/string_view.hpp"  // cx::basic_string_view, cx::literals, cx::make_string_view,
                               // cx::string_view, cx::u16string_view, cx::u32string_view,
                               // cx::wstring_view

#include "catch.hpp"

TEST_CASE("cx::basic_string_view<char> is compliant", "[cx::string_view]")
{
  using namespace cx::literals;

  constexpr auto s1 = cx::basic_string_view<char>{"test"};
  constexpr auto s2 = cx::basic_string_view<char>{"testt", 4};
  constexpr auto s3 = cx::make_string_view("test");
  constexpr auto s4 = cx::make_string_view("testt", 4);
  constexpr auto s5 = cx::string_view{"test"};
  constexpr auto s6 = cx::string_view{"testt", 4};
  constexpr auto s7 = "test"_sv;

  static_assert(std::is_same<decltype(s1), decltype(s2)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s3)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s4)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s5)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s6)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s7)>::value, "");

  static_assert(s1.size() == 4, "");
  static_assert(s2.size() == 4, "");
  static_assert(s3.size() == 4, "");
  static_assert(s4.size() == 4, "");
  static_assert(s5.size() == 4, "");
  static_assert(s6.size() == 4, "");
  static_assert(s7.size() == 4, "");

  static_assert(!s1.empty(), "");
  static_assert(!s2.empty(), "");
  static_assert(!s3.empty(), "");
  static_assert(!s4.empty(), "");
  static_assert(!s5.empty(), "");
  static_assert(!s6.empty(), "");
  static_assert(!s7.empty(), "");

  static_assert(s1 == s2, "");
  static_assert(s1 == s3, "");
  static_assert(s1 == s4, "");
  static_assert(s1 == s5, "");
  static_assert(s1 == s6, "");
  static_assert(s1 == s7, "");

  static_assert(""_sv.size() == 0, "");
  static_assert(""_sv.empty(), "");

  static_assert(""_sv == ""_sv, "");
  static_assert("test"_sv != ""_sv, "");
  static_assert("test"_sv != "testt"_sv, "");
}

TEST_CASE("cx::basic_string_view<char16_t> is compliant", "[cx::string_view]")
{
  using namespace cx::literals;

  constexpr auto s1 = cx::basic_string_view<char16_t>{u"test"};
  constexpr auto s2 = cx::basic_string_view<char16_t>{u"testt", 4};
  constexpr auto s3 = cx::make_string_view(u"test");
  constexpr auto s4 = cx::make_string_view(u"testt", 4);
  constexpr auto s5 = cx::u16string_view{u"test"};
  constexpr auto s6 = cx::u16string_view{u"testt", 4};
  constexpr auto s7 = u"test"_sv;

  static_assert(std::is_same<decltype(s1), decltype(s2)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s3)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s4)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s5)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s6)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s7)>::value, "");

  static_assert(s1.size() == 4, "");
  static_assert(s2.size() == 4, "");
  static_assert(s3.size() == 4, "");
  static_assert(s4.size() == 4, "");
  static_assert(s5.size() == 4, "");
  static_assert(s6.size() == 4, "");
  static_assert(s7.size() == 4, "");

  static_assert(!s1.empty(), "");
  static_assert(!s2.empty(), "");
  static_assert(!s3.empty(), "");
  static_assert(!s4.empty(), "");
  static_assert(!s5.empty(), "");
  static_assert(!s6.empty(), "");
  static_assert(!s7.empty(), "");

  static_assert(s1 == s2, "");
  static_assert(s1 == s3, "");
  static_assert(s1 == s4, "");
  static_assert(s1 == s5, "");
  static_assert(s1 == s6, "");
  static_assert(s1 == s7, "");

  static_assert(u""_sv.size() == 0, "");
  static_assert(u""_sv.empty(), "");

  static_assert(u""_sv == u""_sv, "");
  static_assert(u"test"_sv != u""_sv, "");
  static_assert(u"test"_sv != u"testt"_sv, "");
}

TEST_CASE("cx::basic_string_view<char32_t> is compliant", "[cx::string_view]")
{
  using namespace cx::literals;

  constexpr auto s1 = cx::basic_string_view<char32_t>{U"test"};
  constexpr auto s2 = cx::basic_string_view<char32_t>{U"testt", 4};
  constexpr auto s3 = cx::make_string_view(U"test");
  constexpr auto s4 = cx::make_string_view(U"testt", 4);
  constexpr auto s5 = cx::u32string_view{U"test"};
  constexpr auto s6 = cx::u32string_view{U"testt", 4};
  constexpr auto s7 = U"test"_sv;

  static_assert(std::is_same<decltype(s1), decltype(s2)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s3)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s4)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s5)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s6)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s7)>::value, "");

  static_assert(s1.size() == 4, "");
  static_assert(s2.size() == 4, "");
  static_assert(s3.size() == 4, "");
  static_assert(s4.size() == 4, "");
  static_assert(s5.size() == 4, "");
  static_assert(s6.size() == 4, "");
  static_assert(s7.size() == 4, "");

  static_assert(!s1.empty(), "");
  static_assert(!s2.empty(), "");
  static_assert(!s3.empty(), "");
  static_assert(!s4.empty(), "");
  static_assert(!s5.empty(), "");
  static_assert(!s6.empty(), "");
  static_assert(!s7.empty(), "");

  static_assert(s1 == s2, "");
  static_assert(s1 == s3, "");
  static_assert(s1 == s4, "");
  static_assert(s1 == s5, "");
  static_assert(s1 == s6, "");
  static_assert(s1 == s7, "");

  static_assert(U""_sv.size() == 0, "");
  static_assert(U""_sv.empty(), "");

  static_assert(U""_sv == U""_sv, "");
  static_assert(U"test"_sv != U""_sv, "");
  static_assert(U"test"_sv != U"testt"_sv, "");
}

TEST_CASE("cx::basic_string_view<wchar_t> is compliant", "[cx::string_view]")
{
  using namespace cx::literals;

  constexpr auto s1 = cx::basic_string_view<wchar_t>{L"test"};
  constexpr auto s2 = cx::basic_string_view<wchar_t>{L"testt", 4};
  constexpr auto s3 = cx::make_string_view(L"test");
  constexpr auto s4 = cx::make_string_view(L"testt", 4);
  constexpr auto s5 = cx::wstring_view{L"test"};
  constexpr auto s6 = cx::wstring_view{L"testt", 4};
  constexpr auto s7 = L"test"_sv;

  static_assert(std::is_same<decltype(s1), decltype(s2)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s3)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s4)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s5)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s6)>::value, "");
  static_assert(std::is_same<decltype(s1), decltype(s7)>::value, "");

  static_assert(s1.size() == 4, "");
  static_assert(s2.size() == 4, "");
  static_assert(s3.size() == 4, "");
  static_assert(s4.size() == 4, "");
  static_assert(s5.size() == 4, "");
  static_assert(s6.size() == 4, "");
  static_assert(s7.size() == 4, "");

  static_assert(!s1.empty(), "");
  static_assert(!s2.empty(), "");
  static_assert(!s3.empty(), "");
  static_assert(!s4.empty(), "");
  static_assert(!s5.empty(), "");
  static_assert(!s6.empty(), "");
  static_assert(!s7.empty(), "");

  static_assert(s1 == s2, "");
  static_assert(s1 == s3, "");
  static_assert(s1 == s4, "");
  static_assert(s1 == s5, "");
  static_assert(s1 == s6, "");
  static_assert(s1 == s7, "");

  static_assert(L""_sv.size() == 0, "");
  static_assert(L""_sv.empty(), "");

  static_assert(L""_sv == L""_sv, "");
  static_assert(L"test"_sv != L""_sv, "");
  static_assert(L"test"_sv != L"testt"_sv, "");
}
