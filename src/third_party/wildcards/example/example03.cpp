// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <exception>
#include <iostream>
#include <stdexcept>

#include "cx/string_view.hpp"
#include "wildcards/match.hpp"

template <typename T>
struct valid_id_pattern;

template <>
struct valid_id_pattern<char>
{
  constexpr static cx::string_view value()
  {
    return cx::string_view{"[Tt]est_*"};
  }
};

template <>
struct valid_id_pattern<char16_t>
{
  constexpr static cx::u16string_view value()
  {
    return cx::u16string_view{u"[Tt]est_*"};
  }
};

template <>
struct valid_id_pattern<char32_t>
{
  constexpr static cx::u32string_view value()
  {
    return cx::u32string_view{U"[Tt]est_*"};
  }
};

template <>
struct valid_id_pattern<wchar_t>
{
  constexpr static cx::wstring_view value()
  {
    return cx::wstring_view{L"[Tt]est_*"};
  }
};

template <typename T>
constexpr cx::basic_string_view<T> basic_valid_id(
    cx::basic_string_view<T> s, cx::basic_string_view<T> p = valid_id_pattern<T>::value())
{
  return wildcards::match(s, p) ? s : throw std::logic_error("Invalid ID");
}

constexpr cx::string_view valid_id(cx::string_view s, cx::string_view p)
{
  return basic_valid_id<char>(s, p);
}

constexpr cx::u16string_view valid_id(cx::u16string_view s, cx::u16string_view p)
{
  return basic_valid_id<char16_t>(s, p);
}

constexpr cx::u32string_view valid_id(cx::u32string_view s, cx::u32string_view p)
{
  return basic_valid_id<char32_t>(s, p);
}

constexpr cx::wstring_view valid_id(cx::wstring_view s, cx::wstring_view p)
{
  return basic_valid_id<wchar_t>(s, p);
}

constexpr cx::string_view valid_id(cx::string_view s)
{
  return basic_valid_id<char>(s);
}

constexpr cx::u16string_view valid_id(cx::u16string_view s)
{
  return basic_valid_id<char16_t>(s);
}

constexpr cx::u32string_view valid_id(cx::u32string_view s)
{
  return basic_valid_id<char32_t>(s);
}

constexpr cx::wstring_view valid_id(cx::wstring_view s)
{
  return basic_valid_id<wchar_t>(s);
}

constexpr cx::string_view operator"" _valid_id(const char* str, std::size_t s)
{
  return valid_id(cx::make_string_view(str, s));
}

constexpr cx::u16string_view operator"" _valid_id(const char16_t* str, std::size_t s)
{
  return valid_id(cx::make_string_view(str, s));
}

constexpr cx::u32string_view operator"" _valid_id(const char32_t* str, std::size_t s)
{
  return valid_id(cx::make_string_view(str, s));
}

constexpr cx::wstring_view operator"" _valid_id(const wchar_t* str, std::size_t s)
{
  return valid_id(cx::make_string_view(str, s));
}

int main()
{
  // compile time
  {
    constexpr auto id1 = valid_id("test_something");
    std::cout << "Valid ID: " << id1 << std::endl;

    constexpr auto id2 = "Test_something_else"_valid_id;
    std::cout << "Valid ID: " << id2 << std::endl;

    // constexpr auto id3 = "tst_something_different"_valid_id;  // compile time error
    // std::cout << "Valid ID: " << id3 << std::endl;
  }

  // runtime
  try
  {
    auto id1 = valid_id("test_something");
    std::cout << "Valid ID: " << id1 << std::endl;

    auto id2 = "Test_something_else"_valid_id;
    std::cout << "Valid ID: " << id2 << std::endl;

    auto id3 = "tst_something_different"_valid_id;  // runtime error
    std::cout << "Valid ID: " << id3 << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
