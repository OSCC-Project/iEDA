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
constexpr cx::basic_string_view<T> basic_valid_id(cx::basic_string_view<T> s,
                                                  cx::basic_string_view<T> p)
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

int main()
{
  // compile time
  {
    constexpr auto id1 = valid_id("test_something", "[Tt]est_*");
    std::cout << "Valid ID: " << id1 << std::endl;

    constexpr auto id2 = valid_id("Test_something_else", "[Tt]est_*");
    std::cout << "Valid ID: " << id2 << std::endl;

    // constexpr auto id3 = valid_id("tst_something_different", "[Tt]est_*");  // compile time error
    // std::cout << "Valid ID: " << id3 << std::endl;
  }

  // runtime
  try
  {
    auto id1 = valid_id("test_something", "[Tt]est_*");
    std::cout << "Valid ID: " << id1 << std::endl;

    auto id2 = valid_id("Test_something_else", "[Tt]est_*");
    std::cout << "Valid ID: " << id2 << std::endl;

    auto id3 = valid_id("tst_something_different", "[Tt]est_*");  // runtime error
    std::cout << "Valid ID: " << id3 << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
