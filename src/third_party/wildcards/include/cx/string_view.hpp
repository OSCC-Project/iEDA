// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CX_STRING_VIEW_HPP
#define CX_STRING_VIEW_HPP

#include <cstddef>  // std::size_t
#include <ostream>  // std::basic_ostream

#include "cx/algorithm.hpp"  // cx::equal

namespace cx
{

template <typename T>
class basic_string_view
{
 public:
  using value_type = T;

  constexpr basic_string_view() = default;

  template <std::size_t N>
  constexpr basic_string_view(const T (&str)[N]) : data_{&str[0]}, size_{N - 1}
  {
  }

  constexpr basic_string_view(const T* str, std::size_t s) : data_{str}, size_{s}
  {
  }

  constexpr const T* data() const
  {
    return data_;
  }

  constexpr std::size_t size() const
  {
    return size_;
  }

  constexpr bool empty() const
  {
    return size() == 0;
  }

  constexpr const T* begin() const
  {
    return data_;
  }

  constexpr const T* cbegin() const
  {
    return begin();
  }

  constexpr const T* end() const
  {
    return data_ + size_;
  }

  constexpr const T* cend() const
  {
    return end();
  }

 private:
  const T* data_{nullptr};
  std::size_t size_{0};
};

template <typename T>
constexpr bool operator==(const basic_string_view<T>& lhs, const basic_string_view<T>& rhs)
{
  return equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename T>
constexpr bool operator!=(const basic_string_view<T>& lhs, const basic_string_view<T>& rhs)
{
  return !(lhs == rhs);
}

template <typename T>
std::basic_ostream<T>& operator<<(std::basic_ostream<T>& o, const basic_string_view<T>& s)
{
  o << s.data();
  return o;
}

template <typename T, std::size_t N>
constexpr basic_string_view<T> make_string_view(const T (&str)[N])
{
  return {str, N - 1};
}

template <typename T>
constexpr basic_string_view<T> make_string_view(const T* str, std::size_t s)
{
  return {str, s};
}

using string_view = basic_string_view<char>;
using u16string_view = basic_string_view<char16_t>;
using u32string_view = basic_string_view<char32_t>;
using wstring_view = basic_string_view<wchar_t>;

namespace literals
{

constexpr string_view operator"" _sv(const char* str, std::size_t s)
{
  return {str, s};
}

constexpr u16string_view operator"" _sv(const char16_t* str, std::size_t s)
{
  return {str, s};
}

constexpr u32string_view operator"" _sv(const char32_t* str, std::size_t s)
{
  return {str, s};
}

constexpr wstring_view operator"" _sv(const wchar_t* str, std::size_t s)
{
  return {str, s};
}

}  // namespace literals

}  // namespace cx

#endif  // CX_STRING_VIEW_HPP
