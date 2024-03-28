// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CX_FUNCTIONAL_HPP
#define CX_FUNCTIONAL_HPP

#include <utility>  // std::forward

namespace cx
{

template <typename T>
struct less
{
  constexpr auto operator()(const T& lhs, const T& rhs) const -> decltype(lhs < rhs)
  {
    return lhs < rhs;
  }
};

template <>
struct less<void>
{
  template <typename T, typename U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
      -> decltype(std::forward<T>(lhs) < std::forward<U>(rhs))
  {
    return std::forward<T>(lhs) < std::forward<U>(rhs);
  }
};

template <typename T>
struct equal_to
{
  constexpr auto operator()(const T& lhs, const T& rhs) const -> decltype(lhs == rhs)
  {
    return lhs == rhs;
  }
};

template <>
struct equal_to<void>
{
  template <typename T, typename U>
  constexpr auto operator()(T&& lhs, U&& rhs) const
      -> decltype(std::forward<T>(lhs) == std::forward<U>(rhs))
  {
    return std::forward<T>(lhs) == std::forward<U>(rhs);
  }
};

}  // namespace cx

#endif  // CX_FUNCTIONAL_HPP
