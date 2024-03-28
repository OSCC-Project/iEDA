// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CX_ITERATOR_HPP
#define CX_ITERATOR_HPP

#include <cstddef>           // std::size_t
#include <initializer_list>  // std::initializer_list

namespace cx
{

template <typename It>
constexpr It next(It it)
{
  return it + 1;
}

template <typename It>
constexpr It prev(It it)
{
  return it - 1;
}

template <typename C>
constexpr auto size(const C& c) -> decltype(c.size())
{
  return c.size();
}

template <typename T, std::size_t N>
constexpr std::size_t size(const T (&)[N])
{
  return N;
}

template <typename C>
constexpr auto empty(const C& c) -> decltype(c.empty())
{
  return c.empty();
}

template <typename T, std::size_t N>
constexpr bool empty(const T (&)[N])
{
  return false;
}

template <typename E>
constexpr bool empty(std::initializer_list<E> il)
{
  return il.size() == 0;
}

template <typename C>
constexpr auto begin(const C& c) -> decltype(c.begin())
{
  return c.begin();
}

template <typename C>
constexpr auto begin(C& c) -> decltype(c.begin())
{
  return c.begin();
}

template <typename T, std::size_t N>
constexpr T* begin(T (&array)[N])
{
  return &array[0];
}

template <typename E>
constexpr const E* begin(std::initializer_list<E> il)
{
  return il.begin();
}

template <typename C>
constexpr auto cbegin(const C& c) -> decltype(cx::begin(c))
{
  return cx::begin(c);
}

template <typename C>
constexpr auto end(const C& c) -> decltype(c.end())
{
  return c.end();
}

template <typename C>
constexpr auto end(C& c) -> decltype(c.end())
{
  return c.end();
}

template <typename T, std::size_t N>
constexpr T* end(T (&array)[N])
{
  return &array[N];
}

template <typename E>
constexpr const E* end(std::initializer_list<E> il)
{
  return il.end();
}

template <typename C>
constexpr auto cend(const C& c) -> decltype(cx::end(c))
{
  return cx::end(c);
}

}  // namespace cx

#endif  // CX_ITERATOR_HPP
