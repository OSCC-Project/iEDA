// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CX_ARRAY_HPP
#define CX_ARRAY_HPP

#include <cstddef>    // std::size_t
#include <stdexcept>  // std::out_of_range

#include "config.hpp"        // cfg_constexpr14
#include "cx/algorithm.hpp"  // cx::equal

namespace cx
{

template <typename T, std::size_t N>
struct array
{
  using value_type = T;

  constexpr std::size_t size() const
  {
    return N;
  }

  constexpr bool empty() const
  {
    return size() == 0;
  }

  constexpr const T* begin() const
  {
    return &data[0];
  }

  cfg_constexpr14 T* begin()
  {
    return &data[0];
  }

  constexpr const T* cbegin() const
  {
    return begin();
  }

  constexpr const T* end() const
  {
    return &data[N];
  }

  cfg_constexpr14 T* end()
  {
    return &data[N];
  }

  constexpr const T* cend() const
  {
    return end();
  }

  constexpr const T& operator[](std::size_t pos) const
  {
    return data[pos];
  }

  cfg_constexpr14 T& operator[](std::size_t pos)
  {
    return data[pos];
  }

  constexpr const T& at(std::size_t pos) const
  {
    return pos < size() ? data[pos] : throw std::out_of_range("The given position is out of range");
  }

  cfg_constexpr14 T& at(std::size_t pos)
  {
    return pos < size() ? data[pos] : throw std::out_of_range("The given position is out of range");
  }

  T data[N > 0 ? N : 1];
};

template <typename T, std::size_t N>
constexpr bool operator==(const array<T, N>& lhs, const array<T, N>& rhs)
{
  return equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, std::size_t N>
constexpr bool operator!=(const array<T, N>& lhs, const array<T, N>& rhs)
{
  return !(lhs == rhs);
}

template <std::size_t Index, typename T>
struct tuple_element;

template <std::size_t Index, typename T, std::size_t N>
struct tuple_element<Index, array<T, N>>
{
  static_assert(Index < N, "Index out of bounds in cx::tuple_element<>");

  using type = T;

  constexpr static const T& get(const array<T, N>& a)
  {
    return a[Index];
  }

  constexpr static T& get(array<T, N>& a)
  {
    return a[Index];
  }
};

template <std::size_t Index, typename T>
using tuple_element_t = typename tuple_element<Index, T>::type;

template <std::size_t Index, typename T, std::size_t N>
constexpr const T& get(const array<T, N>& a)
{
  static_assert(Index < N, "Index out of bounds in cx::get<>(const cx::array<>&)");

  return a[Index];
}

template <std::size_t Index, typename T, std::size_t N>
constexpr T& get(array<T, N>& a)
{
  static_assert(Index < N, "Index out of bounds in cx::get<>(cx::array<>&)");

  return a[Index];
}

}  // namespace cx

#endif  // CX_ARRAY_HPP
