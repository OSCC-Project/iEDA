// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CX_UTILITY_HPP
#define CX_UTILITY_HPP

#include <cstddef>      // std::size_t
#include <type_traits>  // std::integral_constant
#include <utility>      // std::forward, std::move

namespace cx
{

template <typename First, typename Second>
struct pair
{
  using first_type = First;
  using second_type = Second;

  constexpr pair() = default;

  constexpr pair(First frst, Second scnd) : first{std::move(frst)}, second{std::move(scnd)}
  {
  }

  First first;
  Second second;
};

template <typename First, typename Second>
constexpr bool operator==(const pair<First, Second>& lhs, const pair<First, Second>& rhs)
{
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <typename First, typename Second>
constexpr bool operator!=(const pair<First, Second>& lhs, const pair<First, Second>& rhs)
{
  return !(lhs == rhs);
}

template <typename First, typename Second>
constexpr pair<First, Second> make_pair(First&& first, Second&& second)
{
  return {std::forward<First>(first), std::forward<Second>(second)};
}

template <typename T>
struct tuple_size;

template <typename First, typename Second>
struct tuple_size<pair<First, Second>> : std::integral_constant<std::size_t, 2>
{
};

template <typename First, typename Second>
struct tuple_size<const pair<First, Second>>
    : std::integral_constant<std::size_t, tuple_size<pair<First, Second>>::value>
{
};

template <std::size_t Index, typename T>
struct tuple_element;

template <typename First, typename Second>
struct tuple_element<0, pair<First, Second>>
{
  using type = First;

  constexpr static const First& get(const pair<First, Second>& p)
  {
    return p.first;
  }

  constexpr static First& get(pair<First, Second>& p)
  {
    return p.first;
  }
};

template <typename First, typename Second>
struct tuple_element<1, pair<First, Second>>
{
  using type = Second;

  constexpr static const Second& get(const pair<First, Second>& p)
  {
    return p.second;
  }

  constexpr static Second& get(pair<First, Second>& p)
  {
    return p.second;
  }
};

template <std::size_t Index, typename T>
using tuple_element_t = typename tuple_element<Index, T>::type;

template <std::size_t Index, typename First, typename Second>
constexpr const tuple_element_t<Index, pair<First, Second>>& get(const pair<First, Second>& p)
{
  return tuple_element<Index, pair<First, Second>>::get(p);
}

template <std::size_t Index, typename First, typename Second>
constexpr tuple_element_t<Index, pair<First, Second>>& get(pair<First, Second>& p)
{
  return tuple_element<Index, pair<First, Second>>::get(p);
}

template <typename First, typename Second>
constexpr const First& get(const pair<First, Second>& p)
{
  return p.first;
}

template <typename First, typename Second>
constexpr First& get(pair<First, Second>& p)
{
  return p.first;
}

template <typename Second, typename First>
constexpr const Second& get(const pair<First, Second>& p)
{
  return p.second;
}

template <typename Second, typename First>
constexpr Second& get(pair<First, Second>& p)
{
  return p.second;
}

}  // namespace cx

#endif  // CX_UTILITY_HPP
