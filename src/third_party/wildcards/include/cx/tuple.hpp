// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CX_TUPLE_HPP
#define CX_TUPLE_HPP

#include <cstddef>      // std::size_t
#include <type_traits>  // std::integral_constant
#include <utility>      // std::forward, std::move

namespace cx
{

template <typename... Types>
struct tuple;

template <typename First, typename... Rest>
struct tuple<First, Rest...> : public tuple<Rest...>
{
  template <std::size_t Index, typename T>
  friend struct tuple_element;

  constexpr tuple() = default;

  constexpr tuple(First first, Rest... rest)
      : tuple<Rest...>{std::move(rest)...}, first_{std::move(first)}
  {
  }

 private:
  First first_;
};

template <>
struct tuple<>
{
};

template <typename... Types>
constexpr tuple<Types...> make_tuple(Types&&... types)
{
  return {std::forward<Types>(types)...};
}

template <typename T>
struct tuple_size;

template <typename... Types>
struct tuple_size<tuple<Types...>> : std::integral_constant<std::size_t, sizeof...(Types)>
{
};

template <typename... Types>
struct tuple_size<const tuple<Types...>>
    : std::integral_constant<std::size_t, tuple_size<tuple<Types...>>::value>
{
};

template <std::size_t Index, typename T>
struct tuple_element;

template <std::size_t Index, typename First, typename... Rest>
struct tuple_element<Index, tuple<First, Rest...>>
{
  using type = typename tuple_element<Index - 1, tuple<Rest...>>::type;

  constexpr static const typename tuple_element<Index - 1, tuple<Rest...>>::type& get(
      const tuple<First, Rest...>& t)
  {
    return tuple_element<Index - 1, tuple<Rest...>>::get(t);
  }

  constexpr static typename tuple_element<Index - 1, tuple<Rest...>>::type& get(
      tuple<First, Rest...>& t)
  {
    return tuple_element<Index - 1, tuple<Rest...>>::get(t);
  }
};

template <typename First, typename... Rest>
struct tuple_element<0, tuple<First, Rest...>>
{
  using type = First;

  constexpr static const First& get(const tuple<First, Rest...>& t)
  {
    return t.first_;
  }

  constexpr static First& get(tuple<First, Rest...>& t)
  {
    return t.first_;
  }
};

template <std::size_t Index, typename T>
using tuple_element_t = typename tuple_element<Index, T>::type;

template <std::size_t Index, typename First, typename... Rest>
constexpr const tuple_element_t<Index, tuple<First, Rest...>>& get(const tuple<First, Rest...>& t)
{
  return tuple_element<Index, tuple<First, Rest...>>::get(t);
}

template <std::size_t Index, typename First, typename... Rest>
constexpr tuple_element_t<Index, tuple<First, Rest...>>& get(tuple<First, Rest...>& t)
{
  return tuple_element<Index, tuple<First, Rest...>>::get(t);
}

namespace detail
{

template <std::size_t Index>
struct tuples
{
  template <typename... Types1, typename... Types2>
  constexpr static bool equal(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
  {
    return get<Index>(lhs) == get<Index>(rhs) && tuples<Index - 1>::equal(lhs, rhs);
  }
};

template <>
struct tuples<0>
{
  template <typename... Types1, typename... Types2>
  constexpr static bool equal(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
  {
    return get<0>(lhs) == get<0>(rhs);
  }
};

}  // namespace detail

template <typename... Types1, typename... Types2>
constexpr bool operator==(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
{
  static_assert(
      sizeof...(Types1) == sizeof...(Types2),
      "Tuples size is not equal in cx::operator==(const cx::tuple<>&, const cx::tuple<>&)");

  return detail::tuples<sizeof...(Types1) -1>::equal(lhs, rhs);
}

template <>
constexpr bool operator==(const tuple<>&, const tuple<>&)
{
  return true;
}

template <typename... Types1, typename... Types2>
constexpr bool operator!=(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
{
  return !(lhs == rhs);
}

}  // namespace cx

#endif  // CX_TUPLE_HPP
