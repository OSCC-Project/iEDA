// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef WILDCARDS_UTILITY_HPP
#define WILDCARDS_UTILITY_HPP

#include <type_traits>  // std::remove_cv, std::remove_reference
#include <utility>      // std::declval

#include "cx/iterator.hpp"  // cx::begin

namespace wildcards
{

template <typename C>
struct const_iterator
{
  using type = typename std::remove_cv<
      typename std::remove_reference<decltype(cx::cbegin(std::declval<C>()))>::type>::type;
};

template <typename C>
using const_iterator_t = typename const_iterator<C>::type;

template <typename C>
struct iterator
{
  using type = typename std::remove_cv<
      typename std::remove_reference<decltype(cx::begin(std::declval<C>()))>::type>::type;
};

template <typename C>
using iterator_t = typename iterator<C>::type;

template <typename It>
struct iterated_item
{
  using type = typename std::remove_cv<
      typename std::remove_reference<decltype(*std::declval<It>())>::type>::type;
};

template <typename It>
using iterated_item_t = typename iterated_item<It>::type;

template <typename C>
struct container_item
{
  using type = typename std::remove_cv<
      typename std::remove_reference<decltype(*cx::begin(std::declval<C>()))>::type>::type;
};

template <typename C>
using container_item_t = typename container_item<C>::type;

}  // namespace wildcards

#endif  // WILDCARDS_UTILITY_HPP
