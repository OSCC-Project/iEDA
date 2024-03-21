// Copyright Tomas Zeman 2019.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef WILDCARDS_MATCHER_HPP
#define WILDCARDS_MATCHER_HPP

#include <cstddef>      // std::size_t
#include <type_traits>  // std::enable_if, std::is_same
#include <utility>      // std::forward, std::move

#include "cx/functional.hpp"      // cx::equal_to
#include "cx/iterator.hpp"        // cx::cbegin, cx::cend
#include "cx/string_view.hpp"     // cx::make_string_view
#include "wildcards/cards.hpp"    // wildcards::cards
#include "wildcards/match.hpp"    // wildcards::detail::make_full_match_result
                                  // wildcards::detail::match
#include "wildcards/utility.hpp"  // wildcards::const_iterator_t, wildcards::container_item_t,

namespace wildcards
{

template <typename Pattern, typename EqualTo = cx::equal_to<void>>
class matcher
{
 public:
  constexpr explicit matcher(Pattern&& pattern, const cards<container_item_t<Pattern>>& c =
                                                    cards<container_item_t<Pattern>>(),
                             const EqualTo& equal_to = EqualTo())
      : p_{cx::cbegin(pattern)},
        pend_{cx::cend(std::forward<Pattern>(pattern))},
        c_{c},
        equal_to_{equal_to}
  {
  }

  constexpr matcher(Pattern&& pattern, const EqualTo& equal_to)
      : p_{cx::cbegin(pattern)},
        pend_{cx::cend(std::forward<Pattern>(pattern))},
        c_{cards<container_item_t<Pattern>>()},
        equal_to_{equal_to}
  {
  }

  template <typename Sequence>
  constexpr full_match_result<const_iterator_t<Sequence>, const_iterator_t<Pattern>> matches(
      Sequence&& sequence) const
  {
    return detail::make_full_match_result(
        cx::cbegin(sequence), cx::cend(sequence), p_, pend_,
        detail::match(cx::cbegin(sequence), cx::cend(std::forward<Sequence>(sequence)), p_, pend_,
                      c_, equal_to_));
  }

 private:
  const_iterator_t<Pattern> p_;
  const_iterator_t<Pattern> pend_;
  cards<container_item_t<Pattern>> c_;
  EqualTo equal_to_;
};

template <typename Pattern, typename EqualTo = cx::equal_to<void>>
constexpr matcher<Pattern, EqualTo> make_matcher(
    Pattern&& pattern,
    const cards<container_item_t<Pattern>>& c = cards<container_item_t<Pattern>>(),
    const EqualTo& equal_to = EqualTo())
{
  return matcher<Pattern, EqualTo>{std::forward<Pattern>(pattern), c, equal_to};
}

template <typename Pattern, typename EqualTo = cx::equal_to<void>,
          typename = typename std::enable_if<!std::is_same<EqualTo, cards_type>::value>::type>
constexpr matcher<Pattern, EqualTo> make_matcher(Pattern&& pattern, const EqualTo& equal_to)
{
  return make_matcher(std::forward<Pattern>(pattern), cards<container_item_t<Pattern>>(), equal_to);
}

namespace literals
{

constexpr auto operator"" _wc(const char* str, std::size_t s)
    -> decltype(make_matcher(cx::make_string_view(str, s + 1)))
{
  return make_matcher(cx::make_string_view(str, s + 1));
}

constexpr auto operator"" _wc(const char16_t* str, std::size_t s)
    -> decltype(make_matcher(cx::make_string_view(str, s + 1)))
{
  return make_matcher(cx::make_string_view(str, s + 1));
}

constexpr auto operator"" _wc(const char32_t* str, std::size_t s)
    -> decltype(make_matcher(cx::make_string_view(str, s + 1)))
{
  return make_matcher(cx::make_string_view(str, s + 1));
}

constexpr auto operator"" _wc(const wchar_t* str, std::size_t s)
    -> decltype(make_matcher(cx::make_string_view(str, s + 1)))
{
  return make_matcher(cx::make_string_view(str, s + 1));
}

}  // namespace literals

}  // namespace wildcards

#endif  // WILDCARDS_MATCHER_HPP
