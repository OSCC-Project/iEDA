// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef WILDCARDS_MATCH_HPP
#define WILDCARDS_MATCH_HPP

#include <stdexcept>    // std::invalid_argument, std::logic_error, std::runtime_error
#include <type_traits>  // std::enable_if, std::is_same
#include <utility>      // std::forward, std::move

#include "config.hpp"             // cfg_HAS_CONSTEXPR14, cfg_HAS_FULL_FEATURED_CONSTEXPR14
#include "cx/functional.hpp"      // cx::equal_to
#include "cx/iterator.hpp"        // cx::cbegin, cx::cend, cx::next, cx::prev
#include "wildcards/cards.hpp"    // wildcards::cards
#include "wildcards/utility.hpp"  // wildcards::const_iterator_t, wildcards::container_item_t,
                                  // wildcards::iterated_item_t

namespace wildcards
{

template <typename SequenceIterator, typename PatternIterator>
struct full_match_result
{
  bool res;
  SequenceIterator s, send, s1;
  PatternIterator p, pend, p1;

  constexpr operator bool() const
  {
    return res;
  }
};

namespace detail
{

template <typename SequenceIterator, typename PatternIterator>
struct match_result
{
  bool res;
  SequenceIterator s;
  PatternIterator p;

  constexpr operator bool() const
  {
    return res;
  }
};

template <typename SequenceIterator, typename PatternIterator>
constexpr match_result<SequenceIterator, PatternIterator> make_match_result(bool res,
                                                                            SequenceIterator s,
                                                                            PatternIterator p)
{
  return {std::move(res), std::move(s), std::move(p)};
}

template <typename SequenceIterator, typename PatternIterator>
constexpr full_match_result<SequenceIterator, PatternIterator> make_full_match_result(
    SequenceIterator s, SequenceIterator send, PatternIterator p, PatternIterator pend,
    match_result<SequenceIterator, PatternIterator> mr)
{
  return {std::move(mr.res), std::move(s),    std::move(send), std::move(mr.s),
          std::move(p),      std::move(pend), std::move(mr.p)};
}

#if !cfg_HAS_FULL_FEATURED_CONSTEXPR14

constexpr bool throw_invalid_argument(const char* what_arg)
{
  return what_arg == nullptr ? false : throw std::invalid_argument(what_arg);
}

template <typename T>
constexpr T throw_invalid_argument(T t, const char* what_arg)
{
  return what_arg == nullptr ? t : throw std::invalid_argument(what_arg);
}

constexpr bool throw_logic_error(const char* what_arg)
{
  return what_arg == nullptr ? false : throw std::logic_error(what_arg);
}

template <typename T>
constexpr T throw_logic_error(T t, const char* what_arg)
{
  return what_arg == nullptr ? t : throw std::logic_error(what_arg);
}

#endif

enum class is_set_state
{
  open,
  not_or_first,
  first,
  next
};

template <typename PatternIterator>
constexpr bool is_set(
    PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    is_set_state state = is_set_state::open)
{
#if cfg_HAS_CONSTEXPR14

  if (!c.set_enabled)
  {
    return false;
  }

  while (p != pend)
  {
    switch (state)
    {
      case is_set_state::open:
        if (*p != c.set_open)
        {
          return false;
        }

        state = is_set_state::not_or_first;
        break;

      case is_set_state::not_or_first:
        if (*p == c.set_not)
        {
          state = is_set_state::first;
        }
        else
        {
          state = is_set_state::next;
        }

        break;

      case is_set_state::first:
        state = is_set_state::next;
        break;

      case is_set_state::next:
        if (*p == c.set_close)
        {
          return true;
        }

        break;

      default:
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
        throw std::logic_error(
            "The program execution should never end up here throwing this exception");
#else
        return throw_logic_error(
            "The program execution should never end up here throwing this exception");
#endif
    }

    p = cx::next(p);
  }

  return false;

#else  // !cfg_HAS_CONSTEXPR14

  return c.set_enabled && p != pend &&
         (state == is_set_state::open
              ? *p == c.set_open && is_set(cx::next(p), pend, c, is_set_state::not_or_first)
              :

              state == is_set_state::not_or_first
                  ? *p == c.set_not ? is_set(cx::next(p), pend, c, is_set_state::first)
                                    : is_set(cx::next(p), pend, c, is_set_state::next)
                  : state == is_set_state::first
                        ? is_set(cx::next(p), pend, c, is_set_state::next)
                        : state == is_set_state::next
                              ? *p == c.set_close ||
                                    is_set(cx::next(p), pend, c, is_set_state::next)
                              : throw std::logic_error("The program execution should never end up "
                                                       "here throwing this exception"));

#endif  // cfg_HAS_CONSTEXPR14
}

enum class set_end_state
{
  open,
  not_or_first,
  first,
  next
};

template <typename PatternIterator>
constexpr PatternIterator set_end(
    PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    set_end_state state = set_end_state::open)
{
#if cfg_HAS_CONSTEXPR14

  if (!c.set_enabled)
  {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
    throw std::invalid_argument("The use of sets is disabled");
#else
    return throw_invalid_argument(p, "The use of sets is disabled");
#endif
  }

  while (p != pend)
  {
    switch (state)
    {
      case set_end_state::open:
        if (*p != c.set_open)
        {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
          throw std::invalid_argument("The given pattern is not a valid set");
#else
          return throw_invalid_argument(p, "The given pattern is not a valid set");
#endif
        }

        state = set_end_state::not_or_first;
        break;

      case set_end_state::not_or_first:
        if (*p == c.set_not)
        {
          state = set_end_state::first;
        }
        else
        {
          state = set_end_state::next;
        }

        break;

      case set_end_state::first:
        state = set_end_state::next;
        break;

      case set_end_state::next:
        if (*p == c.set_close)
        {
          return cx::next(p);
        }

        break;

      default:
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
        throw std::logic_error(
            "The program execution should never end up here throwing this exception");
#else
        return throw_logic_error(
            p, "The program execution should never end up here throwing this exception");
#endif
    }

    p = cx::next(p);
  }

#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
  throw std::invalid_argument("The given pattern is not a valid set");
#else
  return throw_invalid_argument(p, "The given pattern is not a valid set");
#endif

#else  // !cfg_HAS_CONSTEXPR14

  return !c.set_enabled
             ? throw std::invalid_argument("The use of sets is disabled")
             : p == pend
                   ? throw std::invalid_argument("The given pattern is not a valid set")
                   :

                   state == set_end_state::open
                       ? *p == c.set_open
                             ? set_end(cx::next(p), pend, c, set_end_state::not_or_first)
                             : throw std::invalid_argument("The given pattern is not a valid set")
                       :

                       state == set_end_state::not_or_first
                           ? *p == c.set_not ? set_end(cx::next(p), pend, c, set_end_state::first)
                                             : set_end(cx::next(p), pend, c, set_end_state::next)
                           : state == set_end_state::first
                                 ? set_end(cx::next(p), pend, c, set_end_state::next)
                                 : state == set_end_state::next
                                       ? *p == c.set_close
                                             ? cx::next(p)
                                             : set_end(cx::next(p), pend, c, set_end_state::next)
                                       : throw std::logic_error(
                                             "The program execution should never end up "
                                             "here throwing this exception");

#endif  // cfg_HAS_CONSTEXPR14
}

enum class match_set_state
{
  open,
  not_or_first_in,
  first_out,
  next_in,
  next_out
};

template <typename SequenceIterator, typename PatternIterator,
          typename EqualTo = cx::equal_to<void>>
constexpr match_result<SequenceIterator, PatternIterator> match_set(
    SequenceIterator s, SequenceIterator send, PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    const EqualTo& equal_to = EqualTo(), match_set_state state = match_set_state::open)
{
#if cfg_HAS_CONSTEXPR14

  if (!c.set_enabled)
  {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
    throw std::invalid_argument("The use of sets is disabled");
#else
    return throw_invalid_argument(make_match_result(false, s, p), "The use of sets is disabled");
#endif
  }

  while (p != pend)
  {
    switch (state)
    {
      case match_set_state::open:
        if (*p != c.set_open)
        {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
          throw std::invalid_argument("The given pattern is not a valid set");
#else
          return throw_invalid_argument(make_match_result(false, s, p),
                                        "The given pattern is not a valid set");
#endif
        }

        state = match_set_state::not_or_first_in;
        break;

      case match_set_state::not_or_first_in:
        if (*p == c.set_not)
        {
          state = match_set_state::first_out;
        }
        else
        {
          if (s == send)
          {
            return make_match_result(false, s, p);
          }

          if (equal_to(*s, *p))
          {
            return make_match_result(true, s, p);
          }

          state = match_set_state::next_in;
        }

        break;

      case match_set_state::first_out:
        if (s == send || equal_to(*s, *p))
        {
          return make_match_result(false, s, p);
        }

        state = match_set_state::next_out;
        break;

      case match_set_state::next_in:
        if (*p == c.set_close || s == send)
        {
          return make_match_result(false, s, p);
        }

        if (equal_to(*s, *p))
        {
          return make_match_result(true, s, p);
        }

        break;

      case match_set_state::next_out:
        if (*p == c.set_close)
        {
          return make_match_result(true, s, p);
        }

        if (s == send || equal_to(*s, *p))
        {
          return make_match_result(false, s, p);
        }

        break;

      default:
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
        throw std::logic_error(
            "The program execution should never end up here throwing this exception");
#else
        return throw_logic_error(
            make_match_result(false, s, p),
            "The program execution should never end up here throwing this exception");
#endif
    }

    p = cx::next(p);
  }

#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
  throw std::invalid_argument("The given pattern is not a valid set");
#else
  return throw_invalid_argument(make_match_result(false, s, p),
                                "The given pattern is not a valid set");
#endif

#else  // !cfg_HAS_CONSTEXPR14

  return !c.set_enabled
             ? throw std::invalid_argument("The use of sets is disabled")
             : p == pend
                   ? throw std::invalid_argument("The given pattern is not a valid set")
                   : state == match_set_state::open
                         ? *p == c.set_open
                               ? match_set(s, send, cx::next(p), pend, c, equal_to,
                                           match_set_state::not_or_first_in)
                               :

                               throw std::invalid_argument("The given pattern is not a valid set")
                         :

                         state == match_set_state::not_or_first_in
                             ? *p == c.set_not
                                   ? match_set(s, send, cx::next(p), pend, c, equal_to,
                                               match_set_state::first_out)
                                   :

                                   s == send ? make_match_result(false, s, p)
                                             : equal_to(*s, *p)
                                                   ? make_match_result(true, s, p)
                                                   : match_set(s, send, cx::next(p), pend, c,
                                                               equal_to, match_set_state::next_in)

                             :

                             state == match_set_state::first_out
                                 ? s == send || equal_to(*s, *p)
                                       ? make_match_result(false, s, p)
                                       : match_set(s, send, cx::next(p), pend, c, equal_to,
                                                   match_set_state::next_out)

                                 :

                                 state == match_set_state::next_in
                                     ? *p == c.set_close || s == send
                                           ? make_match_result(false, s, p)
                                           : equal_to(*s, *p) ? make_match_result(true, s, p)
                                                              : match_set(s, send, cx::next(p),
                                                                          pend, c, equal_to, state)

                                     :

                                     state == match_set_state::next_out
                                         ? *p == c.set_close
                                               ? make_match_result(true, s, p)
                                               : s == send || equal_to(*s, *p)
                                                     ? make_match_result(false, s, p)
                                                     : match_set(s, send, cx::next(p), pend, c,
                                                                 equal_to, state)

                                         : throw std::logic_error(
                                               "The program execution should never end up "
                                               "here "
                                               "throwing this exception");

#endif  // cfg_HAS_CONSTEXPR14
}

enum class is_alt_state
{
  open,
  next,
  escape
};

template <typename PatternIterator>
constexpr bool is_alt(
    PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    is_alt_state state = is_alt_state::open, int depth = 0)
{
#if cfg_HAS_CONSTEXPR14

  if (!c.alt_enabled)
  {
    return false;
  }

  while (p != pend)
  {
    switch (state)
    {
      case is_alt_state::open:
        if (*p != c.alt_open)
        {
          return false;
        }

        state = is_alt_state::next;
        ++depth;
        break;

      case is_alt_state::next:
        if (*p == c.escape)
        {
          state = is_alt_state::escape;
        }
        else if (c.set_enabled && *p == c.set_open &&
                 is_set(cx::next(p), pend, c, is_set_state::not_or_first))
        {
          p = cx::prev(set_end(cx::next(p), pend, c, set_end_state::not_or_first));
        }
        else if (*p == c.alt_open)
        {
          ++depth;
        }
        else if (*p == c.alt_close)
        {
          --depth;

          if (depth == 0)
          {
            return true;
          }
        }

        break;

      case is_alt_state::escape:
        state = is_alt_state::next;
        break;

      default:
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
        throw std::logic_error(
            "The program execution should never end up here throwing this exception");
#else
        return throw_logic_error(
            p, "The program execution should never end up here throwing this exception");
#endif
    }

    p = cx::next(p);
  }

  return false;

#else  // !cfg_HAS_CONSTEXPR14

  return c.alt_enabled && p != pend &&
         (state == is_alt_state::open
              ? *p == c.alt_open && is_alt(cx::next(p), pend, c, is_alt_state::next, depth + 1)
              : state == is_alt_state::next
                    ? *p == c.escape
                          ? is_alt(cx::next(p), pend, c, is_alt_state::escape, depth)
                          : c.set_enabled && *p == c.set_open &&
                                    is_set(cx::next(p), pend, c, is_set_state::not_or_first)
                                ? is_alt(set_end(cx::next(p), pend, c, set_end_state::not_or_first),
                                         pend, c, state, depth)
                                : *p == c.alt_open
                                      ? is_alt(cx::next(p), pend, c, state, depth + 1)
                                      : *p == c.alt_close
                                            ? depth == 1 ||
                                                  is_alt(cx::next(p), pend, c, state, depth - 1)
                                            : is_alt(cx::next(p), pend, c, state, depth)
                    :

                    state == is_alt_state::escape
                        ? is_alt(cx::next(p), pend, c, is_alt_state::next, depth)
                        : throw std::logic_error(
                              "The program execution should never end up here throwing this "
                              "exception"));

#endif  // cfg_HAS_CONSTEXPR14
}

enum class alt_end_state
{
  open,
  next,
  escape
};

template <typename PatternIterator>
constexpr PatternIterator alt_end(
    PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    alt_end_state state = alt_end_state::open, int depth = 0)
{
#if cfg_HAS_CONSTEXPR14

  if (!c.alt_enabled)
  {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
    throw std::invalid_argument("The use of alternatives is disabled");
#else
    return throw_invalid_argument(p, "The use of alternatives is disabled");
#endif
  }

  while (p != pend)
  {
    switch (state)
    {
      case alt_end_state::open:
        if (*p != c.alt_open)
        {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
          throw std::invalid_argument("The given pattern is not a valid alternative");
#else
          return throw_invalid_argument(p, "The given pattern is not a valid alternative");
#endif
        }

        state = alt_end_state::next;
        ++depth;
        break;

      case alt_end_state::next:
        if (*p == c.escape)
        {
          state = alt_end_state::escape;
        }
        else if (c.set_enabled && *p == c.set_open &&
                 is_set(cx::next(p), pend, c, is_set_state::not_or_first))
        {
          p = cx::prev(set_end(cx::next(p), pend, c, set_end_state::not_or_first));
        }
        else if (*p == c.alt_open)
        {
          ++depth;
        }
        else if (*p == c.alt_close)
        {
          --depth;

          if (depth == 0)
          {
            return cx::next(p);
          }
        }

        break;

      case alt_end_state::escape:
        state = alt_end_state::next;
        break;

      default:
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
        throw std::logic_error(
            "The program execution should never end up here throwing this exception");
#else
        return throw_logic_error(
            p, "The program execution should never end up here throwing this exception");
#endif
    }

    p = cx::next(p);
  }

#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
  throw std::invalid_argument("The given pattern is not a valid alternative");
#else
  return throw_invalid_argument(p, "The given pattern is not a valid alternative");
#endif

#else  // !cfg_HAS_CONSTEXPR14

  return !c.alt_enabled
             ? throw std::invalid_argument("The use of alternatives is disabled")
             : p == pend
                   ? throw std::invalid_argument("The given pattern is not a valid alternative")
                   : state == alt_end_state::open
                         ? *p == c.alt_open
                               ? alt_end(cx::next(p), pend, c, alt_end_state::next, depth + 1)
                               : throw std::invalid_argument(
                                     "The given pattern is not a valid alternative")
                         : state == alt_end_state::next
                               ? *p == c.escape
                                     ? alt_end(cx::next(p), pend, c, alt_end_state::escape, depth)
                                     : c.set_enabled && *p == c.set_open &&
                                               is_set(cx::next(p), pend, c,
                                                      is_set_state::not_or_first)
                                           ? alt_end(set_end(cx::next(p), pend, c,
                                                             set_end_state::not_or_first),
                                                     pend, c, state, depth)
                                           : *p == c.alt_open
                                                 ? alt_end(cx::next(p), pend, c, state, depth + 1)
                                                 : *p == c.alt_close
                                                       ? depth == 1 ? cx::next(p)
                                                                    : alt_end(cx::next(p), pend, c,
                                                                              state, depth - 1)
                                                       : alt_end(cx::next(p), pend, c, state, depth)
                               :

                               state == alt_end_state::escape
                                   ? alt_end(cx::next(p), pend, c, alt_end_state::next, depth)
                                   : throw std::logic_error(
                                         "The program execution should never end up here throwing "
                                         "this "
                                         "exception");

#endif  // cfg_HAS_CONSTEXPR14
}

enum class alt_sub_end_state
{
  next,
  escape
};

template <typename PatternIterator>
constexpr PatternIterator alt_sub_end(
    PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    alt_sub_end_state state = alt_sub_end_state::next, int depth = 1)
{
#if cfg_HAS_CONSTEXPR14

  if (!c.alt_enabled)
  {
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
    throw std::invalid_argument("The use of alternatives is disabled");
#else
    return throw_invalid_argument(p, "The use of alternatives is disabled");
#endif
  }

  while (p != pend)
  {
    switch (state)
    {
      case alt_sub_end_state::next:
        if (*p == c.escape)
        {
          state = alt_sub_end_state::escape;
        }
        else if (c.set_enabled && *p == c.set_open &&
                 is_set(cx::next(p), pend, c, is_set_state::not_or_first))
        {
          p = cx::prev(set_end(cx::next(p), pend, c, set_end_state::not_or_first));
        }
        else if (*p == c.alt_open)
        {
          ++depth;
        }
        else if (*p == c.alt_close)
        {
          --depth;

          if (depth == 0)
          {
            return p;
          }
        }
        else if (*p == c.alt_or)
        {
          if (depth == 1)
          {
            return p;
          }
        }

        break;

      case alt_sub_end_state::escape:
        state = alt_sub_end_state::next;
        break;

      default:
#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
        throw std::logic_error(
            "The program execution should never end up here throwing this exception");
#else
        return throw_logic_error(
            p, "The program execution should never end up here throwing this exception");
#endif
    }

    p = cx::next(p);
  }

#if cfg_HAS_FULL_FEATURED_CONSTEXPR14
  throw std::invalid_argument("The given pattern is not a valid alternative");
#else
  return throw_invalid_argument(p, "The given pattern is not a valid alternative");
#endif

#else  // !cfg_HAS_CONSTEXPR14

  return !c.alt_enabled
             ? throw std::invalid_argument("The use of alternatives is disabled")
             : p == pend
                   ? throw std::invalid_argument("The given pattern is not a valid alternative")
                   : state == alt_sub_end_state::next
                         ? *p == c.escape
                               ? alt_sub_end(cx::next(p), pend, c, alt_sub_end_state::escape, depth)
                               : c.set_enabled && *p == c.set_open &&
                                         is_set(cx::next(p), pend, c, is_set_state::not_or_first)
                                     ? alt_sub_end(set_end(cx::next(p), pend, c,
                                                           set_end_state::not_or_first),
                                                   pend, c, state, depth)
                                     : *p == c.alt_open
                                           ? alt_sub_end(cx::next(p), pend, c, state, depth + 1)
                                           : *p == c.alt_close
                                                 ? depth == 1 ? p : alt_sub_end(cx::next(p), pend,
                                                                                c, state, depth - 1)
                                                 : *p == c.alt_or
                                                       ? depth == 1 ? p
                                                                    : alt_sub_end(cx::next(p), pend,
                                                                                  c, state, depth)
                                                       : alt_sub_end(cx::next(p), pend, c, state,
                                                                     depth)
                         :

                         state == alt_sub_end_state::escape
                             ? alt_sub_end(cx::next(p), pend, c, alt_sub_end_state::next, depth)
                             : throw std::logic_error(
                                   "The program execution should never end up here throwing "
                                   "this "
                                   "exception");

#endif  // cfg_HAS_CONSTEXPR14
}

template <typename SequenceIterator, typename PatternIterator,
          typename EqualTo = cx::equal_to<void>>
constexpr match_result<SequenceIterator, PatternIterator> match(
    SequenceIterator s, SequenceIterator send, PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    const EqualTo& equal_to = EqualTo(), bool partial = false, bool escape = false);

template <typename SequenceIterator, typename PatternIterator,
          typename EqualTo = cx::equal_to<void>>
constexpr match_result<SequenceIterator, PatternIterator> match_alt(
    SequenceIterator s, SequenceIterator send, PatternIterator p1, PatternIterator p1end,
    PatternIterator p2, PatternIterator p2end,
    const cards<iterated_item_t<PatternIterator>>& c = cards<iterated_item_t<PatternIterator>>(),
    const EqualTo& equal_to = EqualTo(), bool partial = false)
{
#if cfg_HAS_CONSTEXPR14

  auto result1 = match(s, send, p1, p1end, c, equal_to, true);

  if (result1)
  {
    auto result2 = match(result1.s, send, p2, p2end, c, equal_to, partial);

    if (result2)
    {
      return result2;
    }
  }

  p1 = cx::next(p1end);

  if (p1 == p2)
  {
    return make_match_result(false, s, p1end);
  }

  return match_alt(s, send, p1, alt_sub_end(p1, p2, c), p2, p2end, c, equal_to, partial);

#else  // !cfg_HAS_CONSTEXPR14

  return match(s, send, p1, p1end, c, equal_to, true) &&
                 match(match(s, send, p1, p1end, c, equal_to, true).s, send, p2, p2end, c, equal_to,
                       partial)
             ? match(match(s, send, p1, p1end, c, equal_to, true).s, send, p2, p2end, c, equal_to,
                     partial)
             : cx::next(p1end) == p2
                   ? make_match_result(false, s, p1end)
                   : match_alt(s, send, cx::next(p1end), alt_sub_end(cx::next(p1end), p2, c), p2,
                               p2end, c, equal_to, partial);

#endif  // cfg_HAS_CONSTEXPR14
}

template <typename SequenceIterator, typename PatternIterator, typename EqualTo>
constexpr match_result<SequenceIterator, PatternIterator> match(
    SequenceIterator s, SequenceIterator send, PatternIterator p, PatternIterator pend,
    const cards<iterated_item_t<PatternIterator>>& c, const EqualTo& equal_to, bool partial,
    bool escape)
{
#if cfg_HAS_CONSTEXPR14

  if (p == pend)
  {
    return make_match_result(partial || s == send, s, p);
  }

  if (escape)
  {
    if (s == send || !equal_to(*s, *p))
    {
      return make_match_result(false, s, p);
    }

    return match(cx::next(s), send, cx::next(p), pend, c, equal_to, partial);
  }

  if (*p == c.anything)
  {
    auto result = match(s, send, cx::next(p), pend, c, equal_to, partial);

    if (result)
    {
      return result;
    }

    if (s == send)
    {
      return make_match_result(false, s, p);
    }

    return match(cx::next(s), send, p, pend, c, equal_to, partial);
  }

  if (*p == c.single)
  {
    if (s == send)
    {
      return make_match_result(false, s, p);
    }

    return match(cx::next(s), send, cx::next(p), pend, c, equal_to, partial);
  }

  if (*p == c.escape)
  {
    return match(s, send, cx::next(p), pend, c, equal_to, partial, true);
  }

  if (c.set_enabled && *p == c.set_open && is_set(cx::next(p), pend, c, is_set_state::not_or_first))
  {
    auto result =
        match_set(s, send, cx::next(p), pend, c, equal_to, match_set_state::not_or_first_in);

    if (!result)
    {
      return result;
    }

    return match(cx::next(s), send, set_end(cx::next(p), pend, c, set_end_state::not_or_first),
                 pend, c, equal_to, partial);
  }

  if (c.alt_enabled && *p == c.alt_open && is_alt(cx::next(p), pend, c, is_alt_state::next, 1))
  {
    auto p_alt_end = alt_end(cx::next(p), pend, c, alt_end_state::next, 1);

    return match_alt(s, send, cx::next(p), alt_sub_end(cx::next(p), p_alt_end, c), p_alt_end, pend,
                     c, equal_to, partial);
  }

  if (s == send || !equal_to(*s, *p))
  {
    return make_match_result(false, s, p);
  }

  return match(cx::next(s), send, cx::next(p), pend, c, equal_to, partial);

#else  // !cfg_HAS_CONSTEXPR14

  return p == pend
             ? make_match_result(partial || s == send, s, p)
             : escape
                   ? s == send || !equal_to(*s, *p)
                         ? make_match_result(false, s, p)
                         : match(cx::next(s), send, cx::next(p), pend, c, equal_to, partial)
                   : *p == c.anything
                         ? match(s, send, cx::next(p), pend, c, equal_to, partial)
                               ? match(s, send, cx::next(p), pend, c, equal_to, partial)
                               : s == send ? make_match_result(false, s, p)
                                           : match(cx::next(s), send, p, pend, c, equal_to, partial)
                         : *p == c.single
                               ? s == send ? make_match_result(false, s, p)
                                           : match(cx::next(s), send, cx::next(p), pend, c,
                                                   equal_to, partial)
                               : *p == c.escape
                                     ? match(s, send, cx::next(p), pend, c, equal_to, partial, true)
                                     : c.set_enabled && *p == c.set_open &&
                                               is_set(cx::next(p), pend, c,
                                                      is_set_state::not_or_first)
                                           ? !match_set(s, send, cx::next(p), pend, c, equal_to,
                                                        match_set_state::not_or_first_in)
                                                 ? match_set(s, send, cx::next(p), pend, c,
                                                             equal_to,
                                                             match_set_state::not_or_first_in)
                                                 : match(cx::next(s), send,
                                                         set_end(cx::next(p), pend, c,
                                                                 set_end_state::not_or_first),
                                                         pend, c, equal_to, partial)
                                           : c.alt_enabled && *p == c.alt_open &&
                                                     is_alt(cx::next(p), pend, c,
                                                            is_alt_state::next, 1)
                                                 ? match_alt(
                                                       s, send, cx::next(p),
                                                       alt_sub_end(cx::next(p),
                                                                   alt_end(cx::next(p), pend, c,
                                                                           alt_end_state::next, 1),
                                                                   c),
                                                       alt_end(cx::next(p), pend, c,
                                                               alt_end_state::next, 1),
                                                       pend, c, equal_to, partial)
                                                 : s == send || !equal_to(*s, *p)
                                                       ? make_match_result(false, s, p)
                                                       : match(cx::next(s), send, cx::next(p), pend,
                                                               c, equal_to, partial);

#endif  // cfg_HAS_CONSTEXPR14
}

}  // namespace detail

template <typename Sequence, typename Pattern, typename EqualTo = cx::equal_to<void>>
constexpr full_match_result<const_iterator_t<Sequence>, const_iterator_t<Pattern>> match(
    Sequence&& sequence, Pattern&& pattern,
    const cards<container_item_t<Pattern>>& c = cards<container_item_t<Pattern>>(),
    const EqualTo& equal_to = EqualTo())
{
  return detail::make_full_match_result(
      cx::cbegin(sequence), cx::cend(sequence), cx::cbegin(pattern), cx::cend(pattern),
      detail::match(cx::cbegin(sequence), cx::cend(std::forward<Sequence>(sequence)),
                    cx::cbegin(pattern), cx::cend(std::forward<Pattern>(pattern)), c, equal_to));
}

template <typename Sequence, typename Pattern, typename EqualTo = cx::equal_to<void>,
          typename = typename std::enable_if<!std::is_same<EqualTo, cards_type>::value>::type>
constexpr full_match_result<const_iterator_t<Sequence>, const_iterator_t<Pattern>> match(
    Sequence&& sequence, Pattern&& pattern, const EqualTo& equal_to)
{
  return match(std::forward<Sequence>(sequence), std::forward<Pattern>(pattern),
               cards<container_item_t<Pattern>>(), equal_to);
}

}  // namespace wildcards

#endif  // WILDCARDS_MATCH_HPP
