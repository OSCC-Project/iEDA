// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "cpp_feature.hpp"

#define cfg_HAS_CONSTEXPR14 (__cpp_constexpr >= 201304)

#if cfg_HAS_CONSTEXPR14
#define cfg_constexpr14 constexpr
#else
#define cfg_constexpr14
#endif

#if cfg_HAS_CONSTEXPR14 && defined(__clang__)
#define cfg_HAS_FULL_FEATURED_CONSTEXPR14 1
#else
#define cfg_HAS_FULL_FEATURED_CONSTEXPR14 0
#endif

#endif  // CONFIG_HPP
