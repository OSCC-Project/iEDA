/**
 * @file StrMap.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-27
 */

#pragma once

#include "Map.hh"
#include "Set.hh"

namespace ieda {

/**
 * @brief The C-style string case lexical cmp.
 *
 */
struct StrCmp
{
  bool operator()(const char* const& lhs, const char* const& rhs) const;
};

/**
 * @brief The C-style string map.
 *
 * @tparam VALUE
 */
template <typename VALUE>
class StrMap : public Map<const char*, VALUE, StrCmp>
{
};

/**
 * @brief The C-style string set.
 *
 */
class StrSet : public Set<const char*, StrCmp>
{
};

}  // namespace ieda
