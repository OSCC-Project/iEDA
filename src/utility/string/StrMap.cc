/**
 * @file StrMap.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-27
 */

#include "StrMap.hh"

#include "Str.hh"

namespace ieda {

/**
 * @brief The fuction for C-style string cmp.
 *
 * @param lhs
 * @param rhs
 * @return true if lhs < rhs
 * @return false if lhs >=rhs
 */
bool StrCmp::operator()(const char* const& lhs, const char* const& rhs) const
{
  return Str::caseCmp(lhs, rhs) < 0;  // if lhs == rhs, should be false to satify partial order.
}
}  // namespace ieda