#ifndef SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALUTIL_HPP_
#define SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALUTIL_HPP_

#include <map>
#include <set>

namespace eval {

class EvalUtil
{
 public:
  template <typename Key, typename Compare = std::less<Key>>
  static bool exist(const std::set<Key, Compare>& set, const Key& key)
  {
    return (set.find(key) != set.end());
  }

  template <typename Key, typename Value, typename Compare = std::less<Key>>
  static bool exist(const std::map<Key, Value, Compare>& map, const Key& key)
  {
    return (map.find(key) != map.end());
  }
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALUTIL_HPP_
