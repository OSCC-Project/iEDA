// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once
/**
 * @File Name: iterableHeap.h
 * @Brief : priority_queue with find method and iterator
 * @Author : GuoFan (guofan@ustc.edu)
 * @Version : 1.0
 * @Creat Date : 2023-09-27
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <queue>
#include <vector>

namespace ieda_contest {

namespace astar {

template <typename T>
struct lowerFCost
{
  bool operator()(T n1, T n2)
  {
    if (n1->get_f_cost() == n2->get_f_cost()) {
      return n1->get_h_cost() > n2->get_h_cost();
    }
    return n1->get_f_cost() > n2->get_f_cost();
  }
};

/// @brief a priority_queue which provide iterators for std::make_heap to repair heap structure
/// @tparam T value type
/// @tparam Container std::vector
/// @tparam Compare for A* algorithm, use f and h value
template <class T, class Container = std::vector<T>, class Compare = lowerFCost<T>>
class iterableHeap : public std::priority_queue<T, Container, Compare>
{
 public:
  typedef typename std::priority_queue<T, Container, Compare>::container_type::const_iterator const_iterator;

  const_iterator find(const T& t) const
  {
    auto first = this->c.cbegin();
    auto last = this->c.cend();

    while (first != last) {
      if (*first == t)
        return first;
      ++first;
    }

    return last;
  }

  typename Container::iterator begin() { return this->c.begin(); }
  typename Container::iterator end() { return this->c.end(); }
};

}  // namespace astar

}  // namespace ieda_contest
