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

#include <queue>

namespace irt {

template <class T, class Container = std::vector<T>, class Compare = std::less<typename Container::value_type>>
class PriorityQueue : public std::priority_queue<T, Container, Compare>
{
 public:
  PriorityQueue() = default;
  ~PriorityQueue() = default;
  // function
  Container::iterator begin() { return this->c.begin(); }
  Container::iterator end() { return this->c.end(); }

 private:
};

}  // namespace irt
