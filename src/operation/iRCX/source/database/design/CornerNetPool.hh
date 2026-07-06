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

#include <vector>

#include "Types.hh"

namespace ircx {

struct CornerNetId
{
  Size corner_idx{kMaxSize};
  Size net_idx{kMaxSize};
};

template <typename T>
class CornerNetPool
{
 public:
  CornerNetPool() = default;
  ~CornerNetPool() = default;

  void init(Size corner_num, Size net_num)
  {
    corner_num_ = corner_num;
    net_num_ = net_num;
    items_.clear();
    items_.resize(corner_num_ * net_num_);
  }

  void clear()
  {
    corner_num_ = 0;
    net_num_ = 0;
    items_.clear();
  }

  bool empty() const { return items_.empty(); }
  Size corner_num() const { return corner_num_; }
  Size net_num() const { return net_num_; }
  Size size() const { return items_.size(); }

  T& at(CornerNetId id) { return items_.at(index(id)); }
  const T& at(CornerNetId id) const { return items_.at(index(id)); }

 private:
  Size index(CornerNetId id) const
  {
    return id.corner_idx * net_num_ + id.net_idx;
  }

  Size corner_num_{0};
  Size net_num_{0};
  std::vector<T> items_;
};

}  // namespace ircx
