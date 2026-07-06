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

#include <span>
#include <utility>
#include <vector>

#include "GroupPool.hh"
#include "Types.hh"

namespace ircx {

struct EdgeEtchInterval {
  Micron a0{0};  // interval start along the edge
  Micron a1{0};  // interval end along the edge

  Micron center{0};
  Micron width{0};

  Micron lo_spacing{kMaxMicron};
  Micron hi_spacing{kMaxMicron};

  Micron thickness{0};
  Micron height{0};
};

class NetEtchProfile
{
 public:
  NetEtchProfile() = default;
  ~NetEtchProfile() = default;

  void appendEdgeIntervals(std::vector<EdgeEtchInterval> intervals)
  {
    edge_interval_groups_.append(std::move(intervals));
  }

  std::span<const EdgeEtchInterval> edgeIntervals(Size edge_id) const
  {
    return edge_interval_groups_.items(edge_id);
  }

  std::span<EdgeEtchInterval> edgeIntervals(Size edge_id)
  {
    return edge_interval_groups_.items(edge_id);
  }

  void clear()
  {
    edge_interval_groups_.clear();
  }

 private:
  GroupPool<EdgeEtchInterval> edge_interval_groups_;
};

}  // namespace ircx
