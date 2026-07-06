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

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "GroupPool.hh"
#include "Types.hh"

namespace ircx {

inline constexpr F32 kDefaultWindowUm = 2;
inline constexpr F32 kDefaultBucketUm = 5;

class TopoEdge;

struct CrossOverlapSub {
  Dbu a0{0};
  Dbu a1{0};
  Size abv_layer{0};  // absolute layer; 0 = none
  Size blw_layer{0};  // absolute layer; 0 = SUBSTRATE
};

struct DiagCoupSub {
  Dbu a0{0};
  Dbu a1{0};
  TopoEdge* neighbor{nullptr};
  Dbu dist{0};         // center-to-center, fixed direction, dbu
  int16_t layer_delta{0};  // target layer = this_edge.layer + layer_delta
};

struct EdgeEnvironmentInterval {
  Dbu a0{0};
  Dbu a1{0};

  Dbu lo_spacing{kMaxDbu};  // center-to-center; kMaxDbu = no neighbor
  Dbu hi_spacing{kMaxDbu};

  const TopoEdge* lo_adjacent{nullptr};
  const TopoEdge* hi_adjacent{nullptr};

  std::vector<CrossOverlapSub> cross_segs;
  std::vector<DiagCoupSub> diag_segs;
};

class NetEnvironment
{
 public:
  NetEnvironment() = default;
  ~NetEnvironment() = default;

  void appendEdgeIntervals(std::vector<EdgeEnvironmentInterval> intervals)
  {
    edge_interval_groups_.append(std::move(intervals));
  }

  std::span<const EdgeEnvironmentInterval> edgeIntervals(Size edge_id) const
  {
    return edge_interval_groups_.items(edge_id);
  }

  void clear()
  {
    edge_interval_groups_.clear();
  }

 private:
  GroupPool<EdgeEnvironmentInterval> edge_interval_groups_;
};

}  // namespace ircx
