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

#include "CTSAPI.hpp"
#include "Params.h"
#include "Topology.h"
#include "ZstNode.h"
#include "pgl.h"

namespace icts {
using std::vector;

class ZeroSkewTree {
 public:
  ZeroSkewTree() {}
  ZeroSkewTree(const ZstParams& params) { _params = params; }
  ~ZeroSkewTree() = default;

  template <typename T>
  void build(Topology<ZstNode<T>>& topo) {
    if (topo.size() > 2) {
      buildMergeSegment(topo);
      findExactPlacement(topo);
    }
  }

 private:
  template <typename T>
  void buildMergeSegment(Topology<ZstNode<T>>& topo) {
    auto pv_itr = topo.postorder_vertexs();
    for (auto itr = pv_itr.first; itr != pv_itr.second; ++itr) {
      auto& zst_node = *itr;

      if (itr.is_leaf()) {
        leafMergeSegment(zst_node);
      } else {
        auto& left = *itr.left();
        auto& right = *itr.right();

        auto parent = merge(left, right);
        zst_node.copy_message(parent);
      }
    }
  }

  template <typename T>
  void findExactPlacement(Topology<ZstNode<T>>& topo) {
    auto pv_itr = topo.preorder_vertexs();
    for (auto itr = pv_itr.first; itr != pv_itr.second; ++itr) {
      auto& node = *itr;
      if (itr.is_root()) {
        placeRoot(node);
      } else {
        Polygon trr;
        auto& parent = *itr.parent();
        Point parent_loc = parent.get_loc();
        int radius = node.get_edge_length();
        pgl::tilted_rect_region(trr, parent_loc, radius);

        vector<Segment> inter_segs;
        Segment seg;
        auto ms = node.get_merge_segment();

        Point node_loc;
        if (pgl::bound_intersects(inter_segs, ms, trr)) {
          pgl::longest_segment(seg, inter_segs);
          node_loc = seg.low();
        } else {
          node_loc = pgl::closest_point(parent_loc, ms);
        }
        node.set_loc(node_loc);
      }
    }
    auto post_itr = topo.postorder_vertexs();
    for (auto itr = post_itr.first; itr != post_itr.second; ++itr) {
      auto& node = *itr;
      if (!itr.is_root()) {
        auto& parent = *itr.parent();
        auto parent_loc = parent.get_loc();
        auto cur_loc = node.get_loc();
        auto dist = pgl::manhattan_distance(parent_loc, cur_loc);
        auto sub_wirelength =
            DataTraits<T>::getSubWirelength(parent.get_data());
        DataTraits<T>::setSubWirelength(parent.get_data(),
                                        std::max(sub_wirelength, dist));
      }
    }
  }

  template <typename T>
  ZstNode<T> merge(ZstNode<T>& left, ZstNode<T>& right) {
    ZstNode<T> node;
    Segment ms_l = left.get_merge_segment();
    Segment ms_r = right.get_merge_segment();
    auto delay_l = left.get_delay();
    auto delay_r = right.get_delay();

    ZstDelayFunc delay_func(ms_l, delay_l, ms_r, delay_r, _params);

    std::pair<Coordinate, Coordinate> edge_pair;
    bool have_min_delay = delay_func.edge_length(edge_pair);
    left.set_edge_length(edge_pair.first);
    right.set_edge_length(edge_pair.second);

    if (have_min_delay) {
      Polygon trr_a, trr_b;
      vector<Segment> inter_set;
      Segment ms;
      pgl::tilted_rect_region(trr_a, ms_l, left.get_edge_length());
      pgl::tilted_rect_region(trr_b, ms_r, right.get_edge_length());
      pgl::bound_intersects(inter_set, trr_a, trr_b);
      pgl::longest_segment(ms, inter_set);
      node.set_merge_segment(ms);

      node.set_delay(delay_func.delay());
    } else {
      // 实现不完整 详见 ZST 论文第五页 图3
      if (left.get_edge_length() == 0) {
        node.set_merge_segment(ms_l);
        node.set_delay(delay_l);
      } else {
        node.set_merge_segment(ms_r);
        node.set_delay(delay_r);
      }
    }

    return node;
  }

  template <typename T>
  void placeRoot(ZstNode<T>& node) {
    auto ms = node.get_merge_segment();
    node.set_loc(ms.low());
  }

  template <typename T>
  void leafMergeSegment(ZstNode<T>& node) {
    auto node_loc = node.get_loc();
    node.set_merge_segment(Segment(node_loc, node_loc));
  }

 private:
  ZstParams _params;
};

}  // namespace icts