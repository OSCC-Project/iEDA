// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PGNetlist.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The pg netlist for wire topo analysis, and esitmate the wire R.
 * @version 0.1
 * @date 2025-02-22
 */
#include "PGNetlist.hh"
#include "log/Log.hh"

namespace iir {

/**
 * @brief build special net to IRPGNetlist.
 * 
 * @param special_net 
 * @return IRPGNetlist 
 */
IRPGNetlist IRPGNetlistBuilder::build(idb::IdbSpecialNet* special_net) {
  IRPGNetlist pg_netlist;
  std::vector<BGSegment> bg_segments;
  std::vector<BGRect> bg_rects;

  auto* idb_wires = special_net->get_wire_list();
  for (auto* idb_wire : idb_wires->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      if (idb_segment->is_via()) {
        auto* idb_via = idb_segment->get_via();
        auto* coord = idb_via->get_coordinate();

        auto top_layer_shape = idb_via->get_top_layer_shape();
        auto top_layer_id = top_layer_shape.get_layer()->get_id() + 1;

        auto bottom_layer_shape = idb_via->get_bottom_layer_shape();
        auto bottom_layer_id = bottom_layer_shape.get_layer()->get_id() + 1;
        
        // build cut segment
        BGPoint cut_start(coord->get_x(), coord->get_y(), bottom_layer_id);
        BGPoint cut_end(coord->get_x(), coord->get_y(), top_layer_id);
        auto cut_path = BGSegment(cut_start, cut_end);
        bg_segments.emplace_back(std::move(cut_path));

        // build top and bottom rect. not used now.
        // auto enclosure_top = idb_via->get_top_layer_shape();
        // for (auto* rect : enclosure_top.get_rect_list()) {
        //   BGPoint low(rect->get_low_x(), rect->get_low_y(), top_layer_id);
        //   BGPoint high(rect->get_high_x(), rect->get_high_y(), top_layer_id);
        //   auto bg_rect = BGRect(low, high);
        //   bg_rects.emplace_back(std::move(bg_rect));
        // }

        // auto enclosure_bottom = idb_via->get_bottom_layer_shape();
        // for (auto* rect : enclosure_bottom.get_rect_list()) {
        //   BGPoint low(rect->get_low_x(), rect->get_low_y(), bottom_layer_id);
        //   BGPoint high(rect->get_high_x(), rect->get_high_y(), bottom_layer_id);
        //   auto bg_rect = BGRect(low, high);
        //   bg_rects.emplace_back(std::move(bg_rect));
        // }
      } else if (idb_segment->is_line()) {
        auto* coord_start = idb_segment->get_point_start();
        auto* coord_end = idb_segment->get_point_second();
        int layer_id = idb_segment->get_layer()->get_id() + 1;

        auto bg_segment =
            BGSegment(BGPoint(coord_start->get_x(), coord_start->get_y(), layer_id),
                           BGPoint(coord_end->get_x(), coord_end->get_y(), layer_id));
        bg_segments.emplace_back(std::move(bg_segment));
      }
    }
  }

  for (int i = 0; auto& bg_seg : bg_segments) {
    _rtree.insert(std::make_pair(BGRect(bg_seg.first, bg_seg.second), i++));
  }

  LOG_INFO << "rect start index: " << bg_segments.size();
  for (int i = bg_segments.size(); auto& bg_rect : bg_rects) {
    _rtree.insert(std::make_pair(bg_rect, i++));
  }

  for (unsigned i = 0; i < bg_segments.size(); ++i) {
    std::vector<BGValue> result_s;

    auto bg_rect = BGRect(bg_segments[i].first, bg_segments[i].second);
    _rtree.query(bgi::intersects(bg_rect), std::back_inserter(result_s));

    LOG_INFO << "bg segment " << bg::dsv(bg_rect);

    for (const auto& r : result_s) {
      if (r.second > i) {
        LOG_INFO << "intersect segment " << bg::dsv(r.first); 

        BGRect intersection_result;
        bg::intersection(bg_rect, r.first, intersection_result);
        LOG_INFO << "bg segment " << i << " intersect with " << r.second
                 << " intersection_result: " << bg::dsv(intersection_result);
      }
    }
  }

  return pg_netlist;
}
}