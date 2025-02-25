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
#include <iostream>
#include <fstream>
#include "string/Str.hh"

namespace iir {

/**
 * @brief print the pg netlist to yaml file.
 * 
 * @param yaml_path 
 */
void IRPGNetlist::printToYaml(std::string yaml_path) {
  std::ofstream file(yaml_path, std::ios::trunc);
  for (unsigned node_id = 0; auto& node : _nodes) {
    const char* node_name = ieda::Str::printf("node_%d", node_id++);
    file << node_name << ":" << "\n";

    file << "  coord: " << "[ " << node.get_coord().first << " "
         << node.get_coord().second << " " << node.get_layer_id() << " ]" << "\n";
  }

  for (unsigned edge_id = 0; auto& edge : _edges) {
    const char* edge_name = ieda::Str::printf("edge_%d", edge_id++);
    file << edge_name << ":" << "\n";

    file << "  node1: " << edge.get_node1()->get_node_id() << "\n";
    file << "  node2: " << edge.get_node2()->get_node_id() << "\n";
  }

  file.close();
}

/**
 * @brief build special net to IRPGNetlist.
 * 
 * @param special_net 
 * @return IRPGNetlist 
 */
IRPGNetlist IRPGNetlistBuilder::build(idb::IdbSpecialNet* special_net) {
  IRPGNetlist pg_netlist;
  std::vector<BGSegment> bg_segments;
  // std::vector<BGRect> bg_rects;

  // build line segment.
  auto* idb_wires = special_net->get_wire_list();
  for (auto* idb_wire : idb_wires->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      // line firstly process, we need know line intersect point first.
      if (idb_segment->is_line()) {
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

  auto line_segment_num = bg_segments.size();
  LOG_INFO << "line segment num: " << bg_segments.size();
  
  // then build wire segment.
  for (auto* idb_wire : idb_wires->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      // via secondly process, via intersect point is fixed.
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
      }
    }
  }

  LOG_INFO << "via segment num: " << bg_segments.size() - line_segment_num;

  for (int i = 0; auto& bg_seg : bg_segments) {
    _rtree.insert(std::make_pair(BGRect(bg_seg.first, bg_seg.second), i++));
  }

  // LOG_INFO << "rect start index: " << bg_segments.size();
  // for (int i = bg_segments.size(); auto& bg_rect : bg_rects) {
  //   _rtree.insert(std::make_pair(bg_rect, i++));
  // }

  // get the wire topo point in line segment.
  std::set<std::tuple<int64_t, int64_t, int64_t>> pg_points;
  std::map<int, std::set<IRPGNode*, IRNodeComparator>> segment_to_point;
  for (unsigned i = 0; i < line_segment_num; ++i) {
    // only get line segment intersect point with other segment.
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

        auto& left_bottom = intersection_result.min_corner();
        auto& right_top = intersection_result.max_corner();

        auto left_x = bg::get<0>(left_bottom);
        auto left_y = bg::get<1>(left_bottom);
        auto left_layer_id = bg::get<2>(left_bottom);
        auto left_tuple = std::make_tuple(left_x, left_y, left_layer_id);

        auto right_x = bg::get<0>(right_top);
        auto right_y = bg::get<1>(right_top);
        auto right_layer_id = bg::get<2>(right_top);
        auto right_tuple = std::make_tuple(right_x, right_y, right_layer_id);

        LOG_FATAL_IF(left_tuple != right_tuple)
            << "instersect box should be one point";

        if (!pg_points.contains(left_tuple)) {
          auto& pg_node = pg_netlist.addNode({left_x, left_y}, left_layer_id);
          segment_to_point[i].insert(&pg_node);
        }
      }
    }
  }

  // build wire topo edge for connect the wire topo point.
  // first we build the line segment edge.
  for (auto& [segment_id, point_set] : segment_to_point) {
    IRPGNode* pg_last_node = nullptr;
    for (auto* pg_node : point_set) {
      if (pg_last_node) {
        pg_netlist.addEdge(pg_last_node, pg_node);
      }

      pg_last_node = pg_node;
    }
  }

  LOG_INFO << "line edge num: " << pg_netlist.getEdgeNum();
  
  // the we build the via segment edge.
  for (unsigned i = line_segment_num; i < bg_segments.size(); ++i) {
    auto bg_start = bg_segments[i].first;
    auto bg_end = bg_segments[i].second;

    auto* via_start_node = pg_netlist.findNode({bg_start.get<0>(), bg_start.get<1>()}, bg_start.get<2>());
    if (!via_start_node) {
      via_start_node = &(pg_netlist.addNode({bg_start.get<0>(), bg_start.get<1>()}, bg_start.get<2>()));
    }
    
    auto* via_end_node = pg_netlist.findNode({bg_end.get<0>(), bg_end.get<1>()}, bg_end.get<2>());
    if (!via_end_node) {
      via_end_node = &(pg_netlist.addNode({bg_end.get<0>(), bg_end.get<1>()}, bg_end.get<2>()));
    }
    
    pg_netlist.addEdge(via_start_node, via_end_node);
  }

  LOG_INFO << "via edge num: " << line_segment_num;
  LOG_INFO << "total edge num: " << pg_netlist.getEdgeNum();

  // for debug.
  pg_netlist.printToYaml("/home/taosimin/ir_example/aes/pg_netlist/aes_pg_netlist.yaml");

  return pg_netlist;
}


}