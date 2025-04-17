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
#include <random>
#include "string/Str.hh"

#include "iir-rust/IRRustC.hh"

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

    file << "  " << node.get_node_name() << ": " << "[ " << node.get_coord().first << " "
         << node.get_coord().second << " " << node.get_layer_id() << " ]" << "\n";
  }

  for (unsigned edge_id = 0; auto& edge : _edges) {
    const char* edge_name = ieda::Str::printf("edge_%d", edge_id++);
    file << edge_name << ":" << "\n";

    file << "  node1: " << edge.get_node1() << "\n";
    file << "  node2: " << edge.get_node2() << "\n";
    file << "  resistance: " << edge.get_resistance() << "\n";
  }

  file.close();
}

/**
 * @brief build the boost geometry segments.
 * 
 * @param special_net 
 * @param line_segment_num stripe segment number.
 * @return std::vector<BGSegment> 
 */
std::vector<BGSegment> IRPGNetlistBuilder::buildBGSegments(
    idb::IdbSpecialNet* special_net, unsigned& line_segment_num) {
  std::vector<BGSegment> bg_segments;
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

  line_segment_num = bg_segments.size();
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
      }
    }
  }


  LOG_INFO << "via segment num: " << bg_segments.size() - line_segment_num;

  for (int i = 0; auto& bg_seg : bg_segments) {
    _rtree.insert(std::make_pair(BGRect(bg_seg.first, bg_seg.second), i++));
  }

  return bg_segments;
}

/**
 * @brief build special net to IRPGNetlist.
 * 
 * @param special_net 
 * @return IRPGNetlist 
 */
void IRPGNetlistBuilder::build(
    idb::IdbSpecialNet* special_net, idb::IdbPin* io_pin,
    std::function<double(unsigned, unsigned)> calc_resistance) {
  IRPGNetlist& pg_netlist = _pg_netlists.emplace_back();
  pg_netlist.set_net_name(special_net->get_net_name());

  unsigned line_segment_num = 0;
  std::vector<BGSegment> bg_segments = buildBGSegments(special_net, line_segment_num);

  // Firstly, get the wire topo point in line segment.
  std::set<std::tuple<int64_t, int64_t, int64_t>> pg_points;
  std::map<int, std::set<IRPGNode*, IRNodeComparator>> segment_to_point;
  for (unsigned i = 0; i < line_segment_num; ++i) {
    // only get line segment intersect point with other segment.
    std::vector<BGValue> result_s;

    auto bg_rect = BGRect(bg_segments[i].first, bg_segments[i].second);
    _rtree.query(bgi::intersects(bg_rect), std::back_inserter(result_s));

    // LOG_INFO << "bg segment " << bg::dsv(bg_rect);

    for (const auto& r : result_s) {
      if (r.second > i) {
        // LOG_INFO << "intersect segment " << bg::dsv(r.first);

        BGRect intersection_result;
        bg::intersection(bg_rect, r.first, intersection_result);
        // LOG_INFO << "bg segment " << i << " intersect with " << r.second
        //          << " intersection_result: " << bg::dsv(intersection_result);

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
          pg_points.insert(left_tuple);
        }
      }
    }
  }

  // calc segment resistance.
  auto calc_segment_resistance = [&calc_resistance](auto* node1, auto* node2) -> double {
    auto [x1, y1] = node1->get_coord();
    auto [x2, y2] = node2->get_coord();
    auto distance = std::abs(x1 - x2) + std::abs(y1 - y2);
    // pg node layer from one first, we need minus one.
    double resistance = calc_resistance(node1->get_layer_id() - 1, distance);

    return resistance;
  };

  // Secondly, build wire topo edge for connect the wire topo point.
  // first we build the line segment edge.
  for (auto& [segment_id, point_set] : segment_to_point) {
    IRPGNode* pg_last_node = nullptr;
    for (auto* pg_node : point_set) {
      if (pg_last_node) {
        auto& pg_edge = pg_netlist.addEdge(pg_last_node, pg_node);

        double resistance = calc_segment_resistance(pg_last_node, pg_node);
        pg_edge.set_resistance(resistance);
      }

      pg_last_node = pg_node;
    }
  }

  unsigned line_edge_num = pg_netlist.getEdgeNum();
  LOG_INFO << "line edge num: " << line_edge_num;
  
  // Then we build the via segment edge.
  std::map<int, std::set<IRPGNode*, IRNodeComparator>> coordy_to_via_segment_nodes; // via nodes sort by coord y, then x.
  // FIXME(to taosimin), should not hard code the instance pin layer.
  unsigned instance_pin_layer = 2;
  // start from via segments.
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
    
    if (bg_start.get<2>() == instance_pin_layer) {
      coordy_to_via_segment_nodes[via_start_node->get_coord().second].insert(via_start_node);
    }
    
    auto& pg_edge = pg_netlist.addEdge(via_start_node, via_end_node);
    // TODO(to taosimin), hard code the via resistance, need know the resistance of via.
    pg_edge.set_resistance(_c_via_resistance);
  }
  
  unsigned via_edge_num = pg_netlist.getEdgeNum() - line_edge_num;
  LOG_INFO << "via edge num: " << via_edge_num;

  // Finally, connect the instance pin list and PG Port.
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  std::uniform_real_distribution<> dis(0.0, 1.0);

  auto instance_pin_list = special_net->get_instance_pin_list()->get_pin_list();
  for (auto* instance_pin : instance_pin_list) {
    // LOG_INFO << "connect instance pin: "
    //          << instance_pin->get_instance()->get_name() << ":"
    //          << instance_pin->get_pin_name();
    
    auto* instance_pin_coord = instance_pin->get_grid_coordinate();
    auto* instance_pin_node = &(pg_netlist.addNode(
        {instance_pin_coord->get_x(), instance_pin_coord->get_y()},
        instance_pin_layer));
    instance_pin_node->set_is_instance_pin();

    std::string node_name = instance_pin->get_instance()->get_name() + ":" +
                            instance_pin->get_pin_name();    
    pg_netlist.addNodeIdToName(instance_pin_node->get_node_id(), std::move(node_name));
    auto& stored_node_name = pg_netlist.getNodeName(instance_pin_node->get_node_id());
    instance_pin_node->set_node_name(stored_node_name.c_str());
    
    int via_last_coord_y = 0;
    std::vector<IRPGNode*> via_connected_nodes;

    // lambda for choose close point.
    auto choose_closer_point = [](int point, int left, int right) {
      int distToLeft = abs(point - left);
      int distToRight = abs(point - right);

      return distToLeft <= distToRight ? left : right;
    };

    for (auto& [coord_y, via_segment_nodes] : coordy_to_via_segment_nodes) {
      // via should be on the same row with the instance pin.
      if (coord_y == instance_pin_coord->get_y()) {
        for (auto* via_segment_node : via_segment_nodes) {
          via_connected_nodes.push_back(via_segment_node);
        }
      } else if (coord_y > instance_pin_coord->get_y()) {
        // choose the nearest distance nodes.
        int choose_y = via_last_coord_y
                           ? choose_closer_point(instance_pin_coord->get_y(),
                                                 via_last_coord_y, coord_y)
                           : coord_y;
        for (auto* via_segment_node : coordy_to_via_segment_nodes[choose_y]) {
          via_connected_nodes.push_back(via_segment_node);
        }
      } else {
        via_last_coord_y = coord_y;
        continue;
      }

      for (auto* via_connected_node : via_connected_nodes) {
        auto& pg_edge =
            pg_netlist.addEdge(via_connected_node, instance_pin_node);
        // hard code the last instance resistance.
        double random_value = dis(gen);
        // random is to disturbance the value for LU decomposition.
        pg_edge.set_resistance(_c_instance_row_resistance + random_value);
      }

      if (via_connected_nodes.size() > 0) {
        // instance pin node has already been connected, break.
        break;
      }
    }
  }

  auto* port_layer_shape = io_pin->get_port_box_list().front();

  // connect io node to the segment node.
  auto layer_id = port_layer_shape->get_layer()->get_id() + 1;
  auto bounding_box = port_layer_shape->get_bounding_box();
  auto middle_point = bounding_box.get_middle_point();

  // create bump node.
  auto* bump_node = &(pg_netlist.addNode(
      {middle_point.get_x(), middle_point.get_y()}, layer_id));
  bump_node->set_is_bump();
  std::string node_name = io_pin->get_pin_name();
  pg_netlist.addNodeIdToName(bump_node->get_node_id(), std::move(node_name));
  auto& stored_node_name = pg_netlist.getNodeName(bump_node->get_node_id());
  bump_node->set_node_name(stored_node_name.c_str());

  // connect bump node to segment node.
  bool is_found = false;
  for (auto& [segment_id, point_set] : segment_to_point) {
    for (auto* pg_node : point_set) {
      if (pg_node->get_layer_id() != layer_id) {
        // should be in one layer
        break;
      }
      // assume pg node should be in one segment with port shape node.
      if ((middle_point.get_x() == pg_node->get_coord().first) ||
          (middle_point.get_y() == pg_node->get_coord().second)) {
        auto& pg_edge = pg_netlist.addEdge(bump_node, pg_node);

        double resistance = calc_segment_resistance(bump_node, pg_node);
        pg_edge.set_resistance(resistance);
        is_found = true;
        break;
      }
    }

    if (is_found) {
      break;
    }
  }

  LOG_FATAL_IF(!is_found) << "bump node is not connected";

  LOG_INFO << "instance pin edge num: " << pg_netlist.getEdgeNum() - via_edge_num - line_edge_num;
  LOG_INFO << "total edge num: " << pg_netlist.getEdgeNum();

  // for debug.
  pg_netlist.printToYaml("/home/taosimin/ir_example/aes/pg_netlist/aes_pg_netlist.yaml");
}

/**
 * @brief create rust pg netlist.
 * 
 */
void IRPGNetlistBuilder::createRustPGNetlist() {
  for (auto &pg_netlist : _pg_netlists) {
    const char* net_name = pg_netlist.get_net_name().c_str();

    auto* rust_pg_netlist = create_pg_netlist(net_name);
    for (auto& pg_node : pg_netlist.get_nodes()) {
      create_pg_node(rust_pg_netlist, &pg_node);
    }

    for (auto& pg_edge : pg_netlist.get_edges()) {
      create_pg_edge(rust_pg_netlist, &pg_edge);
    }

    _rust_pg_netlists.push_back(rust_pg_netlist);
  }
} 

/**
 * @brief estimate rc for the pg netlist.
 * 
 */
void IRPGNetlistBuilder::createRustRCData() {
  auto* rust_pg_netlist_vec_ptr = _rust_pg_netlists.data();
  auto len =  _rust_pg_netlists.size();
  _rust_rc_data = create_rc_data(rust_pg_netlist_vec_ptr, len);
}


}