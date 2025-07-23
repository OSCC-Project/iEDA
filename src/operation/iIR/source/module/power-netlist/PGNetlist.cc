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

#include <fstream>
#include <iostream>
#include <random>

#include "iir-rust/IRRustC.hh"
#include "log/Log.hh"
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
    const char* node_index = ieda::Str::printf("node_%d", node_id++);
    file << node_index << ":"
         << "\n";

    std::string node_name = node.get_node_name();
    if (node.is_via()) {
      node_name += "_via";
    }

    if (node.is_bump()) {
      node_name += "_bump";
    }

    file << "  " << node_name << ": "
         << "[ " << node.get_coord().first << " " << node.get_coord().second
         << " " << node.get_layer_id() << " ]"
         << "\n";
  }

  for (unsigned edge_id = 0; auto& edge : _edges) {
    const char* edge_name = ieda::Str::printf("edge_%d", edge_id++);
    file << edge_name << ":"
         << "\n";

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
    idb::IdbSpecialNet* special_net, unsigned& line_segment_num,
    std::vector<unsigned>& segment_widths) {
  std::vector<BGSegment> bg_segments;
  // build line segment.
  auto* idb_wires = special_net->get_wire_list();
  for (auto* idb_wire : idb_wires->get_wire_list()) {
    if (!idb_wire->get_shiled_name().empty()) {
      // skip the shield wire.
      continue;
    }

    for (auto* idb_segment : idb_wire->get_segment_list()) {
      // line firstly process, we need know line intersect point first.
      if (!idb_segment->is_via()) {
        auto* coord_start = idb_segment->get_point_start();
        auto* coord_end = idb_segment->get_point_second();
        std::string layer_name = idb_segment->get_layer()->get_name();
        unsigned layer_id = getLayerId(layer_name);

        auto bg_segment = BGSegment(
            BGPoint(coord_start->get_x(), coord_start->get_y(), layer_id),
            BGPoint(coord_end->get_x(), coord_end->get_y(), layer_id));
        bg_segments.emplace_back(std::move(bg_segment));
        segment_widths.emplace_back(idb_segment->get_route_width());
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
        std::string top_layer_name = idb_segment->get_layer()->get_name();
        auto top_layer_id = getLayerId(top_layer_name);

        auto bottom_layer_shape = idb_via->get_bottom_layer_shape();
        std::string bottom_layer_name =
            bottom_layer_shape.get_layer()->get_name();
        auto bottom_layer_id = getLayerId(bottom_layer_name);

        // build via segment
        if (bottom_layer_id < top_layer_id) {
          BGPoint via_start(coord->get_x(), coord->get_y(), bottom_layer_id);
          BGPoint via_end(coord->get_x(), coord->get_y(), top_layer_id);

          auto via_path = BGSegment(via_start, via_end);
          bg_segments.emplace_back(std::move(via_path));
          segment_widths.emplace_back(0);  // via segment width is 0.
        } else {
          LOG_FATAL << "bottom layer id is larger than top layer id.";
        }
      }
    }
  }

  LOG_INFO << "via segment num: " << bg_segments.size() - line_segment_num;

  for (int i = 0; auto& bg_seg : bg_segments) {
    _rtree.insert(std::make_pair(BGRect(bg_seg.first, bg_seg.second), i));
    i++;
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
    std::function<double(unsigned, unsigned, unsigned)> calc_resistance) {
  IRPGNetlist& pg_netlist = _pg_netlists.emplace_back();
  auto& special_net_name = special_net->get_net_name();
  pg_netlist.set_net_name(special_net_name);

  LOG_INFO << "building PG netlist for special net " << special_net_name
           << " start";

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, _c_instance_row_resistance * 0.1);

  unsigned line_segment_num = 0;
  std::vector<unsigned> segment_widths;  // store the segment widths in dbu.
  std::vector<BGSegment> bg_segments =
      buildBGSegments(special_net, line_segment_num, segment_widths);

  // FIXME(to taosimin), should not hard code the instance pin layer.
  unsigned instance_pin_layer = 1;

  // Firstly, get the wire topo point in line segment.
  std::set<std::tuple<int64_t, int64_t, int64_t>> pg_points;
  std::map<int, std::set<IRPGNode*, IRNodeComparator>> segment_to_point;
  std::map<int, int>
      intersect_segment_one_layer;  // the segment one layer need connect.
  std::map<int, int> coordy_to_segment_id;  // coord y to segment id, for locate
                                            // instance pin segment.
  for (unsigned i = 0; i < line_segment_num; ++i) {
    // only get line segment intersect point with other segment.
    std::vector<BGValue> result_s;

    auto bg_rect = BGRect(bg_segments[i].first, bg_segments[i].second);
    _rtree.query(bgi::intersects(bg_rect), std::back_inserter(result_s));

    if (bg_segments[i].first.get<2>() == instance_pin_layer) {
      // store the segment id for instance pin layer., the key is coord y.
      coordy_to_segment_id[bg_segments[i].first.get<1>()] = i;
    }

    // LOG_INFO << "bg segment " << bg::dsv(bg_rect);

    int the_line_seg_layer_id = bg_segments[i].first.get<2>();
    for (const auto& r : result_s) {
      int the_result_left_layer_id = r.first.min_corner().get<2>();
      int the_result_right_layer_id = r.first.max_corner().get<2>();
      if (r.second > i) {
        // LOG_INFO << "intersect segment " << bg::dsv(r.first);

        if ((the_result_left_layer_id == the_result_right_layer_id) &&
            (the_result_left_layer_id == the_line_seg_layer_id)) {
          intersect_segment_one_layer[i] = r.second;
          continue;  // skip the segment intersect with itself.
        }

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

        // other segment should have one intersect point with the specify
        // segment.
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
  auto calc_segment_resistance = [&calc_resistance](
                                     auto* node1, auto* node2,
                                     unsigned width_dbu) -> double {
    auto [x1, y1] = node1->get_coord();
    auto [x2, y2] = node2->get_coord();
    auto distance = std::abs(x1 - x2) + std::abs(y1 - y2);
    // pg node layer from one first, we need minus one.
    double resistance =
        calc_resistance(node1->get_layer_id(), distance, width_dbu);

    return resistance;
  };

  // Secondly, build wire topo edge for connect the wire topo point.
  // first we build the line segment edge.
  for (auto& [segment_id, point_set] : segment_to_point) {
    IRPGNode* pg_last_node = nullptr;
    for (auto* pg_node : point_set) {
      if (pg_last_node) {
        auto& pg_edge = pg_netlist.addEdge(pg_last_node, pg_node);

        auto width_dbu = segment_widths[segment_id];

        double resistance =
            calc_segment_resistance(pg_last_node, pg_node, width_dbu);

        double random_value = dis(gen);
        pg_edge.set_resistance(resistance + random_value);
      }

      pg_last_node = pg_node;
    }
  }

  // for the segment intersect in one layer, we need connect the the point
  // of the segment.
  for (auto [seg_id1, seg_id2] : intersect_segment_one_layer) {
    // connect the last point of seg_id1 and first point of seg_id2.
    auto* node1 = *(segment_to_point[seg_id1]
                        .rbegin());  // Get last element of first segment
    LOG_INFO_IF(!node1) << "The nodes of segment " << seg_id1 << " ("
                        << bg_segments[seg_id1].first.get<0>() << " "
                        << bg_segments[seg_id1].first.get<1>() << " "
                        << bg_segments[seg_id1].first.get<2>() << ")"
                        << " is nullptr.";
    auto* node2 = *(segment_to_point[seg_id2]
                        .begin());  // Get first element of second segment
    LOG_INFO_IF(!node2) << "The nodes of segment " << seg_id2 << " ("
                        << bg_segments[seg_id2].first.get<0>() << " "
                        << bg_segments[seg_id2].first.get<1>() << " "
                        << bg_segments[seg_id2].first.get<2>() << ")"
                        << " is nullptr.";

    if (!node1 || !node2) {
      continue;
    }

    if (node1->get_coord().first == node2->get_coord().first) {
      LOG_FATAL_IF(node1->get_coord().second > node2->get_coord().second)
          << "node1 coord y should be less than node2 coord y.";
    } else if (node1->get_coord().second == node2->get_coord().second) {
      LOG_FATAL_IF(node1->get_coord().first > node2->get_coord().first)
          << "node1 coord x should be less than node2 coord x.";
    }

    // Add edge between intersecting segments
    auto& pg_edge = pg_netlist.addEdge(node1, node2);
    auto width_dbu = segment_widths[seg_id1];

    double resistance = calc_segment_resistance(node1, node2, width_dbu);

    double random_value = dis(gen);
    pg_edge.set_resistance(resistance + random_value);
  }

  unsigned line_edge_num = pg_netlist.getEdgeNum();
  LOG_INFO << "line edge num: " << line_edge_num;

  // Then we build the via segment edge.
  std::map<int, std::set<IRPGNode*, IRNodeComparator>>
      coordy_to_via_segment_nodes;  // via nodes sort by coord y, then x.
  // start from via segments.
  for (unsigned i = line_segment_num; i < bg_segments.size(); ++i) {
    auto bg_start = bg_segments[i].first;
    auto bg_end = bg_segments[i].second;

    auto* via_start_node = pg_netlist.findNode(
        {bg_start.get<0>(), bg_start.get<1>()}, bg_start.get<2>());
    if (!via_start_node) {
      via_start_node = &(pg_netlist.addNode(
          {bg_start.get<0>(), bg_start.get<1>()}, bg_start.get<2>()));
      via_start_node->set_is_via();
    }

    auto* via_end_node = pg_netlist.findNode({bg_end.get<0>(), bg_end.get<1>()},
                                             bg_end.get<2>());
    if (!via_end_node) {
      via_end_node = &(pg_netlist.addNode({bg_end.get<0>(), bg_end.get<1>()},
                                          bg_end.get<2>()));
      via_end_node->set_is_via();
    }

    auto via_bottom_layer = bg_start.get<2>();

    if (via_bottom_layer == instance_pin_layer) {
      coordy_to_via_segment_nodes[via_start_node->get_coord().second].insert(
          via_start_node);
    }

    auto& pg_edge = pg_netlist.addEdge(via_start_node, via_end_node);
    double random_value = dis(gen);
    // TODO(to taosimin), hard code the via resistance, need know the
    // resistance of via.
    double via_resistance = getViaResistance(via_bottom_layer);
    pg_edge.set_resistance(via_resistance + random_value);
  }

  unsigned via_edge_num = pg_netlist.getEdgeNum() - line_edge_num;
  LOG_INFO << "via edge num: " << via_edge_num;

  // Finally, connect the instance pin list and PG Port.
  auto instance_pin_list = special_net->get_instance_pin_list()->get_pin_list();
  for (auto* instance_pin : instance_pin_list) {
    // LOG_INFO << "connect instance pin: "
    //          << instance_pin->get_instance()->get_name() << ":"
    //          << instance_pin->get_pin_name();

    auto instance_name = instance_pin->get_instance()->get_name();
    if (!_instance_names.contains(instance_name)) {
      // skip the no power instance.
      LOG_INFO_FIRST_N(10) << "skip the no power instance: " << instance_name
                           << " in wire topo";
      continue;
    }

    auto* instance_pin_coord = instance_pin->get_grid_coordinate();
    auto* instance_pin_node = &(pg_netlist.addNode(
        {instance_pin_coord->get_x(), instance_pin_coord->get_y()},
        instance_pin_layer));
    instance_pin_node->set_is_instance_pin();

    std::string node_name = instance_name + ":" + instance_pin->get_pin_name();
    pg_netlist.addNodeIdToName(instance_pin_node->get_node_id(),
                               std::move(node_name));
    auto& stored_node_name =
        pg_netlist.getNodeName(instance_pin_node->get_node_id());
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

        int coord_y = via_connected_node->get_coord().second;
        LOG_FATAL_IF(!coordy_to_segment_id.contains(coord_y))
            << "not found the line segment id for coord y: " << coord_y;
        int line_segment_id = coordy_to_segment_id[coord_y];
        double width_dbu =
            segment_widths[line_segment_id];  // instance row width.
        // random is to disturbance the value for LU decomposition.

        double resistance = calc_segment_resistance(
            via_connected_node, instance_pin_node, width_dbu);
        pg_edge.set_resistance(resistance + random_value);
      }

      if (via_connected_nodes.size() > 0) {
        // instance pin node has already been connected, break.
        break;
      }
    }
  }

  idb::IdbLayerShape* port_layer_shape = nullptr;
  idb::IdbCoordinate<int32_t> middle_point;
  int layer_id = 0;
  if (io_pin->get_port_box_list().size() > 0) {
    port_layer_shape = io_pin->get_port_box_list().front();
    // connect io node to the segment node.
    auto layer_name = port_layer_shape->get_layer()->get_name();
    layer_id = getLayerId(layer_name);
    auto bounding_box = port_layer_shape->get_bounding_box();
    middle_point = bounding_box.get_middle_point();
  } else {
    auto* io_port = io_pin->get_term()->get_port_list().front();
    if (io_port->get_layer_shape().size() == 0) {
      LOG_FATAL << "io port layer shape is empty";
    }
    port_layer_shape = io_port->get_layer_shape().front();
    auto layer_name = port_layer_shape->get_layer()->get_name();
    layer_id = getLayerId(layer_name);
    middle_point = *(io_port->get_coordinate());
  }

  // create bump node.
  auto* bump_node = &(pg_netlist.addNode(
      {middle_point.get_x(), middle_point.get_y()}, layer_id));
  bump_node->set_is_bump();
  std::string node_name = io_pin->get_pin_name();
  pg_netlist.addNodeIdToName(bump_node->get_node_id(), std::move(node_name));
  auto& stored_node_name = pg_netlist.getNodeName(bump_node->get_node_id());
  bump_node->set_node_name(stored_node_name.c_str());

  pg_netlist.addBumpNode(bump_node);

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

        auto width_dbu = segment_widths[segment_id];

        double resistance =
            calc_segment_resistance(bump_node, pg_node, width_dbu);
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

  LOG_INFO << "net " << special_net_name
           << " bump node location: " << middle_point.get_x() / _dbu << " "
           << middle_point.get_y() / _dbu << " " << getLayerName(layer_id);

  IRNodeLoc bump_node_loc{{middle_point.get_x() / (double)_dbu,
                           middle_point.get_y() / (double)_dbu},
                          getLayerName(layer_id)};
  _net_bump_node_locs[special_net_name] = bump_node_loc;

  LOG_INFO << "instance pin edge num: "
           << pg_netlist.getEdgeNum() - via_edge_num - line_edge_num;
  LOG_INFO << "total edge num: " << pg_netlist.getEdgeNum();

  LOG_INFO << "building PG netlist for special net " << special_net_name
           << " end";

  // for debug.
  // if (special_net_name == "VDD") {
  //   pg_netlist.printToYaml(
  //       "/home/taosimin/iEDA24/iEDA/bin/aes_pg_netlist_06_23.yaml");
  // }
}

/**
 * @brief create rust pg netlist.
 *
 */
void IRPGNetlistBuilder::createRustPGNetlist() {
  for (auto& pg_netlist : _pg_netlists) {
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
unsigned IRPGNetlistBuilder::createRustRCData() {
  auto* rust_pg_netlist_vec_ptr = _rust_pg_netlists.data();
  auto len = _rust_pg_netlists.size();
  _rust_rc_data = create_rc_data(rust_pg_netlist_vec_ptr, len);
  return 1;
}

/**
 * @brief calc every node resistance from bump node.
 *
 */
void IRPGNetlistBuilder::calcResistanceFromBumpNode(std::string net_name) {
  // auto* pg_netlist = getPGNetlist(net_name);

  // TODO(to taosimin), need to calc the resistance from bump node to all
  // segment node.
}

/**
 * @brief get the via resistance, maybe need calc by layer shape.
 *
 * @param bottom_layer_id
 * @return double
 */
double IRPGNetlistBuilder::getViaResistance(unsigned bottom_layer_id) {
  double via_resistance = _c_via_resistance;

  if (bottom_layer_id == 1) {
    via_resistance = 7.597;
  } else if (bottom_layer_id == 2) {
    via_resistance = 3.799;
  } else if (bottom_layer_id == 3) {
    via_resistance = 3.799;
  } else if (bottom_layer_id == 4) {
    via_resistance = 3.799;
  } else if (bottom_layer_id == 5) {
    via_resistance = 3.799;
  } else if (bottom_layer_id == 6) {
    via_resistance = 3.7997;
  } else if (bottom_layer_id == 7) {
    via_resistance = 0.085;
  } else if (bottom_layer_id == 8) {
    via_resistance = 0.017;
  }

  return via_resistance;
}

}  // namespace iir