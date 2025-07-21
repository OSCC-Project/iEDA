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
/**
 * @file lm_net_graph_gen.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-29
 * @brief Construct graph from net data.
 */

#include "lm_net_graph_gen.hh"

#include <algorithm>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <fstream>
#include <functional>
#include <tuple>

#include "IdbPins.h"
#include "idm.h"
#include "log/Log.hh"
#ifdef BUILD_LM_GUI
#include "lm_graph_gui.hh"
#endif

namespace ilm {
void LmNetGraphGenerator::initLayerMap()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();
  auto idb_layer_1st = dmInst->get_config().get_routing_layer_1st();
  auto layers = idb_layers->get_layers();

  // Define the range starting from the 'idb_layer_1st'
  // and ending before the first non-routing, non-cut layer
  auto layer_range = layers | std::views::drop_while([&](auto* layer) -> bool { return layer->get_name() != idb_layer_1st; })
                     | std::views::take_while([&](auto* layer) -> bool { return layer->is_routing() || layer->is_cut(); });

  int index = 0;
  std::ranges::for_each(layer_range, [&](auto* layer) -> void { _layer_map[layer->get_name()] = index++; });

  LOG_INFO << "Layer number : " << index;
}
WireGraph LmNetGraphGenerator::buildGraph(idb::IdbNet* idb_net) const
{
  if (isCornerCase(idb_net)) {
    return buildCornerCaseGraph(idb_net);
  }
  auto topo_graph = buildTopoGraph(idb_net);
  auto wire_graph = buildWireGraph(topo_graph);
  return wire_graph;
}
std::vector<WireGraph> LmNetGraphGenerator::buildGraphs() const
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_nets = idb_design->get_net_list();
  std::vector<WireGraph> graphs;
  std::ranges::transform(idb_nets->get_net_list(), std::back_inserter(graphs),
                         [&](auto* idb_net) -> WireGraph { return buildGraph(idb_net); });
  return graphs;
}
bool LmNetGraphGenerator::isCornerCase(idb::IdbNet* idb_net) const
{
  auto* driver_pin = idb_net->get_driving_pin();
  if (!driver_pin) {
    return true;
  }
  auto io_pins = idb_net->get_io_pins()->get_pin_list();
  if (io_pins.size() != 1) {
    return false;
  }

  auto* io_pin = io_pins.front();
  auto io_shapes = io_pin->get_port_box_list();
  if (io_shapes.empty()) {
    return true;
  }

  return false;
}
WireGraph LmNetGraphGenerator::buildCornerCaseGraph(idb::IdbNet* idb_net) const
{
  // corner case: only one IO pin and one PAD
  WireGraph wire_graph;
  auto* pad = idb_net->get_instance_pin_list()->get_pin_list().front();

  auto* pin_shape = pad->get_port_box_list().front();
  auto rect = pin_shape->get_rect_list().front();
  auto low_x = rect->get_low_x();
  auto low_y = rect->get_low_y();
  auto high_x = rect->get_high_x();
  auto high_y = rect->get_high_y();
  auto layer_id = _layer_map.at(pin_shape->get_layer()->get_name());
  auto io_location = LayoutDefPoint(low_x, low_y, layer_id);

  auto pad_vertex = boost::add_vertex(wire_graph);
  wire_graph[pad_vertex].x = (low_x + high_x) / 2;
  wire_graph[pad_vertex].y = (low_y + high_y) / 2;
  wire_graph[pad_vertex].layer_id = layer_id;
  wire_graph[pad_vertex].is_pin = true;

  auto io_vertex = boost::add_vertex(wire_graph);
  wire_graph[io_vertex].x = (low_x + high_x) / 2;
  wire_graph[io_vertex].y = (low_y + high_y) / 2;
  wire_graph[io_vertex].layer_id = layer_id;
  wire_graph[io_vertex].is_pin = true;

  auto edge = boost::add_edge(pad_vertex, io_vertex, wire_graph).first;
  wire_graph[edge].path.emplace_back(std::make_pair(io_location, io_location));

  return wire_graph;
}
TopoGraph LmNetGraphGenerator::buildTopoGraph(idb::IdbNet* idb_net) const
{
  // LOG_INFO << "Net name : " << idb_net->get_net_name();

  TopoGraph graph;
  // Build Instances' pins and IO pins
  auto* driver_pin = idb_net->get_driving_pin();
  std::vector<idb::IdbPin*> pins;
  std::ranges::copy(idb_net->get_instance_pin_list()->get_pin_list(), std::back_inserter(pins));
  std::ranges::copy(idb_net->get_io_pins()->get_pin_list(), std::back_inserter(pins));
  std::ranges::for_each(pins, [&](auto* idb_pin) -> void {
    auto vertex = boost::add_vertex(graph);
    auto* layout_pin = new LayoutPin(idb_net->get_net_name(), idb_pin->get_pin_name(), idb_pin == driver_pin);
    graph[vertex].content = layout_pin;
    for (auto* layer_shape : idb_pin->get_port_box_list()) {
      auto layer_name = layer_shape->get_layer()->get_name();
      auto layer_id = _layer_map.at(layer_name);
      if (layer_shape->is_via()) {
        std::ranges::for_each(layer_shape->get_rect_list(), [&](auto* rect) -> void { layout_pin->addViaCut(rect, layer_id); });
      } else {
        std::ranges::for_each(layer_shape->get_rect_list(), [&](auto* rect) -> void { layout_pin->addPinShape(rect, layer_id); });
      }
    }
  });

  // Build Wires
  auto* idb_wires = idb_net->get_wire_list();
  for (auto* idb_wire : idb_wires->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      if (idb_segment->is_rect()) {
        auto* coord_start = idb_segment->get_point_start();
        auto* delta_rect = idb_segment->get_delta_rect();
        auto* rect = new idb::IdbRect(delta_rect);
        rect->moveByStep(coord_start->get_x(), coord_start->get_y());
        auto layer_id = _layer_map.at(idb_segment->get_layer()->get_name());
        auto* patch = new LayoutPatch(rect, layer_id);
        delete rect;
        auto vertex = boost::add_vertex(graph);
        graph[vertex].content = patch;
      }
      if (idb_segment->is_via()) {
        std::ranges::for_each(idb_segment->get_via_list(), [&](auto* idb_via) -> void {
          auto* coord = idb_via->get_coordinate();
          auto enclosure_bottom = idb_via->get_bottom_layer_shape();
          auto enclosure_top = idb_via->get_top_layer_shape();
          auto layer_shape = idb_via->get_cut_layer_shape();
          auto layer_name = layer_shape.get_layer()->get_name();
          auto cur_layer_id = _layer_map.at(layer_name);
          std::vector<LayoutDefRect> bottom_shapes;
          std::vector<LayoutDefRect> top_shapes;
          auto* via = new LayoutVia(coord, enclosure_bottom.get_rect_list(), enclosure_top.get_rect_list(), cur_layer_id);
          auto vertex = boost::add_vertex(graph);
          graph[vertex].content = via;
        });
      }
      if (idb_segment->is_wire()) {
        auto* coord_start = idb_segment->get_point_start();
        auto* coord_end = idb_segment->get_point_second();
        auto layer_id = _layer_map.at(idb_segment->get_layer()->get_name());
        auto* wire = new LayoutWire(coord_start, coord_end, layer_id);
        auto vertex = boost::add_vertex(graph);
        graph[vertex].content = wire;
      }
    }
  }
  buildConnections(graph);
  checkConnectivity(graph);
  return graph;
}

TopoGraph LmNetGraphGenerator::buildTopoGraph(idb::IdbSpecialNet* idb_net) const
{
  // LOG_INFO << "Net name : " << idb_net->get_net_name();

  TopoGraph graph;
  // Build Instances' pins and IO pins
  std::vector<idb::IdbPin*> pins;
  std::ranges::copy(idb_net->get_instance_pin_list()->get_pin_list(), std::back_inserter(pins));
  std::ranges::copy(idb_net->get_io_pin_list()->get_pin_list(), std::back_inserter(pins));
  // Corner Case: ignore the bottom layer
  auto is_bottom_pin = [&](auto* idb_pin) -> bool {
    for (auto* layer_shape : idb_pin->get_port_box_list()) {
      auto layer_name = layer_shape->get_layer()->get_name();
      if (_layer_map.at(layer_name) == 0) {
        return true;
      }
    }
    return false;
  };
  std::ranges::for_each(pins, [&](auto* idb_pin) -> void {
    if (is_bottom_pin(idb_pin)) {
      return;
    }
    auto vertex = boost::add_vertex(graph);
    auto* layout_pin = new LayoutPin();
    graph[vertex].content = layout_pin;
    for (auto* layer_shape : idb_pin->get_port_box_list()) {
      auto layer_name = layer_shape->get_layer()->get_name();
      auto layer_id = _layer_map.at(layer_name);
      if (layer_shape->is_via()) {
        std::ranges::for_each(layer_shape->get_rect_list(), [&](auto* rect) -> void { layout_pin->addViaCut(rect, layer_id); });
      } else {
        std::ranges::for_each(layer_shape->get_rect_list(), [&](auto* rect) -> void { layout_pin->addPinShape(rect, layer_id); });
      }
    }
  });

  // Build Wires
  // Corner Case: ignore the bottom layer
  auto is_bottom_segment = [&](auto* idb_segment) -> bool {
    auto layer_name = idb_segment->get_layer()->get_name();
    return _layer_map.at(layer_name) == 0;
  };
  auto* idb_wires = idb_net->get_wire_list();
  for (auto* idb_wire : idb_wires->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      if (is_bottom_segment(idb_segment)) {
        continue;
      }

      if (idb_segment->is_rect()) {
        auto* coord_start = idb_segment->get_point_start();
        auto* delta_rect = idb_segment->get_delta_rect();
        auto* rect = new idb::IdbRect(delta_rect);
        rect->moveByStep(coord_start->get_x(), coord_start->get_y());
        auto layer_id = _layer_map.at(idb_segment->get_layer()->get_name());
        auto* patch = new LayoutPatch(rect, layer_id);
        delete rect;
        auto vertex = boost::add_vertex(graph);
        graph[vertex].content = patch;
      }
      if (idb_segment->is_via()) {
        auto* idb_via = idb_segment->get_via();
        auto* coord = idb_via->get_coordinate();
        auto enclosure_bottom = idb_via->get_bottom_layer_shape();
        auto enclosure_top = idb_via->get_top_layer_shape();
        auto layer_shape = idb_via->get_cut_layer_shape();
        auto layer_name = layer_shape.get_layer()->get_name();
        auto cur_layer_id = _layer_map.at(layer_name);
        std::vector<LayoutDefRect> bottom_shapes;
        std::vector<LayoutDefRect> top_shapes;
        auto* via = new LayoutVia(coord, enclosure_bottom.get_rect_list(), enclosure_top.get_rect_list(), cur_layer_id);
        auto vertex = boost::add_vertex(graph);
        graph[vertex].content = via;
      }
      if (idb_segment->is_line()) {
        auto* coord_start = idb_segment->get_point_start();
        auto* coord_end = idb_segment->get_point_second();
        auto layer_id = _layer_map.at(idb_segment->get_layer()->get_name());
        auto* wire = new LayoutWire(coord_start, coord_end, layer_id);
        auto vertex = boost::add_vertex(graph);
        graph[vertex].content = wire;
      }
    }
  }
  buildConnections(graph);
  checkConnectivity(graph);
  return graph;
}
LayoutShapeManager LmNetGraphGenerator::buildShapeManager(const TopoGraph& graph) const
{
  LayoutShapeManager shape_manager;

  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = graph[v].content;
    if (content->is_patch()) {
      auto* patch = static_cast<LayoutPatch*>(content);
      shape_manager.addShape(patch->rect, v);
    } else if (content->is_wire()) {
      auto* wire = static_cast<LayoutWire*>(content);
      auto z = getZ(wire->start);
      auto [x1, x2] = std::ranges::minmax({getX(wire->start), getX(wire->end)});
      auto [y1, y2] = std::ranges::minmax({getY(wire->start), getY(wire->end)});
      shape_manager.addShape(LayoutDefRect({x1, y1, z}, {x2, y2, z}), v);
    } else if (content->is_via()) {
      auto* via = static_cast<LayoutVia*>(content);
      shape_manager.addShape(via->cut_path, v);
      for (auto& bottom_shape : via->bottom_shapes) {
        shape_manager.addShape(bottom_shape, v);
      }
      for (auto& top_shape : via->top_shapes) {
        shape_manager.addShape(top_shape, v);
      }
    } else if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      for (auto& pin_shape : pin->pin_shapes) {
        shape_manager.addShape(pin_shape, v);
      }
      for (auto& via_cut : pin->via_cuts) {
        shape_manager.addShape(via_cut, v);
      }
    } else {
      LOG_FATAL << "Unknown content type";
    }
  }

  return shape_manager;
}
void LmNetGraphGenerator::buildConnections(TopoGraph& graph) const
{
  auto shape_manager = buildShapeManager(graph);

  // Find intersections
  auto connect = [&](const size_t& ref, const std::vector<size_t>& intersections) -> void {
    LOG_FATAL_IF(intersections.empty()) << "No intersections found";
    for (const auto& id : intersections) {
      if (id != ref && !boost::edge(ref, id, graph).second) {
        // except itself and no duplicate edge
        boost::add_edge(ref, id, graph);
      }
    }
  };

  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = graph[v].content;
    if (content->is_patch()) {
      auto* patch = static_cast<LayoutPatch*>(content);
      // for patch, find the intersections of the rect
      auto intersections = shape_manager.findIntersections(patch->rect);
      connect(v, intersections);
    } else if (content->is_wire()) {
      auto* wire = static_cast<LayoutWire*>(content);
      // for wire, find the intersections of the wire
      auto intersections = shape_manager.findIntersections(wire->start, wire->end);
      connect(v, intersections);
    } else if (content->is_via()) {
      auto* via = static_cast<LayoutVia*>(content);
      // for via, find the intersections of the cut path, bottom shapes and top shapes
      auto cut_intersections = shape_manager.findIntersections(via->cut_path);
      connect(v, cut_intersections);
      for (auto& bottom_shape : via->bottom_shapes) {
        auto intersections = shape_manager.findIntersections(bottom_shape);
        connect(v, intersections);
      }
      for (auto& top_shape : via->top_shapes) {
        auto intersections = shape_manager.findIntersections(top_shape);
        connect(v, intersections);
      }
    } else if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      // for pin, find the intersections of the pin shapes and via cuts
      for (auto& pin_shape : pin->pin_shapes) {
        auto intersections = shape_manager.findIntersections(pin_shape);
        connect(v, intersections);
      }
      for (auto& via_cut : pin->via_cuts) {
        auto intersections = shape_manager.findIntersections(via_cut);
        connect(v, intersections);
      }
    } else {
      LOG_FATAL << "Unknown content type";
    }
  }
}
bool LmNetGraphGenerator::checkConnectivity(const TopoGraph& graph) const
{
  std::vector<int> component(boost::num_vertices(graph));
  auto num = boost::connected_components(graph, component.data());
  if (num > 1) {
    LOG_ERROR << "Topo Graph is not connected";
    // toQt(graph);
    // toPy(graph, "/home/liweiguo/temp/file/temp_net.py");
    return false;
  }
  // toQt(graph);
  // toPy(graph, "/home/liweiguo/temp/file/temp_net.py");
  return true;
}
WireGraph LmNetGraphGenerator::buildWireGraph(const TopoGraph& graph) const
{
  WireGraph wire_graph;
  WireGraphVertexMap point_to_vertex;
  auto build_and_connect = [&](const LayoutDefPoint& start, const LayoutDefPoint& end) -> void {
    if (!point_to_vertex.contains(start)) {
      auto vertex = boost::add_vertex(wire_graph);
      wire_graph[vertex].x = getX(start);
      wire_graph[vertex].y = getY(start);
      wire_graph[vertex].layer_id = getZ(start);
      point_to_vertex[start] = vertex;
    }
    if (!point_to_vertex.contains(end)) {
      auto vertex = boost::add_vertex(wire_graph);
      wire_graph[vertex].x = getX(end);
      wire_graph[vertex].y = getY(end);
      wire_graph[vertex].layer_id = getZ(end);
      point_to_vertex[end] = vertex;
    }
    // check edge exists
    auto start_vertex = point_to_vertex[start];
    auto end_vertex = point_to_vertex[end];
    if (boost::edge(start_vertex, end_vertex, wire_graph).second) {
      return;
    }
    auto [edge, inserted] = boost::add_edge(start_vertex, end_vertex, wire_graph);
    if (inserted) {
      wire_graph[edge].path.emplace_back(std::make_pair(start, end));
    }
  };
  // traversal all vertices
  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = graph[v].content;
    if (content->is_patch()) {
      continue;
    }
    if (content->is_wire()) {
      auto* wire = static_cast<LayoutWire*>(content);
      build_and_connect(wire->start, wire->end);
    } else if (content->is_via()) {
      auto* via = static_cast<LayoutVia*>(content);
      auto cut_path = via->cut_path;
      build_and_connect(cut_path.first, cut_path.second);
    } else if (content->is_pin()) {
      // only connect the pin's via cut
      auto* pin = static_cast<LayoutPin*>(content);
      auto via_cuts = pin->via_cuts;
      std::ranges::for_each(via_cuts, [&](const LayoutDefRect& via_cut) -> void {
        auto center = getCenter(via_cut);
        auto start = LayoutDefPoint(getX(center), getY(center), getZ(center) - 1);
        auto end = LayoutDefPoint(getX(center), getY(center), getZ(center) + 1);
        build_and_connect(start, end);
      });
    } else {
      LOG_FATAL << "Unknown content type";
    }
  }
  // post process
  breakCycle(wire_graph);
  innerConnectivityCompletion(graph, wire_graph, point_to_vertex);
  buildVirtualWire(graph, wire_graph, point_to_vertex);
  markPinVertex(graph, wire_graph);
  reduceWireGraph(wire_graph);
  checkConnectivity(wire_graph);
  checkDriverIsTreeRoot(graph, wire_graph);
  return wire_graph;
}
std::vector<WireGraphVertex> LmNetGraphGenerator::canonicalizeCycle(const std::vector<WireGraphVertex>& cycle) const
{
  auto forward = cycle;
  auto backward = cycle;
  std::reverse(backward.begin() + 1, backward.end());

  return (forward < backward) ? forward : backward;
}
void LmNetGraphGenerator::dfsFindCycles(const WireGraph& graph, WireGraphVertex start, WireGraphVertex current, std::vector<bool>& visited,
                                        std::vector<WireGraphVertex>& path, std::vector<std::vector<WireGraphVertex>>& cycles) const
{
  for (auto neighbor : boost::make_iterator_range(adjacent_vertices(current, graph))) {
    if (neighbor == start && path.size() >= 3) {
      cycles.push_back(path);
    } else if (!visited[neighbor] && neighbor > start) {
      visited[neighbor] = true;
      path.push_back(neighbor);
      dfsFindCycles(graph, start, neighbor, visited, path, cycles);
      path.pop_back();
      visited[neighbor] = false;
    }
  }
}
std::vector<std::vector<WireGraphVertex>> LmNetGraphGenerator::findAllCycles(const WireGraph& graph) const
{
  std::vector<std::vector<WireGraphVertex>> cycles;
  const auto n = boost::num_vertices(graph);
  std::vector<bool> visited(n, false);

  for (WireGraphVertex start = 0; start < n; ++start) {
    visited[start] = true;
    std::vector<WireGraphVertex> path;
    path.push_back(start);
    dfsFindCycles(graph, start, start, visited, path, cycles);
    visited[start] = false;
  }
  std::ranges::for_each(cycles, [&](std::vector<WireGraphVertex>& cycle) -> void { cycle = canonicalizeCycle(cycle); });
  std::sort(cycles.begin(), cycles.end());
  cycles.erase(std::unique(cycles.begin(), cycles.end()), cycles.end());
  return cycles;
}
void LmNetGraphGenerator::breakCycle(WireGraph& graph) const
{
  if (!hasCycle(graph)) {
    return;
  }
  // find all cycles, check cycle length whether is 4, and find the node which degree is 2, then remove one of nodes.
  auto cycles = findAllCycles(graph);
  std::ranges::for_each(cycles, [&](const std::vector<WireGraphVertex>& cycle) -> void {
    LOG_FATAL_IF(cycle.size() != 4) << "Cycle length is " << cycle.size() << ", not 4";

    auto degree_2_node
        = std::find_if(cycle.begin(), cycle.end(), [&](const WireGraphVertex& v) -> bool { return boost::degree(v, graph) == 2; });
    LOG_FATAL_IF(degree_2_node == cycle.end()) << "No degree 2 node found";

    auto v = *degree_2_node;
    boost::clear_vertex(v, graph);
    boost::remove_vertex(v, graph);
  });
}
void LmNetGraphGenerator::innerConnectivityCompletion(const TopoGraph& graph, WireGraph& wire_graph,
                                                      WireGraphVertexMap& point_to_vertex) const
{
  // Temporary structure to group the vertices by flatten shape
  struct FlattenShape
  {
    int low_x;
    int low_y;
    int high_x;
    int high_y;

    bool operator==(const FlattenShape& other) const
    {
      return low_x == other.low_x && low_y == other.low_y && high_x == other.high_x && high_y == other.high_y;
    }
  };

  struct FlattenShapeHash
  {
    size_t operator()(const FlattenShape& shape) const
    {
      return std::hash<int>()(shape.low_x) ^ std::hash<int>()(shape.low_y) ^ std::hash<int>()(shape.high_x)
             ^ std::hash<int>()(shape.high_y);
    }
  };

  auto build_and_connect = [&](const LayoutDefPoint& start, const LayoutDefPoint& end) -> void {
    if (!point_to_vertex.contains(start)) {
      auto vertex = boost::add_vertex(wire_graph);
      wire_graph[vertex].x = getX(start);
      wire_graph[vertex].y = getY(start);
      wire_graph[vertex].layer_id = getZ(start);
      point_to_vertex[start] = vertex;
    }
    if (!point_to_vertex.contains(end)) {
      auto vertex = boost::add_vertex(wire_graph);
      wire_graph[vertex].x = getX(end);
      wire_graph[vertex].y = getY(end);
      wire_graph[vertex].layer_id = getZ(end);
      point_to_vertex[end] = vertex;
    }
    // check edge exists
    auto start_vertex = point_to_vertex[start];
    auto end_vertex = point_to_vertex[end];
    if (boost::edge(start_vertex, end_vertex, wire_graph).second) {
      return;
    }
    auto [edge, inserted] = boost::add_edge(start_vertex, end_vertex, wire_graph);
    if (inserted) {
      wire_graph[edge].path.emplace_back(std::make_pair(start, end));
    }
  };
  // Build an R-tree to locate points within a shape
  LayoutShapeManager shape_manager;
  std::vector<LayoutDefPoint> points;
  for (auto& [point, vertex] : point_to_vertex) {
    points.emplace_back(point);
  }
  for (size_t i = 0; i < points.size(); ++i) {
    shape_manager.addShape(points[i], i);
  }

  auto connectivity_complete = [&](const TopoGraphVertex& v) -> void {
    auto* content = graph[v].content;
    if (!content->is_pin()) {
      return;
    }
    auto* pin = static_cast<LayoutPin*>(content);
    auto pin_shapes = pin->pin_shapes;
    auto via_cuts = pin->via_cuts;
    auto via_cuts_set = std::unordered_set<LayoutDefRect, LayoutDefRectHash, LayoutDefRectEqual>(via_cuts.begin(), via_cuts.end());
    // Group pin shapes by (low_x, low_y, high_x, high_y), if the group has more than one shape, connect them
    std::unordered_map<FlattenShape, std::vector<size_t>, FlattenShapeHash> shape_map;
    for (size_t i = 0; i < pin_shapes.size(); ++i) {
      auto& shape = pin_shapes[i];
      auto low_x = getLowX(shape);
      auto low_y = getLowY(shape);
      auto high_x = getHighX(shape);
      auto high_y = getHighY(shape);
      shape_map[{low_x, low_y, high_x, high_y}].emplace_back(i);
    }
    // Check if the group has more than one shape, if so, further check the connectivity between the neighboring shapes
    for (auto& [shape, indices] : shape_map) {
      if (indices.size() < 2) {
        continue;
      }
      std::vector<LayoutDefRect> flatten_shapes;
      std::ranges::transform(indices, std::back_inserter(flatten_shapes), [&](size_t i) -> LayoutDefRect { return pin_shapes[i]; });
      // sort the shapes by layer, ascending
      std::ranges::sort(flatten_shapes,
                        [&](const LayoutDefRect& lhs, const LayoutDefRect& rhs) -> bool { return getLowZ(lhs) < getLowZ(rhs); });
      // connect the neighboring shapes
      for (size_t i = 0; i < flatten_shapes.size() - 1; ++i) {
        auto low_shape = flatten_shapes[i];
        auto high_shape = flatten_shapes[i + 1];

        auto low_point_ids = shape_manager.findIntersections(low_shape);
        auto high_points = shape_manager.findIntersections(high_shape);

        if (low_point_ids.empty() || high_points.empty()) {
          continue;
        }

        std::ranges::for_each(low_point_ids, [&](size_t low_id) -> void {
          auto low_point = points[low_id];
          auto high_point = LayoutDefPoint(getX(low_point), getY(low_point), getHighZ(high_shape));
          auto via_cut = LayoutDefRect(low_point, high_point);
          if (via_cuts_set.find(via_cut) != via_cuts_set.end()) {
            return;
          }
          via_cuts_set.insert(via_cut);
          build_and_connect(low_point, high_point);
        });
        continue;
      }
    }
    // Update the pin's via cuts
    pin->via_cuts = std::vector<LayoutDefRect>(via_cuts_set.begin(), via_cuts_set.end());
  };
  // Traverse all vertices
  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    connectivity_complete(v);
  }
}
void LmNetGraphGenerator::buildVirtualWire(const TopoGraph& graph, WireGraph& wire_graph, WireGraphVertexMap& point_to_vertex) const
{
  // for each pin and patch in TopoGraph
  std::vector<TopoGraphVertex> pins_to_process;
  std::vector<TopoGraphVertex> patches_to_process;
  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = graph[v].content;
    auto [neighbors_begin, neighbors_end] = boost::adjacent_vertices(v, graph);
    size_t count = 0;
    for (auto it = neighbors_begin; it != neighbors_end; ++it) {
      auto neighbor = *it;
      if (graph[neighbor].content->is_wire() || graph[neighbor].content->is_via()) {
        ++count;
      }
    }
    if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      count += pin->via_cuts.size();
      if (count > 1) {
        pins_to_process.emplace_back(v);
      }
    }
    if (content->is_patch() && count > 1) {
      patches_to_process.emplace_back(v);
    }
  }
  if (pins_to_process.empty() && patches_to_process.empty()) {
    return;
  }
  // if (!patches_to_process.empty()) {
  //   LOG_WARNING << "Patch is not supported";
  // }
  // PIN PROCESS
  std::ranges::for_each(pins_to_process, [&](TopoGraphVertex v) -> void {
    auto* content = graph[v].content;
    auto* pin = static_cast<LayoutPin*>(content);
    // only save the points in the shapes
    auto shape_manager = LayoutShapeManager();
    for (size_t i = 0; i < pin->pin_shapes.size(); ++i) {
      shape_manager.addShape(pin->pin_shapes[i], i);
    }
    auto is_in_shapes = [&](const LayoutDefPoint& point) -> bool {
      auto intersections = shape_manager.findIntersections(point);
      return !intersections.empty();
    };
    // divide pin's shape by layer
    std::unordered_map<int, std::vector<LayoutDefRect>> shapes_by_layer;
    for (auto& pin_shape : pin->pin_shapes) {
      shapes_by_layer[getLowZ(pin_shape)].emplace_back(pin_shape);
    }
    // divide via cuts, pin's via cuts and wire's end points by layer
    std::unordered_map<int, std::vector<LayoutDefPoint>> connections_by_layer;
    auto [neighbors_begin, neighbors_end] = boost::adjacent_vertices(v, graph);
    for (auto it = neighbors_begin; it != neighbors_end; ++it) {
      auto neighbor = *it;
      auto* neighbor_content = graph[neighbor].content;
      if (neighbor_content->is_wire()) {
        auto* wire = static_cast<LayoutWire*>(neighbor_content);
        auto start = LayoutDefPoint(wire->start);
        auto end = LayoutDefPoint(wire->end);
        if (is_in_shapes(start)) {
          connections_by_layer[getZ(start)].emplace_back(start);
        }
        if (is_in_shapes(end)) {
          connections_by_layer[getZ(end)].emplace_back(end);
        }
      } else if (neighbor_content->is_via()) {
        auto* via = static_cast<LayoutVia*>(neighbor_content);
        auto cut_path = via->cut_path;
        auto start = cut_path.first;
        auto end = cut_path.second;
        if (is_in_shapes(start)) {
          connections_by_layer[getZ(start)].emplace_back(start);
        }
        if (is_in_shapes(end)) {
          connections_by_layer[getZ(end)].emplace_back(end);
        }
      }
    }
    // pin's via cuts
    for (auto& via_cut : pin->via_cuts) {
      auto center = getCenter(via_cut);
      auto start = LayoutDefPoint(getX(center), getY(center), getZ(center) - 1);
      auto end = LayoutDefPoint(getX(center), getY(center), getZ(center) + 1);
      connections_by_layer[getZ(center) - 1].emplace_back(start);
      connections_by_layer[getZ(center) + 1].emplace_back(end);
    }
    // for each layer, generate connected points
    auto drop_duplicate = [&](const std::vector<LayoutDefPoint>& points) -> std::vector<LayoutDefPoint> {
      std::unordered_set<LayoutDefPoint, LayoutDefPointHash, LayoutDefPointEqual> point_set;
      std::ranges::copy_if(points, std::inserter(point_set, point_set.end()),
                           [&](const LayoutDefPoint& point) -> bool { return point_set.find(point) == point_set.end(); });
      return std::vector<LayoutDefPoint>(point_set.begin(), point_set.end());
    };
    for (auto& [layer_id, shapes] : shapes_by_layer) {
      auto& connections = connections_by_layer[layer_id];
      connections = drop_duplicate(connections);
      if (connections.size() < 2) {
        continue;
      }
      auto paths = generateShortestPath(connections, shapes);
      std::ranges::for_each(paths, [&](const std::vector<LayoutDefPoint>& path) -> void {
        // add to wrie graph
        auto start = path.front();
        auto end = path.back();
        std::vector<std::pair<LayoutDefPoint, LayoutDefPoint>> path_pairs;
        for (size_t i = 0; i < path.size() - 1; ++i) {
          path_pairs.emplace_back(std::make_pair(path[i], path[i + 1]));
        }
        auto start_vertex = point_to_vertex[start];
        auto end_vertex = point_to_vertex[end];
        if (boost::edge(start_vertex, end_vertex, wire_graph).second) {
          return;
        }
        auto [edge, inserted] = boost::add_edge(start_vertex, end_vertex, wire_graph);
        if (inserted) {
          wire_graph[edge].path = path_pairs;
        }
      });
    }
  });
}
void LmNetGraphGenerator::markPinVertex(const TopoGraph& graph, WireGraph& wire_graph) const
{
  auto shape_manager = LayoutShapeManager();
  auto [v_topo_iter, v_topo_end] = boost::vertices(graph);
  for (; v_topo_iter != v_topo_end; ++v_topo_iter) {
    auto v = *v_topo_iter;
    auto* content = graph[v].content;
    if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      for (auto& pin_shape : pin->pin_shapes) {
        shape_manager.addShape(pin_shape, v);
      }
    }
  }
  // for each node in wire graph, if it's located in the pin shape, mark it as pin
  auto [v_wire_iter, v_wire_end] = boost::vertices(wire_graph);
  for (; v_wire_iter != v_wire_end; ++v_wire_iter) {
    auto v = *v_wire_iter;
    auto x = wire_graph[v].x;
    auto y = wire_graph[v].y;
    auto layer_id = wire_graph[v].layer_id;
    auto point = LayoutDefPoint(x, y, layer_id);
    auto intersections = shape_manager.findIntersections(point);
    if (intersections.empty()) {
      continue;
    }
    auto is_driver_pin = std::ranges::any_of(intersections, [&](size_t i) -> bool {
      auto* content = graph[i].content;
      if (content->is_pin()) {
        auto* pin = static_cast<LayoutPin*>(content);
        return pin->is_driver_pin;
      }
      return false;
    });
    wire_graph[v].is_pin = true;
    wire_graph[v].is_driver_pin = is_driver_pin;
  }
}
void LmNetGraphGenerator::reduceWireGraph(WireGraph& graph, const bool& retain_pin) const
{
  // Step 1: Prepare

  size_t num_vertices = boost::num_vertices(graph);
  std::vector<bool> visited(num_vertices, false);

  auto is_reduce_vertex = [&](WireGraphVertex v) -> bool {
    if (boost::degree(v, graph) != 2) {
      return false;
    }
    auto neighbors = boost::adjacent_vertices(v, graph);
    auto left = *neighbors.first;
    auto right = *std::next(neighbors.first);
    return graph[v].layer_id == graph[left].layer_id && graph[v].layer_id == graph[right].layer_id;
  };

  // Step 2: Start from an endpoint with degree 1 and collect a sub-path
  struct PathToReduce
  {
    std::vector<WireGraphVertex> vertices;
  };

  std::vector<PathToReduce> paths_to_reduce;
  WireGraphVertex start_vertex = 0;
  for (size_t i = 0; i < num_vertices; ++i) {
    if (boost::degree(i, graph) == 1) {
      start_vertex = i;
      break;
    }
  }

  // Step 3: DFS from the start vertex
  std::stack<WireGraphVertex> stack;
  stack.push(start_vertex);
  while (!stack.empty()) {
    WireGraphVertex current = stack.top();
    stack.pop();
    visited[current] = true;

    auto [adj_iter_begin, adj_iter_end] = boost::adjacent_vertices(current, graph);
    for (auto it = adj_iter_begin; it != adj_iter_end; ++it) {
      WireGraphVertex v = *it;
      if (!visited[v] && is_reduce_vertex(v)) {
        PathToReduce path;
        path.vertices.emplace_back(current);
        path.vertices.emplace_back(v);
        while (is_reduce_vertex(v)) {
          visited[v] = true;
          auto neighbors = boost::adjacent_vertices(v, graph);
          auto left = *neighbors.first;
          auto right = *std::next(neighbors.first);
          if (!visited[left]) {
            path.vertices.emplace_back(left);
            v = left;
          } else if (!visited[right]) {
            path.vertices.emplace_back(right);
            v = right;
          } else {
            LOG_ERROR << "Cannot find a valid path to reduce";
            break;
          }
        }
        paths_to_reduce.emplace_back(path);
        stack.push(v);
      } else if (!visited[v]) {
        stack.push(v);
      }
    }
  }

  // Step 4: Remove the paths which include virtual wires
  if (retain_pin) {
    std::erase_if(paths_to_reduce, [&](const PathToReduce& path) -> bool {
      for (size_t i = 0; i < path.vertices.size(); ++i) {
        auto v = path.vertices[i];
        if (graph[v].is_pin) {
          return true;
        }
      }
      return false;
    });
  }

  // Step 5: Reduce the paths
  for (const auto& path : paths_to_reduce) {
    auto vertices = path.vertices;
    if (vertices.size() < 3) {
      continue;
    }
    auto start = vertices.front();
    auto end = vertices.back();
    // add edge between start and end
    auto [edge, inserted] = boost::add_edge(start, end, graph);
    if (inserted) {
      auto& new_path = graph[edge].path;
      new_path.clear();
      for (size_t i = 0; i < vertices.size() - 1; ++i) {
        auto cur = vertices[i];
        auto next = vertices[i + 1];
        auto cur_edge = boost::edge(cur, next, graph).first;
        auto& path = graph[cur_edge].path;
        auto cur_point = LayoutDefPoint(graph[cur].x, graph[cur].y, graph[cur].layer_id);
        auto path_start_point = path.front().first;
        if (bg::equals(cur_point, path_start_point)) {
          new_path.insert(new_path.end(), path.begin(), path.end());
        } else {
          // swap for each pair
          auto reverse_path = path;
          std::ranges::reverse(reverse_path);
          std::ranges::for_each(reverse_path, [&](const std::pair<LayoutDefPoint, LayoutDefPoint>& pair) -> void {
            new_path.emplace_back(std::make_pair(pair.second, pair.first));
          });
        }
      }
    }
  }
  std::vector<WireGraphVertex> vertices_to_remove;
  std::ranges::for_each(paths_to_reduce, [&](const PathToReduce& path) -> void {
    vertices_to_remove.insert(vertices_to_remove.end(), path.vertices.begin() + 1, path.vertices.end() - 1);
  });
  std::ranges::sort(vertices_to_remove, std::greater<>());
  std::ranges::for_each(vertices_to_remove, [&](WireGraphVertex v) -> void {
    boost::clear_vertex(v, graph);
    boost::remove_vertex(v, graph);
  });

  // Step 6: Reduce the redundant path pairs (with the same direction)
  auto calc_direction = [&](const LayoutDefPoint& start, const LayoutDefPoint& end) -> int {
    if (getX(start) == getX(end)) {
      return getY(start) < getY(end) ? 0 : 1;
    } else {
      return getX(start) < getX(end) ? 2 : 3;
    }
  };
  auto remove_redundant_path = [&](std::vector<std::pair<LayoutDefPoint, LayoutDefPoint>>& path) -> void {
    if (path.size() < 2) {
      return;
    }
    std::vector<std::pair<LayoutDefPoint, LayoutDefPoint>> new_path;

    auto it = path.begin();
    while (it != path.end()) {
      auto start = it->first;
      auto end = it->second;
      auto direction = calc_direction(start, end);

      auto next = std::next(it);
      while (next != path.end() && calc_direction(end, next->second) == direction) {
        end = next->second;
        ++next;
      }

      new_path.emplace_back(start, end);

      it = next;
    }

    path = std::move(new_path);
  };
  auto [e_iter, e_end] = boost::edges(graph);
  for (; e_iter != e_end; ++e_iter) {
    auto e = *e_iter;
    auto& path = graph[e].path;
    if (path.size() < 2) {
      continue;
    }
    remove_redundant_path(path);
  }
}

// Helper function to check for cycles in the graph
bool LmNetGraphGenerator::hasCycleUtil(const WireGraph& graph, WireGraphVertex v, std::vector<bool>& visited, WireGraphVertex parent) const
{
  visited[v] = true;

  auto [adj_iter_begin, adj_iter_end] = boost::adjacent_vertices(v, graph);
  for (auto it = adj_iter_begin; it != adj_iter_end; ++it) {
    WireGraphVertex adj_v = *it;
    if (!visited[adj_v]) {
      if (hasCycleUtil(graph, adj_v, visited, v))
        return true;
    } else if (adj_v != parent) {
      return true;
    }
  }
  return false;
}

bool LmNetGraphGenerator::hasCycle(const WireGraph& graph) const
{
  size_t num_vertices = boost::num_vertices(graph);
  std::vector<bool> visited(num_vertices, false);

  for (size_t i = 0; i < num_vertices; ++i) {
    if (!visited[i]) {
      if (hasCycleUtil(graph, i, visited, -1))
        return true;
    }
  }
  return false;
}
bool LmNetGraphGenerator::checkConnectivity(const WireGraph& graph) const
{
  std::vector<int> component(boost::num_vertices(graph));
  auto num = boost::connected_components(graph, component.data());
  if (num > 1) {
    LOG_ERROR << "Wire Graph is not connected";
    // toQt(graph);
    // toPy(graph, "/home/liweiguo/temp/file/temp_net.py");
    return false;
  }
  // toQt(graph);
  // toPy(graph, "/home/liweiguo/temp/file/temp_net.py");
  return true;
}
bool LmNetGraphGenerator::checkDriverIsTreeRoot(const TopoGraph& topo_graph, const WireGraph& wire_graph) const
{
  // build rtree for driver pin shapes
  auto shape_manager = LayoutShapeManager();
  auto [v_iter, v_end] = boost::vertices(topo_graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = topo_graph[v].content;
    if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      if (!pin->is_driver_pin) {
        continue;
      }
      auto pin_shapes = pin->pin_shapes;
      for (size_t i = 0; i < pin_shapes.size(); ++i) {
        shape_manager.addShape(pin_shapes[i], i);
      }
    }
  }
  // check leaf vertices in wire graph whether is in the driver pin shapes
  size_t connected_point_count = 0;
  auto [w_iter, w_end] = boost::vertices(wire_graph);
  for (; w_iter != w_end; ++w_iter) {
    auto v = *w_iter;
    auto x = wire_graph[v].x;
    auto y = wire_graph[v].y;
    auto layer_id = wire_graph[v].layer_id;
    auto point = LayoutDefPoint(x, y, layer_id);
    auto intersections = shape_manager.findIntersections(point);
    if (intersections.empty()) {
      continue;
    }
    ++connected_point_count;
  }
  if (connected_point_count > 1) {
    // LOG_ERROR << "Multiple connections to driver pin are not allowed, please check the design";
    // toPy(topo_graph, "/home/liweiguo/temp/file/temp_topo.py");
    // toPy(wire_graph, "/home/liweiguo/temp/file/temp_wire.py");
    return false;
  }
  return true;
}
std::vector<std::vector<LayoutDefPoint>> LmNetGraphGenerator::generateShortestPath(const std::vector<LayoutDefPoint>& points,
                                                                                   const std::vector<LayoutDefRect>& regions) const
{
  using PointSet = std::unordered_set<LayoutDefPoint, LayoutDefPointHash, LayoutDefPointEqual>;
  PointSet path_point_set(points.begin(), points.end());

  auto shape_manager = LayoutShapeManager();
  for (size_t i = 0; i < regions.size(); ++i) {
    shape_manager.addShape(regions[i], i);
  }

  // 1. generate seg pivot between regions
  std::ranges::for_each(regions, [&](const LayoutDefRect& rect) -> void {
    auto intersections = shape_manager.findIntersections(rect);
    std::vector<LayoutDefSeg> segs;
    std::ranges::for_each(intersections, [&](const size_t& idx) -> void {
      auto other = regions[idx];
      if (boost::geometry::equals(rect, other)) {
        return;
      }
      if (!bg::intersects(rect, other)) {
        return;
      }
      LayoutDefSeg seg;
      bg::intersection(rect, other, seg);
      segs.emplace_back(seg);
    });
    std::ranges::for_each(segs, [&](const LayoutDefSeg& seg) -> void {
      auto pivot = generateSegPivot(seg, rect);
      path_point_set.insert(pivot);
    });
  });

  // 2. generate crossroads points between points and regions
  PointSet add_point_set;
  std::ranges::for_each(points, [&](const LayoutDefPoint& point) -> void {
    auto intersections = shape_manager.findIntersections(point);
    std::ranges::for_each(intersections, [&](const size_t& idx) -> void {
      auto rect = regions[idx];
      auto crossroads = generateCrossroadsPoints(point, rect);
      std::ranges::for_each(crossroads, [&](const LayoutDefPoint& crossroad) -> void { add_point_set.insert(crossroad); });
    });
  });
  std::ranges::for_each(add_point_set, [&](const LayoutDefPoint& point) -> void {
    auto intersections = shape_manager.findIntersections(point);
    std::ranges::for_each(intersections, [&](const size_t& idx) -> void {
      auto rect = regions[idx];
      auto pivot = generatePointPivot(point, rect);
      path_point_set.insert(pivot);
    });
  });

  auto path_point_vec = std::vector<LayoutDefPoint>(path_point_set.begin(), path_point_set.end());
  // 3. generate shortest path which connects all input {points}
  auto paths = findByDijkstra(points, path_point_vec, regions);
  return paths;
}
std::vector<std::vector<LayoutDefPoint>> LmNetGraphGenerator::findByDijkstra(const std::vector<LayoutDefPoint>& points,
                                                                             const std::vector<LayoutDefPoint>& path_points,
                                                                             const std::vector<LayoutDefRect>& regions) const
{
  // pre-process
  using FlatPoint = bg::model::point<int, 2, bg::cs::cartesian>;
  using FlatLine = bg::model::linestring<FlatPoint>;
  using FlatPolygon = bg::model::polygon<FlatPoint>;
  using FlatMultiPolygon = bg::model::multi_polygon<FlatPolygon>;
  FlatMultiPolygon regions_polygon;

  std::ranges::for_each(regions, [&](const LayoutDefRect& rect) -> void {
    FlatPolygon polygon;
    bg::append(polygon.outer(), FlatPoint(getLowX(rect), getLowY(rect)));
    bg::append(polygon.outer(), FlatPoint(getHighX(rect), getLowY(rect)));
    bg::append(polygon.outer(), FlatPoint(getHighX(rect), getHighY(rect)));
    bg::append(polygon.outer(), FlatPoint(getLowX(rect), getHighY(rect)));
    bg::correct(polygon);

    std::vector<FlatPolygon> regions_polygon_list;
    bg::union_(regions_polygon, polygon, regions_polygon_list);

    regions_polygon.clear();
    std::ranges::for_each(regions_polygon_list, [&](const FlatPolygon& poly) -> void { regions_polygon.emplace_back(poly); });
  });

  struct GridGraphVertexProperty
  {
    LayoutDefPoint point;
  };
  struct GridGraphEdgeProperty
  {
    int weight;
  };
  using GridGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, GridGraphVertexProperty, GridGraphEdgeProperty>;
  using GridGraphVertex = boost::graph_traits<GridGraph>::vertex_descriptor;
  using GridVertexMap = std::unordered_map<LayoutDefPoint, GridGraphVertex, LayoutDefPointHash, LayoutDefPointEqual>;
  GridGraph grid;
  GridVertexMap point_to_vertex;
  auto build_and_connect = [&](const LayoutDefPoint& start, const LayoutDefPoint& end) -> void {
    if (!point_to_vertex.contains(start)) {
      auto vertex = boost::add_vertex(grid);
      grid[vertex].point = start;
      point_to_vertex[start] = vertex;
    }
    if (!point_to_vertex.contains(end)) {
      auto vertex = boost::add_vertex(grid);
      grid[vertex].point = end;
      point_to_vertex[end] = vertex;
    }
    // check edge exists
    auto start_vertex = point_to_vertex[start];
    auto end_vertex = point_to_vertex[end];
    if (boost::edge(start_vertex, end_vertex, grid).second) {
      return;
    }
    auto [edge, inserted] = boost::add_edge(start_vertex, end_vertex, grid);
    if (inserted) {
      grid[edge].weight = std::abs(getX(start) - getX(end)) + std::abs(getY(start) - getY(end));
    }
  };
  // build edges between path_points in path_point_set (if e(p1, p2) is vertical or horizontal, and e(p1, p2) is in the regions)
  for (size_t i = 0; i < path_points.size(); ++i) {
    for (size_t j = i + 1; j < path_points.size(); ++j) {
      auto p1 = path_points[i];
      auto p2 = path_points[j];
      if (getX(p1) == getX(p2) || getY(p1) == getY(p2)) {
        FlatLine flat_line;
        bg::append(flat_line, FlatPoint(getX(p1), getY(p1)));
        bg::append(flat_line, FlatPoint(getX(p2), getY(p2)));
        if (!bg::covered_by(flat_line, regions_polygon)) {
          continue;
        }
        build_and_connect(p1, p2);
      }
    }
  }
  // 1. find shortest path between points, by Dijkstra (distance is edge weight)
  auto weight_map = boost::get(&GridGraphEdgeProperty::weight, grid);
  auto find_shortest_path = [&](const LayoutDefPoint& start, const LayoutDefPoint& end) -> std::vector<LayoutDefPoint> {
    auto start_vertex = point_to_vertex[start];
    auto end_vertex = point_to_vertex[end];
    std::vector<GridGraphVertex> predecessors(boost::num_vertices(grid));
    std::vector<int> distances(boost::num_vertices(grid));
    boost::dijkstra_shortest_paths(grid, start_vertex,
                                   boost::predecessor_map(predecessors.data()).distance_map(distances.data()).weight_map(weight_map));
    std::vector<LayoutDefPoint> path;
    for (auto v = end_vertex; v != start_vertex; v = predecessors[v]) {
      path.emplace_back(grid[v].point);
    }
    path.emplace_back(start);
    std::reverse(path.begin(), path.end());
    return path;
  };
  auto calc_distance = [&](const std::vector<LayoutDefPoint>& path) -> int {
    int distance = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
      distance += std::abs(getX(path[i]) - getX(path[i + 1])) + std::abs(getY(path[i]) - getY(path[i + 1]));
    }
    return distance;
  };
  // 2. build points graph by shortest distance (for find the minimum spanning tree/path)
  using PointGraph
      = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int>>;
  using PointGraphEdge = boost::graph_traits<PointGraph>::edge_descriptor;
  PointGraph point_graph;
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      auto p1 = points[i];
      auto p2 = points[j];
      auto path = find_shortest_path(p1, p2);
      auto distance = calc_distance(path);
      auto [edge, inserted] = boost::add_edge(i, j, point_graph);
      if (inserted) {
        boost::put(boost::edge_weight, point_graph, edge, distance);
      }
    }
  }
  // 3. find the minimum spanning tree/path by Kruskal
  std::vector<PointGraphEdge> spanning_tree;
  boost::kruskal_minimum_spanning_tree(point_graph, std::back_inserter(spanning_tree));
  // 4. check if the spanning tree is a path, and convert it to the path
  std::vector<std::vector<LayoutDefPoint>> paths;
  std::ranges::for_each(spanning_tree, [&](PointGraphEdge edge) -> void {
    auto start = points[boost::source(edge, point_graph)];
    auto end = points[boost::target(edge, point_graph)];
    auto path = find_shortest_path(start, end);
    paths.emplace_back(path);
  });
  return paths;
}
std::vector<LayoutDefPoint> LmNetGraphGenerator::generateCrossroadsPoints(const LayoutDefPoint& p, const LayoutDefRect& rect) const
{
  std::unordered_set<LayoutDefPoint, LayoutDefPointHash, LayoutDefPointEqual> points;
  auto x = getX(p);
  auto y = getY(p);
  auto z = getZ(p);
  auto low_x = getLowX(rect);
  auto high_x = getHighX(rect);
  auto low_y = getLowY(rect);
  auto high_y = getHighY(rect);
  points.insert(LayoutDefPoint(x, low_y, z));
  points.insert(LayoutDefPoint(x, high_y, z));
  points.insert(LayoutDefPoint(low_x, y, z));
  points.insert(LayoutDefPoint(high_x, y, z));
  return std::vector<LayoutDefPoint>(points.begin(), points.end());
}
LayoutDefPoint LmNetGraphGenerator::generatePointPivot(const LayoutDefPoint& p, const LayoutDefRect& rect) const
{
  auto x = getX(p);
  auto y = getY(p);
  auto z = getZ(p);
  auto low_x = getLowX(rect);
  auto high_x = getHighX(rect);
  auto low_y = getLowY(rect);
  auto high_y = getHighY(rect);
  if (x == low_x || x == high_x) {
    auto new_x = (low_x + high_x) / 2;
    return LayoutDefPoint(new_x, y, z);
  }
  if (y == low_y || y == high_y) {
    auto new_y = (low_y + high_y) / 2;
    return LayoutDefPoint(x, new_y, z);
  }
  LOG_FATAL << "Invalid point pivot";
  return LayoutDefPoint();
}
LayoutDefPoint LmNetGraphGenerator::generateSegPivot(const LayoutDefSeg& seg, const LayoutDefRect& rect) const
{
  LayoutDefPoint center = getCenter(seg);
  return generatePointPivot(center, rect);
}
void LmNetGraphGenerator::toPy(const TopoGraph& graph, const std::string& path) const
{
  std::ofstream file(path);
  file << "import plotly.graph_objects as go\n";
  file << "import matplotlib.cm as cm\n";
  file << "\n";
  file << "# Create a Plotly Figure\n";
  file << "fig = go.Figure()\n";
  file << "\n";
  size_t patch_count = 0;
  size_t wire_count = 0;
  size_t via_count = 0;
  size_t pin_count = 0;
  std::vector<int> component(boost::num_vertices(graph));
  auto num_components = boost::connected_components(graph, component.data());
  file << "colors = [f'rgb{cm.tab10(i)[:3]}' for i in range(" << num_components << ")]\n";
  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = graph[v].content;
    if (content->is_patch()) {
      auto* patch = static_cast<LayoutPatch*>(content);
      // plot rect
      file << "fig.add_trace(go.Scatter3d(\n";
      file << "    x=[" << getLowX(patch->rect) << ", " << getHighX(patch->rect) << ", " << getHighX(patch->rect) << ", "
           << getLowX(patch->rect) << ", " << getLowX(patch->rect) << "],\n";
      file << "    y=[" << getLowY(patch->rect) << ", " << getLowY(patch->rect) << ", " << getHighY(patch->rect) << ", "
           << getHighY(patch->rect) << ", " << getLowY(patch->rect) << "],\n";
      file << "    z=[" << getLowZ(patch->rect) << ", " << getLowZ(patch->rect) << ", " << getLowZ(patch->rect) << ", "
           << getLowZ(patch->rect) << ", " << getLowZ(patch->rect) << "],\n";
      file << "    mode='lines',\n";
      file << "    line=dict(color='rgb(0,128,0)', width=4),\n";
      file << "    name='Patch " << patch_count++ << "'\n";
      file << "))\n";
    } else if (content->is_wire()) {
      auto* wire = static_cast<LayoutWire*>(content);
      // plot line
      file << "fig.add_trace(go.Scatter3d(\n";
      file << "    x=[" << getX(wire->start) << ", " << getX(wire->end) << "],\n";
      file << "    y=[" << getY(wire->start) << ", " << getY(wire->end) << "],\n";
      file << "    z=[" << getZ(wire->start) << ", " << getZ(wire->end) << "],\n";
      file << "    mode='lines',\n";
      file << "    line=dict(color=colors[" << component[v] << "], width=4),\n";
      file << "    name='Wire " << wire_count++ << "'\n";
      file << "))\n";
    } else if (content->is_via()) {
      auto* via = static_cast<LayoutVia*>(content);
      size_t bottom_count = 0;
      size_t top_count = 0;
      // plot cut path
      file << "fig.add_trace(go.Scatter3d(\n";
      file << "    x=[" << getStartX(via->cut_path) << ", " << getEndX(via->cut_path) << "],\n";
      file << "    y=[" << getStartY(via->cut_path) << ", " << getEndY(via->cut_path) << "],\n";
      file << "    z=[" << getStartZ(via->cut_path) << ", " << getEndZ(via->cut_path) << "],\n";
      file << "    mode='lines',\n";
      file << "    line=dict(color=colors[" << component[v] << "], width=4),\n";
      file << "    name='Via " << via_count << " Cut Path'\n";
      file << "))\n";
      for (auto& bottom_shape : via->bottom_shapes) {
        // plot bottom shapes
        file << "fig.add_trace(go.Scatter3d(\n";
        file << "    x=[" << getLowX(bottom_shape) << ", " << getHighX(bottom_shape) << ", " << getHighX(bottom_shape) << ", "
             << getLowX(bottom_shape) << ", " << getLowX(bottom_shape) << "],\n";
        file << "    y=[" << getLowY(bottom_shape) << ", " << getLowY(bottom_shape) << ", " << getHighY(bottom_shape) << ", "
             << getHighY(bottom_shape) << ", " << getLowY(bottom_shape) << "],\n";
        file << "    z=[" << getLowZ(bottom_shape) << ", " << getLowZ(bottom_shape) << ", " << getLowZ(bottom_shape) << ", "
             << getLowZ(bottom_shape) << ", " << getLowZ(bottom_shape) << "],\n";
        file << "    mode='lines',\n";
        file << "    line=dict(color='rgb(128,128,128)', width=4),\n";
        file << "    name='Via " << via_count << " Bottom " << bottom_count++ << "'\n";
        file << "))\n";
      }
      for (auto& top_shape : via->top_shapes) {
        // plot top shapes
        file << "fig.add_trace(go.Scatter3d(\n";
        file << "    x=[" << getLowX(top_shape) << ", " << getHighX(top_shape) << ", " << getHighX(top_shape) << ", " << getLowX(top_shape)
             << ", " << getLowX(top_shape) << "],\n";
        file << "    y=[" << getLowY(top_shape) << ", " << getLowY(top_shape) << ", " << getHighY(top_shape) << ", " << getHighY(top_shape)
             << ", " << getLowY(top_shape) << "],\n";
        file << "    z=[" << getLowZ(top_shape) << ", " << getLowZ(top_shape) << ", " << getLowZ(top_shape) << ", " << getLowZ(top_shape)
             << ", " << getLowZ(top_shape) << "],\n";
        file << "    mode='lines',\n";
        file << "    line=dict(color='rgb(128,128,128)', width=4),\n";
        file << "    name='Via " << via_count << " Top " << top_count++ << "'\n";
        file << "))\n";
      }
      via_count++;
    } else if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      size_t via_pin_shape_count = 0;
      size_t via_cut_count = 0;
      auto pin_name = pin->net_name + "/" + pin->pin_name;
      auto rgb_color = pin->is_driver_pin ? "rgb(255,0,0)" : "rgb(255, 255, 0)";
      for (auto& pin_shape : pin->pin_shapes) {
        // plot pin shapes
        file << "fig.add_trace(go.Scatter3d(\n";
        file << "    x=[" << getLowX(pin_shape) << ", " << getHighX(pin_shape) << ", " << getHighX(pin_shape) << ", " << getLowX(pin_shape)
             << ", " << getLowX(pin_shape) << "],\n";
        file << "    y=[" << getLowY(pin_shape) << ", " << getLowY(pin_shape) << ", " << getHighY(pin_shape) << ", " << getHighY(pin_shape)
             << ", " << getLowY(pin_shape) << "],\n";
        file << "    z=[" << getLowZ(pin_shape) << ", " << getLowZ(pin_shape) << ", " << getLowZ(pin_shape) << ", " << getLowZ(pin_shape)
             << ", " << getLowZ(pin_shape) << "],\n";
        file << "    mode='lines',\n";
        file << "    line=dict(color='" << rgb_color << "', width=4),\n";
        file << "    name='Pin " << pin_name << " Id " << pin_count << " Shape " << via_pin_shape_count++ << "'\n";
        file << "))\n";
      }

      for (auto& via_cut : pin->via_cuts) {
        auto center = getCenter(via_cut);
        // plot line
        file << "fig.add_trace(go.Scatter3d(\n";
        file << "    x=[" << getX(center) << ", " << getX(center) << "],\n";
        file << "    y=[" << getY(center) << ", " << getY(center) << "],\n";
        file << "    z=[" << getLowZ(via_cut) << ", " << getHighZ(via_cut) << "],\n";
        file << "    mode='lines',\n";
        file << "    line=dict(color='" << rgb_color << "', width=4),\n";
        file << "    name='Pin " << pin_name << " Id " << pin_count << " Via Cut " << via_cut_count++ << "'\n";
        file << "))\n";
      }
      pin_count++;
    }
  }
  // Set up the layout with axis labels and title
  file << "\n";
  file << "fig.update_layout(\n";
  file << "    scene=dict(\n";
  file << "        xaxis_title='X',\n";
  file << "        yaxis_title='Y',\n";
  file << "        zaxis_title='Layer ID'\n";
  file << "    ),\n";
  file << "    title='Temp Net'\n";
  file << ")\n";
  file << "\n";

  // Show the plot
  file << "fig.show()\n";
}
void LmNetGraphGenerator::toPy(const WireGraph& graph, const std::string& path) const
{
  std::ofstream file(path);
  file << "import plotly.graph_objects as go\n";
  file << "import matplotlib.cm as cm\n";
  file << "\n";
  file << "# Create a Plotly Figure\n";
  file << "fig = go.Figure()\n";
  file << "\n";
  std::vector<int> component(boost::num_vertices(graph));
  auto num_components = boost::connected_components(graph, component.data());
  file << "colors = [f'rgb{cm.tab10(i)[:3]}' for i in range(" << num_components << ")]\n";
  size_t wire_idx = 0;
  size_t via_idx = 0;
  auto [e_iter, e_end] = boost::edges(graph);
  for (; e_iter != e_end; ++e_iter) {
    auto e = *e_iter;
    auto u = boost::source(e, graph);
    auto v = boost::target(e, graph);
    size_t path_idx = 0;
    // plot line
    for (const auto& [start, end] : graph[e].path) {
      file << "fig.add_trace(go.Scatter3d(\n";
      file << "    x=[" << getX(start) << ", " << getX(end) << "],\n";
      file << "    y=[" << getY(start) << ", " << getY(end) << "],\n";
      file << "    z=[" << getZ(start) << ", " << getZ(end) << "],\n";
      file << "    mode='lines',\n";
      file << "    line=dict(color=colors[" << component[u] << "], width=4),\n";
      if (graph[u].layer_id == graph[v].layer_id) {
        file << "    name='Component " << component[u] << ", Wire " << wire_idx << ", Path " << path_idx++ << "'\n";
      } else {
        file << "    name='Component " << component[u] << ", Via " << via_idx << "'\n";
      }
      file << "))\n";
    }
    if (graph[u].layer_id == graph[v].layer_id) {
      wire_idx++;
    } else {
      via_idx++;
    }
  }

  // Set up the layout with axis labels and title
  file << "\n";
  file << "fig.update_layout(\n";
  file << "    scene=dict(\n";
  file << "        xaxis_title='X',\n";
  file << "        yaxis_title='Y',\n";
  file << "        zaxis_title='Layer ID'\n";
  file << "    ),\n";
  file << "    title='Temp Net'\n";
  file << ")\n";
  file << "\n";

  // Show the plot
  file << "fig.show()\n";
}
void LmNetGraphGenerator::toQt(const TopoGraph& graph, const bool& component_mode) const
{
  // Create the Qt application if needed (assumes one is not already running)
#ifdef BUILD_LM_GUI
  int argc = 0;
  char** argv = nullptr;
  auto app = QApplication(argc, argv);

  // Create the widget.
  LmGraphWidget* widget = new LmGraphWidget();

  // Compute connected components for color selection if needed.
  std::vector<int> component(boost::num_vertices(graph));

  size_t patch_count = 0;
  size_t wire_count = 0;
  size_t via_count = 0;
  size_t pin_count = 0;

  auto [v_iter, v_end] = boost::vertices(graph);
  for (; v_iter != v_end; ++v_iter) {
    auto v = *v_iter;
    auto* content = graph[v].content;
    if (content->is_patch()) {
      auto* patch = static_cast<LayoutPatch*>(content);
      // Add patch as a rectangle (default green).
      auto label = component_mode ? "Component " + std::to_string(component[v]) : "Patch";
      widget->addRect(getLowX(patch->rect), getLowY(patch->rect), getLowZ(patch->rect), getHighX(patch->rect), getHighY(patch->rect),
                      getLowZ(patch->rect), "Patch " + std::to_string(patch_count), label);
      ++patch_count;
    } else if (content->is_wire()) {
      auto* wire = static_cast<LayoutWire*>(content);
      // Add wire (default red).
      auto label = component_mode ? "Component " + std::to_string(component[v]) : "Wire";
      widget->addWire(getX(wire->start), getY(wire->start), getZ(wire->start), getX(wire->end), getY(wire->end), getZ(wire->end),
                      "Wire " + std::to_string(wire_count), label);
      ++wire_count;
    } else if (content->is_via()) {
      auto* via = static_cast<LayoutVia*>(content);
      // Add the via cut path (default blue).
      auto via_label = component_mode ? "Component " + std::to_string(component[v]) : "Via Cut Path";
      widget->addVia(getStartX(via->cut_path), getStartY(via->cut_path), getStartZ(via->cut_path), getEndZ(via->cut_path),
                     "Via " + std::to_string(via_count) + " Cut Path", via_label);
      // Add bottom shapes as rectangles (gray).
      auto via_bottom_label = component_mode ? "Component " + std::to_string(component[v]) : "Via Bottom";
      for (auto& bottom_shape : via->bottom_shapes) {
        widget->addRect(getLowX(bottom_shape), getLowY(bottom_shape), getLowZ(bottom_shape), getHighX(bottom_shape), getHighY(bottom_shape),
                        getLowZ(bottom_shape), "Via " + std::to_string(via_count) + " Bottom", via_bottom_label);
      }
      // Add top shapes as rectangles (gray).
      auto via_top_label = component_mode ? "Component " + std::to_string(component[v]) : "Via Top";
      for (auto& top_shape : via->top_shapes) {
        widget->addRect(getLowX(top_shape), getLowY(top_shape), getLowZ(top_shape), getHighX(top_shape), getHighY(top_shape),
                        getLowZ(top_shape), "Via " + std::to_string(via_count) + " Top", via_top_label);
      }
      ++via_count;
    } else if (content->is_pin()) {
      auto* pin = static_cast<LayoutPin*>(content);
      // Add each pin shape as a rectangle (yellow).
      auto pin_label = component_mode ? "Component " + std::to_string(component[v]) : "Pin Shape";
      for (auto& pin_shape : pin->pin_shapes) {
        widget->addRect(getLowX(pin_shape), getLowY(pin_shape), getLowZ(pin_shape), getHighX(pin_shape), getHighY(pin_shape),
                        getLowZ(pin_shape), "Pin " + std::to_string(pin_count) + " Shape", pin_label);
      }
      // Add each via cut as a vertical wire (yellow).
      auto via_cut_label = component_mode ? "Component " + std::to_string(component[v]) : "Pin Via Cut";
      for (auto& via_cut : pin->via_cuts) {
        auto center = getCenter(via_cut);
        widget->addVia(getX(center), getY(center), getLowZ(via_cut), getHighZ(via_cut), "Pin " + std::to_string(pin_count) + " Via Cut",
                       via_cut_label);
      }
      ++pin_count;
    }
  }
  // Setup the view.
  widget->autoScale();
  widget->initView();
  widget->showAxes();
  widget->resize(1600, 1200);
  widget->show();

  app.exec();
#endif
}
void LmNetGraphGenerator::toQt(const WireGraph& graph) const
{
#ifdef BUILD_LM_GUI
  // Create the Qt application if needed (assumes one is not already running)
  int argc = 0;
  char** argv = nullptr;
  auto app = QApplication(argc, argv);

  // Create the widget.
  LmGraphWidget* widget = new LmGraphWidget();

  // Compute connected components for color selection if needed.
  std::vector<int> component(boost::num_vertices(graph));

  size_t wire_idx = 0;
  size_t via_idx = 0;

  auto [e_iter, e_end] = boost::edges(graph);
  for (; e_iter != e_end; ++e_iter) {
    auto e = *e_iter;
    auto u = boost::source(e, graph);
    auto v = boost::target(e, graph);
    size_t path_idx = 0;
    // plot line
    for (const auto& [start, end] : graph[e].path) {
      auto label = "Component " + std::to_string(component[u]);
      auto start_z = getZ(start);
      auto end_z = getZ(end);
      if (start_z != end_z) {
        widget->addVia(getX(start), getY(start), start_z, end_z, "Via " + std::to_string(via_idx), label);
      } else {
        widget->addWire(getX(start), getY(start), getZ(start), getX(end), getY(end), getZ(end), "Wire " + std::to_string(wire_idx), label);
      }
      ++path_idx;
    }
    if (graph[u].layer_id == graph[v].layer_id) {
      wire_idx++;
    } else {
      via_idx++;
    }
  }

  // Setup the view.
  widget->autoScale();
  widget->initView();
  widget->showAxes();
  widget->resize(1600, 1200);
  widget->show();

  app.exec();
#endif
}
void LmNetGraphGenerator::toJs(const std::vector<TopoGraph>& graphs, const std::string& path) const
{
  std::ofstream file(path);
  file << "{\n";
  file << "  \"shapes\": [\n";

  bool first_shape = true;

  for (size_t graph_idx = 0; graph_idx < graphs.size(); ++graph_idx) {
    const auto& graph = graphs[graph_idx];

    // Calculate connected components for this graph to determine colors
    std::vector<int> component(boost::num_vertices(graph));
    auto num_components = boost::connected_components(graph, component.data());

    // Counters for naming shapes within this graph
    size_t patch_count = 0;
    size_t wire_count = 0;
    size_t via_count = 0;
    size_t pin_count = 0;

    auto [v_iter, v_end] = boost::vertices(graph);
    for (; v_iter != v_end; ++v_iter) {
      auto v = *v_iter;
      auto* content = graph[v].content;

      if (content->is_patch()) {
        auto* patch = static_cast<LayoutPatch*>(content);

        if (!first_shape)
          file << ",\n";
        first_shape = false;

        file << "    {\n";
        file << "      \"type\": \"Rect\",\n";
        file << "      \"x1\": " << getLowX(patch->rect) << ",\n";
        file << "      \"y1\": " << getLowY(patch->rect) << ",\n";
        file << "      \"z1\": " << getLowZ(patch->rect) << ",\n";
        file << "      \"x2\": " << getHighX(patch->rect) << ",\n";
        file << "      \"y2\": " << getHighY(patch->rect) << ",\n";
        file << "      \"z2\": " << getLowZ(patch->rect) << ",\n";
        file << "      \"comment\": \"Graph " << graph_idx << " Patch " << patch_count << "\",\n";
        file << "      \"shapeClass\": \"Graph" << graph_idx << "_Patches\",\n";
        file << "      \"color\": { \"r\": 0, \"g\": 0.5, \"b\": 0 }\n";
        file << "    }";

        patch_count++;

      } else if (content->is_wire()) {
        auto* wire = static_cast<LayoutWire*>(content);

        if (!first_shape)
          file << ",\n";
        first_shape = false;

        // Generate color based on component ID (use tab10 colormap approximation)
        double hue = (component[v] % 10) / 10.0;
        double r = 0.5 + 0.5 * std::sin(hue * 6.28);
        double g = 0.5 + 0.5 * std::sin((hue + 0.33) * 6.28);
        double b = 0.5 + 0.5 * std::sin((hue + 0.66) * 6.28);

        file << "    {\n";
        file << "      \"type\": \"Wire\",\n";
        file << "      \"x1\": " << getX(wire->start) << ",\n";
        file << "      \"y1\": " << getY(wire->start) << ",\n";
        file << "      \"z1\": " << getZ(wire->start) << ",\n";
        file << "      \"x2\": " << getX(wire->end) << ",\n";
        file << "      \"y2\": " << getY(wire->end) << ",\n";
        file << "      \"z2\": " << getZ(wire->end) << ",\n";
        file << "      \"comment\": \"Graph " << graph_idx << " Wire " << wire_count << " Component " << component[v] << "\",\n";
        if (graph_idx == 102) {
          file << "      \"shapeClass\": \"Graph " << graph_idx << " Wire " << wire_count << " Component " << component[v] << "\",\n";
        } else {
          file << "      \"shapeClass\": \"Graph" << graph_idx << "_Wires_Comp" << component[v] << "\",\n";
        }
        // file << "      \"shapeClass\": \"Graph" << graph_idx << "_Wires_Comp" << component[v] << "\",\n";
        file << "      \"color\": { \"r\": " << r << ", \"g\": " << g << ", \"b\": " << b << " }\n";
        file << "    }";

        wire_count++;

      } else if (content->is_via()) {
        auto* via = static_cast<LayoutVia*>(content);

        // Generate color based on component ID
        double hue = (component[v] % 10) / 10.0;
        double r = 0.5 + 0.5 * std::sin(hue * 6.28);
        double g = 0.5 + 0.5 * std::sin((hue + 0.33) * 6.28);
        double b = 0.5 + 0.5 * std::sin((hue + 0.66) * 6.28);

        // Add cut path as a via
        if (!first_shape)
          file << ",\n";
        first_shape = false;

        file << "    {\n";
        file << "      \"type\": \"Via\",\n";
        file << "      \"x1\": " << getStartX(via->cut_path) << ",\n";
        file << "      \"y1\": " << getStartY(via->cut_path) << ",\n";
        file << "      \"z1\": " << getStartZ(via->cut_path) << ",\n";
        file << "      \"x2\": " << getEndX(via->cut_path) << ",\n";
        file << "      \"y2\": " << getEndY(via->cut_path) << ",\n";
        file << "      \"z2\": " << getEndZ(via->cut_path) << ",\n";
        file << "      \"comment\": \"Graph " << graph_idx << " Via " << via_count << " Cut Path\",\n";
        if (graph_idx == 102) {
          file << "      \"shapeClass\": \"Graph " << graph_idx << " Via " << via_count << " Cut Path Component " << component[v]
               << "\",\n";
        } else {
          file << "      \"shapeClass\": \"Graph" << graph_idx << "_Vias_Comp" << component[v] << "\",\n";
        }
        // file << "      \"shapeClass\": \"Graph" << graph_idx << "_Vias_Comp" << component[v] << "\",\n";
        file << "      \"color\": { \"r\": " << r << ", \"g\": " << g << ", \"b\": " << b << " }\n";
        file << "    }";

        // Add bottom shapes as rectangles
        size_t bottom_count = 0;
        for (auto& bottom_shape : via->bottom_shapes) {
          if (!first_shape)
            file << ",\n";
          first_shape = false;

          file << "    {\n";
          file << "      \"type\": \"Rect\",\n";
          file << "      \"x1\": " << getLowX(bottom_shape) << ",\n";
          file << "      \"y1\": " << getLowY(bottom_shape) << ",\n";
          file << "      \"z1\": " << getLowZ(bottom_shape) << ",\n";
          file << "      \"x2\": " << getHighX(bottom_shape) << ",\n";
          file << "      \"y2\": " << getHighY(bottom_shape) << ",\n";
          file << "      \"z2\": " << getLowZ(bottom_shape) << ",\n";
          file << "      \"comment\": \"Graph " << graph_idx << " Via " << via_count << " Bottom " << bottom_count << "\",\n";
          if (graph_idx == 102) {
            file << "      \"shapeClass\": \"Graph " << graph_idx << " Via " << via_count << " Bottoms Component " << component[v]
                 << "\",\n";
          } else {
            file << "      \"shapeClass\": \"Graph" << graph_idx << "_ViaBottoms\",\n";
          }
          // file << "      \"shapeClass\": \"Graph" << graph_idx << "_ViaBottoms\",\n";
          file << "      \"color\": { \"r\": 0.5, \"g\": 0.5, \"b\": 0.5 }\n";
          file << "    }";
          bottom_count++;
        }

        // Add top shapes as rectangles
        size_t top_count = 0;
        for (auto& top_shape : via->top_shapes) {
          if (!first_shape)
            file << ",\n";
          first_shape = false;

          file << "    {\n";
          file << "      \"type\": \"Rect\",\n";
          file << "      \"x1\": " << getLowX(top_shape) << ",\n";
          file << "      \"y1\": " << getLowY(top_shape) << ",\n";
          file << "      \"z1\": " << getLowZ(top_shape) << ",\n";
          file << "      \"x2\": " << getHighX(top_shape) << ",\n";
          file << "      \"y2\": " << getHighY(top_shape) << ",\n";
          file << "      \"z2\": " << getLowZ(top_shape) << ",\n";
          file << "      \"comment\": \"Graph " << graph_idx << " Via " << via_count << " Top " << top_count << "\",\n";
          if (graph_idx == 102) {
            file << "      \"shapeClass\": \"Graph " << graph_idx << " Via " << via_count << " Tops Component " << component[v] << "\",\n";
          } else {
            file << "      \"shapeClass\": \"Graph" << graph_idx << "_ViaTops\",\n";
          }
          // file << "      \"shapeClass\": \"Graph" << graph_idx << "_ViaTops\",\n";
          file << "      \"color\": { \"r\": 0.5, \"g\": 0.5, \"b\": 0.5 }\n";
          file << "    }";
          top_count++;
        }

        via_count++;

      } else if (content->is_pin()) {
        auto* pin = static_cast<LayoutPin*>(content);
        auto pin_name = pin->net_name + "/" + pin->pin_name;

        // Pin color: red for driver pins, yellow for non-driver pins
        double r = pin->is_driver_pin ? 1.0 : 1.0;
        double g = pin->is_driver_pin ? 0.0 : 1.0;
        double b = 0.0;
        std::string pin_class = pin->is_driver_pin ? "DriverPins" : "ReceiverPins";

        // Add pin shapes as rectangles
        size_t pin_shape_count = 0;
        for (auto& pin_shape : pin->pin_shapes) {
          if (!first_shape)
            file << ",\n";
          first_shape = false;

          file << "    {\n";
          file << "      \"type\": \"Rect\",\n";
          file << "      \"x1\": " << getLowX(pin_shape) << ",\n";
          file << "      \"y1\": " << getLowY(pin_shape) << ",\n";
          file << "      \"z1\": " << getLowZ(pin_shape) << ",\n";
          file << "      \"x2\": " << getHighX(pin_shape) << ",\n";
          file << "      \"y2\": " << getHighY(pin_shape) << ",\n";
          file << "      \"z2\": " << getLowZ(pin_shape) << ",\n";
          file << "      \"comment\": \"Graph " << graph_idx << " Pin " << pin_name << " Shape " << pin_shape_count << "\",\n";
          file << "      \"shapeClass\": \"Graph" << graph_idx << "_" << pin_class << "\",\n";
          file << "      \"color\": { \"r\": " << r << ", \"g\": " << g << ", \"b\": " << b << " }\n";
          file << "    }";
          pin_shape_count++;
        }

        // Add via cuts as vias
        size_t via_cut_count = 0;
        for (auto& via_cut : pin->via_cuts) {
          auto center = getCenter(via_cut);

          if (!first_shape)
            file << ",\n";
          first_shape = false;

          file << "    {\n";
          file << "      \"type\": \"Via\",\n";
          file << "      \"x1\": " << getX(center) << ",\n";
          file << "      \"y1\": " << getY(center) << ",\n";
          file << "      \"z1\": " << getLowZ(via_cut) << ",\n";
          file << "      \"x2\": " << getX(center) << ",\n";
          file << "      \"y2\": " << getY(center) << ",\n";
          file << "      \"z2\": " << getHighZ(via_cut) << ",\n";
          file << "      \"comment\": \"Graph " << graph_idx << " Pin " << pin_name << " Via Cut " << via_cut_count << "\",\n";
          file << "      \"shapeClass\": \"Graph" << graph_idx << "_" << pin_class << "_Vias\",\n";
          file << "      \"color\": { \"r\": " << r << ", \"g\": " << g << ", \"b\": " << b << " }\n";
          file << "    }";
          via_cut_count++;
        }

        pin_count++;
      }
    }
  }

  file << "\n  ]\n";
  file << "}\n";
}
}  // namespace ilm