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
 * @file lm_net_graph_gen.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-29
 * @brief Construct graph from net data.
 *        Remark:
 *          Wire, start from a point to another point (Vertex)
 *          Via, a via from one layer to another layer, include top/bottom rects (Edge)
 *          Patch, a rectangle area in a layer (Edge)
 */

#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <iostream>
#include <vector>

#include "IdbGeometry.h"
#include "IdbNet.h"

namespace ilm {

// Alias for simplicity
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using LayoutDefPoint = bg::model::point<int, 3, bg::cs::cartesian>;
using LayoutDefSeg = bg::model::segment<LayoutDefPoint>;
using LayoutDefRect = bg::model::box<LayoutDefPoint>;

struct LayoutDefPointHash
{
  std::size_t operator()(const LayoutDefPoint& point) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, point.get<0>());
    boost::hash_combine(seed, point.get<1>());
    boost::hash_combine(seed, point.get<2>());
    return seed;
  }
};

struct LayoutDefPointEqual
{
  bool operator()(const LayoutDefPoint& lhs, const LayoutDefPoint& rhs) const { return boost::geometry::equals(lhs, rhs); }
};

struct LayoutDefSegHash
{
  std::size_t operator()(const LayoutDefSeg& seg) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, seg.first.get<0>());
    boost::hash_combine(seed, seg.first.get<1>());
    boost::hash_combine(seed, seg.first.get<2>());
    boost::hash_combine(seed, seg.second.get<0>());
    boost::hash_combine(seed, seg.second.get<1>());
    boost::hash_combine(seed, seg.second.get<2>());
    return seed;
  }
};

struct LayoutDefSegEqual
{
  bool operator()(const LayoutDefSeg& lhs, const LayoutDefSeg& rhs) const { return boost::geometry::equals(lhs, rhs); }
};

struct LayoutDefRectHash
{
  std::size_t operator()(const LayoutDefRect& rect) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, rect.min_corner().get<0>());
    boost::hash_combine(seed, rect.min_corner().get<1>());
    boost::hash_combine(seed, rect.min_corner().get<2>());
    boost::hash_combine(seed, rect.max_corner().get<0>());
    boost::hash_combine(seed, rect.max_corner().get<1>());
    boost::hash_combine(seed, rect.max_corner().get<2>());
    return seed;
  }
};

struct LayoutDefRectEqual
{
  bool operator()(const LayoutDefRect& lhs, const LayoutDefRect& rhs) const { return boost::geometry::equals(lhs, rhs); }
};

struct LayoutBase
{
 public:
  LayoutBase(bool is_wire, bool is_via, bool is_patch, bool is_pin) : is_wire(is_wire), is_via(is_via), is_patch(is_patch), is_pin(is_pin)
  {
  }
  virtual ~LayoutBase() = default;
  bool is_wire;
  bool is_via;
  bool is_patch;
  bool is_pin;
};

struct LayoutWire : public LayoutBase
{
 public:
  LayoutWire(idb::IdbCoordinate<int32_t>* s, idb::IdbCoordinate<int32_t>* e, int layer_id) : LayoutBase{true, false, false, false}
  {
    start = LayoutDefPoint(s->get_x(), s->get_y(), layer_id);
    end = LayoutDefPoint(e->get_x(), e->get_y(), layer_id);
  }
  LayoutDefPoint start;
  LayoutDefPoint end;
};

struct LayoutVia : public LayoutBase
{
 public:
  LayoutVia(idb::IdbCoordinate<int32_t>* coord, std::vector<idb::IdbRect*> b_shapes, std::vector<idb::IdbRect*> t_shapes, int layer_id)
      : LayoutBase{false, true, false, false}
  {
    LayoutDefPoint cut_start(coord->get_x(), coord->get_y(), layer_id - 1);
    LayoutDefPoint cut_end(coord->get_x(), coord->get_y(), layer_id + 1);
    cut_path = LayoutDefSeg(cut_start, cut_end);
    for (auto* rect : b_shapes) {
      LayoutDefPoint low(rect->get_low_x(), rect->get_low_y(), layer_id - 1);
      LayoutDefPoint high(rect->get_high_x(), rect->get_high_y(), layer_id - 1);
      bottom_shapes.push_back(LayoutDefRect(low, high));
    }
    for (auto* rect : t_shapes) {
      LayoutDefPoint low(rect->get_low_x(), rect->get_low_y(), layer_id + 1);
      LayoutDefPoint high(rect->get_high_x(), rect->get_high_y(), layer_id + 1);
      top_shapes.push_back(LayoutDefRect(low, high));
    }
  }
  LayoutDefSeg cut_path;
  std::vector<LayoutDefRect> bottom_shapes;
  std::vector<LayoutDefRect> top_shapes;
};

struct LayoutPatch : public LayoutBase
{
 public:
  LayoutPatch(idb::IdbRect* r, int layer_id) : LayoutBase{false, false, true, false}
  {
    LayoutDefPoint low(r->get_low_x(), r->get_low_y(), layer_id);
    LayoutDefPoint high(r->get_high_x(), r->get_high_y(), layer_id);
    rect = LayoutDefRect(low, high);
  }
  LayoutDefRect rect;
};

struct LayoutPin : public LayoutBase
{
 public:
  LayoutPin() : LayoutBase{false, false, false, true} {}
  void addPinShape(idb::IdbRect* r, int layer_id)
  {
    LayoutDefPoint low(r->get_low_x(), r->get_low_y(), layer_id);
    LayoutDefPoint high(r->get_high_x(), r->get_high_y(), layer_id);
    pin_shapes.push_back(LayoutDefRect(low, high));
  }
  void addViaCut(idb::IdbRect* r, int layer_id)
  {
    LayoutDefPoint low(r->get_low_x(), r->get_low_y(), layer_id - 1);
    LayoutDefPoint high(r->get_high_x(), r->get_high_y(), layer_id + 1);
    via_cuts.push_back(LayoutDefRect(low, high));
  }
  std::vector<LayoutDefRect> pin_shapes;
  std::vector<LayoutDefRect> via_cuts;
};

// R-tree value type
using RTreeVal = std::pair<LayoutDefRect, size_t>;
// Main class to store shapes and perform intersection searches
class LayoutShapeManager
{
 public:
  LayoutShapeManager() = default;
  ~LayoutShapeManager() = default;

  void addShape(const LayoutDefSeg& seg, const size_t& vertex_id)
  {
    LayoutDefRect box(seg.first, seg.second);
    addShape(box, vertex_id);
  }
  void addShape(const LayoutDefRect& box, const size_t& vertex_id) { _rtree.insert(std::make_pair(box, vertex_id)); }

  std::vector<size_t> findIntersections(const LayoutDefRect& box) const
  {
    std::vector<RTreeVal> result;
    _rtree.query(bgi::intersects(box), std::back_inserter(result));
    std::unordered_set<size_t> vertex_ids;
    for (const auto& [rect, vertex_id] : result) {
      vertex_ids.insert(vertex_id);
    }
    return std::vector<size_t>(vertex_ids.begin(), vertex_ids.end());
  }

  std::vector<size_t> findIntersections(const LayoutDefPoint& point) const
  {
    LayoutDefRect box(point, point);
    return findIntersections(box);
  }

  std::vector<size_t> findIntersections(const LayoutDefPoint& start, const LayoutDefPoint& end) const
  {
    LayoutDefRect box(start, end);
    return findIntersections(box);
  }

  std::vector<size_t> findIntersections(const LayoutDefSeg& seg) const
  {
    LayoutDefRect box(seg.first, seg.second);
    return findIntersections(box);
  }

  void clear() { _rtree.clear(); }

 private:
  bgi::rtree<RTreeVal, bgi::quadratic<16>> _rtree;  // Spatial index
};

struct TopoGraphVertexProperty
{
  LayoutBase* content;
};

using TopoGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, TopoGraphVertexProperty>;
using TopoGraphVertex = boost::graph_traits<TopoGraph>::vertex_descriptor;
using TopoGraphEdge = boost::graph_traits<TopoGraph>::edge_descriptor;

struct WireGraphVertexProperty
{
  int x;
  int y;
  int layer_id;
};

struct WireGraphEdgeProperty
{
  bool is_virtual = false;
  std::vector<std::pair<LayoutDefPoint, LayoutDefPoint>> path;
};

using WireGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, WireGraphVertexProperty, WireGraphEdgeProperty>;
using WireGraphVertex = boost::graph_traits<WireGraph>::vertex_descriptor;
using WireGraphEdge = boost::graph_traits<WireGraph>::edge_descriptor;
using WireGraphVertexMap = std::unordered_map<LayoutDefPoint, WireGraphVertex, LayoutDefPointHash, LayoutDefPointEqual>;

class LmNetGraphGenerator
{
 public:
  LmNetGraphGenerator() { initLayerMap(); };
  ~LmNetGraphGenerator() = default;

  void initLayerMap();
  WireGraph buildGraph(idb::IdbNet* idb_net) const;
  std::vector<WireGraph> buildGraphs() const;

  // Topo Graph
  TopoGraph buildTopoGraph(idb::IdbNet* idb_net) const;
  void buildConnections(TopoGraph& graph) const;
  bool checkConnectivity(const TopoGraph& graph) const;

  // Wire Graph
  WireGraph buildWireGraph(const TopoGraph& graph) const;
  void buildVirtualWire(const TopoGraph& graph, WireGraph& wire_graph, WireGraphVertexMap& point_to_vertex) const;
  void reduceWireGraph(WireGraph& graph, const bool& retain_pin = true) const;
  bool hasCycleUtil(const WireGraph& graph, WireGraphVertex v, std::vector<bool>& visited, WireGraphVertex parent) const;
  bool hasCycle(const WireGraph& graph) const;
  bool checkConnectivity(const WireGraph& graph) const;
  std::vector<std::vector<LayoutDefPoint>> generateShortestPath(const std::vector<LayoutDefPoint>& points,
                                                                const std::vector<LayoutDefRect>& regions) const;
  std::vector<std::vector<LayoutDefPoint>> findByDijkstra(const std::vector<LayoutDefPoint>& points,
                                                          const std::vector<LayoutDefPoint>& path_points,
                                                          const std::vector<LayoutDefRect>& regions) const;
  std::vector<LayoutDefPoint> generateCrossroadsPoints(const LayoutDefPoint& p, const LayoutDefRect& rect) const;
  LayoutDefPoint generatePointPivot(const LayoutDefPoint& p, const LayoutDefRect& rect) const;
  LayoutDefPoint generateSegPivot(const LayoutDefSeg& seg, const LayoutDefRect& rect) const;

  // debug
  void toPy(const TopoGraph& graph, const std::string& path) const;
  void toPy(const WireGraph& graph, const std::string& path) const;

 private:
  int getX(const LayoutDefPoint& point) const { return bg::get<0>(point); }
  int getY(const LayoutDefPoint& point) const { return bg::get<1>(point); }
  int getZ(const LayoutDefPoint& point) const { return bg::get<2>(point); }
  int getStartX(const LayoutDefSeg& seg) const { return bg::get<0, 0>(seg); }
  int getStartY(const LayoutDefSeg& seg) const { return bg::get<0, 1>(seg); }
  int getStartZ(const LayoutDefSeg& seg) const { return bg::get<0, 2>(seg); }
  int getEndX(const LayoutDefSeg& seg) const { return bg::get<1, 0>(seg); }
  int getEndY(const LayoutDefSeg& seg) const { return bg::get<1, 1>(seg); }
  int getEndZ(const LayoutDefSeg& seg) const { return bg::get<1, 2>(seg); }
  int getLowX(const LayoutDefRect& rect) const { return bg::get<bg::min_corner, 0>(rect); }
  int getLowY(const LayoutDefRect& rect) const { return bg::get<bg::min_corner, 1>(rect); }
  int getLowZ(const LayoutDefRect& rect) const { return bg::get<bg::min_corner, 2>(rect); }
  int getHighX(const LayoutDefRect& rect) const { return bg::get<bg::max_corner, 0>(rect); }
  int getHighY(const LayoutDefRect& rect) const { return bg::get<bg::max_corner, 1>(rect); }
  int getHighZ(const LayoutDefRect& rect) const { return bg::get<bg::max_corner, 2>(rect); }
  LayoutDefPoint getCenter(const LayoutDefSeg& seg) const
  {
    return LayoutDefPoint((getStartX(seg) + getEndX(seg)) / 2, (getStartY(seg) + getEndY(seg)) / 2, (getStartZ(seg) + getEndZ(seg)) / 2);
  }
  LayoutDefPoint getCenter(const LayoutDefRect& rect) const
  {
    return LayoutDefPoint((getLowX(rect) + getHighX(rect)) / 2, (getLowY(rect) + getHighY(rect)) / 2, (getLowZ(rect) + getHighZ(rect)) / 2);
  }

  std::unordered_map<std::string, int> _layer_map;
};

}  // namespace ilm