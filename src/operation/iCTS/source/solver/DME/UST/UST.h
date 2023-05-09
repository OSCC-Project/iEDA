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
#include <unordered_map>
#include <vector>

#include "CTSAPI.hpp"
#include "CtsEnum.h"
#include "Operator.h"
#include "Params.h"
#include "SkewScheduler.h"
#include "UstNode.h"

namespace icts {

class UsefulSkewTree {
 public:
  typedef std::pair<Segment, Segment> EdgePair;
  UsefulSkewTree(const UstParams& params, SkewScheduler* skew_scheduler)
      : _params(params), _skew_scheduler(skew_scheduler) {}

  ~UsefulSkewTree() = default;

  template <typename T>
  void build(Topology<UstNode<T>>& topo) {
    if (topo.size() > 2) {
      bottomUpMerging(topo);
      topDownEmbedding(topo);
    }
  }

 private:
  template <typename T>
  void joiningSegment(UstNode<T>& node_a, UstNode<T>& node_b);

  template <typename T>
  void leafInit(UstNode<T>& node) const;
  std::pair<Segment, Segment> closestEdge(const Polygon& mr_a,
                                          const Polygon& mr_b) const;
  bool shortestDistanceRegion(Polygon& sdr, const Segment& seg_a,
                              const Segment& seg_b);
  template <typename T>
  SkewCommit determineSkewCommit(UstNode<T>& t_u, UstNode<T>& t_1,
                                 UstNode<T>& t_2);

  template <typename T>
  UstDelayFunc makeDelayFunc(UstNode<T>& node_a, UstNode<T>& node_b);

  vector<Point> snakeByDistance(const Point& begin, const Point& end,
                                const int& dist) const;

  template <typename T>
  void wireDetour(UstNode<T>& node_v, UstNode<T>& node_a, UstNode<T>& node_b,
                  const SkewRange& fsr);

  template <typename Coord, typename T>
  bool feasibleMergeRegion(CtsPolygon<Coord>& fmr, UstNode<T>& node_a,
                           UstNode<T>& node_b, const SkewRange& fsr);
  template <typename T>
  void feasibleMergeRegion(UstNode<T>& node_v, UstNode<T>& node_a,
                           UstNode<T>& node_b, const SkewRange& fsr);
  template <typename T>
  void placeRoot(UstNode<T>& node);
  template <typename T>
  void bottomUpMerging(Topology<UstNode<T>>& topo);
  template <typename T>
  void topDownEmbedding(Topology<UstNode<T>>& topo);

 private:
  UstParams _params;
  SkewScheduler* _skew_scheduler;
};
template <typename T>
void UsefulSkewTree::leafInit(UstNode<T>& node) const {
  Point leaf_loc = node.get_loc();
  node.set_merge_region({leaf_loc});
  auto name = DataTraits<T>::getId(node.get_data());
  auto before_delay = _skew_scheduler->find_before_delay(name);
  if (before_delay) {
    // buffer
    node.add_point_delay(leaf_loc, before_delay.value());
  } else {
    // sink
    auto id = _skew_scheduler->find_matrix_id(name);
    // auto init_cap = CTSAPIInst.getSinkCap(name);
    auto init_cap = 0;
    node.add_point_delay(leaf_loc, UstDelay(0, 0, id, id, init_cap));
  }
}

template <typename T>
inline void UsefulSkewTree::joiningSegment(UstNode<T>& node_a,
                                           UstNode<T>& node_b) {
  auto mr_a = node_a.get_merge_region();
  auto mr_b = node_b.get_merge_region();

  // merge region intersect
  if (!gtl::empty(mr_a & mr_b)) {
    Segment join_seg;
    gtl::scale_up(mr_a, 10);
    gtl::scale_up(mr_b, 10);
    PolygonSet polyset;
    polyset += mr_a & mr_b;
    auto poly = polyset.front();
    gtl::scale_down(poly, 10);
    pgl::longest_segment(join_seg, poly.get_edges());
    if (pgl::rectilinear(join_seg)) {
      join_seg = {join_seg.low(), join_seg.low()};
    }
    node_a.set_joining_segment(join_seg);
    node_b.set_joining_segment(join_seg);
    return;
  }
  // not intersect
  auto edge = closestEdge(mr_a, mr_b);
  auto join_seg_a = edge.first;
  auto join_seg_b = edge.second;
  // 计算最近边中的有效的部分，有效部分作为joining segment
  // two closest edge are manhattan arc
  if (!pgl::manhattan_arc(join_seg_a) || !pgl::manhattan_arc(join_seg_b)) {
    auto point_pair = pgl::closest_point_pair(join_seg_a, join_seg_b);
    join_seg_a = Segment(point_pair.first, point_pair.first);
    join_seg_b = Segment(point_pair.second, point_pair.second);
  }
  Polygon trr;
  Coordinate radius = pgl::manhattan_distance(join_seg_a, join_seg_b);
  assert(pgl::tilted_rect_region(trr, join_seg_b, radius));
  assert(pgl::bound_intersect(join_seg_a, join_seg_a, trr));
  assert(pgl::tilted_rect_region(trr, join_seg_a, radius));
  assert(pgl::bound_intersect(join_seg_b, join_seg_b, trr));

  node_a.set_joining_segment(join_seg_a);
  node_b.set_joining_segment(join_seg_b);
}

template <typename T>
inline SkewCommit UsefulSkewTree::determineSkewCommit(UstNode<T>& t_u,
                                                      UstNode<T>& t_1,
                                                      UstNode<T>& t_2) {
  auto js_u = t_u.get_joining_segment();
  // determine skew(id_1,id_2) = x
  auto l_u = js_u.low();
  auto js_1 = t_1.get_joining_segment();
  auto js_2 = t_2.get_joining_segment();
  auto l_1 = pgl::closest_point(l_u, js_1);
  auto l_2 = pgl::closest_point(l_u, js_2);
  UstDelay delay_1;
  UstDelay delay_2;
  t_1.get_point_delay(delay_1, l_1);
  t_2.get_point_delay(delay_2, l_2);
  UstDelay delay;
  if (!t_u.find_point_delay(delay, l_u)) {
    UstDelayFunc delay_func(js_1, delay_1, js_2, delay_2, _params);
    delay = delay_func.delay(l_u);
  }
  auto left_delay = delay.get_left_delay();
  auto right_delay = delay.get_right_delay();
  Skew constraint = left_delay - right_delay;
  return SkewCommit(delay_1.get_left_id(), delay_2.get_right_id(), constraint);
}

template <typename T>
UstDelayFunc UsefulSkewTree::makeDelayFunc(UstNode<T>& node_a,
                                           UstNode<T>& node_b) {
  auto join_seg_a = node_a.get_joining_segment();
  auto join_seg_b = node_b.get_joining_segment();
  UstDelay delay_a;
  UstDelay delay_b;
  node_a.get_point_delay(delay_a, join_seg_a.low());
  node_b.get_point_delay(delay_b, join_seg_b.low());
  UstDelayFunc delay_func(join_seg_a, delay_a, join_seg_b, delay_b, _params);
  return delay_func;
}

template <typename Coord, typename T>
bool UsefulSkewTree::feasibleMergeRegion(CtsPolygon<Coord>& fmr,
                                         UstNode<T>& node_a, UstNode<T>& node_b,
                                         const SkewRange& fsr) {
  Polygon short_dist_region;
  auto join_seg_a = node_a.get_joining_segment();
  auto join_seg_b = node_b.get_joining_segment();
  if (!shortestDistanceRegion(short_dist_region, join_seg_a, join_seg_b)) {
    assert(false);
  }
  // 计算 SDR上所有顶点的延迟时间
  auto delay_func = makeDelayFunc(node_a, node_b);
  vector<CtsPoint<Coord>> points;
  for (auto edge : short_dist_region.get_edges()) {
    auto pair_a = pgl::closest_point_pair(edge, join_seg_a);
    auto pair_b = pgl::closest_point_pair(edge, join_seg_b);
    auto direct_edge = CtsSegment<Coord>(pair_a.first, pair_b.first);
    auto endpoints = delay_func.feasibleSkewEndpoint(direct_edge, fsr);
    for (auto point : endpoints) {
      points.emplace_back(point);
    }
  }
  if (points.empty()) {
    return false;
  }
  fmr = CtsPolygon<Coord>(points.begin(), points.end());
  pgl::simplify_polygon(fmr);
  return true;
}

inline vector<Point> UsefulSkewTree::snakeByDistance(const Point& begin,
                                                     const Point& end,
                                                     const int& dist) const {
  auto x = end.x() - begin.x();
  auto y = end.y() - begin.y();
  auto snake_dist = dist - std::abs(x) - std::abs(y);

  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto die = db_wrapper->get_core_bounding_box();
  auto direction = x > 0 ? 1 : -1;
  vector<Point> snake_points = {begin};
  auto snake_p1 = Point(begin.x() + direction * snake_dist / 2, begin.y());
  auto snake_p2 = Point(begin.x() + direction * snake_dist / 2, end.y());
  if (!(die.is_in(snake_p1) && die.is_in(snake_p2))) {
    snake_p1 = Point(end.x() - direction * snake_dist / 2, begin.y());
    snake_p2 = Point(end.x() - direction * snake_dist / 2, end.y());
  }
  snake_points.emplace_back(snake_p1);
  snake_points.emplace_back(snake_p2);
  snake_points.emplace_back(end);
  return snake_points;
}

template <typename T>
void UsefulSkewTree::wireDetour(UstNode<T>& node_v, UstNode<T>& node_a,
                                UstNode<T>& node_b, const SkewRange& fsr) {
  auto point_a = node_a.get_joining_segment().low();
  auto point_b = node_b.get_joining_segment().low();
  node_a.set_joining_segment(Segment(point_a, point_a));
  node_b.set_joining_segment(Segment(point_b, point_b));
  auto delay_func = makeDelayFunc(node_a, node_b);
  auto skew_left = delay_func.delay(point_a).get_skew();
  auto skew_demand_left = std::abs(fsr.second - skew_left);
  auto skew_right = delay_func.delay(point_b).get_skew();
  auto skew_demand_right = std::abs(fsr.first - skew_right);
  auto dist = pgl::manhattan_distance(point_a, point_b);
  UstDelay snake_delay;
  if (skew_demand_left > skew_demand_right) {
    auto left_snake_dist = delay_func.detourLeft(fsr.first);
    auto snake_points = snakeByDistance(point_b, point_a, left_snake_dist);
    DataTraits<T>::setDetour(node_a.get_data());
    DataTraits<T>::setDetourPoints(node_a.get_data(), snake_points);
    node_a.set_extra_wirelength(left_snake_dist - dist);
    node_v.set_merge_region(Polygon({point_b}));
    snake_delay = delay_func.elmoreDelay(point_b, left_snake_dist - dist, 0);
    node_v.add_point_delay(point_b, snake_delay);
  } else {
    auto right_snake_dist = delay_func.detourRight(fsr.second);
    auto snake_points = snakeByDistance(point_a, point_b, right_snake_dist);
    DataTraits<T>::setDetour(node_b.get_data());
    DataTraits<T>::setDetourPoints(node_b.get_data(), snake_points);
    node_b.set_extra_wirelength(right_snake_dist - dist);
    node_v.set_merge_region(Polygon({point_a}));
    snake_delay = delay_func.elmoreDelay(point_a, 0, right_snake_dist - dist);
    node_v.add_point_delay(point_a, snake_delay);
  }
}

template <typename T>
inline void UsefulSkewTree::feasibleMergeRegion(UstNode<T>& node_v,
                                                UstNode<T>& node_a,
                                                UstNode<T>& node_b,
                                                const SkewRange& fsr) {
  Polygon fmr;
  bool feasible = feasibleMergeRegion(fmr, node_a, node_b, fsr);
  if (feasible) {
    auto delay_func = makeDelayFunc(node_a, node_b);
    node_v.set_delay_func(delay_func);
    node_v.set_merge_region(fmr);
  } else {
    wireDetour(node_v, node_a, node_b, fsr);
  }
}

template <typename T>
inline void UsefulSkewTree::bottomUpMerging(Topology<UstNode<T>>& topo) {
  auto topo_itr = topo.postorder_vertexs();
  for (auto itr = topo_itr.first; itr != topo_itr.second; ++itr) {
    auto& t_v = *itr;
    if (itr.is_leaf()) {
      leafInit(t_v);
      continue;
    }

    auto& t_u = *itr.left();
    auto& t_w = *itr.right();
    bool matrix_update = false;

    if (!itr.left().is_leaf()) {
      // select l_u in mr(u) that is the closest to mr(w)
      joiningSegment(t_u, t_w);
      auto& t_1 = *itr.left().left();
      auto& t_2 = *itr.left().right();
      SkewCommit skew_commit = determineSkewCommit(t_u, t_1, t_2);
      // update D by skew commitment
      matrix_update = _skew_scheduler->updateDistanceMatrix(skew_commit);
    }

    if (!itr.right().is_leaf()) {
      auto& t_1 = *itr.right().left();
      auto& t_2 = *itr.right().right();
      if (matrix_update) {
        // update mr(w)
        auto fsr = _skew_scheduler->get_feasible_skew_range(
            DataTraits<T>::getId(t_1.get_data()),
            DataTraits<T>::getId(t_2.get_data()));
        feasibleMergeRegion(t_w, t_1, t_2, fsr);
      }

      // perform steps
      joiningSegment(t_u, t_w);
      SkewCommit skew_commit = determineSkewCommit(t_w, t_1, t_2);
      // update D by skew commitment
      _skew_scheduler->updateDistanceMatrix(skew_commit);
    }
    auto fsr = _skew_scheduler->get_feasible_skew_range(
        DataTraits<T>::getId(t_u.get_data()),
        DataTraits<T>::getId(t_w.get_data()));
    joiningSegment(t_u, t_w);
    feasibleMergeRegion(t_v, t_u, t_w, fsr);
  }
}

template <typename T>
inline void UsefulSkewTree::placeRoot(UstNode<T>& node) {
  auto& merge_region = node.get_merge_region();
  PropagationTime best_skew = std::numeric_limits<PropagationTime>::max();
  UstDelay best_delay;
  Point best_point(0, 0);
  for (auto edge : merge_region.get_edges()) {
    auto edge_points = pgl::edge_to_points(edge);
    for (auto itr = edge_points.begin(); itr != edge_points.end(); ++itr) {
      auto& point = *itr;
      UstDelay delay;
      node.get_point_delay(delay, point);
      auto skew = delay.get_skew();
      auto fsr = _skew_scheduler->get_feasible_skew_range(delay.get_left_id(),
                                                          delay.get_right_id());
      if (fsr.first <= skew && skew <= fsr.second) {
        if (std::abs(skew) < best_skew) {
          best_skew = std::abs(skew);
          best_delay = delay;
          best_point = point;
        }
      }
    }
  }
  assert(best_skew != std::numeric_limits<PropagationTime>::max());
  node.set_loc(best_point);
  auto name = DataTraits<T>::getId(node.get_data());
  _skew_scheduler->insert_delay(name, best_delay);
  return;
}

template <typename T>
inline void UsefulSkewTree::topDownEmbedding(Topology<UstNode<T>>& topo) {
  auto pre_itr = topo.preorder_vertexs();
  for (auto itr = pre_itr.first; itr != pre_itr.second; ++itr) {
    auto& node = *itr;
    if (itr.is_root()) {
      placeRoot(node);
    } else {
      auto& parent = *itr.parent();
      Point parent_loc = parent.get_loc();
      Segment& join_seg = node.get_joining_segment();
      auto closest_point = pgl::closest_point(parent_loc, join_seg);
      node.set_loc(closest_point);
    }
  }
  auto post_itr = topo.postorder_vertexs();
  for (auto itr = post_itr.first; itr != post_itr.second; ++itr) {
    auto& node = *itr;
    if (!itr.is_root()) {
      auto& parent = *itr.parent();
      auto parent_loc = parent.get_loc();
      auto cur_loc = node.get_loc();
      auto dist = pgl::manhattan_distance(parent_loc, cur_loc) +
                  node.get_extra_wirelength();
      auto sub_wirelength = DataTraits<T>::getSubWirelength(parent.get_data());
      DataTraits<T>::setSubWirelength(parent.get_data(),
                                      std::max(sub_wirelength, dist));
    }
  }
}

inline std::pair<Segment, Segment> UsefulSkewTree::closestEdge(
    const Polygon& mr_a, const Polygon& mr_b) const {
  vector<EdgePair> edge_pairs = pgl::closest_edges(mr_a, mr_b);
  assert(!edge_pairs.empty());
  for (auto& edge_pair : edge_pairs) {
    if (pgl::cross_product(edge_pair.first, edge_pair.second) == 0 &&
        pgl::manhattan_arc(edge_pair.first) &&
        pgl::manhattan_arc(edge_pair.second)) {
      return edge_pair;
    }
  }
  for (auto& edge_pair : edge_pairs) {
    if (pgl::cross_product(edge_pair.first, edge_pair.second) == 0) {
      return edge_pair;
    }
  }
  for (auto& edge_pair : edge_pairs) {
    if (pgl::dot_product(edge_pair.first, edge_pair.second) == 0) {
      return edge_pair;
    }
  }
  return edge_pairs.front();
}

inline bool UsefulSkewTree::shortestDistanceRegion(Polygon& sdr,
                                                   const Segment& seg_a,
                                                   const Segment& seg_b) {
  if (seg_a == seg_b) {
    sdr = seg_a.low() == seg_a.high() ? Polygon({seg_a.low()})
                                      : Polygon({seg_a.low(), seg_a.high()});
    return true;
  }
  if (seg_a.low() == seg_a.high() && seg_b.low() == seg_b.high() &&
      pgl::rectilinear(seg_a.low(), seg_b.low())) {
    sdr = CtsPolygon<int64_t>({seg_a.low(), seg_b.low()});
    return true;
  }
  CtsRectangle<int64_t> bbox;
  vector<CtsPoint<int64_t>> points{seg_a.low(), seg_a.high(), seg_b.low(),
                                   seg_b.high()};
  pgl::convex_hull(points);
  pgl::extents(bbox, points);

  CtsPolygon<int64_t> trr_a, trr_b;
  Coordinate radius = pgl::manhattan_distance(seg_a, seg_b);
  pgl::tilted_rect_region(trr_a, seg_a, radius);
  pgl::tilted_rect_region(trr_b, seg_b, radius);

  gtl::scale_up(bbox, SCALE_FACTOR);
  gtl::scale_up(trr_a, SCALE_FACTOR);
  gtl::scale_up(trr_b, SCALE_FACTOR);

  CtsPolygonSet<int64_t> polyset;
  polyset += bbox & (trr_a & trr_b);

  CtsPolygon<int64_t> poly;
  if (!polyset.empty()) {
    poly = polyset.front();
    gtl::scale_down(poly, SCALE_FACTOR);
    sdr = poly;
  }
  return !polyset.empty();
}
}  // namespace icts