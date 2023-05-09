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
#define SCALE_FACTOR 1000

#include <boost/polygon/gtl.hpp>
#include <fstream>
#include <set>
#include <utility>
#include <vector>

#include "BstNode.h"
#include "CtsEnum.h"
#include "GDSPloter.h"
#include "Operator.h"
#include "Params.h"
#include "Topology.h"
#include "pgl.h"

namespace icts {

using namespace boost::polygon::operators;
struct ManhattanArcTag {};
struct RectilinearTag {};

class BoundedSkewTree {
 public:
  BoundedSkewTree() {}
  BoundedSkewTree(const BstParams &params) { _params = params; }
  ~BoundedSkewTree() = default;

  template <typename T>
  void build(Topology<BstNode<T>> &topo) {
    if (topo.size() > 2) {
      buildMergingRegion(topo);
      findExactPlacement(topo);
    }
  }
  icts::BstParams &get_params() { return _params; }

 public:
  typedef std::pair<Segment, Segment> EdgePair;

  double get_skew_bound() const { return _params.get_skew_bound(); }

  template <typename T>
  void buildMergingRegion(Topology<BstNode<T>> &topo);

  template <typename T>
  void findExactPlacement(Topology<BstNode<T>> &topo);

  template <typename T>
  void placeRoot(BstNode<T> &node);

  template <typename T>
  void leafMergeRegion(BstNode<T> &node) const {
    Point leaf_loc = node.get_loc();
    node.set_merge_region({leaf_loc});
    node.add_point_delay(leaf_loc, BstDelay{0, 0});
  }

  template <typename T>
  BstNode<T> merge(BstNode<T> &node_a, BstNode<T> &node_b);
  template <typename T>
  void joiningSegment(BstNode<T> &node_a, BstNode<T> &node_b);
  template <typename T>
  bool feasibleMergeRegion(BstNode<T> &node_v, BstNode<T> &node_a,
                           BstNode<T> &node_b, ManhattanArcTag tag);
  template <typename T>
  bool feasibleMergeRegion(BstNode<T> &node_v, BstNode<T> &node_a,
                           BstNode<T> &node_b, RectilinearTag tag);
  template <typename T>
  bool calculateMergeRegion(BstNode<T> &node_v, BstNode<T> &node_a,
                            BstNode<T> &node_b);

  bool shortestDistanceRegion(Polygon &sdr, const Segment &seg_a,
                              const Segment &seg_b);

  template <typename Coord, typename T>
  bool feasibleMergeRegion(CtsPolygon<Coord> &fmr, BstNode<T> &node_a,
                           BstNode<T> &node_b, RectilinearTag tag);
  template <typename Coord, typename T>
  bool feasibleMergeRegion(CtsPolygon<Coord> &fmr, BstNode<T> &node_a,
                           BstNode<T> &node_b, ManhattanArcTag tag);

  // return: 0 manhanttan arc, 1 rectinear
  SegmentType joinSegmentType(const Segment &join_seg_a,
                              const Segment &join_seg_b) const;
  // 找到最近的两条边
  std::pair<Segment, Segment> closestEdge(const Polygon &mr_a,
                                          const Polygon &mr_b) const;

  gtl::orientation_2d parallelLineOrient(const Segment &join_seg_a,
                                         const Segment &join_seg_b) const {
    gtl::orientation_2d orient =
        pgl::horizontal(join_seg_a) ? gtl::VERTICAL : gtl::HORIZONTAL;
    if (join_seg_a.low() == join_seg_a.high() &&
        join_seg_b.low() == join_seg_b.high()) {
      Segment seg(join_seg_a.low(), join_seg_b.low());
      orient = pgl::horizontal(seg) ? gtl::HORIZONTAL : gtl::VERTICAL;
    }
    return orient;
  }

 private:
  BstParams _params;
};

template <typename T>
inline void BoundedSkewTree::buildMergingRegion(Topology<BstNode<T>> &topo) {
  auto pv_itr = topo.postorder_vertexs();
  for (auto itr = pv_itr.first; itr != pv_itr.second; ++itr) {
    auto &bst_node = *itr;

    if (itr.is_leaf()) {
      leafMergeRegion(bst_node);
    } else {
      auto &left = *itr.left();
      auto &right = *itr.right();
      auto parent = merge(left, right);
      bst_node.copy_message(parent);
    }
  }
}

template <typename T>
inline void BoundedSkewTree::findExactPlacement(Topology<BstNode<T>> &topo) {
  auto pv_itr = topo.preorder_vertexs();

  for (auto itr = pv_itr.first; itr != pv_itr.second; ++itr) {
    auto &node = *itr;
    if (itr.is_root()) {
      placeRoot(node);
    } else {
      auto &parent = *itr.parent();
      Point parent_loc = parent.get_loc();
      Segment &join_seg = node.get_joining_segment();
      if (node.edge_len_determined() == false) {
        auto dist = pgl::manhattan_distance(parent_loc, join_seg);
        node.set_edge_length(dist);
      }
      auto closest_point = pgl::closest_point(parent_loc, join_seg);
      node.set_loc(closest_point);
    }
  }
}

template <typename T>
inline void BoundedSkewTree::placeRoot(BstNode<T> &node) {
  auto &merge_region = node.get_merge_region();

  BstDelay delay(0, 0);
  Point min_point = *merge_region.begin();
  node.get_point_delay(delay, min_point);
  Coordinate min_skew = delay._max_t - delay._min_t;
  for (auto it = merge_region.begin(); it != merge_region.end(); ++it) {
    auto &point = *it;
    node.get_point_delay(delay, point);
    auto skew = delay._max_t - delay._min_t;
    if (skew < min_skew) {
      min_skew = skew;
      min_point = point;
    }
  }
  node.set_loc(min_point);
}

template <typename T>
inline BstNode<T> BoundedSkewTree::merge(BstNode<T> &node_a,
                                         BstNode<T> &node_b) {
  // 计算得到的结果是两个完整的平行边
  joiningSegment(node_a, node_b);
  auto &join_seg_a = node_a.get_joining_segment();
  auto &join_seg_b = node_b.get_joining_segment();

  BstNode<T> node_v;
  if (joinSegmentType(join_seg_a, join_seg_b) == SegmentType::kMANHATAN_ARC) {
    feasibleMergeRegion(node_v, node_a, node_b, ManhattanArcTag());
  } else {
    feasibleMergeRegion(node_v, node_a, node_b, RectilinearTag());
  }

  if (node_v.get_merge_region().empty()) {
    calculateMergeRegion(node_v, node_a, node_b);
  }

  return node_v;
}

template <typename T>
inline void BoundedSkewTree::joiningSegment(BstNode<T> &node_a,
                                            BstNode<T> &node_b) {
  auto &mr_a = node_a.get_merge_region();
  auto &mr_b = node_b.get_merge_region();

  // merge region intersect
  if (!gtl::empty(mr_a & mr_b)) {
    Segment join_seg;
    pgl::bound_intersect(join_seg, mr_a, mr_b);
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
  Polygon trr;
  Coordinate radius = pgl::manhattan_distance(join_seg_a, join_seg_b);
  if (pgl::rectilinear(join_seg_a) && pgl::rectilinear(join_seg_b)) {
    gtl::orientation_2d orient =
        pgl::horizontal(join_seg_a) ? gtl::HORIZONTAL : gtl::VERTICAL;
    auto interval_a = pgl::interval(join_seg_a, orient);
    auto interval_b = pgl::interval(join_seg_b, orient);

    if (pgl::dot_product(join_seg_a, join_seg_b) == 0 ||
        gtl::intersects(interval_a, interval_b) == false) {
      auto point_pair = pgl::closest_point_pair(join_seg_a, join_seg_b);
      join_seg_a = Segment(point_pair.first, point_pair.first);
      join_seg_b = Segment(point_pair.second, point_pair.second);
    } else {
      auto interval = interval_a & interval_b;
      join_seg_a.set(gtl::LOW, join_seg_a.low().set(orient, interval.low()));
      join_seg_a.set(gtl::HIGH, join_seg_a.high().set(orient, interval.high()));
      join_seg_b.set(gtl::LOW, join_seg_b.low().set(orient, interval.low()));
      join_seg_b.set(gtl::HIGH, join_seg_b.high().set(orient, interval.high()));
    }
  } else {
    if (!pgl::manhattan_arc(join_seg_a) || !pgl::manhattan_arc(join_seg_b)) {
      auto point_pair = pgl::closest_point_pair(join_seg_a, join_seg_b);
      if (!pgl::manhattan_arc(join_seg_a)) {
        join_seg_a.low(point_pair.first);
        join_seg_a.high(point_pair.first);
      }
      if (!pgl::manhattan_arc(join_seg_b)) {
        join_seg_b.low(point_pair.second);
        join_seg_b.high(point_pair.second);
      }
    }

    assert(pgl::tilted_rect_region(trr, join_seg_b, radius));
    assert(pgl::bound_intersect(join_seg_a, join_seg_a, trr));
    assert(pgl::tilted_rect_region(trr, join_seg_a, radius));
    assert(pgl::bound_intersect(join_seg_b, join_seg_b, trr));
  }

  node_a.set_joining_segment(join_seg_a);
  node_b.set_joining_segment(join_seg_b);
}

template <typename T>
inline bool BoundedSkewTree::calculateMergeRegion(BstNode<T> &node_v,
                                                  BstNode<T> &node_a,
                                                  BstNode<T> &node_b) {
  auto &join_seg_a = node_a.get_joining_segment();
  auto &join_seg_b = node_b.get_joining_segment();
  auto dist = pgl::manhattan_distance(join_seg_a, join_seg_b);
  auto skew_bound = get_skew_bound();

  Pair<PointDelay> pda, pdb;
  node_a.get_point_delay(pda, node_a.subordinateEdge(join_seg_a));
  node_b.get_point_delay(pdb, node_b.subordinateEdge(join_seg_b));
  BstDelayFunc func_a(pda.first, pda.second, DelayFuncType::kOLD);
  BstDelayFunc func_b(pdb.first, pdb.second, DelayFuncType::kOLD);

  auto turn_points_a = func_a.skewTurnPoint();
  auto turn_points_b = func_b.skewTurnPoint();
  auto min_skew_a = func_a.minSkew();
  auto min_skew_b = func_b.minSkew();

  // msr: minimum skew region
  Segment msr = Segment(turn_points_a.first, turn_points_a.second);
  BstDelayFunc func = func_a;
  node_a.set_edge_length(0);
  node_b.set_edge_length(dist + min_skew_a - skew_bound);
  if (min_skew_a > min_skew_b) {
    func = func_b;
    msr = Segment(turn_points_b.first, turn_points_b.second);
    node_a.set_edge_length(dist + min_skew_b - skew_bound);
    node_b.set_edge_length(0);
  }
  node_v.add_point_delay(msr.low(), func.delayTime(msr.low()));
  node_v.add_point_delay(msr.high(), func.delayTime(msr.high()));
  node_v.set_merge_region({msr.low(), msr.high()});

  auto &merge_region = node_v.get_merge_region();
  for (auto it = merge_region.begin(); it != merge_region.end(); ++it) {
    auto &point = *it;
    BstDelay delay;
    node_v.get_point_delay(delay, point);
    delay._min_t = delay._max_t - skew_bound;
    node_v.update_point_delay(point, delay);
  }
  return true;
}

inline SegmentType BoundedSkewTree::joinSegmentType(
    const Segment &join_seg_a, const Segment &join_seg_b) const {
  if (join_seg_a.low() == join_seg_a.high() &&
      join_seg_b.low() == join_seg_b.high()) {
    if (join_seg_a.low().x() == join_seg_b.low().x() ||
        join_seg_a.low().y() == join_seg_b.low().y()) {
      return SegmentType::kRECTILINEAR;
    } else {
      return SegmentType::kMANHATAN_ARC;
    }
  } else if (join_seg_a.low() == join_seg_a.high() ||
             join_seg_b.low() == join_seg_b.high()) {
    return SegmentType::kMANHATAN_ARC;
  } else {
    if (pgl::rectilinear(join_seg_a) && pgl::rectilinear(join_seg_b)) {
      return SegmentType::kRECTILINEAR;
    } else {
      return SegmentType::kMANHATAN_ARC;
    }
  }
}

template <typename T>
inline bool BoundedSkewTree::feasibleMergeRegion(BstNode<T> &node_v,
                                                 BstNode<T> &node_a,
                                                 BstNode<T> &node_b,
                                                 ManhattanArcTag tag) {
  CtsPolygon<double> ffmr;
  Polygon fmr;

  bool ans = feasibleMergeRegion(ffmr, node_a, node_b, tag);
  if (!ans) {
    return false;
  }

  pgl::round_off_coord(fmr, ffmr);
  node_v.set_merge_region(fmr);

  Pair<PointDelay> pda, pdb;
  auto &join_seg_a = node_a.get_joining_segment();
  auto &join_seg_b = node_b.get_joining_segment();
  node_a.get_point_delay(pda, node_a.subordinateEdge(join_seg_a));
  node_b.get_point_delay(pdb, node_b.subordinateEdge(join_seg_b));
  BstDelayFunc func_a(pda.first, pda.second, DelayFuncType::kOLD);
  BstDelayFunc func_b(pdb.first, pdb.second, DelayFuncType::kOLD);

  for (auto it = fmr.begin(); it != fmr.end(); ++it) {
    Point point = *it;
    auto point_a = pgl::closest_point(point, join_seg_a);
    auto point_b = pgl::closest_point(point, join_seg_b);
    auto delay_a = func_a.delayTime(point_a);
    auto delay_b = func_b.delayTime(point_b);

    BstDelayFunc delay_func(point_a, delay_a, point_b, delay_b,
                            DelayFuncType::kNEW);
    auto delay = delay_func.delayTime(point);
    node_v.add_point_delay(point, delay);
  }
  return true;
}

template <typename Coord, typename T>
bool BoundedSkewTree::feasibleMergeRegion(CtsPolygon<Coord> &fmr,
                                          BstNode<T> &node_a,
                                          BstNode<T> &node_b,
                                          ManhattanArcTag tag) {
  Polygon short_dist_region;
  auto &join_seg_a = node_a.get_joining_segment();
  auto &join_seg_b = node_b.get_joining_segment();
  shortestDistanceRegion(short_dist_region, join_seg_a, join_seg_b);

  // 计算 SDR上所有顶点的延迟时间
  vector<CtsPoint<Coord>> points;
  auto skew_bound = get_skew_bound();
  for (auto it = short_dist_region.begin(); it != short_dist_region.end();
       ++it) {
    Point point = *it;
    if (gtl::contains(join_seg_a, point) || gtl::contains(join_seg_b, point)) {
      continue;
    }

    auto point_a = pgl::closest_point(point, join_seg_a);
    auto point_b = pgl::closest_point(point, join_seg_b);
    BstDelay delay_a, delay_b;
    assert(node_a.get_point_delay(delay_a, point_a));
    assert(node_b.get_point_delay(delay_b, point_b));
    BstDelayFunc delay_func(point_a, delay_a, point_b, delay_b,
                            DelayFuncType::kNEW);

    auto have_bound_point = delay_func.boundPoints(points, skew_bound, point);
    if (have_bound_point) {
      auto bound_point_l = points[points.size() - 2];
      auto bound_point_r = points[points.size() - 1];
      if (!pgl::rectilinear(bound_point_l, bound_point_r)) {
        points.emplace_back(point);
      }
    }
  }

  if (points.empty()) {
    return false;
  }
  pgl::convex_hull(points);
  fmr = CtsPolygon<Coord>(points.begin(), points.end());
  return true;
}

template <typename Coord, typename T>
inline bool BoundedSkewTree::feasibleMergeRegion(CtsPolygon<Coord> &fmr,
                                                 BstNode<T> &node_a,
                                                 BstNode<T> &node_b,
                                                 RectilinearTag tag) {
  Pair<PointDelay> pda, pdb;
  auto &join_seg_a = node_a.get_joining_segment();
  auto &join_seg_b = node_b.get_joining_segment();
  node_a.get_point_delay(pda, node_a.subordinateEdge(join_seg_a));
  node_b.get_point_delay(pdb, node_b.subordinateEdge(join_seg_b));
  BstDelayFunc func_a(pda.first, pda.second, DelayFuncType::kOLD);
  BstDelayFunc func_b(pdb.first, pdb.second, DelayFuncType::kOLD);

  auto orient = parallelLineOrient(join_seg_a, join_seg_b);
  std::set<Point> points;
  points.insert(join_seg_a.low());
  points.insert(join_seg_a.high());

  auto turn_point_a = func_a.skewTurnPoint();
  auto turn_point_b = func_b.skewTurnPoint();
  points.insert(turn_point_a.first);
  points.insert(turn_point_a.second);

  auto coord_a = join_seg_a.low().get(orient);
  turn_point_b.first.set(orient, coord_a);
  turn_point_b.second.set(orient, coord_a);
  points.insert(turn_point_b.first);
  points.insert(turn_point_b.second);

  // 删除不在 joining segment 上的点
  for (auto itr = points.begin(); itr != points.end();) {
    if (!gtl::contains(join_seg_a, *itr)) {
      itr = points.erase(itr);
    } else {
      itr++;
    }
  }

  // 计算每一对skew turn point所形成的feasible merge region
  vector<CtsPoint<Coord>> vertexs;
  auto skew_bound = get_skew_bound();
  Coordinate coord_b = join_seg_b.low().get(orient);
  for (auto &point_a : points) {
    Point point_b = point_a;
    point_b.set(orient, coord_b);

    auto delay_a = func_a.delayTime(point_a);
    auto delay_b = func_b.delayTime(point_b);
    BstDelayFunc delay_func(point_a, delay_a, point_b, delay_b,
                            DelayFuncType::kNEW);
    delay_func.boundPoints(vertexs, skew_bound);
  }

  if (vertexs.empty()) {
    return false;
  }
  pgl::convex_hull(vertexs);
  fmr = CtsPolygon<Coord>(vertexs.begin(), vertexs.end());
  return true;
}

template <typename T>
inline bool BoundedSkewTree::feasibleMergeRegion(BstNode<T> &node_v,
                                                 BstNode<T> &node_a,
                                                 BstNode<T> &node_b,
                                                 RectilinearTag tag) {
  auto &join_seg_a = node_a.get_joining_segment();
  auto &join_seg_b = node_b.get_joining_segment();
  gtl::orientation_2d orient = parallelLineOrient(join_seg_a, join_seg_b);
  auto coord_a = join_seg_a.low().get(orient);
  auto coord_b = join_seg_b.low().get(orient);
  CtsPolygon<Coordinate> fmr;
  CtsPolygon<double> ffmr;

  bool ans = feasibleMergeRegion(ffmr, node_a, node_b, tag);
  if (ans == false) {
    return false;
  }
  pgl::round_off_coord(fmr, ffmr);

  std::vector<Point> points(fmr.begin(), fmr.end());
  if (points.size() == 2) {
    gtl::orientation_2d orient = parallelLineOrient(join_seg_a, join_seg_b);
    Point point_a = points.front();
    Point point_b = points.front();
    point_a.set(orient, coord_a);
    point_b.set(orient, coord_b);
    Segment seg;
    pgl::intersect(seg, Segment(point_a, point_b),
                   Segment(points.front(), points.back()));
    fmr = Polygon({seg.low(), seg.high()});
  } else {
    PolygonSet polyset;
    Rectangle rect = pgl::extends(join_seg_a, join_seg_b);
    polyset += rect & fmr;
    if (!polyset.empty()) {
      fmr = polyset.front();
    }
  }
  node_v.set_merge_region(fmr);

  // 获取joining segment上的延迟函数
  Pair<PointDelay> pda, pdb;
  node_a.get_point_delay(pda, node_a.subordinateEdge(join_seg_a));
  node_b.get_point_delay(pdb, node_b.subordinateEdge(join_seg_b));
  BstDelayFunc func_a(pda.first, pda.second, DelayFuncType::kOLD);
  BstDelayFunc func_b(pdb.first, pdb.second, DelayFuncType::kOLD);

  // get all parallel line segment composed of skew turning points
  // 计算每一对skew turn point所形成的feasible merge region
  for (auto itr = fmr.begin(); itr != fmr.end(); ++itr) {
    Point point = *itr;
    Point point_a = point;
    Point point_b = point;
    point_a.set(orient, coord_a);
    point_b.set(orient, coord_b);
    auto delay_a = func_a.delayTime(point_a);
    auto delay_b = func_b.delayTime(point_b);
    BstDelayFunc delay_func(point_a, delay_a, point_b, delay_b,
                            DelayFuncType::kNEW);

    auto delay = delay_func.delayTime(point);
    node_v.add_point_delay(point, delay);
  }
  return true;
}

inline bool BoundedSkewTree::shortestDistanceRegion(Polygon &sdr,
                                                    const Segment &seg_a,
                                                    const Segment &seg_b) {
  if (seg_a == seg_b) {
    return false;
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

inline std::pair<Segment, Segment> BoundedSkewTree::closestEdge(
    const Polygon &mr_a, const Polygon &mr_b) const {
  vector<EdgePair> edge_pairs = pgl::closest_edges(mr_a, mr_b);
  assert(!edge_pairs.empty());
  for (auto &edge_pair : edge_pairs) {
    if (pgl::cross_product(edge_pair.first, edge_pair.second) == 0 &&
        pgl::manhattan_arc(edge_pair.first) &&
        pgl::manhattan_arc(edge_pair.second)) {
      return edge_pair;
    }
  }
  for (auto &edge_pair : edge_pairs) {
    if (pgl::cross_product(edge_pair.first, edge_pair.second) == 0) {
      return edge_pair;
    }
  }
  for (auto &edge_pair : edge_pairs) {
    if (pgl::dot_product(edge_pair.first, edge_pair.second) == 0) {
      return edge_pair;
    }
  }
  return edge_pairs.front();
}

}  // namespace icts