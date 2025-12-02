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

#include "Direction.hpp"
#include "EXTPlanarRect.hpp"
#include "GridMap.hpp"
#include "LayerCoord.hpp"
#include "MTree.hpp"
#include "Orientation.hpp"
#include "PlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "RTHeader.hpp"
#include "ScaleAxis.hpp"
#include "ScaleGrid.hpp"
#include "Segment.hpp"
#include "ViaMaster.hpp"
#include "json.hpp"

namespace irt {

#define RTUTIL (irt::Utility::getInst())

class Utility
{
 public:
  static void initInst();
  static Utility& getInst();
  static void destroyInst();
  // function

#if 1  // 方向方位计算

  // 判断线段方向
  static Direction getDirection(PlanarCoord start_coord, PlanarCoord end_coord)
  {
    if (start_coord == end_coord) {
      return Direction::kProximal;
    }

    bool is_h = (start_coord.get_y() == end_coord.get_y());
    bool is_v = (start_coord.get_x() == end_coord.get_x());
    return is_h ? Direction::kHorizontal : is_v ? Direction::kVertical : Direction::kOblique;
  }

  // 判断线段是否为一个点
  static bool isProximal(const PlanarCoord& start_coord, const PlanarCoord& end_coord) { return getDirection(start_coord, end_coord) == Direction::kProximal; }

  // 判断线段是否为水平线
  static bool isHorizontal(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return getDirection(start_coord, end_coord) == Direction::kHorizontal;
  }

  // 判断线段是否为竖直线
  static bool isVertical(const PlanarCoord& start_coord, const PlanarCoord& end_coord) { return getDirection(start_coord, end_coord) == Direction::kVertical; }

  // 判断线段是否为斜线
  static bool isOblique(const PlanarCoord& start_coord, const PlanarCoord& end_coord) { return getDirection(start_coord, end_coord) == Direction::kOblique; }

  // 判断线段是否为直角线
  static bool isRightAngled(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return isProximal(start_coord, end_coord) || isHorizontal(start_coord, end_coord) || isVertical(start_coord, end_coord);
  }

  static std::vector<Orientation> getOrientationList(const PlanarCoord& start_coord, const PlanarCoord& end_coord,
                                                     Orientation point_orientation = Orientation::kNone)
  {
    std::set<Orientation> orientation_set;
    orientation_set.insert(getOrientation(start_coord, PlanarCoord(start_coord.get_x(), end_coord.get_y()), point_orientation));
    orientation_set.insert(getOrientation(start_coord, PlanarCoord(end_coord.get_x(), start_coord.get_y()), point_orientation));
    orientation_set.erase(Orientation::kNone);
    return std::vector<Orientation>(orientation_set.begin(), orientation_set.end());
  }

  // 判断线段方向 从start到end
  static Orientation getOrientation(const LayerCoord& start_coord, const LayerCoord& end_coord, Orientation point_orientation = Orientation::kNone)
  {
    Orientation orientation;

    if (start_coord.get_layer_idx() == end_coord.get_layer_idx()) {
      if (isProximal(start_coord, end_coord)) {
        orientation = point_orientation;
      } else if (isHorizontal(start_coord, end_coord)) {
        orientation = (start_coord.get_x() - end_coord.get_x()) > 0 ? Orientation::kWest : Orientation::kEast;
      } else if (isVertical(start_coord, end_coord)) {
        orientation = (start_coord.get_y() - end_coord.get_y()) > 0 ? Orientation::kSouth : Orientation::kNorth;
      } else {
        orientation = Orientation::kOblique;
      }
    } else {
      if (isProximal(start_coord, end_coord)) {
        orientation = (start_coord.get_layer_idx() - end_coord.get_layer_idx()) > 0 ? Orientation::kBelow : Orientation::kAbove;
      } else {
        orientation = Orientation::kOblique;
      }
    }
    return orientation;
  }

  static Orientation getOrientation(const PlanarCoord& start_coord, const PlanarCoord& end_coord, Orientation point_orientation = Orientation::kNone)
  {
    return getOrientation(LayerCoord(start_coord), LayerCoord(end_coord));
  }

  static Orientation getOppositeOrientation(Orientation orientation)
  {
    Orientation opposite_orientation;
    switch (orientation) {
      case Orientation::kEast:
        opposite_orientation = Orientation::kWest;
        break;
      case Orientation::kWest:
        opposite_orientation = Orientation::kEast;
        break;
      case Orientation::kSouth:
        opposite_orientation = Orientation::kNorth;
        break;
      case Orientation::kNorth:
        opposite_orientation = Orientation::kSouth;
        break;
      case Orientation::kAbove:
        opposite_orientation = Orientation::kBelow;
        break;
      case Orientation::kBelow:
        opposite_orientation = Orientation::kAbove;
        break;
      default:
        RTLOG.error(Loc::current(), "The orientation is error!");
        break;
    }
    return opposite_orientation;
  }

  static std::vector<Orientation> convertToOrientation(Direction direction)
  {
    std::vector<Orientation> orient_list;
    if (direction == Direction::kHorizontal) {
      orient_list = {Orientation::kEast, Orientation::kWest};
    } else if (direction == Direction::kVertical) {
      orient_list = {Orientation::kSouth, Orientation::kNorth};
    }
    return orient_list;
  }

#endif

#if 1  // 距离线长计算

  // 获得两坐标的曼哈顿距离
  static int32_t getManhattanDistance(LayerCoord start_coord, LayerCoord end_coord)
  {
    return std::abs(start_coord.get_x() - end_coord.get_x()) + std::abs(start_coord.get_y() - end_coord.get_y())
           + std::abs(start_coord.get_layer_idx() - end_coord.get_layer_idx());
  }

  // 获得两坐标的曼哈顿距离
  static int32_t getManhattanDistance(PlanarCoord start_coord, PlanarCoord end_coord)
  {
    return std::abs(start_coord.get_x() - end_coord.get_x()) + std::abs(start_coord.get_y() - end_coord.get_y());
  }

  // 获得线段和点的最短曼哈顿距离
  static int32_t getManhattanDistance(Segment<PlanarCoord>& seg, PlanarCoord& coord)
  {
    int32_t coord_x = coord.get_x();
    int32_t coord_y = coord.get_y();
    int32_t first_coord_x = seg.get_first().get_x();
    int32_t first_coord_y = seg.get_first().get_y();
    int32_t second_coord_x = seg.get_second().get_x();
    int32_t second_coord_y = seg.get_second().get_y();

    if (first_coord_y == second_coord_y && first_coord_x <= coord_x && coord_x <= second_coord_x) {
      return std::abs(first_coord_y - coord_y);
    } else if (first_coord_x == second_coord_x && first_coord_y <= coord_y && coord_y <= second_coord_y) {
      return std::abs(first_coord_x - coord_x);
    } else {
      return std::min(getManhattanDistance(coord, seg.get_first()), getManhattanDistance(coord, seg.get_second()));
    }
  }

  // 获得两个线段的最短曼哈顿距离
  static int32_t getManhattanDistance(Segment<PlanarCoord>& seg1, Segment<PlanarCoord>& seg2)
  {
    if (isIntersection(seg1, seg2)) {
      return 0;
    }
    return std::min(getManhattanDistance(seg1, seg2.get_first()), getManhattanDistance(seg1, seg2.get_second()));
  }

  // 获得两个矩形的欧式距离
  static double getEuclideanDistance(PlanarRect& a, PlanarRect& b)
  {
    int32_t x_spacing = std::max(b.get_ll_x() - a.get_ur_x(), a.get_ll_x() - b.get_ur_x());
    int32_t y_spacing = std::max(b.get_ll_y() - a.get_ur_y(), a.get_ll_y() - b.get_ur_y());

    if (x_spacing > 0 && y_spacing > 0) {
      return std::sqrt((double) (x_spacing * x_spacing + y_spacing * y_spacing));
    } else {
      return std::max(std::max(x_spacing, y_spacing), 0);
    }
  }

  // 获得线段的长度
  static int32_t getManhattanDistance(Segment<PlanarCoord> segment) { return getManhattanDistance(segment.get_first(), segment.get_second()); }

  // 获得坐标集合的H树线长
  static int32_t getHTreeLength(std::vector<PlanarCoord>& coord_list)
  {
    std::sort(coord_list.begin(), coord_list.end(), CmpPlanarCoordByXASC());
    PlanarCoord tree_axis_coord = getBalanceCoord(coord_list);
    // 计算H树长度
    int32_t h_tree_length = 0;
    int32_t pre_x = 0;
    int32_t balance_x = tree_axis_coord.get_x();
    int32_t balance_y = tree_axis_coord.get_y();
    int32_t min_x = balance_x;
    int32_t max_x = balance_x;
    int32_t min_y = balance_y;
    int32_t max_y = balance_y;
    for (size_t j = 0; j < coord_list.size(); j++) {
      const PlanarCoord& coord = coord_list[j];
      int32_t x = coord.get_x();
      int32_t y = coord.get_y();
      if (pre_x != x) {
        h_tree_length += std::abs(max_y - min_y);
        pre_x = x;
        min_y = balance_y;
        max_y = balance_y;
      }
      min_x = x < min_x ? x : min_x;
      max_x = max_x < x ? x : max_x;
      min_y = y < min_y ? y : min_y;
      max_y = max_y < y ? y : max_y;
    }
    h_tree_length += std::abs(max_y - min_y);
    h_tree_length += std::abs(max_x - min_x);
    return h_tree_length;
  }

  // 获得坐标集合的V树线长
  static int32_t getVTreeLength(std::vector<PlanarCoord>& coord_list)
  {
    std::sort(coord_list.begin(), coord_list.end(), CmpPlanarCoordByYASC());
    PlanarCoord tree_axis_coord = getBalanceCoord(coord_list);
    // 计算H树长度
    int32_t v_tree_length = 0;
    int32_t pre_y = 0;
    int32_t balance_x = tree_axis_coord.get_x();
    int32_t balance_y = tree_axis_coord.get_y();
    int32_t min_x = balance_x;
    int32_t max_x = balance_x;
    int32_t min_y = balance_y;
    int32_t max_y = balance_y;
    for (size_t j = 0; j < coord_list.size(); j++) {
      const PlanarCoord& coord = coord_list[j];
      int32_t x = coord.get_x();
      int32_t y = coord.get_y();
      if (pre_y != y) {
        v_tree_length += std::abs(max_x - min_x);
        pre_y = y;
        min_x = balance_x;
        max_x = balance_x;
      }
      min_x = x < min_x ? x : min_x;
      max_x = max_x < x ? x : max_x;
      min_y = y < min_y ? y : min_y;
      max_y = max_y < y ? y : max_y;
    }
    v_tree_length += std::abs(max_y - min_y);
    v_tree_length += std::abs(max_x - min_x);
    return v_tree_length;
  }

  // 获得两个线段的最短距离点对
  static Segment<PlanarCoord> getShortenCoordPair(Segment<PlanarCoord>& seg1, Segment<PlanarCoord>& seg2)
  {
    if (isIntersection(seg1, seg2)) {
      PlanarCoord coord = getIntersection(seg1, seg2);
      return Segment<PlanarCoord>(coord, coord);
    }

    Segment<PlanarCoord> candidate_seg1 = getShortenCoordPair(seg1, seg2.get_first());
    Segment<PlanarCoord> candidate_seg2 = getShortenCoordPair(seg1, seg2.get_second());

    if (getManhattanDistance(candidate_seg1) < getManhattanDistance(candidate_seg2)) {
      return candidate_seg1;
    } else {
      return candidate_seg2;
    }
  }

  // 获得线段和点的最短距离点对
  static Segment<PlanarCoord> getShortenCoordPair(Segment<PlanarCoord>& seg, PlanarCoord& coord)
  {
    int32_t coord_x = coord.get_x();
    int32_t coord_y = coord.get_y();
    int32_t first_coord_x = seg.get_first().get_x();
    int32_t first_coord_y = seg.get_first().get_y();
    int32_t second_coord_x = seg.get_second().get_x();
    int32_t second_coord_y = seg.get_second().get_y();

    if (first_coord_y == second_coord_y && first_coord_x <= coord_x && coord_x <= second_coord_x) {
      return Segment<PlanarCoord>(PlanarCoord(coord_x, first_coord_y), coord);
    } else if (first_coord_x == second_coord_x && first_coord_y <= coord_y && coord_y <= second_coord_y) {
      return Segment<PlanarCoord>(PlanarCoord(first_coord_x, coord_y), coord);
    }

    if (getManhattanDistance(coord, seg.get_first()) < getManhattanDistance(coord, seg.get_second())) {
      return Segment<PlanarCoord>(coord, seg.get_first());
    } else {
      return Segment<PlanarCoord>(coord, seg.get_second());
    }
  }

#endif

#if 1  // 位置关系计算

  // 判断两个线段是否平行
  static bool isParallel(Segment<PlanarCoord>& seg1, Segment<PlanarCoord>& seg2)
  {
    return getDirection(seg1.get_first(), seg1.get_second()) == getDirection(seg2.get_first(), seg2.get_second());
  }

  // 判断两个线段是否相交
  static bool isIntersection(Segment<PlanarCoord>& lhs, Segment<PlanarCoord>& rhs)
  {
    PlanarCoord intersection_point = getIntersection(lhs, rhs);
    if (intersection_point.get_x() == -1 && intersection_point.get_y() == -1) {
      return false;
    }
    return false;
  }

  // 判断矩形是否与线段相交
  static bool isIntersection(PlanarRect& rect, Segment<PlanarCoord>& seg)
  {
    std::vector<Segment<PlanarCoord>> edge_list = rect.getEdgeList();

    for (size_t i = 0; i < edge_list.size(); i++) {
      PlanarCoord intersection_point = getIntersection(seg, edge_list[i]);
      if (intersection_point.get_x() == -1 && intersection_point.get_y() == -1) {
        continue;
      }
      return true;
    }

    return false;
  }

  /**
   *  ！在检测DRC中
   *  如果a与b中有膨胀矩形,那么则用isOpenOverlap
   *  如果a与b中都是真实矩形,那么用isClosedOverlap
   *
   *  isOpenOverlap:不考虑边的overlap
   */
  static bool isOpenOverlap(const PlanarRect& a, const PlanarRect& b) { return isOverlap(a, b, false); }

  /**
   *  ！在检测DRC中
   *  如果a与b中有膨胀矩形,那么则用isOpenOverlap
   *  如果a与b中都是真实矩形,那么用isClosedOverlap
   *
   *  isClosedOverlap:考虑边的overlap
   */
  static bool isClosedOverlap(const PlanarRect& a, const PlanarRect& b) { return isOverlap(a, b, true); }

  // 判断两个矩形是否重叠
  static bool isOverlap(const PlanarRect& a, const PlanarRect& b, bool consider_edge = true)
  {
    int32_t x_spacing = std::max(b.get_ll_x() - a.get_ur_x(), a.get_ll_x() - b.get_ur_x());
    int32_t y_spacing = std::max(b.get_ll_y() - a.get_ur_y(), a.get_ll_y() - b.get_ur_y());

    if (x_spacing == 0 && y_spacing <= 0) {
      return consider_edge;
    } else if (x_spacing <= 0 && y_spacing == 0) {
      return consider_edge;
    } else {
      return (x_spacing < 0 && y_spacing < 0);
    }
  }

  static bool isOverlap(PlanarRect& a, Segment<PlanarCoord>& b, bool consider_edge = true)
  {
    int32_t first_x = b.get_first().get_x();
    int32_t second_x = b.get_second().get_x();
    swapByASC(first_x, second_x);
    int32_t first_y = b.get_first().get_y();
    int32_t second_y = b.get_second().get_y();
    swapByASC(first_y, second_y);

    int32_t x_spacing = std::max(first_x - a.get_ur_x(), a.get_ll_x() - second_x);
    int32_t y_spacing = std::max(first_y - a.get_ur_y(), a.get_ll_y() - second_y);

    if (x_spacing == 0 && y_spacing <= 0) {
      return consider_edge;
    } else if (x_spacing <= 0 && y_spacing == 0) {
      return consider_edge;
    } else {
      return (x_spacing < 0 && y_spacing < 0);
    }
  }

  // 判断coord是否在segment内
  static bool isInside(const Segment<LayerCoord>& segment, const LayerCoord& coord)
  {
    int32_t first_x = segment.get_first().get_x();
    int32_t first_y = segment.get_first().get_y();
    int32_t first_layer_idx = segment.get_first().get_layer_idx();
    int32_t second_x = segment.get_second().get_x();
    int32_t second_y = segment.get_second().get_y();
    int32_t second_layer_idx = segment.get_second().get_layer_idx();

    swapByASC(first_x, second_x);
    swapByASC(first_y, second_y);
    swapByASC(first_layer_idx, second_layer_idx);

    return (first_x <= coord.get_x() && coord.get_x() <= second_x && first_y <= coord.get_y() && coord.get_y() <= second_y
            && first_layer_idx <= coord.get_layer_idx() && coord.get_layer_idx() <= second_layer_idx);
  }

  // 判断coord是否在rect内,可以选择是否包含边界
  static bool isInside(const PlanarRect& rect, const PlanarCoord& coord, bool boundary = true)
  {
    int32_t coord_x = coord.get_x(), coord_y = coord.get_y();
    int32_t rect_ll_x = rect.get_ll_x(), rect_ll_y = rect.get_ll_y();
    int32_t rect_ur_x = rect.get_ur_x(), rect_ur_y = rect.get_ur_y();
    if (boundary) {
      return (rect_ll_x <= coord_x && coord_x <= rect_ur_x) && (rect_ll_y <= coord_y && coord_y <= rect_ur_y);
    }
    return (rect_ll_x < coord_x && coord_x < rect_ur_x) && (rect_ll_y < coord_y && coord_y < rect_ur_y);
  }

  // 线段在矩形内
  static bool isInside(const PlanarRect& master, const Segment<PlanarCoord>& seg)
  {
    return isInside(master, seg.get_first()) && isInside(master, seg.get_second());
  }

  /**
   * 矩形在矩形内
   *          ________________
   *         |   Master       |
   *         |  ——————————    |
   *         |  |  rect   |   |
   *         |  ——————————    |
   *         |________________|
   */
  static bool isInside(const PlanarRect& master, const PlanarRect& rect) { return (isInside(master, rect.get_ll()) && isInside(master, rect.get_ur())); }

  static void addOffset(PlanarCoord& coord, PlanarCoord& offset_coord) { addOffset(coord, offset_coord.get_x(), offset_coord.get_y()); }

  static void addOffset(PlanarCoord& coord, int32_t x_offset, int32_t y_offset)
  {
    coord.set_x(coord.get_x() + x_offset);
    coord.set_y(coord.get_y() + y_offset);
  }

  static void minusOffset(PlanarCoord& coord, PlanarCoord& offset_coord) { minusOffset(coord, offset_coord.get_x(), offset_coord.get_y()); }

  static void minusOffset(PlanarCoord& coord, int32_t x_offset, int32_t y_offset)
  {
    coord.set_x((coord.get_x() - x_offset) < 0 ? 0 : (coord.get_x() - x_offset));
    coord.set_y((coord.get_y() - y_offset) < 0 ? 0 : (coord.get_y() - y_offset));
  }

  // 三个坐标是否共线
  static bool isCollinear(PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    return getDirection(first_coord, second_coord) == getDirection(second_coord, third_coord);
  }

  // 坐标集合是否共线
  static bool isCollinear(std::vector<PlanarCoord>& coord_list)
  {
    if (coord_list.empty()) {
      RTLOG.error(Loc::current(), "The coord list is empty!");
    } else if (coord_list.size() <= 2) {
      return true;
    } else {
      Direction pre_direction = getDirection(coord_list[0], coord_list[1]);
      for (size_t i = 2; i < coord_list.size(); i++) {
        Direction curr_direction = getDirection(coord_list[i - 1], coord_list[i]);
        if (pre_direction != curr_direction) {
          return false;
        }
        pre_direction = curr_direction;
      }
      return true;
    }
  }

  // 是否是凸角位置
  static bool isConvexCorner(PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    if (isCollinear(first_coord, second_coord, third_coord)) {
      return false;
    }

    return crossProduct(first_coord, second_coord, third_coord) < 0;
  }

  // 是否是凹角位置
  static bool isConcaveCorner(PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    if (isCollinear(first_coord, second_coord, third_coord)) {
      return false;
    }

    return crossProduct(first_coord, second_coord, third_coord) > 0;
  }

#endif

#if 1  // 形状有关计算

  static PlanarRect getRect(PlanarCoord start_coord, PlanarCoord end_coord)
  {
    PlanarRect rect;
    rect.set_ll_x(std::min(start_coord.get_x(), end_coord.get_x()));
    rect.set_ll_y(std::min(start_coord.get_y(), end_coord.get_y()));
    rect.set_ur_x(std::max(start_coord.get_x(), end_coord.get_x()));
    rect.set_ur_y(std::max(start_coord.get_y(), end_coord.get_y()));
    return rect;
  }

  // 三个点的叉乘
  static int32_t crossProduct(PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    return (second_coord.get_x() - first_coord.get_x()) * (third_coord.get_y() - first_coord.get_y())
           - (second_coord.get_y() - first_coord.get_y()) * (third_coord.get_x() - first_coord.get_x());
  }

  // 获得两个线段的交点
  static PlanarCoord getIntersection(PlanarCoord first_coord1, PlanarCoord second_coord1, PlanarCoord first_coord2, PlanarCoord second_coord2)
  {
    Segment<PlanarCoord> seg1(first_coord1, second_coord1);
    Segment<PlanarCoord> seg2(first_coord2, second_coord2);

    return getIntersection(seg1, seg2);
  }

  // 获得两个线段的交点
  static PlanarCoord getIntersection(Segment<PlanarCoord>& seg1, Segment<PlanarCoord>& seg2)
  {
    double ax = seg1.get_first().get_x();
    double ay = seg1.get_first().get_y();
    double bx = seg1.get_second().get_x();
    double by = seg1.get_second().get_y();
    double cx = seg2.get_first().get_x();
    double cy = seg2.get_first().get_y();
    double dx = seg2.get_second().get_x();
    double dy = seg2.get_second().get_y();

    double acx = cx - ax;
    double acy = cy - ay;
    double abx = bx - ax;
    double aby = by - ay;
    double adx = dx - ax;
    double ady = dy - ay;

    double cax = ax - cx;
    double cay = ay - cy;
    double cbx = bx - cx;
    double cby = by - cy;
    double cdx = dx - cx;
    double cdy = dy - cy;

    // 叉积运算
    auto cross = [](double ux, double uy, double vx, double vy) { return ux * vy - vx * uy; };
    // 判断线段 (ux, uy) -- (vx, vy) 是否包含 (mx, my)
    auto both_side = [&](double mx, double my, double ux, double uy, double vx, double vy) { return (ux - mx) * (vx - mx) <= 0 && (uy - my) * (vy - my) <= 0; };
    // 共线处理
    if (cross(cax, cay, cbx, cby) == 0) {
      bool c_in_ab = both_side(cx, cy, ax, ay, bx, by);
      bool d_in_ab = both_side(dx, dy, ax, ay, bx, by);
      if (c_in_ab) {
        return PlanarCoord(static_cast<int32_t>(cx), static_cast<int32_t>(cy));
      }
      if (d_in_ab) {
        return PlanarCoord(static_cast<int32_t>(dx), static_cast<int32_t>(dy));
      }
      return PlanarCoord(-1, -1);
    }
    // T 形处理
    if (cross(adx, ady, abx, aby) == 0) {
      bool a_in_cd = both_side(ax, ay, cx, cy, dx, dy);
      bool b_in_cd = both_side(bx, by, cx, cy, dx, dy);
      bool c_in_ab = both_side(cx, cy, ax, ay, bx, by);
      bool d_in_ab = both_side(dx, dy, ax, ay, bx, by);
      if (a_in_cd) {
        return PlanarCoord(static_cast<int32_t>(ax), static_cast<int32_t>(ay));
      }
      if (b_in_cd) {
        return PlanarCoord(static_cast<int32_t>(bx), static_cast<int32_t>(by));
      }
      if (c_in_ab) {
        return PlanarCoord(static_cast<int32_t>(cx), static_cast<int32_t>(cy));
      }
      if (d_in_ab) {
        return PlanarCoord(static_cast<int32_t>(dx), static_cast<int32_t>(dy));
      }
      return PlanarCoord(-1, -1);
    }
    // 判断两条线段是否有公共点
    auto intersect = [&]() { return cross(acx, acy, abx, aby) * cross(adx, ady, abx, aby) <= 0 && cross(cax, cay, cdx, cdy) * cross(cbx, cby, cdx, cdy) <= 0; };
    if (!intersect()) {
      return PlanarCoord(-1, -1);
    }
    // 计算三角形 PQM 的面积
    auto get_coord_area = [&](double px, double py, double qx, double qy, double mx, double my) -> double {
      double mpx = px - mx;
      double mpy = py - my;
      double mqx = qx - mx;
      double mqy = qy - my;
      return fabs(static_cast<double>(0.5) * cross(mpx, mpy, mqx, mqy));
    };
    // 定比分点
    double ck = get_coord_area(ax, ay, bx, by, cx, cy);
    double dk = get_coord_area(ax, ay, bx, by, dx, dy);
    double k = ck / dk;
    double rx = (cx + k * dx) / (1 + k);
    double ry = (cy + k * dy) / (1 + k);
    rx = std::round(rx);
    ry = std::round(ry);

    return PlanarCoord(static_cast<int32_t>(rx), static_cast<int32_t>(ry));
  }

  // 获得矩形与线段的交点
  static std::vector<PlanarCoord> getIntersection(PlanarRect& rect, Segment<PlanarCoord>& seg)
  {
    std::vector<Segment<PlanarCoord>> edge_list = rect.getEdgeList();

    std::vector<PlanarCoord> intersection_point_list;
    for (size_t i = 0; i < edge_list.size(); i++) {
      PlanarCoord intersection_point = getIntersection(seg, edge_list[i]);
      if (intersection_point.get_x() == -1 && intersection_point.get_y() == -1) {
        continue;
      }
      intersection_point_list.push_back(intersection_point);
    }
    std::sort(intersection_point_list.begin(), intersection_point_list.end(), CmpPlanarCoordByXASC());
    intersection_point_list.erase(std::unique(intersection_point_list.begin(), intersection_point_list.end()), intersection_point_list.end());
    return intersection_point_list;
  }

  // 获得矩形和线段的overlap
  static Segment<PlanarCoord> getOverlap(PlanarRect a, Segment<PlanarCoord> b)
  {
    int32_t first_x = b.get_first().get_x();
    int32_t second_x = b.get_second().get_x();
    swapByASC(first_x, second_x);
    int32_t first_y = b.get_first().get_y();
    int32_t second_y = b.get_second().get_y();
    swapByASC(first_y, second_y);

    int32_t overlap_ll_x = std::max(first_x, a.get_ll_x());
    int32_t overlap_ur_x = std::min(second_x, a.get_ur_x());
    int32_t overlap_ll_y = std::max(first_y, a.get_ll_y());
    int32_t overlap_ur_y = std::min(second_y, a.get_ur_y());

    if (overlap_ll_x > overlap_ur_x || overlap_ll_y > overlap_ur_y) {
      return Segment<PlanarCoord>(PlanarCoord(), PlanarCoord());
    } else {
      return Segment<PlanarCoord>(PlanarCoord(overlap_ll_x, overlap_ll_y), PlanarCoord(overlap_ur_x, overlap_ur_y));
    }
  }

  // 获得两个矩形的overlap矩形
  static PlanarRect getOverlap(PlanarRect a, PlanarRect b)
  {
    int32_t overlap_ll_x = std::max(a.get_ll_x(), b.get_ll_x());
    int32_t overlap_ur_x = std::min(a.get_ur_x(), b.get_ur_x());
    int32_t overlap_ll_y = std::max(a.get_ll_y(), b.get_ll_y());
    int32_t overlap_ur_y = std::min(a.get_ur_y(), b.get_ur_y());

    if (overlap_ll_x > overlap_ur_x || overlap_ll_y > overlap_ur_y) {
      return PlanarRect(0, 0, 0, 0);
    } else {
      return PlanarRect(overlap_ll_x, overlap_ll_y, overlap_ur_x, overlap_ur_y);
    }
  }

  // 获得两个矩形的overlap矩形
  static std::vector<PlanarRect> getOverlap(std::vector<PlanarRect> a_rect_list, std::vector<PlanarRect> b_rect_list)
  {
    std::vector<PlanarRect> overlap_rect_list;
    for (const PlanarRect& a_rect : a_rect_list) {
      for (const PlanarRect& b_rect : b_rect_list) {
        if (isClosedOverlap(a_rect, b_rect)) {
          overlap_rect_list.push_back(getOverlap(a_rect, b_rect));
        }
      }
    }
    // rect去重
    std::sort(overlap_rect_list.begin(), overlap_rect_list.end(), CmpPlanarRectByXASC());
    overlap_rect_list.erase(std::unique(overlap_rect_list.begin(), overlap_rect_list.end()), overlap_rect_list.end());
    return overlap_rect_list;
  }

  static Segment<LayerCoord> getOverlap(const LayerRect& rect, const Segment<LayerCoord>& segment)
  {
    int32_t rect_ll_x = rect.get_ll_x();
    int32_t rect_ll_y = rect.get_ll_y();
    int32_t rect_ur_x = rect.get_ur_x();
    int32_t rect_ur_y = rect.get_ur_y();
    int32_t rect_layer_idx = rect.get_layer_idx();

    int32_t first_x = segment.get_first().get_x();
    int32_t first_y = segment.get_first().get_y();
    int32_t first_layer_idx = segment.get_first().get_layer_idx();
    int32_t second_x = segment.get_second().get_x();
    int32_t second_y = segment.get_second().get_y();
    int32_t second_layer_idx = segment.get_second().get_layer_idx();
    swapByASC(first_x, second_x);
    swapByASC(first_y, second_y);
    swapByASC(first_layer_idx, second_layer_idx);

    if (first_y == second_y && first_layer_idx == second_layer_idx) {
      if (first_y >= rect_ll_y && first_y <= rect_ur_y && rect_layer_idx == first_layer_idx) {
        int32_t overlap_min_x = std::max(rect_ll_x, std::min(first_x, second_x));
        int32_t overlap_max_x = std::min(rect_ur_x, std::max(first_x, second_x));

        if (overlap_min_x <= overlap_max_x) {
          return Segment<LayerCoord>(LayerCoord(overlap_min_x, first_y, rect_layer_idx), LayerCoord(overlap_max_x, first_y, rect_layer_idx));
        }
      }
    } else if (first_x == second_x && first_layer_idx == second_layer_idx) {
      if (first_x >= rect_ll_x && first_x <= rect_ur_x && rect_layer_idx == first_layer_idx) {
        int32_t overlap_min_y = std::max(rect_ll_y, std::min(first_y, second_y));
        int32_t overlap_max_y = std::min(rect_ur_y, std::max(first_y, second_y));

        if (overlap_min_y <= overlap_max_y) {
          return Segment<LayerCoord>(LayerCoord(first_x, overlap_min_y, rect_layer_idx), LayerCoord(first_x, overlap_max_y, rect_layer_idx));
        }
      }
    } else if (first_x == second_x && first_y == second_y) {
      if (first_x >= rect_ll_x && first_x <= rect_ur_x && first_y >= rect_ll_y && first_y <= rect_ur_y
          && rect_layer_idx >= std::min(first_layer_idx, second_layer_idx) && rect_layer_idx <= std::max(first_layer_idx, second_layer_idx)) {
        return Segment<LayerCoord>(LayerCoord(first_x, first_y, rect_layer_idx), LayerCoord(first_x, first_y, rect_layer_idx));
      }
    }
    return Segment<LayerCoord>(LayerCoord(-1, -1, -1), LayerCoord(-1, -1, -1));
  }

  static bool isOverlap(const LayerRect& rect, const Segment<LayerCoord>& segment)
  {
    int32_t rect_ll_x = rect.get_ll_x();
    int32_t rect_ll_y = rect.get_ll_y();
    int32_t rect_ur_x = rect.get_ur_x();
    int32_t rect_ur_y = rect.get_ur_y();
    int32_t rect_layer_idx = rect.get_layer_idx();

    int32_t first_x = segment.get_first().get_x();
    int32_t first_y = segment.get_first().get_y();
    int32_t first_layer_idx = segment.get_first().get_layer_idx();
    int32_t second_x = segment.get_second().get_x();
    int32_t second_y = segment.get_second().get_y();
    int32_t second_layer_idx = segment.get_second().get_layer_idx();
    swapByASC(first_x, second_x);
    swapByASC(first_y, second_y);
    swapByASC(first_layer_idx, second_layer_idx);

    if (first_y == second_y && first_layer_idx == second_layer_idx) {
      if (first_y >= rect_ll_y && first_y <= rect_ur_y && rect_layer_idx == first_layer_idx) {
        int32_t overlap_min_x = std::max(rect_ll_x, std::min(first_x, second_x));
        int32_t overlap_max_x = std::min(rect_ur_x, std::max(first_x, second_x));

        if (overlap_min_x <= overlap_max_x) {
          return true;
        }
      }
    } else if (first_x == second_x && first_layer_idx == second_layer_idx) {
      if (first_x >= rect_ll_x && first_x <= rect_ur_x && rect_layer_idx == first_layer_idx) {
        int32_t overlap_min_y = std::max(rect_ll_y, std::min(first_y, second_y));
        int32_t overlap_max_y = std::min(rect_ur_y, std::max(first_y, second_y));

        if (overlap_min_y <= overlap_max_y) {
          return true;
        }
      }
    } else if (first_x == second_x && first_y == second_y) {
      if (first_x >= rect_ll_x && first_x <= rect_ur_x && first_y >= rect_ll_y && first_y <= rect_ur_y
          && rect_layer_idx >= std::min(first_layer_idx, second_layer_idx) && rect_layer_idx <= std::max(first_layer_idx, second_layer_idx)) {
        return true;
      }
    }
    return false;
  }

  /**
   *  分开矩形,将master矩形用rect进行分开,并不是求差集
   *       ┌────────────────────────────────────┐  split  ┌────────────────────────────────────┐
   *       │ master                             │ ──────> │ c                                  │
   *       │           ┌─────────────────┐      │         └────────────────────────────────────┘
   *       └───────────┼─────────────────┼──────┘
   *                   │ rect            │
   *        split│     └─────────────────┘  │split
   *             ▼                          ▼
   *       ┌───────────┐                 ┌──────┐
   *       │           │                 │      │
   *       │     a     │                 │  b   │
   *       └───────────┘                 └──────┘
   *  如上图所示,输入master和rect
   *  若split方向为horizontal,将得到a和b,可以理解为在横向上分开
   *  若split方向为vertical,将得到c
   */
  static std::vector<PlanarRect> getSplitRectList(const PlanarRect& master, const PlanarRect& rect, Direction split_direction)
  {
    std::vector<PlanarRect> split_rect_list;

    if (split_direction == Direction::kHorizontal) {
      if (master.get_ll_x() < rect.get_ll_x()) {
        PlanarRect split_rect = master;
        split_rect.set_ur_x(rect.get_ll_x());
        split_rect_list.push_back(split_rect);
      }
      if (rect.get_ur_x() < master.get_ur_x()) {
        PlanarRect split_rect = master;
        split_rect.set_ll_x(rect.get_ur_x());
        split_rect_list.push_back(split_rect);
      }
    } else {
      if (master.get_ll_y() < rect.get_ll_y()) {
        PlanarRect split_rect = master;
        split_rect.set_ur_y(rect.get_ll_y());
        split_rect_list.push_back(split_rect);
      }
      if (rect.get_ur_y() < master.get_ur_y()) {
        PlanarRect split_rect = master;
        split_rect.set_ll_y(rect.get_ur_y());
        split_rect_list.push_back(split_rect);
      }
    }
    return split_rect_list;
  }

  static PlanarRect getEnlargedRect(PlanarCoord center_coord, int32_t enlarge_size)
  {
    return getEnlargedRect(center_coord, enlarge_size, enlarge_size, enlarge_size, enlarge_size);
  }

  static PlanarRect getEnlargedRect(PlanarCoord center_coord, int32_t ll_x_minus_offset, int32_t ll_y_minus_offset, int32_t ur_x_add_offset,
                                    int32_t ur_y_add_offset)
  {
    PlanarRect rect(center_coord, center_coord);
    minusOffset(rect.get_ll(), ll_x_minus_offset, ll_y_minus_offset);
    addOffset(rect.get_ur(), ur_x_add_offset, ur_y_add_offset);
    return rect;
  }

  /**
   * 以enlarge_size扩大线段
   *                               ┌────────────────────┐
   *                               │                    │
   *    ────────────────     ->    │   ──────────────   │
   *                               │                    │
   *                               └────────────────────┘
   *        segment                      rectangle
   */
  static PlanarRect getEnlargedRect(PlanarCoord start_coord, PlanarCoord end_coord, int32_t enlarge_size)
  {
    if (!CmpPlanarCoordByXASC()(start_coord, end_coord)) {
      std::swap(start_coord, end_coord);
    }
    PlanarRect rect(start_coord, end_coord);

    if (isRightAngled(start_coord, end_coord)) {
      rect = getEnlargedRect(rect, enlarge_size);
    } else {
      RTLOG.error(Loc::current(), "The segment is oblique!");
    }
    return rect;
  }

  // 以enlarge_size扩大线段
  static PlanarRect getEnlargedRect(Segment<PlanarCoord> segment, int32_t enlarge_size)
  {
    return getEnlargedRect(segment.get_first(), segment.get_second(), enlarge_size);
  }

  static bool hasRegularRect(const PlanarRect& rect, const PlanarRect& border)
  {
    // 计算新的左下角坐标
    int32_t new_ll_x = std::max(rect.get_ll_x(), border.get_ll_x());
    int32_t new_ll_y = std::max(rect.get_ll_y(), border.get_ll_y());

    // 计算新的右上角坐标
    int32_t new_ur_x = std::min(rect.get_ur_x(), border.get_ur_x());
    int32_t new_ur_y = std::min(rect.get_ur_y(), border.get_ur_y());

    // 检查新的坐标是否有效
    return new_ll_x <= new_ur_x && new_ll_y <= new_ur_y;
  }

  static PlanarRect getRegularRect(PlanarRect rect, PlanarRect border)
  {
    PlanarRect regular_rect;
    regular_rect.set_ll(std::max(rect.get_ll_x(), border.get_ll_x()), std::max(rect.get_ll_y(), border.get_ll_y()));
    regular_rect.set_ur(std::min(rect.get_ur_x(), border.get_ur_x()), std::min(rect.get_ur_y(), border.get_ur_y()));
    return regular_rect;
  }

  static LayerRect getRegularRect(LayerRect rect, PlanarRect border)
  {
    LayerRect regular_rect;
    regular_rect.set_ll(std::max(rect.get_ll_x(), border.get_ll_x()), std::max(rect.get_ll_y(), border.get_ll_y()));
    regular_rect.set_ur(std::min(rect.get_ur_x(), border.get_ur_x()), std::min(rect.get_ur_y(), border.get_ur_y()));
    regular_rect.set_layer_idx(rect.get_layer_idx());
    return regular_rect;
  }

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, int32_t enlarge_size)
  {
    return getEnlargedRect(rect, enlarge_size, enlarge_size, enlarge_size, enlarge_size);
  }

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, int32_t ll_x_minus_offset, int32_t ll_y_minus_offset, int32_t ur_x_add_offset, int32_t ur_y_add_offset)
  {
    minusOffset(rect.get_ll(), ll_x_minus_offset, ll_y_minus_offset);
    addOffset(rect.get_ur(), ur_x_add_offset, ur_y_add_offset);
    return rect;
  }

  // 偏移矩形
  static PlanarRect getOffsetRect(PlanarRect rect, PlanarCoord offset_coord)
  {
    int32_t offset_x = offset_coord.get_x();
    int32_t offset_y = offset_coord.get_y();

    addOffset(rect.get_ll(), offset_x, offset_y);
    addOffset(rect.get_ur(), offset_x, offset_y);
    return rect;
  }

  /**
   * 将矩形的原点坐标进行转换
   *       ______________ (300,300)             ______________ (200,200)
   *      |              |                     |              |
   *      |              |                     |              |
   *      |              |                     |              |
   *      |______________|                     |______________|
   *      (200,200)                            (100,100)
   *   ·(100,100)                          ·(0,0)
   */
  static PlanarRect getRelativeRectByOriginCoord(PlanarRect rect, PlanarCoord origin_coord)
  {
    int32_t offset_x = origin_coord.get_x();
    int32_t offset_y = origin_coord.get_y();

    minusOffset(rect.get_ll(), offset_x, offset_y);
    minusOffset(rect.get_ur(), offset_x, offset_y);
    return rect;
  }

  static bool hasShrinkedRect(PlanarRect rect, int32_t shrinked_size)
  {
    return hasShrinkedRect(rect, shrinked_size, shrinked_size, shrinked_size, shrinked_size);
  }

  static bool hasShrinkedRect(PlanarRect rect, int32_t x_shrinked_size, int32_t y_shrinked_size)
  {
    return hasShrinkedRect(rect, x_shrinked_size, y_shrinked_size, x_shrinked_size, y_shrinked_size);
  }

  static bool hasShrinkedRect(PlanarRect rect, int32_t ll_x_add_offset, int32_t ll_y_add_offset, int32_t ur_x_minus_offset, int32_t ur_y_minus_offset)
  {
    addOffset(rect.get_ll(), ll_x_add_offset, ll_y_add_offset);
    minusOffset(rect.get_ur(), ur_x_minus_offset, ur_y_minus_offset);
    return rect.get_ll_x() <= rect.get_ur_x() && rect.get_ll_y() <= rect.get_ur_y();
  }

  static PlanarRect getShrinkedRect(PlanarRect rect, int32_t shrinked_size)
  {
    return getShrinkedRect(rect, shrinked_size, shrinked_size, shrinked_size, shrinked_size);
  }

  static PlanarRect getShrinkedRect(PlanarRect rect, int32_t x_shrinked_size, int32_t y_shrinked_size)
  {
    return getShrinkedRect(rect, x_shrinked_size, y_shrinked_size, x_shrinked_size, y_shrinked_size);
  }

  static PlanarRect getShrinkedRect(PlanarRect rect, int32_t ll_x_add_offset, int32_t ll_y_add_offset, int32_t ur_x_minus_offset, int32_t ur_y_minus_offset)
  {
    addOffset(rect.get_ll(), ll_x_add_offset, ll_y_add_offset);
    minusOffset(rect.get_ur(), ur_x_minus_offset, ur_y_minus_offset);
    return rect;
  }

  // 获得坐标集合的外接矩形
  static PlanarRect getBoundingBox(const std::vector<LayerCoord>& coord_list)
  {
    std::vector<PlanarCoord> planar_coord_list;
    planar_coord_list.insert(planar_coord_list.end(), coord_list.begin(), coord_list.end());
    return getBoundingBox(planar_coord_list);
  }

  // 获得坐标集合的外接矩形
  static PlanarRect getBoundingBox(const std::vector<PlanarCoord>& coord_list)
  {
    PlanarRect bounding_box;
    if (coord_list.empty()) {
      RTLOG.warn(Loc::current(), "The coord list size is empty!");
    } else {
      int32_t ll_x = INT32_MAX;
      int32_t ll_y = INT32_MAX;
      int32_t ur_x = INT32_MIN;
      int32_t ur_y = INT32_MIN;
      for (size_t i = 0; i < coord_list.size(); i++) {
        const PlanarCoord& coord = coord_list[i];

        ll_x = std::min(ll_x, coord.get_x());
        ll_y = std::min(ll_y, coord.get_y());
        ur_x = std::max(ur_x, coord.get_x());
        ur_y = std::max(ur_y, coord.get_y());
      }
      bounding_box.set_ll(ll_x, ll_y);
      bounding_box.set_ur(ur_x, ur_y);
    }
    return bounding_box;
  }

  static PlanarRect getBoundingBox(const std::vector<PlanarRect>& rect_list)
  {
    int32_t ll_x = INT32_MAX;
    int32_t ll_y = INT32_MAX;
    int32_t ur_x = INT32_MIN;
    int32_t ur_y = INT32_MIN;

    for (size_t i = 0; i < rect_list.size(); i++) {
      ll_x = std::min(ll_x, rect_list[i].get_ll_x());
      ll_y = std::min(ll_y, rect_list[i].get_ll_y());
      ur_x = std::max(ur_x, rect_list[i].get_ur_x());
      ur_y = std::max(ur_y, rect_list[i].get_ur_y());
    }
    return PlanarRect(ll_x, ll_y, ur_x, ur_y);
  }

  // 获得坐标集合的重心
  static LayerCoord getBalanceCoord(const std::vector<LayerCoord>& coord_list)
  {
    if (coord_list.empty()) {
      return LayerCoord(-1, -1, -1);
    }
    std::vector<int32_t> x_list;
    std::vector<int32_t> y_list;
    std::vector<int32_t> layer_idx_list;
    x_list.reserve(coord_list.size());
    y_list.reserve(coord_list.size());
    layer_idx_list.reserve(coord_list.size());
    for (const LayerCoord& coord : coord_list) {
      x_list.push_back(coord.get_x());
      y_list.push_back(coord.get_y());
      layer_idx_list.push_back(coord.get_layer_idx());
    }

    return LayerCoord(getAverage(x_list), getAverage(y_list), getAverage(layer_idx_list));
  }

  // 获得坐标集合的重心
  static PlanarCoord getBalanceCoord(const std::vector<PlanarCoord>& coord_list)
  {
    if (coord_list.empty()) {
      return PlanarCoord(-1, -1);
    }
    std::vector<int32_t> x_value_list;
    std::vector<int32_t> y_value_list;
    x_value_list.reserve(coord_list.size());
    y_value_list.reserve(coord_list.size());
    for (const PlanarCoord& coord : coord_list) {
      x_value_list.push_back(coord.get_x());
      y_value_list.push_back(coord.get_y());
    }

    return PlanarCoord(getAverage(x_value_list), getAverage(y_value_list));
  }

#endif

#if 1  // 与多叉树有关的计算

  // 获得所有结点
  template <typename T>
  static std::vector<TNode<T>*> getNodeList(MTree<T>& tree)
  {
    return getNodeList(tree.get_root());
  }

  // 获得所有结点
  template <typename T>
  static std::vector<TNode<T>*> getNodeList(TNode<T>* root)
  {
    std::vector<TNode<T>*> node_list;
    std::vector<std::vector<TNode<T>*>> level_list = getLevelOrder(root);
    for (size_t i = 0; i < level_list.size(); i++) {
      for (size_t j = 0; j < level_list[i].size(); j++) {
        node_list.push_back(level_list[i][j]);
      }
    }
    return node_list;
  }

  // 以层序遍历获取树的所有结点,可以控制遍历最大深度
  template <typename T>
  static std::vector<std::vector<TNode<T>*>> getLevelOrder(MTree<T>& tree, int32_t max_level = -1)
  {
    return getLevelOrder(tree.get_root(), max_level);
  }

  // 以层序遍历获取树的所有结点,可以控制遍历最大深度
  template <typename T>
  static std::vector<std::vector<TNode<T>*>> getLevelOrder(TNode<T>* root, int32_t max_level = -1)
  {
    if (root == nullptr) {
      return {};
    }
    std::vector<std::vector<TNode<T>*>> level_list;
    std::queue<TNode<T>*> node_queue = initQueue(root);
    std::vector<TNode<T>*> level;
    int32_t level_node_num = 1;
    while (!node_queue.empty()) {
      TNode<T>* node = getFrontAndPop(node_queue);
      level.push_back(node);
      addListToQueue(node_queue, node->get_child_list());
      if (--level_node_num == 0) {
        level_list.push_back(std::move(level));
        level.clear();
        level_node_num = static_cast<int32_t>(node_queue.size());
      }
      if (max_level == -1) {
        continue;
      }
      if (static_cast<int32_t>(level_list.size()) >= max_level) {
        break;
      }
    }
    return level_list;
  }

  // 判断树的深度是否超过max_level
  template <typename T>
  static bool isDeeperThan(MTree<T>& tree, int32_t max_level)
  {
    return isDeeperThan(tree.get_root(), max_level);
  }

  // 判断树的深度是否超过max_level
  template <typename T>
  static bool isDeeperThan(TNode<T>* root, int32_t max_level)
  {
    if (root == nullptr) {
      return max_level < 0;
    }

    std::queue<TNode<T>*> node_queue = initQueue(root);
    int32_t level_num = 0;
    int32_t level_node_num = 1;
    while (!node_queue.empty()) {
      TNode<T>* node = getFrontAndPop(node_queue);
      addListToQueue(node_queue, node->get_child_list());
      if (--level_node_num == 0) {
        ++level_num;
        level_node_num = node_queue.size();
      }
      if (level_num > max_level) {
        return true;
      }
    }
    return false;
  }

  // 对树结点内的值进行转换,需要自定义转换函数
  template <typename T, typename U, typename... Args>
  static MTree<U> convertTree(MTree<T>& old_tree, const std::function<U(T&, Args&...)>& convert, Args&... args)
  {
    return MTree<U>(convertTree(old_tree.get_root(), convert, args...));
  }

  // 对树结点内的值进行转换,需要自定义转换函数
  template <typename T, typename U, typename... Args>
  static TNode<U>* convertTree(TNode<T>* old_root, const std::function<U(T&, Args&...)>& convert, Args&... args)
  {
    if (old_root == nullptr) {
      return nullptr;
    }

    TNode<U>* new_root = new TNode<U>(convert(old_root->value(), args...));
    std::queue<TNode<T>*> old_node_queue = initQueue(old_root);
    std::queue<TNode<U>*> new_node_queue = initQueue(new_root);
    while (!old_node_queue.empty()) {
      TNode<T>* old_node = getFrontAndPop(old_node_queue);
      TNode<U>* new_node = getFrontAndPop(new_node_queue);
      std::vector<TNode<T>*>& child_list = old_node->get_child_list();
      for (size_t i = 0; i < child_list.size(); i++) {
        new_node->addChild(new TNode<U>(convert(child_list[i]->value(), args...)));
      }
      addListToQueue(old_node_queue, old_node->get_child_list());
      addListToQueue(new_node_queue, new_node->get_child_list());
    }

    return new_root;
  }

  // 获得树的边集合
  template <typename T>
  static std::vector<Segment<TNode<T>*>> getSegListByTree(MTree<T>& tree)
  {
    return getSegListByTree(tree.get_root());
  }

  // 获得树的边集合
  template <typename T>
  static std::vector<Segment<TNode<T>*>> getSegListByTree(TNode<T>* root)
  {
    if (root == nullptr) {
      return {};
    }

    std::vector<Segment<TNode<T>*>> segment_list;
    std::vector<std::vector<TNode<T>*>> level_list = getLevelOrder(root);
    for (size_t i = 0; i < level_list.size(); i++) {
      for (size_t j = 0; j < level_list[i].size(); j++) {
        if (level_list[i][j]->isLeafNode()) {
          continue;
        }
        std::vector<TNode<T>*>& child_list = level_list[i][j]->get_child_list();
        for (size_t k = 0; k < child_list.size(); k++) {
          segment_list.emplace_back(level_list[i][j], child_list[k]);
        }
      }
    }
    return segment_list;
  }

  // 通过树根节点和边集构建一棵树,也会消除多个连通分量
  template <typename T>
  static MTree<T> getTreeBySegList(const T& root_value, const std::vector<Segment<T>>& segment_list)
  {
#define PLAN_A
#ifdef PLAN_A
    std::map<T, int32_t, CmpLayerCoordByXASC> coord_map;
    std::vector<T> unique_coord_list;
    std::vector<std::vector<int32_t>> adj_list;
    int32_t idx = 0;
    for (const auto& segment : segment_list) {
      const T& first = segment.get_first();
      const T& second = segment.get_second();
      int32_t first_idx = -1;
      int32_t second_idx = -1;
      if (!exist(coord_map, first)) {
        first_idx = idx++;
        coord_map[first] = first_idx;
        unique_coord_list.push_back(first);
        adj_list.push_back({});
      }
      if (!exist(coord_map, second)) {
        second_idx = idx++;
        coord_map[second] = second_idx;
        unique_coord_list.push_back(second);
        adj_list.push_back({});
      }
      if (first_idx == -1) {
        first_idx = coord_map[first];
      }
      if (second_idx == -1) {
        second_idx = coord_map[second];
      }
      adj_list[first_idx].push_back(second_idx);
      adj_list[second_idx].push_back(first_idx);
    }

    if (!exist(coord_map, root_value)) {
      return MTree<T>(new TNode<T>(root_value));
    }

    std::vector<TNode<T>*> node_list(unique_coord_list.size(), nullptr);
    std::vector<bool> visited(unique_coord_list.size(), false);

    int32_t root_idx = coord_map[root_value];
    std::queue<int32_t> node_queue = initQueue(root_idx);
    visited[root_idx] = true;
    TNode<T>* root = new TNode(root_value);
    node_list[root_idx] = root;
    while (!node_queue.empty()) {
      int32_t curr_idx = getFrontAndPop(node_queue);
      TNode<T>* curr_node = node_list[curr_idx];
      std::vector<int32_t> neighbor_idx_list;
      for (const auto& neighbor_idx : adj_list[curr_idx]) {
        if (!visited[neighbor_idx]) {
          neighbor_idx_list.push_back(neighbor_idx);
          visited[neighbor_idx] = true;
          TNode<T>* child_node = new TNode(unique_coord_list[neighbor_idx]);
          curr_node->addChild(child_node);
          node_list[neighbor_idx] = child_node;
        }
      }
      addListToQueue(node_queue, neighbor_idx_list);
    }
    return MTree<T>(root);
#else
    std::vector<std::pair<bool, Segment<T>>> visited_value_pair_list;
    std::set<T, CmpLayerCoordByXASC> visited_node;  // 需要加一个cmp的传入
    visited_value_pair_list.reserve(segment_list.size());
    for (size_t i = 0; i < segment_list.size(); i++) {
      visited_value_pair_list.emplace_back(false, segment_list[i]);
    }
    TNode<T>* root = new TNode(root_value);
    visited_node.insert(root_value);
    std::queue<TNode<T>*> node_queue = initQueue(root);
    while (!node_queue.empty()) {
      TNode<T>* node = getFrontAndPop(node_queue);
      T& value = node->value();

      std::vector<TNode<T>*> next_node_list;
      for (size_t i = 0; i < visited_value_pair_list.size(); i++) {
        std::pair<bool, Segment<T>>& visited_value_pair = visited_value_pair_list[i];
        if (visited_value_pair.first == true) {
          continue;
        }
        T& value1 = visited_value_pair.second.get_first();
        T& value2 = visited_value_pair.second.get_second();
        if (value == value1 || value == value2) {
          T target_val = (value == value1 ? value2 : value1);
          if (exist(visited_node, target_val)) {
            continue;
          }
          visited_node.insert(target_val);
          TNode<T>* child_node = (value == value1 ? new TNode(value2) : new TNode(value1));
          next_node_list.push_back(child_node);
          node->addChild(child_node);
          visited_value_pair.first = true;
        }
      }
      addListToQueue(node_queue, next_node_list);
    }
    return MTree<T>(root);
#endif
  }

#endif

#if 1  // 与格子图有关的计算
  template <typename T>
  static GridMap<T> sliceMap(GridMap<T>& source_map, EXTPlanarRect& source_rect, EXTPlanarRect& target_rect, T fill_value)
  {
    if (source_rect.getXSize() != source_map.get_x_size() || source_rect.getYSize() != source_map.get_y_size()) {
      RTLOG.error(Loc::current(), "The rect size is inconsistent with map size!");
    }

    GridMap<T> target_map(target_rect.getXSize(), target_rect.getYSize(), fill_value);

    int32_t offset_x = source_rect.get_grid_ll_x() - target_rect.get_grid_ll_x();
    int32_t offset_y = source_rect.get_grid_ll_y() - target_rect.get_grid_ll_y();

    for (int32_t x = 0; x < source_map.get_x_size(); x++) {
      for (int32_t y = 0; y < source_map.get_y_size(); y++) {
        target_map[x + offset_x][y + offset_y] = source_map[x][y];
      }
    }
    return target_map;
  }
#endif

#if 1  // 与Axis有关的计算

  // 如果与边缘相交,则取内的,不取边缘上
  static PlanarRect getOpenGCellGridRect(const PlanarRect& real_rect, ScaleAxis& gcell_axis)
  {
    int32_t real_ll_x = real_rect.get_ll_x();
    int32_t real_ur_x = real_rect.get_ur_x();
    int32_t grid_ll_x = getGCellGridLB(real_ll_x, gcell_axis.get_x_grid_list());
    int32_t grid_ur_x = 0;
    if (real_ll_x == real_ur_x) {
      grid_ur_x = grid_ll_x;
    } else {
      grid_ur_x = getGCellGridRT(real_ur_x, gcell_axis.get_x_grid_list());
    }
    int32_t real_ll_y = real_rect.get_ll_y();
    int32_t real_ur_y = real_rect.get_ur_y();
    int32_t grid_ll_y = getGCellGridLB(real_ll_y, gcell_axis.get_y_grid_list());
    int32_t grid_ur_y = 0;
    if (real_ll_y == real_ur_y) {
      grid_ur_y = grid_ll_y;
    } else {
      grid_ur_y = getGCellGridRT(real_ur_y, gcell_axis.get_y_grid_list());
    }
    PlanarRect grid_rect;
    grid_rect.set_ll(grid_ll_x, grid_ll_y);
    grid_rect.set_ur(grid_ur_x, grid_ur_y);
    return grid_rect;
  }

  // 能取到边缘上
  static PlanarRect getClosedGCellGridRect(const PlanarRect& real_rect, ScaleAxis& gcell_axis)
  {
    PlanarRect border;
    int32_t min_x = gcell_axis.get_x_grid_list().front().get_start_line();
    int32_t max_x = gcell_axis.get_x_grid_list().back().get_end_line();
    int32_t min_y = gcell_axis.get_y_grid_list().front().get_start_line();
    int32_t max_y = gcell_axis.get_y_grid_list().back().get_end_line();
    border.set_ll(min_x, min_y);
    border.set_ur(max_x, max_y);

    PlanarRect new_rect = getEnlargedRect(real_rect, 1);
    if (!RTUTIL.hasRegularRect(new_rect, border)) {
      RTLOG.error(Loc::current(), "This shape is outside the border!");
    }
    new_rect = getRegularRect(new_rect, border);
    return getOpenGCellGridRect(new_rect, gcell_axis);
  }

  // 能取到边缘上
  static PlanarRect getClosedGCellGridRect(const Segment<LayerCoord>& real_segment, ScaleAxis& gcell_axis)
  {
    int32_t ll_x = real_segment.get_first().get_x();
    int32_t ll_y = real_segment.get_first().get_y();
    int32_t ur_x = real_segment.get_second().get_x();
    int32_t ur_y = real_segment.get_second().get_y();

    swapByASC(ll_x, ur_x);
    swapByASC(ll_y, ur_y);
    PlanarRect real_rect(ll_x, ll_y, ur_x, ur_y);
    return getClosedGCellGridRect(real_rect, gcell_axis);
  }

  static PlanarCoord getGCellGridCoordByBBox(const PlanarCoord& real_coord, ScaleAxis& gcell_axis, EXTPlanarRect& bounding_box)
  {
    return PlanarCoord(
        (real_coord.get_x() == bounding_box.get_real_ur_x() ? bounding_box.get_grid_ur_x() : getGCellGridLB(real_coord.get_x(), gcell_axis.get_x_grid_list())),
        (real_coord.get_y() == bounding_box.get_real_ur_y() ? bounding_box.get_grid_ur_y() : getGCellGridLB(real_coord.get_y(), gcell_axis.get_y_grid_list())));
  }

  // [ll , ur)
  static int32_t getGCellGridLB(const int32_t real_coord, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      RTLOG.error(Loc::current(), "The gcell grid list is empty!");
    }
    if (real_coord < gcell_grid_list.front().get_start_line()) {
      RTLOG.error(Loc::current(), "The real coord '", real_coord, "' < gcell grid '", gcell_grid_list.front().get_start_line(), "'!");
    }
    if (gcell_grid_list.back().get_end_line() < real_coord) {
      RTLOG.error(Loc::current(), "The gcell grid '", gcell_grid_list.back().get_end_line(), "' < real coord '", real_coord, "'!");
    }
    // gcell_grid_list 要求有序
    int32_t gcell_grid_idx = 0;
    for (ScaleGrid& gcell_grid : gcell_grid_list) {
      int32_t start_line = gcell_grid.get_start_line();
      int32_t step_length = gcell_grid.get_step_length();
      int32_t end_line = gcell_grid.get_end_line();

      if (start_line <= real_coord && real_coord < end_line) {
        double grid_num = static_cast<double>(real_coord - start_line) / step_length;
        gcell_grid_idx += static_cast<int32_t>(grid_num);
        return gcell_grid_idx;
      } else {
        gcell_grid_idx += gcell_grid.get_step_num();
      }
    }
    return gcell_grid_idx - 1;
  }

  // (ll , ur]
  static int32_t getGCellGridRT(const int32_t real_coord, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      RTLOG.error(Loc::current(), "The gcell grid list is empty!");
    }
    if (real_coord < gcell_grid_list.front().get_start_line()) {
      RTLOG.error(Loc::current(), "The real coord '", real_coord, "' < gcell grid '", gcell_grid_list.front().get_start_line(), "'!");
    }
    if (gcell_grid_list.back().get_end_line() < real_coord) {
      RTLOG.error(Loc::current(), "The gcell grid '", gcell_grid_list.back().get_end_line(), "' < real coord '", real_coord, "'!");
    }
    // gcell_grid_list 要求有序
    int32_t gcell_grid_idx = 0;
    for (ScaleGrid& gcell_grid : gcell_grid_list) {
      int32_t start_line = gcell_grid.get_start_line();
      int32_t step_length = gcell_grid.get_step_length();
      int32_t end_line = gcell_grid.get_end_line();

      if (start_line < real_coord && real_coord <= end_line) {
        double grid_num = static_cast<double>(real_coord - start_line) / step_length;
        gcell_grid_idx += static_cast<int32_t>(grid_num);
        if (isInteger(grid_num)) {
          gcell_grid_idx -= 1;
        }
        return gcell_grid_idx;
      } else {
        gcell_grid_idx += gcell_grid.get_step_num();
      }
    }
    return 0;
  }

  static PlanarRect getRealRectByGCell(PlanarRect grid_rect, ScaleAxis& gcell_axis)
  {
    return getRealRectByGCell(grid_rect.get_ll(), grid_rect.get_ur(), gcell_axis);
  }

  static PlanarRect getRealRectByGCell(PlanarCoord first_coord, PlanarCoord second_coord, ScaleAxis& gcell_axis)
  {
    if (first_coord == second_coord) {
      return getRealRectByGCell(first_coord, gcell_axis);
    }

    std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
    std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();

    swapByASC(first_x, second_x);
    swapByASC(first_y, second_y);

    return PlanarRect(getRealLBByGCell(first_x, x_grid_list), getRealLBByGCell(first_y, y_grid_list), getRealRTByGCell(second_x, x_grid_list),
                      getRealRTByGCell(second_y, y_grid_list));
  }

  static PlanarRect getRealRectByGCell(PlanarCoord grid_coord, ScaleAxis& gcell_axis)
  {
    return getRealRectByGCell(grid_coord.get_x(), grid_coord.get_y(), gcell_axis);
  }

  static PlanarRect getRealRectByGCell(int32_t x, int32_t y, ScaleAxis& gcell_axis)
  {
    std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
    std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();

    return PlanarRect(getRealLBByGCell(x, x_grid_list), getRealLBByGCell(y, y_grid_list), getRealRTByGCell(x, x_grid_list), getRealRTByGCell(y, y_grid_list));
  }

  static int32_t getRealLBByGCell(int32_t grid, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      RTLOG.error(Loc::current(), "The gcell grid list is empty!");
    }

    for (size_t i = 0; i < gcell_grid_list.size(); i++) {
      ScaleGrid& gcell_grid = gcell_grid_list[i];

      if (grid < gcell_grid.get_step_num()) {
        return gcell_grid.get_start_line() + grid * gcell_grid.get_step_length();
      } else {
        grid -= gcell_grid.get_step_num();
      }
    }
    RTLOG.error(Loc::current(), "The grid coord outside grid list!");
    return 0;
  }

  static int32_t getRealRTByGCell(int32_t grid, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      RTLOG.error(Loc::current(), "The gcell grid list is empty!");
    }

    for (size_t i = 0; i < gcell_grid_list.size(); i++) {
      ScaleGrid& gcell_grid = gcell_grid_list[i];

      if (grid < gcell_grid.get_step_num()) {
        return gcell_grid.get_start_line() + (grid + 1) * gcell_grid.get_step_length();
      } else {
        grid -= gcell_grid.get_step_num();
      }
    }
    RTLOG.error(Loc::current(), "The grid coord outside grid list!");
    return 0;
  }

  /**
   * 计算边界包含的刻度列表,如果边界与刻度重合,那么也会包含在内
   */
  static std::vector<int32_t> getScaleList(int32_t begin_line, int32_t end_line, std::vector<ScaleGrid>& scale_grid_list)
  {
    swapByASC(begin_line, end_line);

    std::vector<int32_t> scale_line_list;
    for (ScaleGrid& scale_grid : scale_grid_list) {
      if (scale_grid.get_step_length() == 0) {
        if (begin_line <= scale_grid.get_start_line() && scale_grid.get_start_line() <= end_line) {
          scale_line_list.push_back(scale_grid.get_start_line());
        }
      } else {
        for (int32_t scale_line = scale_grid.get_start_line(); scale_line <= scale_grid.get_end_line(); scale_line += scale_grid.get_step_length()) {
          if (begin_line <= scale_line && scale_line <= end_line) {
            scale_line_list.push_back(scale_line);
          }
        }
      }
    }
    std::sort(scale_line_list.begin(), scale_line_list.end());
    scale_line_list.erase(std::unique(scale_line_list.begin(), scale_line_list.end()), scale_line_list.end());
    return scale_line_list;
  }

  /**
   * 反取begin_line和end_line之外的数据,比begin_line小的存pre_scale_list,比end_line大的存post_scale_set
   */
  static void getScaleList(int32_t begin_line, int32_t end_line, std::vector<ScaleGrid>& scale_grid_list, std::vector<int32_t>& pre_scale_list,
                           std::vector<int32_t>& post_scale_set)
  {
    swapByASC(begin_line, end_line);

    std::vector<int32_t> scale_line_list;
    for (ScaleGrid& scale_grid : scale_grid_list) {
      if (scale_grid.get_step_length() == 0) {
        if (scale_grid.get_start_line() < begin_line) {
          pre_scale_list.push_back(scale_grid.get_start_line());
        }
        if (end_line < scale_grid.get_start_line()) {
          post_scale_set.push_back(scale_grid.get_start_line());
        }
      } else {
        for (int32_t scale_line = scale_grid.get_start_line(); scale_line <= scale_grid.get_end_line(); scale_line += scale_grid.get_step_length()) {
          if (scale_line < begin_line) {
            pre_scale_list.push_back(scale_line);
          }
          if (end_line < scale_line) {
            post_scale_set.push_back(scale_line);
          }
        }
      }
    }
    std::sort(pre_scale_list.begin(), pre_scale_list.end());
    pre_scale_list.erase(std::unique(pre_scale_list.begin(), pre_scale_list.end()), pre_scale_list.end());
    std::sort(post_scale_set.begin(), post_scale_set.end());
    post_scale_set.erase(std::unique(post_scale_set.begin(), post_scale_set.end()), post_scale_set.end());
  }

  static bool existTrackGrid(const PlanarCoord& real_coord, ScaleAxis& track_axis)
  {
    PlanarCoord grid_coord = getTrackGrid(real_coord, track_axis);
    return ((grid_coord.get_x() != -1) && (grid_coord.get_y() != -1));
  }

  static PlanarCoord getTrackGrid(const PlanarCoord& real_coord, ScaleAxis& track_axis)
  {
    return getTrackGrid(PlanarRect(real_coord, real_coord), track_axis).get_ll();
  }

  static bool existTrackGrid(const PlanarRect& real_rect, ScaleAxis& track_axis)
  {
    PlanarRect grid_rect = getTrackGrid(real_rect, track_axis);
    return ((grid_rect.get_ll_x() != -1) && (grid_rect.get_ll_y() != -1) && (grid_rect.get_ur_x() != -1) && (grid_rect.get_ur_y() != -1));
  }

  static PlanarRect getTrackGrid(const PlanarRect& real_rect, ScaleAxis& track_axis)
  {
    std::set<int32_t> x_pre_set;
    std::set<int32_t> x_mid_set;
    std::set<int32_t> x_post_set;
    getTrackIndexSet(track_axis.get_x_grid_list(), real_rect.get_ll_x(), real_rect.get_ur_x(), x_pre_set, x_mid_set, x_post_set);

    std::set<int32_t> y_pre_set;
    std::set<int32_t> y_mid_set;
    std::set<int32_t> y_post_set;
    getTrackIndexSet(track_axis.get_y_grid_list(), real_rect.get_ll_y(), real_rect.get_ur_y(), y_pre_set, y_mid_set, y_post_set);

    PlanarRect grid_rect;
    if (x_mid_set.empty() || y_mid_set.empty()) {
      grid_rect.set_ll(-1, -1);
      grid_rect.set_ur(-1, -1);
    } else {
      grid_rect.set_ll(*x_mid_set.begin(), *y_mid_set.begin());
      grid_rect.set_ur(*x_mid_set.rbegin(), *y_mid_set.rbegin());
    }
    return grid_rect;
  }

  static std::map<std::pair<std::shared_ptr<std::set<int32_t>>, std::shared_ptr<std::set<int32_t>>>, std::set<Orientation>> getTrackGridOrientationMap(
      const PlanarRect& real_rect, ScaleAxis& track_axis)
  {
    std::shared_ptr<std::set<int32_t>> x_pre_set = std::make_shared<std::set<int32_t>>();
    std::shared_ptr<std::set<int32_t>> x_mid_set = std::make_shared<std::set<int32_t>>();
    std::shared_ptr<std::set<int32_t>> x_post_set = std::make_shared<std::set<int32_t>>();
    getTrackIndexSet(track_axis.get_x_grid_list(), real_rect.get_ll_x(), real_rect.get_ur_x(), *x_pre_set, *x_mid_set, *x_post_set);

    std::shared_ptr<std::set<int32_t>> y_pre_set = std::make_shared<std::set<int32_t>>();
    std::shared_ptr<std::set<int32_t>> y_mid_set = std::make_shared<std::set<int32_t>>();
    std::shared_ptr<std::set<int32_t>> y_post_set = std::make_shared<std::set<int32_t>>();
    getTrackIndexSet(track_axis.get_y_grid_list(), real_rect.get_ll_y(), real_rect.get_ur_y(), *y_pre_set, *y_mid_set, *y_post_set);

    std::map<std::pair<std::shared_ptr<std::set<int32_t>>, std::shared_ptr<std::set<int32_t>>>, std::set<Orientation>> grid_orientation_map;
    grid_orientation_map[{std::make_pair(x_pre_set, y_mid_set)}] = {Orientation::kEast};
    grid_orientation_map[{std::make_pair(x_post_set, y_mid_set)}] = {Orientation::kWest};
    grid_orientation_map[{std::make_pair(x_mid_set, y_pre_set)}] = {Orientation::kNorth};
    grid_orientation_map[{std::make_pair(x_mid_set, y_post_set)}] = {Orientation::kSouth};
    grid_orientation_map[{std::make_pair(x_mid_set, y_mid_set)}]
        = {Orientation::kEast, Orientation::kWest, Orientation::kNorth, Orientation::kSouth, Orientation::kAbove, Orientation::kBelow};
    return grid_orientation_map;
  }

  static void getTrackIndexSet(std::vector<ScaleGrid>& scale_grid_list, int32_t ll_scale, int32_t ur_scale, std::set<int32_t>& pre_index_set,
                               std::set<int32_t>& mid_index_set, std::set<int32_t>& post_index_set)
  {
    int32_t pre_index = -1;
    int32_t post_index = -1;
    int32_t pre_scale = INT32_MIN;
    int32_t post_scale = INT32_MAX;

    int32_t index = -1;
    int32_t index_scale = -1;
    for (ScaleGrid& scale_grid : scale_grid_list) {
      for (int32_t i = 0; i <= scale_grid.get_step_num(); ++i) {
        int32_t scale = scale_grid.get_start_line() + i * scale_grid.get_step_length();
        if (index_scale != scale) {
          ++index;
          index_scale = scale;
        } else {
          continue;
        }
        if (scale < ll_scale) {
          if (pre_scale < scale) {
            pre_scale = scale;
            pre_index = index;
          }
        } else if (ur_scale < scale) {
          if (scale < post_scale) {
            post_scale = scale;
            post_index = index;
          }
          goto here;
        } else {
          mid_index_set.insert(index);
        }
      }
    }
  here:
    if (pre_index != -1) {
      pre_index_set.insert(pre_index);
    }
    if (post_index != -1) {
      post_index_set.insert(post_index);
    }
  }

  static void getTrackScaleSet(std::vector<ScaleGrid>& scale_grid_list, int32_t ll_scale, int32_t ur_scale, std::set<int32_t>& pre_scale_set,
                               std::set<int32_t>& mid_scale_set, std::set<int32_t>& post_scale_set)
  {
    int32_t pre_scale = INT32_MIN;
    int32_t post_scale = INT32_MAX;

    int32_t index_scale = -1;
    for (ScaleGrid& scale_grid : scale_grid_list) {
      for (int32_t i = 0; i <= scale_grid.get_step_num(); ++i) {
        int32_t scale = scale_grid.get_start_line() + i * scale_grid.get_step_length();
        if (index_scale != scale) {
          index_scale = scale;
        } else {
          continue;
        }
        if (scale < ll_scale) {
          if (pre_scale < scale) {
            pre_scale = scale;
          }
        } else if (ur_scale < scale) {
          if (scale < post_scale) {
            post_scale = scale;
          }
          goto here;
        } else {
          mid_scale_set.insert(index_scale);
        }
      }
    }
  here:
    if (pre_scale != INT32_MIN) {
      pre_scale_set.insert(pre_scale);
    }
    if (post_scale != INT32_MAX) {
      post_scale_set.insert(post_scale);
    }
  }

  static std::vector<ScaleGrid> makeScaleGridList(std::vector<int32_t>& scale_list)
  {
    std::vector<ScaleGrid> scale_grid_list;

    if (scale_list.size() == 1) {
      ScaleGrid scale_grid;
      scale_grid.set_start_line(scale_list.front());
      scale_grid.set_step_length(0);
      scale_grid.set_step_num(0);
      scale_grid.set_end_line(scale_list.front());
      scale_grid_list.push_back(scale_grid);
    } else {
      for (size_t i = 1; i < scale_list.size(); i++) {
        int32_t pre_scale = scale_list[i - 1];
        int32_t curr_scale = scale_list[i];

        ScaleGrid scale_grid;
        scale_grid.set_start_line(pre_scale);
        scale_grid.set_step_length(curr_scale - pre_scale);
        scale_grid.set_step_num(1);
        scale_grid.set_end_line(curr_scale);
        scale_grid_list.push_back(scale_grid);
      }
    }

    // merge
    merge(scale_grid_list, [](ScaleGrid& sentry, ScaleGrid& soldier) {
      if (sentry.get_step_length() != soldier.get_step_length()) {
        return false;
      }
      sentry.set_start_line(std::min(sentry.get_start_line(), soldier.get_start_line()));
      sentry.set_step_num(sentry.get_step_num() + 1);
      sentry.set_end_line(std::max(sentry.get_end_line(), soldier.get_end_line()));
      return true;
    });
    return scale_grid_list;
  }

#endif

#if 1  // irt使用的函数

  // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
  static MTree<PlanarCoord> getTreeByFullFlow(PlanarCoord& root_coord, std::vector<Segment<PlanarCoord>>& segment_list,
                                              std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>& key_coord_pin_map)
  {
    std::vector<PlanarCoord> candidate_root_coord_list{root_coord};
    return getTreeByFullFlow(candidate_root_coord_list, segment_list, key_coord_pin_map);
  }

  // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
  static MTree<PlanarCoord> getTreeByFullFlow(std::vector<PlanarCoord>& candidate_root_coord_list, std::vector<Segment<PlanarCoord>>& segment_list,
                                              std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>& key_coord_pin_map)
  {
    std::vector<LayerCoord> candidate_root_coord_list_temp;
    for (PlanarCoord& candidate_root_coord : candidate_root_coord_list) {
      candidate_root_coord_list_temp.push_back(candidate_root_coord);
    }
    std::vector<Segment<LayerCoord>> segment_list_temp;
    for (Segment<PlanarCoord>& segment : segment_list) {
      segment_list_temp.emplace_back(segment.get_first(), segment.get_second());
    }
    std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map_temp;
    for (auto& [key_coord, pin_idx_set] : key_coord_pin_map) {
      key_coord_pin_map_temp[key_coord] = pin_idx_set;
    }
    MTree<LayerCoord> coord_tree = getTreeByFullFlow(candidate_root_coord_list_temp, segment_list_temp, key_coord_pin_map_temp);
    std::function<PlanarCoord(LayerCoord&)> convert = [](LayerCoord& coord) { return coord.get_planar_coord(); };
    return convertTree(coord_tree, convert);
  }

  // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
  static MTree<LayerCoord> getTreeByFullFlow(LayerCoord& root_coord, std::vector<Segment<LayerCoord>>& segment_list,
                                             std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::vector<LayerCoord> candidate_root_coord_list{root_coord};
    return getTreeByFullFlow(candidate_root_coord_list, segment_list, key_coord_pin_map);
  }

  // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
  static MTree<LayerCoord> getTreeByFullFlow(std::vector<LayerCoord>& candidate_root_coord_list, std::vector<Segment<LayerCoord>>& segment_list,
                                             std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    // 判断是否有斜线段
    if (!passCheckingOblique(segment_list)) {
      RTLOG.error(Loc::current(), "There are oblique segments in segment_list!");
    }
    // 删除点线段
    erasePointSegment(segment_list);
    // 融合重叠的线段
    mergeOverlapSegment(segment_list, key_coord_pin_map);
    // 从候选的root坐标列表中得到树root结点
    LayerCoord root_value = getRootCoord(candidate_root_coord_list, segment_list, key_coord_pin_map);
    // 构建坐标树
    MTree<LayerCoord> coord_tree = getTreeBySegList(root_value, segment_list);
    // 删除无效(没有关键坐标的子树)的结点
    eraseInvalidNode(coord_tree, key_coord_pin_map);
    // 融合中间(平面进行切点融合,通孔进行层切割)结点
    mergeMiddleNode(coord_tree, key_coord_pin_map);
    // 检查树中是否有斜线
    if (!passCheckingOblique(coord_tree)) {
      RTLOG.error(Loc::current(), "There are oblique segments in tree!");
    }
    // 检查树是否到达所有的关键坐标
    if (!passCheckingConnectivity(coord_tree, key_coord_pin_map)) {
      RTLOG.error(Loc::current(), "The key points unreachable!");
    }
    return coord_tree;
  }

  // 判断是否有斜线
  static bool passCheckingOblique(std::vector<Segment<LayerCoord>>& segment_list)
  {
    for (Segment<LayerCoord>& segment : segment_list) {
      Orientation orientation = getOrientation(segment.get_first(), segment.get_second());
      if (orientation == Orientation::kOblique) {
        return false;
      }
    }
    return true;
  }

  // 删除点线段
  static void erasePointSegment(std::vector<Segment<LayerCoord>>& segment_list)
  {
    std::vector<Segment<LayerCoord>> new_segment_list;
    for (Segment<LayerCoord>& segment : segment_list) {
      if (segment.get_first() == segment.get_second()) {
        continue;
      }
      new_segment_list.push_back(segment);
    }
    segment_list = new_segment_list;
  }

  // 融合重叠的线段
  static void mergeOverlapSegment(std::vector<Segment<LayerCoord>>& segment_list,
                                  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    auto getListByBound = [](const std::set<int32_t>& set, int32_t lower_bound, int32_t upper_bound) {
      return std::vector<int32_t>(set.lower_bound(lower_bound), set.upper_bound(upper_bound));
    };

    std::vector<Segment<LayerCoord>> h_segment_list;
    std::vector<Segment<LayerCoord>> v_segment_list;
    std::vector<Segment<LayerCoord>> p_segment_list;

    for (Segment<LayerCoord>& segment : segment_list) {
      if (isHorizontal(segment.get_first(), segment.get_second())) {
        h_segment_list.push_back(segment);
      } else if (isVertical(segment.get_first(), segment.get_second())) {
        v_segment_list.push_back(segment);
      } else if (isProximal(segment.get_first(), segment.get_second())) {
        p_segment_list.push_back(segment);
      }
    }
    // 先切柱子
    std::vector<Segment<LayerCoord>> p_segment_list_temp;
    for (Segment<LayerCoord>& p_segment : p_segment_list) {
      PlanarCoord& planar_coord = p_segment.get_first().get_planar_coord();
      int32_t first_layer_idx = p_segment.get_first().get_layer_idx();
      int32_t second_layer_idx = p_segment.get_second().get_layer_idx();
      swapByASC(first_layer_idx, second_layer_idx);
      for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        p_segment_list_temp.emplace_back(LayerCoord(planar_coord, layer_idx), LayerCoord(planar_coord, layer_idx + 1));
      }
    }
    p_segment_list = p_segment_list_temp;

    // 初始化平面切点
    std::map<int32_t, std::set<int32_t>> x_cut_set_map;
    std::map<int32_t, std::set<int32_t>> y_cut_set_map;
    for (Segment<LayerCoord>& h_segment : h_segment_list) {
      LayerCoord& first_coord = h_segment.get_first();
      LayerCoord& second_coord = h_segment.get_second();
      int32_t layer_idx = first_coord.get_layer_idx();

      x_cut_set_map[layer_idx].insert(first_coord.get_x());
      x_cut_set_map[layer_idx].insert(second_coord.get_x());
      y_cut_set_map[layer_idx].insert(first_coord.get_y());
    }
    for (Segment<LayerCoord>& v_segment : v_segment_list) {
      LayerCoord& first_coord = v_segment.get_first();
      LayerCoord& second_coord = v_segment.get_second();
      int32_t layer_idx = first_coord.get_layer_idx();

      y_cut_set_map[layer_idx].insert(first_coord.get_y());
      y_cut_set_map[layer_idx].insert(second_coord.get_y());
      x_cut_set_map[layer_idx].insert(first_coord.get_x());
    }
    for (Segment<LayerCoord>& p_segment : p_segment_list) {
      LayerCoord& first_coord = p_segment.get_first();
      int32_t first_layer_idx = first_coord.get_layer_idx();

      LayerCoord& second_coord = p_segment.get_second();
      int32_t second_layer_idx = second_coord.get_layer_idx();

      x_cut_set_map[first_layer_idx].insert(first_coord.get_x());
      y_cut_set_map[first_layer_idx].insert(first_coord.get_y());
      x_cut_set_map[second_layer_idx].insert(second_coord.get_x());
      y_cut_set_map[second_layer_idx].insert(second_coord.get_y());
    }
    for (auto& [key_coord, pin_idx_set] : key_coord_pin_map) {
      int32_t layer_idx = key_coord.get_layer_idx();
      x_cut_set_map[layer_idx].insert(key_coord.get_x());
      y_cut_set_map[layer_idx].insert(key_coord.get_y());
    }

    // 切割平面的h线
    std::vector<Segment<LayerCoord>> h_segment_list_temp;
    for (Segment<LayerCoord>& h_segment : h_segment_list) {
      int32_t first_x = h_segment.get_first().get_x();
      int32_t second_x = h_segment.get_second().get_x();
      int32_t y = h_segment.get_first().get_y();
      int32_t layer_idx = h_segment.get_first().get_layer_idx();

      swapByASC(first_x, second_x);
      std::vector<int32_t> x_list = getListByBound(x_cut_set_map[layer_idx], first_x, second_x);
      for (size_t i = 1; i < x_list.size(); i++) {
        LayerCoord first_coord(x_list[i - 1], y, layer_idx);
        LayerCoord second_coord(x_list[i], y, layer_idx);
        h_segment_list_temp.emplace_back(first_coord, second_coord);
      }
    }
    h_segment_list = h_segment_list_temp;

    // 切割平面的v线
    std::vector<Segment<LayerCoord>> v_segment_list_temp;
    for (Segment<LayerCoord>& v_segment : v_segment_list) {
      int32_t first_y = v_segment.get_first().get_y();
      int32_t second_y = v_segment.get_second().get_y();
      int32_t x = v_segment.get_first().get_x();
      int32_t layer_idx = v_segment.get_first().get_layer_idx();

      swapByASC(first_y, second_y);
      std::vector<int32_t> y_list = getListByBound(y_cut_set_map[layer_idx], first_y, second_y);
      for (size_t i = 1; i < y_list.size(); i++) {
        LayerCoord first_coord(x, y_list[i - 1], layer_idx);
        LayerCoord second_coord(x, y_list[i], layer_idx);
        v_segment_list_temp.emplace_back(first_coord, second_coord);
      }
    }
    v_segment_list = v_segment_list_temp;

    auto mergeSegmentList = [](std::vector<Segment<LayerCoord>>& segment_list) {
      for (Segment<LayerCoord>& segment : segment_list) {
        SortSegmentInnerXASC()(segment);
      }
      std::sort(segment_list.begin(), segment_list.end(), CmpSegmentXASC());
      merge(segment_list, [](Segment<LayerCoord>& sentry, Segment<LayerCoord>& soldier) {
        return (sentry.get_first() == soldier.get_first()) && (sentry.get_second() == soldier.get_second());
      });
    };
    mergeSegmentList(h_segment_list);
    mergeSegmentList(v_segment_list);
    mergeSegmentList(p_segment_list);

    segment_list.clear();
    segment_list.insert(segment_list.end(), h_segment_list.begin(), h_segment_list.end());
    segment_list.insert(segment_list.end(), v_segment_list.begin(), v_segment_list.end());
    segment_list.insert(segment_list.end(), p_segment_list.begin(), p_segment_list.end());
  }

  // 从候选的root坐标列表中得到树root结点
  static LayerCoord getRootCoord(std::vector<LayerCoord>& candidate_root_coord_list, std::vector<Segment<LayerCoord>>& segment_list,
                                 std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    LayerCoord root_coord;

    for (Segment<LayerCoord>& segment : segment_list) {
      for (LayerCoord& candidate_root_coord : candidate_root_coord_list) {
        if (!isInside(segment, candidate_root_coord)) {
          continue;
        }
        return candidate_root_coord;
      }
    }
    if (!segment_list.empty()) {
      RTLOG.error(Loc::current(), "The segment_list not covered root!");
    }
    int32_t max_pin_num = INT32_MIN;
    for (auto& [key_coord, pin_idx_set] : key_coord_pin_map) {
      int32_t pin_num = static_cast<int32_t>(pin_idx_set.size());
      if (max_pin_num < pin_num) {
        max_pin_num = pin_num;
        root_coord = key_coord;
      }
    }
    if (max_pin_num == INT32_MIN) {
      root_coord = candidate_root_coord_list.front();
    }
    return root_coord;
  }

  // 删除无效(没有关键坐标的子树)的结点
  static void eraseInvalidNode(MTree<LayerCoord>& coord_tree, std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::vector<TNode<LayerCoord>*> erase_node_list;
    std::map<TNode<LayerCoord>*, TNode<LayerCoord>*> curr_to_parent_node_map;
    std::queue<TNode<LayerCoord>*> node_queue = initQueue(coord_tree.get_root());
    while (!node_queue.empty()) {
      TNode<LayerCoord>* node = getFrontAndPop(node_queue);
      std::vector<TNode<LayerCoord>*> child_list = node->get_child_list();
      addListToQueue(node_queue, child_list);

      for (TNode<LayerCoord>* child_node : child_list) {
        curr_to_parent_node_map[child_node] = node;

        if (child_node->isLeafNode() && !exist(key_coord_pin_map, child_node->value())) {
          erase_node_list.push_back(child_node);
          TNode<LayerCoord>* parent_node = curr_to_parent_node_map[child_node];
          parent_node->delChild(child_node);

          while (parent_node->isLeafNode() && !exist(key_coord_pin_map, parent_node->value())) {
            erase_node_list.push_back(parent_node);
            TNode<LayerCoord>* child_node = parent_node;
            parent_node = curr_to_parent_node_map[parent_node];
            parent_node->delChild(child_node);
          }
        }
      }
    }
    for (TNode<LayerCoord>* erase_node : erase_node_list) {
      delete erase_node;
    }
  }

  // 融合中间(平面进行切点融合,通孔进行层切割)结点
  static void mergeMiddleNode(MTree<LayerCoord>& coord_tree, std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::vector<TNode<LayerCoord>*> merge_node_list;
    std::map<TNode<LayerCoord>*, TNode<LayerCoord>*> middle_to_start_node_map;
    std::queue<TNode<LayerCoord>*> node_queue = initQueue(coord_tree.get_root());
    while (!node_queue.empty()) {
      TNode<LayerCoord>* node = getFrontAndPop(node_queue);
      addListToQueue(node_queue, node->get_child_list());
      int32_t node_layer_idx = node->value().get_layer_idx();
      PlanarCoord& node_coord = node->value().get_planar_coord();

      for (TNode<LayerCoord>* child_node : node->get_child_list()) {
        int32_t child_node_layer_idx = child_node->value().get_layer_idx();
        PlanarCoord& child_node_coord = child_node->value().get_planar_coord();

        if (node_layer_idx == child_node_layer_idx && node_coord != child_node_coord) {
          middle_to_start_node_map[child_node] = node;
          if (!exist(middle_to_start_node_map, node)) {
            continue;
          }
          TNode<LayerCoord>* parent_node = middle_to_start_node_map[node];
          if (getDirection(parent_node->value().get_planar_coord(), node_coord) == getDirection(node_coord, child_node_coord) && node->getChildrenNum() == 1
              && !exist(key_coord_pin_map, node->value())) {
            parent_node->delChild(node);
            parent_node->addChild(child_node);
            merge_node_list.push_back(node);
            middle_to_start_node_map[child_node] = parent_node;
          }
        }
      }
    }
    for (TNode<LayerCoord>* merge_node : merge_node_list) {
      delete merge_node;
    }
  }

  // 检查树中是否有斜线
  static bool passCheckingOblique(MTree<LayerCoord>& coord_tree)
  {
    for (TNode<LayerCoord>* coord_node : getNodeList(coord_tree)) {
      LayerCoord& coord = coord_node->value();

      PlanarCoord& first_planar_coord = coord.get_planar_coord();
      int32_t first_layer_idx = coord.get_layer_idx();
      PlanarCoord& second_planar_coord = coord.get_planar_coord();
      int32_t second_layer_idx = coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx && isRightAngled(first_planar_coord, second_planar_coord)) {
        continue;
      } else if (first_layer_idx != second_layer_idx && first_planar_coord == second_planar_coord) {
        continue;
      }
      return false;
    }
    return true;
  }

  // 检查树是否到达所有的关键坐标
  static bool passCheckingConnectivity(MTree<LayerCoord>& coord_tree, std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::map<int32_t, bool> visited_map;
    for (auto& [key_coord, pin_idx_set] : key_coord_pin_map) {
      for (int32_t pin_idx : pin_idx_set) {
        visited_map[pin_idx] = false;
      }
    }
    for (TNode<LayerCoord>* coord_node : getNodeList(coord_tree)) {
      LayerCoord coord = coord_node->value();
      if (!exist(key_coord_pin_map, coord)) {
        continue;
      }
      for (int32_t pin_idx : key_coord_pin_map[coord]) {
        visited_map[pin_idx] = true;
      }
    }
    bool is_connectivity = true;
    for (auto& [pin_idx, is_visited] : visited_map) {
      if (is_visited == false) {
        RTLOG.warn(Loc::current(), "The pin idx ", pin_idx, " unreachable!");
        is_connectivity = false;
      }
    }
    return is_connectivity;
  }

  // 获得配置的值
  template <typename T>
  static T getConfigValue(std::map<std::string, std::any>& config_map, const std::string& config_name, const T& default_value)
  {
    T value;
    if (exist(config_map, config_name)) {
      value = std::any_cast<T>(config_map[config_name]);
    } else {
      RTLOG.warn(Loc::current(), "The config '", config_name, "' uses the default value!");
      value = default_value;
    }
    return value;
  }

  template <typename T>
  static bool passCheckingConnectivity(std::vector<T>& key_coord_list, std::vector<Segment<T>>& routing_segment_list)
  {
    if (key_coord_list.empty()) {
      RTLOG.error(Loc::current(), "The key_coord_list is empty!");
    }
    std::vector<std::pair<bool, T>> visited_coord_pair_list;
    std::vector<std::pair<bool, Segment<T>>> visited_segment_pair_list;
    visited_coord_pair_list.reserve(key_coord_list.size());
    visited_segment_pair_list.reserve(routing_segment_list.size());
    for (size_t i = 0; i < key_coord_list.size(); i++) {
      visited_coord_pair_list.emplace_back(false, key_coord_list[i]);
    }
    for (size_t i = 0; i < routing_segment_list.size(); i++) {
      visited_segment_pair_list.emplace_back(false, routing_segment_list[i]);
    }
    std::queue<T> node_queue = initQueue(key_coord_list.front());
    while (!node_queue.empty()) {
      T node = getFrontAndPop(node_queue);
      for (size_t i = 0; i < visited_coord_pair_list.size(); i++) {
        std::pair<bool, T>& visited_coord_pair = visited_coord_pair_list[i];
        if (visited_coord_pair.first == true) {
          continue;
        }
        if (node == visited_coord_pair.second) {
          visited_coord_pair.first = true;
        }
      }
      std::vector<T> next_node_list;
      for (size_t i = 0; i < visited_segment_pair_list.size(); i++) {
        std::pair<bool, Segment<T>>& visited_segment_pair = visited_segment_pair_list[i];
        if (visited_segment_pair.first == true) {
          continue;
        }
        T& node1 = visited_segment_pair.second.get_first();
        T& node2 = visited_segment_pair.second.get_second();
        if (node == node1 || node == node2) {
          T child_node = (node == node1 ? node2 : node1);
          next_node_list.push_back(child_node);
          visited_segment_pair.first = true;
        }
      }
      addListToQueue(node_queue, next_node_list);
    }
    bool is_connected = true;
    for (auto& [visited, coord] : visited_coord_pair_list) {
      is_connected = visited;
      if (is_connected == false) {
        break;
      }
    }
    return is_connected;
  }

  static void printTableList(const std::vector<fort::char_table>& table_list)
  {
    std::vector<std::vector<std::string>> print_table_list;
    for (const fort::char_table& table : table_list) {
      if (!table.is_empty()) {
        print_table_list.push_back(splitString(table.to_string(), '\n'));
      }
    }

    int32_t max_size = INT_MIN;
    for (std::vector<std::string>& table : print_table_list) {
      max_size = std::max(max_size, static_cast<int32_t>(table.size()));
    }
    for (std::vector<std::string>& table : print_table_list) {
      for (int32_t i = static_cast<int32_t>(table.size()); i < max_size; i++) {
        std::string table_str;
        table_str.append(table.front().length(), ' ');
        table.push_back(table_str);
      }
    }

    for (int32_t i = 0; i < max_size; i++) {
      std::string table_str;
      for (std::vector<std::string>& table : print_table_list) {
        table_str += table[i];
        table_str += " ";
      }
      RTLOG.info(Loc::current(), table_str);
    }
  }

  // 计算rect在master上覆盖的面积占master总面积的比例
  static double getOverlapRatio(PlanarRect& master, PlanarRect& rect)
  {
    double ratio = 0;
    if (isOpenOverlap(master, rect)) {
      ratio = getOverlap(master, rect).getArea() / master.getArea();
    }
    return ratio;
  }

  static LayerCoord getFirstEqualCoord(std::vector<LayerCoord> a_list, std::vector<LayerCoord> b_list)
  {
    return getFirstEqualItem(a_list, b_list, CmpLayerCoordByXASC());
  }

#endif

#if 1  // boost数据结构工具函数

#if 1  // int类型

  static GTLPointInt convertToGTLPointInt(const PlanarCoord& coord) { return GTLPointInt(coord.get_x(), coord.get_y()); }

  static PlanarCoord convertToPlanarCoord(const GTLPointInt& gtl_point) { return PlanarCoord(gtl::x(gtl_point), gtl::y(gtl_point)); }

  static PlanarRect convertToPlanarRect(GTLRectInt& gtl_rect) { return PlanarRect(gtl::xl(gtl_rect), gtl::yl(gtl_rect), gtl::xh(gtl_rect), gtl::yh(gtl_rect)); }

  static PlanarRect convertToPlanarRect(BGRectInt& boost_box)
  {
    return PlanarRect(boost_box.min_corner().x(), boost_box.min_corner().y(), boost_box.max_corner().x(), boost_box.max_corner().y());
  }

  static BGRectInt convertToBGRectInt(const PlanarRect& rect)
  {
    return BGRectInt(BGPointInt(rect.get_ll_x(), rect.get_ll_y()), BGPointInt(rect.get_ur_x(), rect.get_ur_y()));
  }

  static BGRectInt convertToBGRectInt(GTLRectInt& gtl_rect)
  {
    return BGRectInt(BGPointInt(gtl::xl(gtl_rect), gtl::yl(gtl_rect)), BGPointInt(gtl::xh(gtl_rect), gtl::yh(gtl_rect)));
  }

  static GTLRectInt convertToGTLRectInt(const PlanarRect& rect) { return GTLRectInt(rect.get_ll_x(), rect.get_ll_y(), rect.get_ur_x(), rect.get_ur_y()); }

  static GTLRectInt convertToGTLRectInt(BGRectInt& boost_box)
  {
    return GTLRectInt(boost_box.min_corner().x(), boost_box.min_corner().y(), boost_box.max_corner().x(), boost_box.max_corner().y());
  }

  static int32_t getLength(BGRectInt& a) { return std::abs(a.max_corner().x() - a.min_corner().x()); }

  static int32_t getWidth(BGRectInt& a) { return std::abs(a.max_corner().y() - a.min_corner().y()); }

  static PlanarCoord getCenter(BGRectInt& a)
  {
    int32_t center_x = std::abs(a.max_corner().x() + a.min_corner().x()) / 2;
    int32_t center_y = std::abs(a.max_corner().y() + a.min_corner().y()) / 2;
    return PlanarCoord(center_x, center_y);
  }

  static BGRectInt enlargeBGRectInt(BGRectInt& a, int32_t enlarge_size)
  {
    return BGRectInt(BGPointInt(a.min_corner().x() - enlarge_size, a.min_corner().y() - enlarge_size),
                     BGPointInt(a.max_corner().x() + enlarge_size, a.max_corner().y() + enlarge_size));
  }

  static void offsetBGRectInt(BGRectInt& boost_box, PlanarCoord& coord)
  {
    boost_box.min_corner().set<0>(boost_box.min_corner().x() + coord.get_x());
    boost_box.min_corner().set<1>(boost_box.min_corner().y() + coord.get_y());

    boost_box.max_corner().set<0>(boost_box.max_corner().x() + coord.get_x());
    boost_box.max_corner().set<1>(boost_box.max_corner().y() + coord.get_y());
  }

  static bool isOverlap(BGRectInt& a, BGRectInt& b, bool consider_edge = true)
  {
    int32_t a_ll_x = a.min_corner().x(), a_ll_y = a.min_corner().y();
    int32_t a_ur_x = a.max_corner().x(), a_ur_y = a.max_corner().y();

    int32_t b_ll_x = b.min_corner().x(), b_ll_y = b.min_corner().y();
    int32_t b_ur_x = b.max_corner().x(), b_ur_y = b.max_corner().y();

    int32_t x_spacing = std::max(b_ll_x - a_ur_x, a_ll_x - b_ur_x);
    int32_t y_spacing = std::max(b_ll_y - a_ur_y, a_ll_y - b_ur_y);

    if (x_spacing == 0 || y_spacing == 0) {
      return consider_edge;
    } else {
      return (x_spacing < 0 && y_spacing < 0);
    }
  }

  static BGRectInt getOverlap(BGRectInt& a, BGRectInt& b)
  {
    int32_t overlap_ll_x = std::max(a.min_corner().x(), b.min_corner().x());
    int32_t overlap_ll_y = std::max(a.min_corner().y(), b.min_corner().y());
    int32_t overlap_ur_x = std::min(a.max_corner().x(), b.max_corner().x());
    int32_t overlap_ur_y = std::min(a.max_corner().y(), b.max_corner().y());

    if (overlap_ll_x > overlap_ur_x || overlap_ll_y > overlap_ur_y) {
      return BGRectInt(BGPointInt(0, 0), BGPointInt(0, 0));
    } else {
      return BGRectInt(BGPointInt(overlap_ll_x, overlap_ll_y), BGPointInt(overlap_ur_x, overlap_ur_y));
    }
  }

  static bool isHorizontal(BGRectInt a) { return (a.max_corner().x() - a.min_corner().x()) >= (a.max_corner().y() - a.min_corner().y()); }

  static int32_t getDiagonalLength(BGRectInt& a)
  {
    int32_t length = getLength(a);
    int32_t width = getWidth(a);
    return (int32_t) std::sqrt((length * length + width * width));
  }

  static int32_t getEuclideanDistance(BGRectInt& a, BGRectInt& b)
  {
    int32_t a_ll_x = a.min_corner().x(), a_ll_y = a.min_corner().y();
    int32_t a_ur_x = a.max_corner().x(), a_ur_y = a.max_corner().y();

    int32_t b_ll_x = b.min_corner().x(), b_ll_y = b.min_corner().y();
    int32_t b_ur_x = b.max_corner().x(), b_ur_y = b.max_corner().y();

    int32_t x_spacing = std::max(b_ll_x - a_ur_x, a_ll_x - b_ur_x);
    int32_t y_spacing = std::max(b_ll_y - a_ur_y, a_ll_y - b_ur_y);

    if (x_spacing > 0 && y_spacing > 0) {
      return (int32_t) std::sqrt((x_spacing * x_spacing + y_spacing * y_spacing));
    } else {
      return std::max(std::max(x_spacing, y_spacing), 0);
    }
  }

  static std::vector<PlanarRect> getMaxRectList(const std::vector<PlanarRect>& master_list)
  {
    GTLPolySetInt gtl_poly_set;
    for (const PlanarRect& master : master_list) {
      gtl_poly_set += convertToGTLRectInt(master);
    }
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
    std::vector<PlanarRect> result_list;
    result_list.reserve(gtl_rect_list.size());
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      result_list.push_back(convertToPlanarRect(gtl_rect));
    }
    return result_list;
  }

#endif

#if 1  // double类型

#if 1  // cutting

  static std::vector<PlanarRect> getOpenCuttingRectListByBoost(const PlanarRect& master, const std::vector<PlanarRect>& rect_list)
  {
    std::vector<PlanarRect> master_list{master};
    return getCuttingRectListByBoost(master_list, rect_list, true);
  }

  static std::vector<PlanarRect> getOpenCuttingRectListByBoost(const std::vector<PlanarRect>& master_list, const std::vector<PlanarRect>& rect_list)
  {
    return getCuttingRectListByBoost(master_list, rect_list, true);
  }

  static std::vector<PlanarRect> getClosedCuttingRectListByBoost(const std::vector<PlanarRect>& master_list, const std::vector<PlanarRect>& rect_list)
  {
    return getCuttingRectListByBoost(master_list, rect_list, false);
  }

  static std::vector<PlanarRect> getCuttingRectListByBoost(const std::vector<PlanarRect>& master_list, const std::vector<PlanarRect>& rect_list, bool is_open)
  {
    std::vector<PlanarRect> result_list;
    if (!is_open) {
      // 先保存master_list中的特殊矩形(点 线)
      for (const PlanarRect& master : master_list) {
        if (master.get_ll_x() == master.get_ur_x() || master.get_ll_y() == master.get_ur_y()) {
          result_list.push_back(master);
        }
      }
    }
    /**
     * 下面每个字母表示一个独立的直角多边形
     * 求解(A ∪ B) - (D ∪ E ∪ F)
     * 转((A - D) ∩ (A - E) ∩ (A - F)) ∪ ((B - D) ∩ (B - E) ∩ (B - F))
     * 其中利用(A - D)、(A - E)等式中结果不可能出现线,实现boost结果传递
     */
    // 将输入解析
    // 其中master_poly_list为(A ∪ B)
    std::vector<BGPolyDBL> master_poly_list = getBGPolyDBLList(master_list);
    // 其中rect_poly_list为(D ∪ E ∪ F)
    std::vector<BGPolyDBL> rect_poly_list = getBGPolyDBLList(rect_list);

    // 计算((A - D) ∩ (A - E) ∩ (A - F)) ∪ ((B - D) ∩ (B - E) ∩ (B - F))
    BGMultiPolyDBL top_multi_poly;
    BGMultiLineDBL top_multi_line;
    BGMultiPointDBL top_multi_point;
    for (BGPolyDBL& master_poly : master_poly_list) {
      // 计算(A - D)和(A - E)和(A - F)
      std::vector<BGMultiPolyDBL> diff_multi_poly_list;
      {
        if (rect_poly_list.empty()) {
          BGMultiPolyDBL diff_multi_poly;
          diff_multi_poly.push_back(master_poly);
          diff_multi_poly_list.push_back(diff_multi_poly);
        } else {
          for (BGPolyDBL& rect_poly : rect_poly_list) {
            // 计算(A - D)
            BGMultiPolyDBL diff_multi_poly;
            bg::difference(master_poly, rect_poly, diff_multi_poly);
            if (diff_multi_poly.empty()) {
              // 当(A - D)为空,后续(A - D) ∩ (A - E) ∩ (A - F)结果为空,直接跳过
              diff_multi_poly_list.clear();
              break;
            } else {
              diff_multi_poly_list.push_back(diff_multi_poly);
            }
          }
        }
        for (BGMultiPolyDBL& diff_multi_poly : diff_multi_poly_list) {
          removeError(diff_multi_poly);
        }
      }
      if (diff_multi_poly_list.empty()) {
        continue;
      }
      // 计算(A - D) ∩ (A - E) ∩ (A - F)
      BGMultiPolyDBL mid_multi_poly;
      BGMultiLineDBL mid_multi_line;
      BGMultiPointDBL mid_multi_point;
      {
        // 用(A - D)初始化
        mid_multi_poly = diff_multi_poly_list.front();
        for (size_t i = 1; i < diff_multi_poly_list.size(); i++) {
          BGMultiPolyDBL& curr_multi_poly = diff_multi_poly_list[i];
          // (A - D) ∩ (A - E)
          BGMultiPolyDBL mid_multi_poly_temp;
          // 与顶层poly相交
          bg::intersection(mid_multi_poly, curr_multi_poly, mid_multi_poly_temp);
          if (!is_open) {
            bg::intersection(mid_multi_poly, curr_multi_poly, mid_multi_line);
            bg::intersection(mid_multi_poly, curr_multi_poly, mid_multi_point);
          }
          mid_multi_poly = mid_multi_poly_temp;
          removeError(mid_multi_poly);
        }
      }
      // 计算((A - D) ∩ (A - E) ∩ (A - F)) ∪ ((B - D) ∩ (B - E) ∩ (B - F))
      {
        top_multi_poly.insert(top_multi_poly.end(), mid_multi_poly.begin(), mid_multi_poly.end());
        top_multi_line.insert(top_multi_line.end(), mid_multi_line.begin(), mid_multi_line.end());
        top_multi_point.insert(top_multi_point.end(), mid_multi_point.begin(), mid_multi_point.end());
      }
    }
    // 生成对应的矩形结果
    for (PlanarRect& rect : getRTRectListByBGMultiPolyDBL(top_multi_poly)) {
      result_list.push_back(rect);
    }
    for (PlanarRect& rect : getRTRectListByBGMultiLineDBL(top_multi_line)) {
      result_list.push_back(rect);
    }
    for (PlanarRect& rect : getRTRectListByBGMultiPointDBL(top_multi_point)) {
      result_list.push_back(rect);
    }
    if (!is_open) {
      std::vector<PlanarRect> special_rect_list;
      std::vector<PlanarRect> result_list_temp;
      result_list_temp.reserve(result_list.size());
      for (PlanarRect& result_rect : result_list) {
        if (result_rect.get_ll_x() == result_rect.get_ur_x() || result_rect.get_ll_y() == result_rect.get_ur_y()) {
          special_rect_list.push_back(result_rect);
        } else {
          result_list_temp.push_back(result_rect);
        }
      }
      result_list = std::move(result_list_temp);
      // 对结果中的特殊矩形进行切割
      for (PlanarRect& cutting_special_rect : getCuttingSpecialRectList(special_rect_list, rect_list)) {
        result_list.push_back(cutting_special_rect);
      }
    }
    // rect去重
    std::sort(result_list.begin(), result_list.end(), CmpPlanarRectByXASC());
    result_list.erase(std::unique(result_list.begin(), result_list.end()), result_list.end());
    return result_list;
  }

  static std::vector<PlanarRect> getCuttingSpecialRectList(const std::vector<PlanarRect>& special_rect_list, const std::vector<PlanarRect>& rect_list)
  {
    std::vector<PlanarRect> result_list;
    for (const PlanarRect& special_rect : special_rect_list) {
      if (special_rect.get_ll() == special_rect.get_ur()) {
        /**
         * 对于点矩形, 在其中一个rect内(不包含边界)则被删除
         */
        PlanarCoord point = special_rect.get_ll();
        bool exist_inside = false;
        for (const PlanarRect& rect : rect_list) {
          if (rect.get_ll_x() < point.get_x() && point.get_x() < rect.get_ur_x() && rect.get_ll_y() < point.get_y() && point.get_y() < rect.get_ur_y()) {
            exist_inside = true;
            break;
          }
        }
        if (!exist_inside) {
          result_list.push_back(special_rect);
        }
      } else {
        /**
         * 对于线矩形, overlap长度不为0则cut
         * segment_list内要保证 first < second
         */
        std::vector<Segment<PlanarCoord>> segment_list = {{special_rect.get_ll(), special_rect.get_ur()}};
        for (const PlanarRect& rect : rect_list) {
          int32_t rect_ll_x = rect.get_ll_x();
          int32_t rect_ur_x = rect.get_ur_x();
          int32_t rect_ll_y = rect.get_ll_y();
          int32_t rect_ur_y = rect.get_ur_y();

          std::vector<Segment<PlanarCoord>> segment_list_temp;
          for (Segment<PlanarCoord>& segment : segment_list) {
            if (isHorizontal(segment.get_first(), segment.get_second())) {
              int32_t seg_first_x = segment.get_first().get_x();
              int32_t seg_second_x = segment.get_second().get_x();
              int32_t seg_y = segment.get_first().get_y();
              if (rect_ll_y <= seg_y && seg_y <= rect_ur_y && seg_first_x < rect_ur_x && rect_ll_x < seg_second_x) {
                if (seg_first_x <= rect_ll_x) {
                  // 提出左突出
                  segment_list_temp.emplace_back(segment.get_first(), PlanarCoord(rect_ll_x, seg_y));
                }
                if (rect_ur_x <= seg_second_x) {
                  // 提出右突出
                  segment_list_temp.emplace_back(PlanarCoord(rect_ur_x, seg_y), segment.get_second());
                }
              } else {
                segment_list_temp.push_back(segment);
              }
            } else {
              int32_t seg_first_y = segment.get_first().get_y();
              int32_t seg_second_y = segment.get_second().get_y();
              int32_t seg_x = segment.get_first().get_x();
              if (rect_ll_x <= seg_x && seg_x <= rect_ur_x && seg_first_y < rect_ur_y && rect_ll_y < seg_second_y) {
                if (seg_first_y <= rect_ll_y) {
                  // 提出下突出
                  segment_list_temp.emplace_back(segment.get_first(), PlanarCoord(seg_x, rect_ll_y));
                }
                if (rect_ur_y <= seg_second_y) {
                  // 提出上突出
                  segment_list_temp.emplace_back(PlanarCoord(seg_x, rect_ur_y), segment.get_second());
                }
              } else {
                segment_list_temp.push_back(segment);
              }
            }
          }
          segment_list = segment_list_temp;
        }
        for (Segment<PlanarCoord>& segment : segment_list) {
          result_list.emplace_back(segment.get_first(), segment.get_second());
        }
      }
    }
    return result_list;
  }

#endif

#if 1  // reduce

  static std::vector<PlanarRect> getOpenShrinkedRectListByBoost(const std::vector<PlanarRect>& master_list, int32_t ll_x_add_offset, int32_t ll_y_add_offset,
                                                                int32_t ur_x_minus_offset, int32_t ur_y_minus_offset)
  {
    return getShrinkedRectListByBoost(master_list, ll_x_add_offset, ll_y_add_offset, ur_x_minus_offset, ur_y_minus_offset, true);
  }

  static std::vector<PlanarRect> getClosedShrinkedRectListByBoost(const std::vector<PlanarRect>& master_list, int32_t shrinked_offset)
  {
    return getShrinkedRectListByBoost(master_list, shrinked_offset, shrinked_offset, shrinked_offset, shrinked_offset, false);
  }

  static std::vector<PlanarRect> getClosedShrinkedRectListByBoost(const std::vector<PlanarRect>& master_list, int32_t ll_x_add_offset, int32_t ll_y_add_offset,
                                                                  int32_t ur_x_minus_offset, int32_t ur_y_minus_offset)
  {
    return getShrinkedRectListByBoost(master_list, ll_x_add_offset, ll_y_add_offset, ur_x_minus_offset, ur_y_minus_offset, false);
  }

  static std::vector<PlanarRect> getShrinkedRectListByBoost(const std::vector<PlanarRect>& master_list, int32_t ll_x_add_offset, int32_t ll_y_add_offset,
                                                            int32_t ur_x_minus_offset, int32_t ur_y_minus_offset, bool is_open)
  {
    std::vector<PlanarRect> result_list;

    GTLPolySetInt master_poly;
    for (const PlanarRect& master : master_list) {
      master_poly += convertToGTLRectInt(master);
    }
    if (!is_open) {
      // 提取点矩形,线段矩形
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::HORIZONTAL);
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::VERTICAL);

      std::vector<PlanarRect> candidate_rect_list;
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        candidate_rect_list.emplace_back(convertToPlanarRect(gtl_rect));
      }
      for (PlanarRect candidate_rect : candidate_rect_list) {
        PlanarCoord& ll = candidate_rect.get_ll();
        PlanarCoord& ur = candidate_rect.get_ur();
        addOffset(ll, ll_x_add_offset, ll_y_add_offset);
        minusOffset(ur, ur_x_minus_offset, ur_y_minus_offset);
        // 去除不是矩形的
        if (candidate_rect.isIncorrect()) {
          continue;
        }
        // 去除面积不为0的
        if (candidate_rect.getArea() > 0) {
          continue;
        }
        result_list.push_back(candidate_rect);
      }
    }
    // 获得常规收缩的矩形
    {
      master_poly.shrink(ll_x_add_offset, ur_x_minus_offset, ll_y_add_offset, ur_y_minus_offset);

      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::HORIZONTAL);
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::VERTICAL);

      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        result_list.emplace_back(convertToPlanarRect(gtl_rect));
      }
    }
    // rect去重
    std::sort(result_list.begin(), result_list.end(), CmpPlanarRectByXASC());
    result_list.erase(std::unique(result_list.begin(), result_list.end()), result_list.end());

    return result_list;
  }

#endif

#if 1  // merge

  static std::vector<PlanarRect> mergeRectListByBoost(const std::vector<PlanarRect>& master_list, Direction direction)
  {
    // 先保存master_list中的特殊矩形
    std::vector<PlanarRect> special_rect_list;
    for (const PlanarRect& master : master_list) {
      if (master.get_ll_x() == master.get_ur_x() || master.get_ll_y() == master.get_ur_y()) {
        // 特殊矩形
        special_rect_list.push_back(master);
      }
    }
    // 对常规矩形做merge
    std::vector<PlanarRect> regular_rect_list;
    GTLPolySetInt master_poly;
    for (const PlanarRect& master : master_list) {
      master_poly += convertToGTLRectInt(master);
    }
    std::vector<GTLRectInt> gtl_rect_list;
    if (direction == Direction::kHorizontal) {
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::HORIZONTAL);
    } else if (direction == Direction::kVertical) {
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::VERTICAL);
    }
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      regular_rect_list.emplace_back(convertToPlanarRect(gtl_rect));
    }
    std::vector<PlanarRect> result_list;
    // 将常规矩形减去特殊矩形
    for (PlanarRect& result_rect : getCuttingSpecialRectList(special_rect_list, regular_rect_list)) {
      result_list.push_back(result_rect);
    }
    for (PlanarRect& result_rect : regular_rect_list) {
      result_list.push_back(result_rect);
    }
    return result_list;
  }

#endif

#if 1  // aux

  static std::vector<BGPolyDBL> getBGPolyDBLList(const std::vector<PlanarRect>& rect_list)
  {
    std::vector<BGPolyDBL> bg_poly_list;
    for (const PlanarRect& rect : rect_list) {
      BGPolyDBL bg_poly;
      for (BGPointDBL& bg_point : getPointList(rect)) {
        bg::append(bg_poly.outer(), bg_point);
      }
      bg_poly_list.push_back(bg_poly);
    }
    return bg_poly_list;
  }

  static std::vector<BGPointDBL> getPointList(PlanarRect rect)
  {
    // 要求顺时针
    std::vector<BGPointDBL> point_list;
    point_list.emplace_back(rect.get_ll_x(), rect.get_ll_y());
    point_list.emplace_back(rect.get_ll_x(), rect.get_ur_y());
    point_list.emplace_back(rect.get_ur_x(), rect.get_ur_y());
    point_list.emplace_back(rect.get_ur_x(), rect.get_ll_y());
    point_list.emplace_back(rect.get_ll_x(), rect.get_ll_y());
    return point_list;
  }

  static void removeError(BGMultiPolyDBL& multipoly)
  {
    for (auto& poly : multipoly) {
      for (auto& ring : poly.outer()) {
        ring.x(getIntScale(ring.x()));
        ring.y(getIntScale(ring.y()));
      }
      for (auto& ring : poly.inners()) {
        for (auto& point : ring) {
          point.x(getIntScale(point.x()));
          point.y(getIntScale(point.y()));
        }
      }
    }
  }

  static int32_t getIntScale(double double_scale)
  {
    int32_t integer_scale = static_cast<int32_t>(std::round(double_scale));
    if (std::abs(double_scale - integer_scale) > RT_ERROR) {
      RTLOG.error(Loc::current(), "Exceeding the error range of a double!");
    }
    return integer_scale;
  }

  static std::vector<PlanarRect> getRTRectListByBGMultiPolyDBL(const BGMultiPolyDBL& bg_multi_poly)
  {
    std::vector<PlanarRect> rect_list;

    GTLPolySetInt gtl_poly_set;
    for (const BGPolyDBL& bg_poly : bg_multi_poly) {
      // 将double类型转int32_t
      std::vector<GTLPointInt> gtl_point_list;
      for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
        BGPointDBL bg_point = bg_poly.outer()[i];
        gtl_point_list.emplace_back(getIntScale(bg_point.x()), getIntScale(bg_point.y()));
      }
      // 检查是否有斜线
      for (size_t i = 1; i < gtl_point_list.size(); i++) {
        GTLPointInt& pre_coord = gtl_point_list[i - 1];
        GTLPointInt& curr_coord = gtl_point_list[i];
        if (gtl::x(pre_coord) != gtl::x(curr_coord) && gtl::y(pre_coord) != gtl::y(curr_coord)) {
          RTLOG.error(Loc::current(), "The segment is oblique!");
        }
      }
      // 生成poly_90
      GTLPolyInt gtl_poly;
      gtl_poly.set(gtl_point_list.begin(), gtl_point_list.end());
      gtl_poly_set.insert(gtl_poly);
    }
    // 横竖切割
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, gtl_poly_set, gtl::HORIZONTAL);
    gtl::get_rectangles(gtl_rect_list, gtl_poly_set, gtl::VERTICAL);
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      rect_list.emplace_back(convertToPlanarRect(gtl_rect));
    }
    return rect_list;
  }

  static std::vector<PlanarRect> getRTRectListByBGMultiLineDBL(const BGMultiLineDBL& bg_multi_line)
  {
    std::vector<PlanarRect> rect_list;

    for (const BGLineDBL& bg_line : bg_multi_line) {
      int32_t ll_x = getIntScale(bg_line[0].x());
      int32_t ll_y = getIntScale(bg_line[0].y());
      int32_t ur_x = getIntScale(bg_line[1].x());
      int32_t ur_y = getIntScale(bg_line[1].y());
      swapByASC(ll_x, ur_x);
      swapByASC(ll_y, ur_y);
      rect_list.emplace_back(ll_x, ll_y, ur_x, ur_y);
    }

    return rect_list;
  }

  static std::vector<PlanarRect> getRTRectListByBGMultiPointDBL(const BGMultiPointDBL& bg_multi_point)
  {
    std::vector<PlanarRect> rect_list;

    for (const BGPointDBL& bg_point : bg_multi_point) {
      PlanarCoord coord(getIntScale(bg_point.x()), getIntScale(bg_point.y()));
      rect_list.emplace_back(coord, coord);
    }

    return rect_list;
  }

#endif

#endif

#endif

#if 1  // std数据结构工具函数

  template <typename T, typename Compare>
  static T getFirstEqualItem(std::vector<T> a_list, std::vector<T> b_list, Compare cmp)
  {
    std::sort(a_list.begin(), a_list.end(), cmp);
    a_list.erase(std::unique(a_list.begin(), a_list.end()), a_list.end());
    std::sort(b_list.begin(), b_list.end(), cmp);
    b_list.erase(std::unique(b_list.begin(), b_list.end()), b_list.end());
    for (size_t i = 0, j = 0; i < a_list.size() && j < b_list.size();) {
      if (a_list[i] == b_list[j]) {
        return a_list[i];
        break;
      } else if (cmp(a_list[i], b_list[j])) {
        i++;
      } else {
        j++;
      }
    }
    return T();
  }

  template <typename Key, typename Value>
  static Value getValueByAny(std::map<Key, std::any>& map, const Key& key, const Value& default_value)
  {
    Value value;
    if (exist(map, key)) {
      value = std::any_cast<Value>(map[key]);
    } else {
      value = default_value;
    }
    return value;
  }

  template <typename Key, typename Value>
  static Value getValue(std::map<Key, Value>& map, const Key& key, const Value& default_value)
  {
    Value value;
    if (exist(map, key)) {
      value = map[key];
    } else {
      value = default_value;
    }
    return value;
  }

  template <typename T>
  static T getAverage(const std::vector<T>& value_list)
  {
    if (value_list.empty()) {
      return 0;
    }
    double average = 0;
    for (size_t i = 0; i < value_list.size(); i++) {
      average += value_list[i];
    }
    average /= static_cast<int32_t>(value_list.size());
    if constexpr (std::is_same<T, int32_t>::value) {
      average = std::round(average);
    }
    return T(average);
  }

  template <typename T>
  static void erase(std::vector<T>& list, const std::function<bool(T&)>& erase_if)
  {
    erase(list, erase_if);
  }

  template <typename T, typename EraseIf>
  static void erase(std::vector<T>& list, EraseIf erase_if)
  {
    size_t save_idx = 0;
    size_t sentry_idx = 0;
    while (sentry_idx < list.size()) {
      T& sentry = list[sentry_idx];
      if (!erase_if(sentry)) {
        list[save_idx] = std::move(sentry);
        ++save_idx;
      }
      sentry_idx++;
    }
    list.erase(list.begin() + save_idx, list.end());
  }

  template <typename T>
  static void merge(std::vector<T>& list, const std::function<bool(T&, T&)>& merge_if)
  {
    merge(list, merge_if);
  }

  template <typename T, typename MergeIf>
  static void merge(std::vector<T>& list, MergeIf merge_if)
  {
    size_t save_idx = 0;
    size_t sentry_idx = 0;
    size_t soldier_idx = sentry_idx + 1;
    while (sentry_idx < list.size()) {
      T& sentry = list[sentry_idx];
      while (soldier_idx < list.size()) {
        T& soldier = list[soldier_idx];
        if (!merge_if(sentry, soldier)) {
          break;
        }
        ++soldier_idx;
      }
      list[save_idx] = std::move(sentry);
      ++save_idx;
      if (!(soldier_idx < list.size())) {
        break;
      }
      sentry_idx = soldier_idx;
      soldier_idx = sentry_idx + 1;
    }
    list.erase(list.begin() + save_idx, list.end());
  }

  template <typename T>
  static bool isDifferentSign(T a, T b)
  {
    return a & b ? (a ^ b) < 0 : false;
  }

  static int32_t getFirstDigit(int32_t n)
  {
    n = n >= 100000000 ? (n / 100000000) : n;
    n = n >= 10000 ? (n / 10000) : n;
    n = n >= 100 ? (n / 100) : n;
    n = n >= 10 ? (n / 10) : n;
    return n;
  }

  static int32_t getDigitNum(int32_t n)
  {
    int32_t count = 0;

    while (n != 0) {
      n /= 10;
      count++;
    }
    return count;
  }

  static int32_t getBatchSize(size_t total_size) { return getBatchSize(static_cast<int32_t>(total_size)); }

  static int32_t getBatchSize(int32_t total_size)
  {
    int32_t batch_size = 10000;

    if (total_size < 0) {
      RTLOG.error(Loc::current(), "The total of size < 0!");
    } else if (total_size <= 10) {
      batch_size = 5;
    } else {
      batch_size = std::max(5, total_size / 10);
      int32_t factor = static_cast<int32_t>(std::pow(10, getDigitNum(batch_size) - 1));
      batch_size = batch_size / factor * factor;
    }
    return batch_size;
  }

  static bool isDivisible(int32_t dividend, int32_t divisor)
  {
    if (dividend % divisor == 0) {
      return true;
    }
    return false;
  }

  static bool isDivisible(double dividend, double divisor)
  {
    double merchant = dividend / divisor;
    return equalDoubleByError(merchant, static_cast<int32_t>(merchant), RT_ERROR);
  }

  template <typename T, typename Compare>
  static void swapByCMP(T& a, T& b, Compare cmp)
  {
    if (!cmp(a, b)) {
      std::swap(a, b);
    }
  }

  template <typename T>
  static void swapByASC(T& a, T& b)
  {
    swapByCMP(a, b, std::less<T>());
  }

  static int32_t getOffset(const int32_t start, const int32_t end)
  {
    int32_t offset = 0;
    if (start < end) {
      offset = 1;
    } else if (start > end) {
      offset = -1;
    } else {
      RTLOG.warn(Loc::current(), "The step == 0!");
    }
    return offset;
  }

  template <typename T>
  static std::queue<T> initQueue(const T& t)
  {
    std::vector<T> list{t};
    return initQueue(list);
  }

  template <typename T>
  static std::queue<T> initQueue(std::vector<T>& list)
  {
    std::queue<T> queue;
    addListToQueue(queue, list);
    return queue;
  }

  template <typename T>
  static T getFrontAndPop(std::queue<T>& queue)
  {
    T node = queue.front();
    queue.pop();
    return node;
  }

  template <typename T>
  static void addListToQueue(std::queue<T>& queue, std::vector<T>& list)
  {
    for (size_t i = 0; i < list.size(); i++) {
      queue.push(list[i]);
    }
  }

  template <typename T>
  static void reverseList(std::vector<T>& list)
  {
    reverseList(list, 0, static_cast<int32_t>(list.size()) - 1);
  }

  template <typename T>
  static void reverseList(std::vector<T>& list, int32_t start_idx, int32_t end_idx)
  {
    while (start_idx < end_idx) {
      std::swap(list[start_idx], list[end_idx]);
      start_idx++;
      end_idx--;
    }
  }

  template <typename T>
  static bool isNanOrInf(T a)
  {
    return (std::isnan(a) || std::isinf(a));
  }

  static bool equalDoubleByError(double a, double b, double error) { return std::abs(a - b) < error; }

  template <typename T>
  static bool sameSign(T a, T b)
  {
    return std::signbit(a) == std::signbit(b);
  }

  template <typename T>
  static bool diffSign(T a, T b)
  {
    return !sameSign(a, b);
  }

  template <typename Key>
  static bool exist(const std::vector<Key>& vector, const Key& key)
  {
    for (size_t i = 0; i < vector.size(); i++) {
      if (vector[i] == key) {
        return true;
      }
    }
    return false;
  }

  template <typename Key, typename Compare = std::less<Key>>
  static bool exist(const std::set<Key, Compare>& set, const Key& key)
  {
    return (set.find(key) != set.end());
  }

  template <typename Key, typename Hash = std::hash<Key>>
  static bool exist(const std::unordered_set<Key, Hash>& set, const Key& key)
  {
    return (set.find(key) != set.end());
  }

  template <typename Key, typename Value, typename Compare = std::less<Key>>
  static bool exist(const std::map<Key, Value, Compare>& map, const Key& key)
  {
    return (map.find(key) != map.end());
  }

  template <typename Key, typename Value, typename Hash = std::hash<Key>>
  static bool exist(const std::unordered_map<Key, Value, Hash>& map, const Key& key)
  {
    return (map.find(key) != map.end());
  }

  template <typename T = nlohmann::json>
  static T getData(nlohmann::json value, std::vector<std::string> flag_list)
  {
    if (flag_list.empty()) {
      RTLOG.error(Loc::current(), "The flag list is empty!");
    }
    for (size_t i = 0; i < flag_list.size(); i++) {
      value = value[flag_list[i]];
    }
    if (!value.is_null()) {
      return value;
    }
    std::string key;
    for (size_t i = 0; i < flag_list.size(); i++) {
      key += flag_list[i] + ".";
    }
    RTLOG.error(Loc::current(), "The configuration file key '", key, "' does not exist!");
    return value;
  }

  template <typename T = nlohmann::json>
  static T getData(nlohmann::json value, std::string flag)
  {
    if (flag.empty()) {
      RTLOG.error(Loc::current(), "The flag is empty!");
    }
    value = value[flag];
    if (!value.is_null()) {
      return value;
    }
    RTLOG.error(Loc::current(), "The configuration file key '", flag, "' does not exist!");
    return value;
  }

  /**
   * @description: sigmoid
   * ---------------------
   * │ accuracy │ value  │
   * │  0.9999  │ 9.2102 │
   * │  0.999   │ 6.9068 │
   * │  0.99    │ 4.5951 │
   * │  0.9     │ 2.1972 │
   * ---------------------
   *
   * return 1.0 / { 1 + e ^ [ accuracy * (1 - 2 * value / threshold) ] }
   * notice : The closer the <value> is to the <threshold>, the closer the return value is to 1
   *
   */
  static double sigmoid(double value, double threshold)
  {
    if (-0.01 < threshold && threshold < 0) {
      threshold = -0.01;
    } else if (0 <= threshold && threshold < 0.01) {
      threshold = 0.01;
    }
    double result = (1.0 / (1 + std::exp(4.5951 * (1 - 2 * value / threshold))));
    if (std::isnan(result)) {
      RTLOG.error(Loc::current(), "The value is nan!");
    }
    return result;
  }

  template <typename T, typename U>
  static double getRatio(T a, U b)
  {
    return (b > 0 ? static_cast<double>(a) / static_cast<double>(b) : 0.0);
  }

  template <typename T, typename U>
  static std::string getPercentage(T a, U b)
  {
    return getString(formatByTwoDecimalPlaces(getRatio(a, b) * 100), "%");
  }

  static std::ifstream* getInputFileStream(std::string file_path) { return getFileStream<std::ifstream>(file_path); }

  static std::ofstream* getOutputFileStream(std::string file_path) { return getFileStream<std::ofstream>(file_path); }

  template <typename T>
  static T* getFileStream(std::string file_path)
  {
    T* file = new T(file_path);
    if (!file->is_open()) {
      RTLOG.error(Loc::current(), "Failed to open file '", file_path, "'!");
    }
    return file;
  }

  template <typename T>
  static void closeFileStream(T* t)
  {
    if (t != nullptr) {
      t->close();
      delete t;
    }
  }

  template <typename T, typename... Args>
  static std::string getString(T value, Args... args)
  {
    std::stringstream oss;
    pushStream(oss, value, args...);
    std::string string = oss.str();
    oss.clear();
    return string;
  }

  template <typename Stream, typename T, typename... Args>
  static void pushStream(Stream* stream, T t, Args... args)
  {
    pushStream(*stream, t, args...);
  }

  template <typename Stream, typename T, typename... Args>
  static void pushStream(Stream& stream, T t, Args... args)
  {
    stream << t;
    pushStream(stream, args...);
  }

  template <typename Stream, typename T>
  static void pushStream(Stream& stream, T t)
  {
    stream << t;
  }

  static std::string escapeBackslash(std::string a)
  {
    std::regex re(R"(\\)");
    return std::regex_replace(a, re, "");
  }

  static bool isInteger(double a) { return equalDoubleByError(a, static_cast<int32_t>(a), RT_ERROR); }

  static void checkFile(std::string file_path)
  {
    if (!std::filesystem::exists(file_path)) {
      RTLOG.error(Loc::current(), "The file ", file_path, " does not exist!");
    }
  }

  static void createDirByFile(std::string file_path) { createDir(dirname((char*) file_path.c_str())); }

  static void createDir(std::string dir_path)
  {
    if (!std::filesystem::exists(dir_path)) {
      std::error_code system_error;
      if (!std::filesystem::create_directories(dir_path, system_error)) {
        RTLOG.error(Loc::current(), "Failed to create directory '", dir_path, "', system_error:", system_error.message());
      }
    }
  }

  static bool existFile(const std::string& file_path) { return std::filesystem::exists(file_path); }

  static void changePermissions(const std::string& dir_path, std::filesystem::perms permissions)
  {
    std::error_code system_error;
    std::filesystem::permissions(dir_path, permissions);
    if (system_error) {
      RTLOG.error(Loc::current(), "Failed to change permissions for '", dir_path, "', system_error: ", system_error.message());
    }
  }

  static void removeDir(const std::string& dir_path)
  {
    std::error_code system_error;

    // 检查文件夹是否存在
    if (std::filesystem::exists(dir_path, system_error)) {
      // 尝试删除文件夹
      if (!std::filesystem::remove_all(dir_path, system_error)) {
        RTLOG.error(Loc::current(), "Failed to remove directory '", dir_path, "'. Error: ", system_error.message());
      }
    }
  }

  static std::string getFileName(std::string file_path)
  {
    size_t loc = file_path.find_last_of('/');
    if (loc == std::string::npos) {
      return file_path;
    }
    return file_path.substr(loc + 1);
  }

  static std::string getSpaceByTabNum(int32_t tab_num)
  {
    std::string all = "";
    for (int32_t i = 0; i < tab_num; i++) {
      all += "  ";
    }
    return all;
  }

  static std::string getHex(int32_t number)
  {
    std::string result;

    std::stringstream ss;
    ss << std::hex << number;
    ss >> result;
    return result;
  }

  static std::vector<std::string> splitString(std::string a, char tok)
  {
    std::vector<std::string> result_list;

    std::stringstream ss(a);
    std::string result_token;
    while (std::getline(ss, result_token, tok)) {
      if (result_token == "") {
        continue;
      }
      result_list.push_back(result_token);
    }
    return result_list;
  }

  std::string getCompressedBase62(uint64_t origin)
  {
    std::string base = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::string result = "";
    while (origin != 0) {
      result.push_back(base[origin % base.size()]);
      origin /= base.size();
    }
    return result;
  }

  uint64_t getDecompressedBase62(std::string origin)
  {
    std::string base = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::map<char, uint64_t> base_map;
    for (size_t i = 0; i < base.size(); i++) {
      base_map.insert(std::make_pair(base[i], i));
    }

    uint64_t result = 0;
    for (int32_t i = static_cast<int32_t>(origin.size()) - 1; i >= 0; i--) {
      result = result * base.size() + base_map[origin[i]];
    }
    return result;
  }

  std::string getCompressedBase128(uint64_t origin)
  {
    std::string result = "";
    while (origin != 0) {
      result.push_back(static_cast<char>(origin % 128));
      origin /= 128;
    }
    return result;
  }

  uint64_t getDecompressedBase128(std::string origin)
  {
    uint64_t result = 0;
    for (int32_t i = static_cast<int32_t>(origin.size()) - 1; i >= 0; i--) {
      result = result * 128 + static_cast<uint64_t>(origin[i]);
    }
    return result;
  }

  static std::string getTimestamp()
  {
    std::string timestamp_string;

    time_t now = time(nullptr);
    tm* t = localtime(&now);
    char* buffer = new char[32];
    strftime(buffer, 32, "%Y%m%d %H:%M:%S", t);
    timestamp_string = buffer;
    delete[] buffer;
    buffer = nullptr;

    return timestamp_string;
  }

  static std::string formatSec(double sec)
  {
    std::string sec_string;

    int32_t integer_sec = static_cast<int32_t>(std::round(sec));
    int32_t h = integer_sec / 3600;
    int32_t m = (integer_sec % 3600) / 60;
    int32_t s = (integer_sec % 3600) % 60;
    char* buffer = new char[32];
    sprintf(buffer, "%02d:%02d:%02d", h, m, s);
    sec_string = buffer;
    delete[] buffer;
    buffer = nullptr;

    return sec_string;
  }

  static std::string formatByTwoDecimalPlaces(double digit)
  {
    std::string digit_string;

    char* buffer = new char[32];
    sprintf(buffer, "%02.2f", digit);
    digit_string = buffer;
    delete[] buffer;
    buffer = nullptr;

    return digit_string;
  }

  template <typename T>
  static std::set<T> getDifference(std::set<T>& master, std::set<T>& set)
  {
    std::vector<T> master_list(master.begin(), master.end());
    std::vector<T> set_list(set.begin(), set.end());

    std::vector<T> result;
    std::set_difference(master_list.begin(), master_list.end(), set_list.begin(), set_list.end(), std::back_inserter(result));

    return std::set<T>(result.begin(), result.end());
  }

  /**
   * 从多个list中,每个选择一个元素并生成所有可能的组合,非递归
   */
  template <typename T>
  static std::vector<std::vector<T>> getCombList(std::vector<std::vector<T>>& list_list)
  {
    if (list_list.empty()) {
      return {};
    }
    std::vector<std::vector<T>> comb_list = {{}};
    for (const std::vector<T>& list : list_list) {
      std::vector<std::vector<T>> new_comb_list;
      for (const std::vector<T>& comb : comb_list) {
        for (const T& item : list) {
          std::vector<T> new_comb = comb;
          new_comb.push_back(item);
          new_comb_list.push_back(new_comb);
        }
      }
      comb_list = new_comb_list;
    }
    return comb_list;
  }

  /**
   * 从一个list中,生成只包含n个元素所有可能的组合,非递归
   */
  template <typename T>
  static std::vector<std::vector<T>> getCombList(std::vector<T>& list, int32_t n)
  {
    int32_t list_num = static_cast<int32_t>(list.size());
    if (n <= 0 || list_num < n) {
      return {};
    }
    std::vector<std::vector<T>> comb_list;
    for (int32_t mask = 0; mask < (1 << list_num); ++mask) {
      int32_t count = 0;
      for (int32_t i = 0; i < list_num; ++i) {
        if (mask & (1 << i)) {
          count++;
        }
      }
      if (count != n) {
        continue;
      }
      std::vector<T> curr_comb;
      for (int32_t i = 0; i < list_num; ++i) {
        if (mask & (1 << i)) {
          curr_comb.push_back(list[i]);
        }
      }
      comb_list.push_back(curr_comb);
    }
    return comb_list;
  }

#endif

 private:
  // self
  static Utility* _util_instance;

  Utility() = default;
  Utility(const Utility& other) = delete;
  Utility(Utility&& other) = delete;
  ~Utility() = default;
  Utility& operator=(const Utility& other) = delete;
  Utility& operator=(Utility&& other) = delete;
  // function
};

}  // namespace irt
