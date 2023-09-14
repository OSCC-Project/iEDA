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
#include "Logger.hpp"
#include "MTree.hpp"
#include "Monitor.hpp"
#include "Orientation.hpp"
#include "PlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "RTU.hpp"
#include "ScaleAxis.hpp"
#include "ScaleGrid.hpp"
#include "Segment.hpp"
#include "ViaMaster.hpp"
#include "ViaNode.hpp"
#include "json.hpp"

namespace irt {

class RTUtil
{
 public:
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
  static bool isProximal(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return getDirection(start_coord, end_coord) == Direction::kProximal;
  }

  // 判断线段是否为水平线
  static bool isHorizontal(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return getDirection(start_coord, end_coord) == Direction::kHorizontal;
  }

  // 判断线段是否为竖直线
  static bool isVertical(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return getDirection(start_coord, end_coord) == Direction::kVertical;
  }

  // 判断线段是否为斜线
  static bool isOblique(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return getDirection(start_coord, end_coord) == Direction::kOblique;
  }

  // 判断线段是否为直角线
  static bool isRightAngled(const PlanarCoord& start_coord, const PlanarCoord& end_coord)
  {
    return isProximal(start_coord, end_coord) || isHorizontal(start_coord, end_coord) || isVertical(start_coord, end_coord);
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
      LOG_INST.error(Loc::current(), "The coord list is empty!");
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

  static std::vector<Orientation> getOrientationList(const PlanarCoord& start_coord, const PlanarCoord& end_coord,
                                                     Orientation point_orientation = Orientation::kNone)
  {
    std::vector<Orientation> orientation_list;
    orientation_list.push_back(getOrientation(start_coord, PlanarCoord(start_coord.get_x(), end_coord.get_y()), point_orientation));
    orientation_list.push_back(getOrientation(start_coord, PlanarCoord(end_coord.get_x(), start_coord.get_y()), point_orientation));
    return orientation_list;
  }

  // 判断线段方向 从start到end
  static Orientation getOrientation(const LayerCoord& start_coord, const LayerCoord& end_coord,
                                    Orientation point_orientation = Orientation::kNone)
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
        orientation = (start_coord.get_layer_idx() - end_coord.get_layer_idx()) > 0 ? Orientation::kDown : Orientation::kUp;
      } else {
        orientation = Orientation::kOblique;
      }
    }
    return orientation;
  }

  static Orientation getOrientation(const PlanarCoord& start_coord, const PlanarCoord& end_coord,
                                    Orientation point_orientation = Orientation::kNone)
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
      case Orientation::kUp:
        opposite_orientation = Orientation::kDown;
        break;
      case Orientation::kDown:
        opposite_orientation = Orientation::kUp;
        break;
      default:
        LOG_INST.error(Loc::current(), "The orientation is error!");
        break;
    }
    return opposite_orientation;
  }

  static std::vector<Orientation> convertToOrientation(Direction direction)
  {
    std::vector<Orientation> orien_list;
    if (direction == Direction::kHorizontal) {
      orien_list = {Orientation::kEast, Orientation::kWest};
    } else if (direction == Direction::kVertical) {
      orien_list = {Orientation::kSouth, Orientation::kNorth};
    }
    return orien_list;
  }

#endif

#if 1  // 距离线长计算

  // 获得两坐标的曼哈顿距离
  static irt_int getManhattanDistance(LayerCoord start_coord, LayerCoord end_coord)
  {
    return std::abs(start_coord.get_x() - end_coord.get_x()) + std::abs(start_coord.get_y() - end_coord.get_y())
           + std::abs(start_coord.get_layer_idx() - end_coord.get_layer_idx());
  }

  // 获得两坐标的曼哈顿距离
  static irt_int getManhattanDistance(PlanarCoord start_coord, PlanarCoord end_coord)
  {
    return std::abs(start_coord.get_x() - end_coord.get_x()) + std::abs(start_coord.get_y() - end_coord.get_y());
  }

  // 获得线段和点的最短曼哈顿距离
  static irt_int getManhattanDistance(Segment<PlanarCoord>& seg, PlanarCoord& coord)
  {
    irt_int coord_x = coord.get_x();
    irt_int coord_y = coord.get_y();
    irt_int first_coord_x = seg.get_first().get_x();
    irt_int first_coord_y = seg.get_first().get_y();
    irt_int second_coord_x = seg.get_second().get_x();
    irt_int second_coord_y = seg.get_second().get_y();

    if (first_coord_y == second_coord_y && first_coord_x <= coord_x && coord_x <= second_coord_x) {
      return std::abs(first_coord_y - coord_y);
    } else if (first_coord_x == second_coord_x && first_coord_y <= coord_y && coord_y <= second_coord_y) {
      return std::abs(first_coord_x - coord_x);
    } else {
      return std::min(getManhattanDistance(coord, seg.get_first()), getManhattanDistance(coord, seg.get_second()));
    }
  }

  // 获得两个线段的最短曼哈顿距离
  static irt_int getManhattanDistance(Segment<PlanarCoord>& seg1, Segment<PlanarCoord>& seg2)
  {
    if (isIntersection(seg1, seg2)) {
      return 0;
    }
    return std::min(getManhattanDistance(seg1, seg2.get_first()), getManhattanDistance(seg1, seg2.get_second()));
  }

  // 获得两个矩形的欧式距离
  static double getEuclideanDistance(PlanarRect& a, PlanarRect& b)
  {
    irt_int x_spacing = std::max(b.get_lb_x() - a.get_rt_x(), a.get_lb_x() - b.get_rt_x());
    irt_int y_spacing = std::max(b.get_lb_y() - a.get_rt_y(), a.get_lb_y() - b.get_rt_y());

    if (x_spacing > 0 && y_spacing > 0) {
      return std::sqrt((double) (x_spacing * x_spacing + y_spacing * y_spacing));
    } else {
      return std::max(std::max(x_spacing, y_spacing), 0);
    }
  }

  // 获得线段的长度
  static irt_int getManhattanDistance(Segment<PlanarCoord> segment)
  {
    return getManhattanDistance(segment.get_first(), segment.get_second());
  }

  // 获得坐标集合的H树线长
  static irt_int getHTreeLength(std::vector<PlanarCoord>& coord_list)
  {
    std::sort(coord_list.begin(), coord_list.end(), CmpPlanarCoordByXASC());
    PlanarCoord tree_axis_coord = getBalanceCoord(coord_list);
    // 计算H树长度
    irt_int h_tree_length = 0;
    irt_int pre_x = 0;
    irt_int balance_x = tree_axis_coord.get_x();
    irt_int balance_y = tree_axis_coord.get_y();
    irt_int min_x = balance_x;
    irt_int max_x = balance_x;
    irt_int min_y = balance_y;
    irt_int max_y = balance_y;
    for (size_t j = 0; j < coord_list.size(); j++) {
      const PlanarCoord& coord = coord_list[j];
      irt_int x = coord.get_x();
      irt_int y = coord.get_y();
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
  static irt_int getVTreeLength(std::vector<PlanarCoord>& coord_list)
  {
    std::sort(coord_list.begin(), coord_list.end(), CmpPlanarCoordByYASC());
    PlanarCoord tree_axis_coord = getBalanceCoord(coord_list);
    // 计算H树长度
    irt_int v_tree_length = 0;
    irt_int pre_y = 0;
    irt_int balance_x = tree_axis_coord.get_x();
    irt_int balance_y = tree_axis_coord.get_y();
    irt_int min_x = balance_x;
    irt_int max_x = balance_x;
    irt_int min_y = balance_y;
    irt_int max_y = balance_y;
    for (size_t j = 0; j < coord_list.size(); j++) {
      const PlanarCoord& coord = coord_list[j];
      irt_int x = coord.get_x();
      irt_int y = coord.get_y();
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
   *  如果a与b中有膨胀矩形，那么则用isOpenOverlap
   *  如果a与b中都是真实矩形，那么用isClosedOverlap
   *
   *  isOpenOverlap：不考虑边的overlap
   */
  static bool isOpenOverlap(const PlanarRect& a, const PlanarRect& b) { return isOverlap(a, b, false); }

  /**
   *  ！在检测DRC中
   *  如果a与b中有膨胀矩形，那么则用isOpenOverlap
   *  如果a与b中都是真实矩形，那么用isClosedOverlap
   *
   *  isClosedOverlap：考虑边的overlap
   */
  static bool isClosedOverlap(const PlanarRect& a, const PlanarRect& b) { return isOverlap(a, b, true); }

  // 判断两个矩形是否重叠
  static bool isOverlap(const PlanarRect& a, const PlanarRect& b, bool consider_edge = true)
  {
    irt_int x_spacing = std::max(b.get_lb_x() - a.get_rt_x(), a.get_lb_x() - b.get_rt_x());
    irt_int y_spacing = std::max(b.get_lb_y() - a.get_rt_y(), a.get_lb_y() - b.get_rt_y());

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
    irt_int first_x = b.get_first().get_x();
    irt_int second_x = b.get_second().get_x();
    swapASC(first_x, second_x);
    irt_int first_y = b.get_first().get_y();
    irt_int second_y = b.get_second().get_y();
    swapASC(first_y, second_y);

    irt_int x_spacing = std::max(first_x - a.get_rt_x(), a.get_lb_x() - second_x);
    irt_int y_spacing = std::max(first_y - a.get_rt_y(), a.get_lb_y() - second_y);

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
    irt_int first_x = segment.get_first().get_x();
    irt_int first_y = segment.get_first().get_y();
    irt_int first_layer_idx = segment.get_first().get_layer_idx();
    irt_int second_x = segment.get_second().get_x();
    irt_int second_y = segment.get_second().get_y();
    irt_int second_layer_idx = segment.get_second().get_layer_idx();

    swapASC(first_x, second_x);
    swapASC(first_y, second_y);
    swapASC(first_layer_idx, second_layer_idx);

    return (first_x <= coord.get_x() && coord.get_x() <= second_x && first_y <= coord.get_y() && coord.get_y() <= second_y
            && first_layer_idx <= coord.get_layer_idx() && coord.get_layer_idx() <= second_layer_idx);
  }

  // 判断coord是否在rect内，可以选择是否包含边界
  static bool isInside(const PlanarRect& rect, const PlanarCoord& coord, bool boundary = true)
  {
    irt_int coord_x = coord.get_x(), coord_y = coord.get_y();
    irt_int rect_lb_x = rect.get_lb_x(), rect_lb_y = rect.get_lb_y();
    irt_int rect_rt_x = rect.get_rt_x(), rect_rt_y = rect.get_rt_y();
    if (boundary) {
      return (rect_lb_x <= coord_x && coord_x <= rect_rt_x) && (rect_lb_y <= coord_y && coord_y <= rect_rt_y);
    }
    return (rect_lb_x < coord_x && coord_x < rect_rt_x) && (rect_lb_y < coord_y && coord_y < rect_rt_y);
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
  static bool isInside(const PlanarRect& master, const PlanarRect& rect)
  {
    return (isInside(master, rect.get_lb()) && isInside(master, rect.get_rt()));
  }

#endif

#if 1  // 形状有关计算

  // 获得两个线段的交点
  static PlanarCoord getIntersection(PlanarCoord first_coord1, PlanarCoord second_coord1, PlanarCoord first_coord2,
                                     PlanarCoord second_coord2)
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
    auto both_side = [&](double mx, double my, double ux, double uy, double vx, double vy) {
      return (ux - mx) * (vx - mx) <= 0 && (uy - my) * (vy - my) <= 0;
    };
    // 共线处理
    if (cross(cax, cay, cbx, cby) == 0) {
      bool c_in_ab = both_side(cx, cy, ax, ay, bx, by);
      bool d_in_ab = both_side(dx, dy, ax, ay, bx, by);
      if (c_in_ab) {
        return PlanarCoord(static_cast<irt_int>(cx), static_cast<irt_int>(cy));
      }
      if (d_in_ab) {
        return PlanarCoord(static_cast<irt_int>(dx), static_cast<irt_int>(dy));
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
        return PlanarCoord(static_cast<irt_int>(ax), static_cast<irt_int>(ay));
      }
      if (b_in_cd) {
        return PlanarCoord(static_cast<irt_int>(bx), static_cast<irt_int>(by));
      }
      if (c_in_ab) {
        return PlanarCoord(static_cast<irt_int>(cx), static_cast<irt_int>(cy));
      }
      if (d_in_ab) {
        return PlanarCoord(static_cast<irt_int>(dx), static_cast<irt_int>(dy));
      }
      return PlanarCoord(-1, -1);
    }
    // 判断两条线段是否有公共点
    auto intersect = [&]() {
      return cross(acx, acy, abx, aby) * cross(adx, ady, abx, aby) <= 0 && cross(cax, cay, cdx, cdy) * cross(cbx, cby, cdx, cdy) <= 0;
    };
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

    return PlanarCoord(static_cast<irt_int>(rx), static_cast<irt_int>(ry));
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
    intersection_point_list.erase(std::unique(intersection_point_list.begin(), intersection_point_list.end()),
                                  intersection_point_list.end());
    return intersection_point_list;
  }

  // 获得矩形和线段的overlap
  static Segment<PlanarCoord> getOverlap(PlanarRect a, Segment<PlanarCoord> b)
  {
    irt_int first_x = b.get_first().get_x();
    irt_int second_x = b.get_second().get_x();
    swapASC(first_x, second_x);
    irt_int first_y = b.get_first().get_y();
    irt_int second_y = b.get_second().get_y();
    swapASC(first_y, second_y);

    irt_int overlap_lb_x = std::max(first_x, a.get_lb_x());
    irt_int overlap_rt_x = std::min(second_x, a.get_rt_x());
    irt_int overlap_lb_y = std::max(first_y, a.get_lb_y());
    irt_int overlap_rt_y = std::min(second_y, a.get_rt_y());

    if (overlap_lb_x > overlap_rt_x || overlap_lb_y > overlap_rt_y) {
      return Segment<PlanarCoord>(PlanarCoord(), PlanarCoord());
    } else {
      return Segment<PlanarCoord>(PlanarCoord(overlap_lb_x, overlap_lb_y), PlanarCoord(overlap_rt_x, overlap_rt_y));
    }
  }

  // 获得两个矩形的overlap矩形
  static PlanarRect getOverlap(PlanarRect a, PlanarRect b)
  {
    irt_int overlap_lb_x = std::max(a.get_lb_x(), b.get_lb_x());
    irt_int overlap_rt_x = std::min(a.get_rt_x(), b.get_rt_x());
    irt_int overlap_lb_y = std::max(a.get_lb_y(), b.get_lb_y());
    irt_int overlap_rt_y = std::min(a.get_rt_y(), b.get_rt_y());

    if (overlap_lb_x > overlap_rt_x || overlap_lb_y > overlap_rt_y) {
      return PlanarRect(0, 0, 0, 0);
    } else {
      return PlanarRect(overlap_lb_x, overlap_lb_y, overlap_rt_x, overlap_rt_y);
    }
  }

  static std::vector<PlanarRect> getOverlap(const PlanarRect& master, const std::vector<PlanarRect>& rect_list)
  {
    return getOverlap({master}, rect_list);
  }

  static std::vector<PlanarRect> getOverlap(const std::vector<PlanarRect>& master_list, const PlanarRect& rect)
  {
    return getOverlap(master_list, {rect});
  }

  static std::vector<PlanarRect> getOverlap(const std::vector<PlanarRect>& master_list, const std::vector<PlanarRect>& rect_list)
  {
    gtl::polygon_90_set_data<irt_int> master_poly;
    for (const PlanarRect& master : master_list) {
      master_poly += convertToGTLRect(master);
    }
    gtl::polygon_90_set_data<irt_int> rect_poly;
    for (const PlanarRect& rect : rect_list) {
      rect_poly += convertToGTLRect(rect);
    }

    master_poly *= rect_poly;

    std::vector<gtl::rectangle_data<irt_int>> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, master_poly);

    std::vector<PlanarRect> overlap_rect_list;
    for (gtl::rectangle_data<irt_int>& overlap_rect : gtl_rect_list) {
      overlap_rect_list.emplace_back(convertToPlanarRect(overlap_rect));
    }
    return overlap_rect_list;
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

  /**
   *  分开矩形，将master矩形用rect进行分开，并不是求差集
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
   *  如上图所示，输入master和rect
   *  若split方向为horizontal，将得到a和b，可以理解为在横向上分开
   *  若split方向为vertical，将得到c
   */
  static std::vector<PlanarRect> getSplitRectList(const PlanarRect& master, const PlanarRect& rect, Direction split_direction)
  {
    std::vector<PlanarRect> split_rect_list;

    if (split_direction == Direction::kHorizontal) {
      if (master.get_lb_x() < rect.get_lb_x()) {
        PlanarRect split_rect = master;
        split_rect.set_rt_x(rect.get_lb_x());
        split_rect_list.push_back(split_rect);
      }
      if (rect.get_rt_x() < master.get_rt_x()) {
        PlanarRect split_rect = master;
        split_rect.set_lb_x(rect.get_rt_x());
        split_rect_list.push_back(split_rect);
      }
    } else {
      if (master.get_lb_y() < rect.get_lb_y()) {
        PlanarRect split_rect = master;
        split_rect.set_rt_y(rect.get_lb_y());
        split_rect_list.push_back(split_rect);
      }
      if (rect.get_rt_y() < master.get_rt_y()) {
        PlanarRect split_rect = master;
        split_rect.set_lb_y(rect.get_rt_y());
        split_rect_list.push_back(split_rect);
      }
    }
    return split_rect_list;
  }

  /**
   *  切割矩形，将master矩形用rect进行切割，求差集
   *       ┌────────────────────────────────────┐
   *       │ master                             │
   *       │           ┌─────────────────┐      │
   *       └───────────┼─────────────────┼──────┘
   *                   │ rect            │
   *        cut  │     └─────────────────┘  │cut
   *             ▼                          ▼
   *       ┌───────────┐┌────────────────┐┌──────┐
   *       │           ││       c        ││      │
   *       │     a     │└────────────────┘│  b   │
   *       └───────────┘                  └──────┘
   *  如上图所示，输入master和rect，切割后得到a b c三个矩形
   */
  static std::vector<PlanarRect> getCuttingRectList(const PlanarRect& master, const PlanarRect& rect)
  {
    std::vector<PlanarRect> master_list = {master};
    std::vector<PlanarRect> rect_list = {rect};
    return getCuttingRectList(master_list, rect_list);
  }

  static std::vector<PlanarRect> getCuttingRectList(const PlanarRect& master, const std::vector<PlanarRect>& rect_list)
  {
    std::vector<PlanarRect> master_list = {master};
    return getCuttingRectList(master_list, rect_list);
  }

  static std::vector<PlanarRect> getCuttingRectList(const std::vector<PlanarRect>& master_list, const PlanarRect& rect)
  {
    std::vector<PlanarRect> rect_list = {rect};
    return getCuttingRectList(master_list, rect_list);
  }

  static std::vector<PlanarRect> getCuttingRectList(const std::vector<PlanarRect>& master_list, const std::vector<PlanarRect>& rect_list)
  {
    gtl::polygon_90_set_data<irt_int> master_poly;
    for (const PlanarRect& master : master_list) {
      master_poly += convertToGTLRect(master);
    }
    gtl::polygon_90_set_data<irt_int> rect_poly;
    for (const PlanarRect& rect : rect_list) {
      rect_poly += convertToGTLRect(rect);
    }

    master_poly -= rect_poly;

    std::vector<gtl::rectangle_data<irt_int>> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, master_poly);

    std::vector<PlanarRect> cutting_rect_list;
    for (gtl::rectangle_data<irt_int>& gtl_rect : gtl_rect_list) {
      cutting_rect_list.emplace_back(convertToPlanarRect(gtl_rect));
    }
    return cutting_rect_list;
  }

  static std::vector<PlanarRect> getMergeRectList(const std::vector<PlanarRect>& rect_list, Direction direction = Direction::kHorizontal)
  {
    gtl::polygon_90_set_data<irt_int> rect_poly;
    for (const PlanarRect& rect : rect_list) {
      rect_poly += convertToGTLRect(rect);
    }

    std::vector<gtl::rectangle_data<irt_int>> gtl_rect_list;
    if (direction == Direction::kHorizontal) {
      gtl::get_rectangles(gtl_rect_list, rect_poly, gtl::orientation_2d_enum::HORIZONTAL);
    } else if (direction == Direction::kVertical) {
      gtl::get_rectangles(gtl_rect_list, rect_poly, gtl::orientation_2d_enum::VERTICAL);
    } else {
      LOG_INST.error(Loc::current(), "The direction is error!");
    }

    std::vector<PlanarRect> merge_rect_list;
    for (gtl::rectangle_data<irt_int>& gtl_rect : gtl_rect_list) {
      merge_rect_list.emplace_back(convertToPlanarRect(gtl_rect));
    }
    return merge_rect_list;
  }

  static PlanarRect getEnlargedRect(PlanarCoord center_coord, irt_int enlarge_size)
  {
    return getEnlargedRect(center_coord, enlarge_size, enlarge_size, enlarge_size, enlarge_size);
  }

  static PlanarRect getEnlargedRect(PlanarCoord center_coord, irt_int lb_x_minus_offset, irt_int lb_y_minus_offset, irt_int rt_x_add_offset,
                                    irt_int rt_y_add_offset)
  {
    PlanarRect rect(center_coord, center_coord);
    minusOffset(rect.get_lb(), lb_x_minus_offset, lb_y_minus_offset);
    addOffset(rect.get_rt(), rt_x_add_offset, rt_y_add_offset);
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
  static PlanarRect getEnlargedRect(PlanarCoord start_coord, PlanarCoord end_coord, irt_int enlarge_size)
  {
    if (!CmpPlanarCoordByXASC()(start_coord, end_coord)) {
      std::swap(start_coord, end_coord);
    }
    PlanarRect rect(start_coord, end_coord);

    if (isRightAngled(start_coord, end_coord)) {
      rect = getEnlargedRect(rect, enlarge_size);
    } else {
      LOG_INST.error(Loc::current(), "The segment is oblique!");
    }
    return rect;
  }

  // 以enlarge_size扩大线段
  static PlanarRect getEnlargedRect(Segment<PlanarCoord> segment, irt_int enlarge_size)
  {
    return getEnlargedRect(segment.get_first(), segment.get_second(), enlarge_size);
  }

  // 在有最大外边界约束下扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, irt_int enlarge_size, PlanarRect border)
  {
    PlanarRect enalrged_rect = getEnlargedRect(rect, enlarge_size);

    enalrged_rect.set_lb(std::max(enalrged_rect.get_lb_x(), border.get_lb_x()), std::max(enalrged_rect.get_lb_y(), border.get_lb_y()));
    enalrged_rect.set_rt(std::min(enalrged_rect.get_rt_x(), border.get_rt_x()), std::min(enalrged_rect.get_rt_y(), border.get_rt_y()));

    return enalrged_rect;
  }

  // 在有最大外边界约束下扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, irt_int lb_x_minus_offset, irt_int lb_y_minus_offset, irt_int rt_x_add_offset,
                                    irt_int rt_y_add_offset, PlanarRect border)
  {
    PlanarRect enalrged_rect = getEnlargedRect(rect, lb_x_minus_offset, lb_y_minus_offset, rt_x_add_offset, rt_y_add_offset);
    enalrged_rect = getRegularRect(enalrged_rect, border);
    return enalrged_rect;
  }

  static PlanarRect getRegularRect(PlanarRect rect, PlanarRect border)
  {
    PlanarRect regular_rect;
    regular_rect.set_lb(std::max(rect.get_lb_x(), border.get_lb_x()), std::max(rect.get_lb_y(), border.get_lb_y()));
    regular_rect.set_rt(std::min(rect.get_rt_x(), border.get_rt_x()), std::min(rect.get_rt_y(), border.get_rt_y()));
    return regular_rect;
  }

  static LayerRect getRegularRect(LayerRect rect, PlanarRect border)
  {
    LayerRect regular_rect;
    regular_rect.set_lb(std::max(rect.get_lb_x(), border.get_lb_x()), std::max(rect.get_lb_y(), border.get_lb_y()));
    regular_rect.set_rt(std::min(rect.get_rt_x(), border.get_rt_x()), std::min(rect.get_rt_y(), border.get_rt_y()));
    regular_rect.set_layer_idx(rect.get_layer_idx());
    return regular_rect;
  }

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, irt_int enlarge_size)
  {
    return getEnlargedRect(rect, enlarge_size, enlarge_size, enlarge_size, enlarge_size);
  }

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, irt_int lb_x_minus_offset, irt_int lb_y_minus_offset, irt_int rt_x_add_offset,
                                    irt_int rt_y_add_offset)
  {
    minusOffset(rect.get_lb(), lb_x_minus_offset, lb_y_minus_offset);
    addOffset(rect.get_rt(), rt_x_add_offset, rt_y_add_offset);
    return rect;
  }

  static std::vector<PlanarRect> getReducedRect(const PlanarRect& rect, irt_int reduce_size)
  {
    std::vector<PlanarRect> rect_list = {rect};
    return getReducedRect(rect_list, reduce_size);
  }

  static std::vector<PlanarRect> getReducedRect(const PlanarRect& rect, irt_int lb_x_add_offset, irt_int lb_y_add_offset,
                                                irt_int rt_x_minus_offset, irt_int rt_y_minus_offset)
  {
    std::vector<PlanarRect> rect_list = {rect};
    return getReducedRect(rect_list, lb_x_add_offset, lb_y_add_offset, rt_x_minus_offset, rt_y_minus_offset);
  }

  static std::vector<PlanarRect> getReducedRect(const std::vector<PlanarRect>& rect_list, irt_int reduce_size)
  {
    return getReducedRect(rect_list, reduce_size, reduce_size, reduce_size, reduce_size);
  }

  static std::vector<PlanarRect> getReducedRect(const std::vector<PlanarRect>& rect_list, irt_int lb_x_add_offset, irt_int lb_y_add_offset,
                                                irt_int rt_x_minus_offset, irt_int rt_y_minus_offset)
  {
    gtl::polygon_90_set_data<irt_int> rect_poly;
    for (const PlanarRect& rect : rect_list) {
      rect_poly += convertToGTLRect(rect);
    }
    rect_poly.shrink(lb_x_add_offset, rt_x_minus_offset, lb_y_add_offset, rt_y_minus_offset);

    std::vector<gtl::rectangle_data<irt_int>> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, rect_poly);

    std::vector<PlanarRect> reduced_rect_list;
    for (gtl::rectangle_data<irt_int>& gtl_rect : gtl_rect_list) {
      reduced_rect_list.emplace_back(convertToPlanarRect(gtl_rect));
    }
    return reduced_rect_list;
  }

  // 偏移矩形
  static PlanarRect getOffsetRect(PlanarRect rect, PlanarCoord offset_coord)
  {
    irt_int offset_x = offset_coord.get_x();
    irt_int offset_y = offset_coord.get_y();

    addOffset(rect.get_lb(), offset_x, offset_y);
    addOffset(rect.get_rt(), offset_x, offset_y);
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
    irt_int offset_x = origin_coord.get_x();
    irt_int offset_y = origin_coord.get_y();

    minusOffset(rect.get_lb(), offset_x, offset_y);
    minusOffset(rect.get_rt(), offset_x, offset_y);
    return rect;
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
    std::vector<std::vector<TNode<T>*>> level_list = getlevelOrder(root);
    for (size_t i = 0; i < level_list.size(); i++) {
      for (size_t j = 0; j < level_list[i].size(); j++) {
        node_list.push_back(level_list[i][j]);
      }
    }
    return node_list;
  }

  // 以层序遍历获取树的所有结点，可以控制遍历最大深度
  template <typename T>
  static std::vector<std::vector<TNode<T>*>> getlevelOrder(MTree<T>& tree, irt_int max_level = -1)
  {
    return getlevelOrder(tree.get_root(), max_level);
  }

  // 以层序遍历获取树的所有结点，可以控制遍历最大深度
  template <typename T>
  static std::vector<std::vector<TNode<T>*>> getlevelOrder(TNode<T>* root, irt_int max_level = -1)
  {
    if (root == nullptr) {
      return {};
    }
    std::vector<std::vector<TNode<T>*>> level_list;
    std::queue<TNode<T>*> node_queue = initQueue(root);
    std::vector<TNode<T>*> level;
    irt_int level_node_num = 1;
    while (!node_queue.empty()) {
      TNode<T>* node = getFrontAndPop(node_queue);
      level.push_back(node);
      addListToQueue(node_queue, node->get_child_list());
      if (--level_node_num == 0) {
        level_list.push_back(std::move(level));
        level.clear();
        level_node_num = static_cast<irt_int>(node_queue.size());
      }
      if (max_level == -1) {
        continue;
      }
      if (static_cast<irt_int>(level_list.size()) >= max_level) {
        break;
      }
    }
    return level_list;
  }

  // 判断树的深度是否超过max_level
  template <typename T>
  static bool isDeeperThan(MTree<T>& tree, irt_int max_level)
  {
    return isDeeperThan(tree.get_root(), max_level);
  }

  // 判断树的深度是否超过max_level
  template <typename T>
  static bool isDeeperThan(TNode<T>* root, irt_int max_level)
  {
    if (root == nullptr) {
      return max_level < 0;
    }

    std::queue<TNode<T>*> node_queue = initQueue(root);
    irt_int level_num = 0;
    irt_int level_node_num = 1;
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

  // 对树结点内的值进行转换，需要自定义转换函数
  template <typename T, typename U, typename... Args>
  static MTree<U> convertTree(MTree<T>& old_tree, const std::function<U(T&, Args&...)>& convert, Args&... args)
  {
    return MTree<U>(convertTree(old_tree.get_root(), convert, args...));
  }

  // 对树结点内的值进行转换，需要自定义转换函数
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
    std::vector<std::vector<TNode<T>*>> level_list = getlevelOrder(root);
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

  // 通过树根节点和边集构建一棵树，也会消除多个连通分量
  template <typename T>
  static MTree<T> getTreeBySegList(const T& root_value, const std::vector<Segment<T>>& segment_list)
  {
    std::vector<std::pair<bool, Segment<T>>> visited_value_pair_list;
    visited_value_pair_list.reserve(segment_list.size());
    for (size_t i = 0; i < segment_list.size(); i++) {
      visited_value_pair_list.emplace_back(false, segment_list[i]);
    }

    irt_int remain_num = static_cast<irt_int>(visited_value_pair_list.size());

    TNode<T>* root = new TNode(root_value);
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
          TNode<T>* child_node = (value == value1 ? new TNode(value2) : new TNode(value1));
          next_node_list.push_back(child_node);
          node->addChild(child_node);
          visited_value_pair.first = true;
          remain_num--;
        }
      }
      addListToQueue(node_queue, next_node_list);
    }
    if (remain_num > 0) {
      LOG_INST.error(Loc::current(), "There are segments remaining, the tree has not been fully constructed!");
    }
    return MTree<T>(root);
  }

#endif

#if 1  // 与格子图有关的计算
  template <typename T>
  static GridMap<T> sliceMap(GridMap<T>& source_map, EXTPlanarRect& source_rect, EXTPlanarRect& target_rect, T fill_value)
  {
    if (source_rect.getXSize() != source_map.get_x_size() || source_rect.getYSize() != source_map.get_y_size()) {
      LOG_INST.error(Loc::current(), "The rect size is inconsistent with map size!");
    }

    GridMap<T> target_map(target_rect.getXSize(), target_rect.getYSize(), fill_value);

    irt_int offset_x = source_rect.get_grid_lb_x() - target_rect.get_grid_lb_x();
    irt_int offset_y = source_rect.get_grid_lb_y() - target_rect.get_grid_lb_y();

    for (irt_int x = 0; x < source_map.get_x_size(); x++) {
      for (irt_int y = 0; y < source_map.get_y_size(); y++) {
        target_map[x + offset_x][y + offset_y] = source_map[x][y];
      }
    }
    return target_map;
  }
#endif

#if 1  // 与GCell有关的计算

  // 如果与边缘相交，则取内的，不取边缘上
  static PlanarRect getOpenGridRect(const PlanarRect& real_rect, ScaleAxis& gcell_axis)
  {
    std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();

    irt_int real_lb_x = real_rect.get_lb_x();
    irt_int real_rt_x = real_rect.get_rt_x();

    irt_int grid_lb_x = getGridLB(real_lb_x, x_grid_list);
    irt_int grid_rt_x = 0;
    if (real_lb_x == real_rt_x) {
      grid_rt_x = grid_lb_x;
    } else {
      grid_rt_x = getGridRT(real_rt_x, x_grid_list);
    }

    std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();

    irt_int real_lb_y = real_rect.get_lb_y();
    irt_int real_rt_y = real_rect.get_rt_y();

    irt_int grid_lb_y = getGridLB(real_lb_y, y_grid_list);
    irt_int grid_rt_y = 0;
    if (real_lb_y == real_rt_y) {
      grid_rt_y = grid_lb_y;
    } else {
      grid_rt_y = getGridRT(real_rt_y, y_grid_list);
    }

    PlanarRect grid_rect;
    grid_rect.set_lb(grid_lb_x, grid_lb_y);
    grid_rect.set_rt(grid_rt_x, grid_rt_y);
    return grid_rect;
  }

  // 能取到边缘上
  static PlanarRect getClosedGridRect(const PlanarRect& real_rect, ScaleAxis& gcell_axis)
  {
    irt_int min_x = gcell_axis.get_x_grid_list().front().get_start_line();
    irt_int max_x = gcell_axis.get_x_grid_list().back().get_end_line();
    irt_int min_y = gcell_axis.get_y_grid_list().front().get_start_line();
    irt_int max_y = gcell_axis.get_y_grid_list().back().get_end_line();
    PlanarRect new_rect = getEnlargedRect(real_rect, 1, PlanarRect(min_x, min_y, max_x, max_y));
    return getOpenGridRect(new_rect, gcell_axis);
  }

  static PlanarCoord getGridCoord(const PlanarCoord& real_coord, ScaleAxis& gcell_axis, EXTPlanarRect& bounding_box)
  {
    return PlanarCoord((real_coord.get_x() == bounding_box.get_real_rt_x() ? bounding_box.get_grid_rt_x()
                                                                           : getGridLB(real_coord.get_x(), gcell_axis.get_x_grid_list())),
                       (real_coord.get_y() == bounding_box.get_real_rt_y() ? bounding_box.get_grid_rt_y()
                                                                           : getGridLB(real_coord.get_y(), gcell_axis.get_y_grid_list())));
  }

  // [lb , rt)
  static irt_int getGridLB(const irt_int real_coord, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      LOG_INST.error(Loc::current(), "The gcell grid list is empty!");
    }
    if (real_coord < gcell_grid_list.front().get_start_line()) {
      LOG_INST.error(Loc::current(), "The real coord '", real_coord, "' < gcell grid '", gcell_grid_list.front().get_start_line(), "'!");
    }
    if (gcell_grid_list.back().get_end_line() < real_coord) {
      LOG_INST.error(Loc::current(), "The gcell grid '", gcell_grid_list.back().get_end_line(), "' < real coord '", real_coord, "'!");
    }
    // gcell_grid_list 要求有序
    irt_int gcell_grid_idx = 0;
    for (size_t i = 0; i < gcell_grid_list.size(); i++) {
      ScaleGrid& gcell_grid = gcell_grid_list[i];
      irt_int start_line = gcell_grid.get_start_line();
      irt_int step_length = gcell_grid.get_step_length();
      irt_int end_line = gcell_grid.get_end_line();

      if (start_line <= real_coord && real_coord < end_line) {
        double grid_num = static_cast<double>(real_coord - start_line) / step_length;
        gcell_grid_idx += static_cast<irt_int>(grid_num);
        return gcell_grid_idx;
      } else {
        gcell_grid_idx += gcell_grid.get_step_num();
      }
    }
    return gcell_grid_idx - 1;
  }

  // (lb , rt]
  static irt_int getGridRT(const irt_int real_coord, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      LOG_INST.error(Loc::current(), "The gcell grid list is empty!");
    }
    if (real_coord < gcell_grid_list.front().get_start_line()) {
      LOG_INST.error(Loc::current(), "The real coord '", real_coord, "' < gcell grid '", gcell_grid_list.front().get_start_line(), "'!");
    }
    if (gcell_grid_list.back().get_end_line() < real_coord) {
      LOG_INST.error(Loc::current(), "The gcell grid '", gcell_grid_list.back().get_end_line(), "' < real coord '", real_coord, "'!");
    }
    // gcell_grid_list 要求有序
    irt_int gcell_grid_idx = 0;
    for (size_t i = 0; i < gcell_grid_list.size(); i++) {
      ScaleGrid& gcell_grid = gcell_grid_list[i];
      irt_int start_line = gcell_grid.get_start_line();
      irt_int step_length = gcell_grid.get_step_length();
      irt_int end_line = gcell_grid.get_end_line();

      if (start_line < real_coord && real_coord <= end_line) {
        double grid_num = static_cast<double>(real_coord - start_line) / step_length;
        gcell_grid_idx += static_cast<irt_int>(grid_num);
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

  static PlanarRect getRealRect(PlanarRect grid_rect, ScaleAxis& gcell_axis)
  {
    return getRealRect(grid_rect.get_lb(), grid_rect.get_rt(), gcell_axis);
  }

  static PlanarRect getRealRect(PlanarCoord first_coord, PlanarCoord second_coord, ScaleAxis& gcell_axis)
  {
    if (first_coord == second_coord) {
      return getRealRect(first_coord, gcell_axis);
    }

    std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
    std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();

    irt_int first_x = first_coord.get_x();
    irt_int first_y = first_coord.get_y();
    irt_int second_x = second_coord.get_x();
    irt_int second_y = second_coord.get_y();

    swapASC(first_x, second_x);
    swapASC(first_y, second_y);

    return PlanarRect(getRealLB(first_x, x_grid_list), getRealLB(first_y, y_grid_list), getRealRT(second_x, x_grid_list),
                      getRealRT(second_y, y_grid_list));
  }

  static PlanarRect getRealRect(PlanarCoord grid_coord, ScaleAxis& gcell_axis)
  {
    return getRealRect(grid_coord.get_x(), grid_coord.get_y(), gcell_axis);
  }

  static PlanarRect getRealRect(irt_int x, irt_int y, ScaleAxis& gcell_axis)
  {
    std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
    std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();

    return PlanarRect(getRealLB(x, x_grid_list), getRealLB(y, y_grid_list), getRealRT(x, x_grid_list), getRealRT(y, y_grid_list));
  }

  static irt_int getRealLB(irt_int grid, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      LOG_INST.error(Loc::current(), "The gcell grid list is empty!");
    }

    for (size_t i = 0; i < gcell_grid_list.size(); i++) {
      ScaleGrid& gcell_grid = gcell_grid_list[i];

      if (grid < gcell_grid.get_step_num()) {
        return gcell_grid.get_start_line() + grid * gcell_grid.get_step_length();
      } else {
        grid -= gcell_grid.get_step_num();
      }
    }
    LOG_INST.error(Loc::current(), "The grid coord outside grid list!");
    return 0;
  }

  static irt_int getRealRT(irt_int grid, std::vector<ScaleGrid>& gcell_grid_list)
  {
    if (gcell_grid_list.empty()) {
      LOG_INST.error(Loc::current(), "The gcell grid list is empty!");
    }

    for (size_t i = 0; i < gcell_grid_list.size(); i++) {
      ScaleGrid& gcell_grid = gcell_grid_list[i];

      if (grid < gcell_grid.get_step_num()) {
        return gcell_grid.get_start_line() + (grid + 1) * gcell_grid.get_step_length();
      } else {
        grid -= gcell_grid.get_step_num();
      }
    }
    LOG_INST.error(Loc::current(), "The grid coord outside grid list!");
    return 0;
  }

#endif

#if 1  // 与Track有关的计算

  static bool existGrid(const PlanarCoord& real_coord, ScaleAxis& track_axis)
  {
    PlanarRect real_rect(real_coord.get_x(), real_coord.get_y(), real_coord.get_x(), real_coord.get_y());
    return existGrid(real_rect, track_axis);
  }

  static PlanarCoord getGridCoord(const PlanarCoord& real_coord, ScaleAxis& track_axis)
  {
    PlanarRect real_rect(real_coord.get_x(), real_coord.get_y(), real_coord.get_x(), real_coord.get_y());
    PlanarRect grid_rect = getGridRect(real_rect, track_axis);
    return PlanarCoord(grid_rect.get_lb_x(), grid_rect.get_lb_y());
  }

  static bool existGrid(const PlanarRect& real_rect, ScaleAxis& track_axis)
  {
    PlanarRect grid_rect = getGridRect(real_rect, track_axis);
    return (grid_rect.get_lb_x() != -1 && grid_rect.get_lb_y() != -1 && grid_rect.get_rt_x() != -1 && grid_rect.get_rt_y() != -1);
  }

  static PlanarRect getGridRect(const PlanarRect& real_rect, ScaleAxis& track_axis)
  {
    std::vector<irt_int> x_idx_list;
    std::vector<irt_int> x_scale_list = getTrackScaleList(track_axis.get_x_grid_list());
    for (irt_int x_idx = 0; x_idx < static_cast<irt_int>(x_scale_list.size()); x_idx++) {
      irt_int x_scale = x_scale_list[x_idx];
      if (real_rect.get_lb_x() <= x_scale && x_scale <= real_rect.get_rt_x()) {
        x_idx_list.push_back(x_idx);
      }
    }
    std::vector<irt_int> y_idx_list;
    std::vector<irt_int> y_scale_list = getTrackScaleList(track_axis.get_y_grid_list());
    for (irt_int y_idx = 0; y_idx < static_cast<irt_int>(y_scale_list.size()); y_idx++) {
      irt_int y_scale = y_scale_list[y_idx];
      if (real_rect.get_lb_y() <= y_scale && y_scale <= real_rect.get_rt_y()) {
        y_idx_list.push_back(y_idx);
      }
    }

    if (x_idx_list.empty() || y_idx_list.empty()) {
      return PlanarRect(-1, -1, -1, -1);
    }
    return PlanarRect(x_idx_list.front(), y_idx_list.front(), x_idx_list.back(), y_idx_list.back());
  }

  static std::vector<irt_int> getTrackScaleList(std::vector<ScaleGrid>& scale_grid_list)
  {
    std::vector<irt_int> scale_list;
    for (ScaleGrid& scale_grid : scale_grid_list) {
      for (irt_int scale = scale_grid.get_start_line(); scale <= scale_grid.get_end_line(); scale += scale_grid.get_step_length()) {
        scale_list.push_back(scale);
      }
    }
    std::sort(scale_list.begin(), scale_list.end());
    scale_list.erase(std::unique(scale_list.begin(), scale_list.end()), scale_list.end());
    return scale_list;
  }

  /**
   * 若rect包含trackgrid则不处理
   * 否则将其扩大到周围最近的track上
   */
  static PlanarRect getTrackGridRect(PlanarRect& rect, ScaleAxis& track_axis, PlanarRect& border)
  {
    if (existGrid(getOverlap(rect, border), track_axis)) {
      return rect;
    }
    return getNearestTrackRect(rect, track_axis, border);
  }

  /**
   * 将矩形扩大到周围最近的track上
   */
  static PlanarRect getNearestTrackRect(PlanarRect& rect, ScaleAxis& track_axis, PlanarRect& border)
  {
    irt_int lb_x = rect.get_lb_x();
    irt_int rt_x = rect.get_rt_x();
    irt_int lb_y = rect.get_lb_y();
    irt_int rt_y = rect.get_rt_y();
    {
      std::vector<irt_int> x_scale_list = getTrackScaleList(track_axis.get_x_grid_list());
      irt_int x_scale_size = static_cast<irt_int>(x_scale_list.size()) - 1;
      // 将lb_x扩展到左侧最近的x_track上
      int x_index = 0;
      for (; x_index <= x_scale_size; x_index++) {
        if (x_scale_list[x_index] >= lb_x) {
          if (x_index > 0) {
            lb_x = x_scale_list[x_index - 1];
          }
          break;
        }
      }
      if (x_index > x_scale_size) {
        lb_x = x_scale_list.back();
      }
      // 将rt_x扩展到右侧最近的x_track上
      for (; x_index <= x_scale_size; x_index++) {
        if (x_scale_list[x_index] > rt_x) {
          rt_x = x_scale_list[x_index];
          break;
        }
      }
    }
    {
      std::vector<irt_int> y_scale_list = getTrackScaleList(track_axis.get_y_grid_list());
      irt_int y_scale_size = static_cast<irt_int>(y_scale_list.size()) - 1;
      // 将lb_y扩展到左侧最近的y_track上
      int y_index = 0;
      for (; y_index <= y_scale_size; y_index++) {
        if (y_scale_list[y_index] >= lb_y) {
          if (y_index > 0) {
            lb_y = y_scale_list[y_index - 1];
          }
          break;
        }
      }
      if (y_index > y_scale_size) {
        lb_y = y_scale_list.back();
      }
      // 将rt_y扩展到右侧最近的y_track上
      for (; y_index <= y_scale_size; y_index++) {
        if (y_scale_list[y_index] > rt_y) {
          rt_y = y_scale_list[y_index];
          break;
        }
      }
    }
    return getOverlap(PlanarRect(lb_x, lb_y, rt_x, rt_y), border);
  }

  // 计算刻度，包含边界
  static std::vector<irt_int> getClosedScaleList(irt_int begin_line, irt_int end_line, std::vector<ScaleGrid>& scale_grid_list)
  {
    return getScaleList(begin_line, end_line, scale_grid_list, true, true);
  }

  // 计算刻度，不包含边界
  static std::vector<irt_int> getOpenScaleList(irt_int begin_line, irt_int end_line, std::vector<ScaleGrid>& scale_grid_list)
  {
    return getScaleList(begin_line, end_line, scale_grid_list, false, false);
  }

  // 计算刻度，可选择是否包含边界
  static std::vector<irt_int> getScaleList(irt_int begin_line, irt_int end_line, std::vector<ScaleGrid>& track_grid_list, bool lb_boundary,
                                           bool rt_boundary)
  {
    std::vector<irt_int> scale_line_list;
    for (ScaleGrid& track_grid : track_grid_list) {
      std::vector<irt_int> curr_scale_line_list = getScaleList(begin_line, end_line, track_grid, lb_boundary, rt_boundary);
      scale_line_list.insert(scale_line_list.end(), curr_scale_line_list.begin(), curr_scale_line_list.end());
    }
    std::sort(scale_line_list.begin(), scale_line_list.end());
    scale_line_list.erase(std::unique(scale_line_list.begin(), scale_line_list.end()), scale_line_list.end());
    return scale_line_list;
  }

  static std::vector<irt_int> getScaleList(irt_int begin_line, irt_int end_line, ScaleGrid& scale_grid, bool lb_boundary, bool rt_boundary)
  {
    swapASC(begin_line, end_line);

    std::vector<irt_int> scale_line_list;
    irt_int scale_start = scale_grid.get_start_line();
    irt_int scale_pitch = scale_grid.get_step_length();
    irt_int scale_end = scale_grid.get_end_line();

    irt_int overlap_begin_line = std::max(scale_start, begin_line);
    irt_int overlap_end_line = std::min(scale_end, end_line);
    if (overlap_end_line < overlap_begin_line) {
      return scale_line_list;
    }

    irt_int begin_scale_idx = static_cast<irt_int>(std::ceil((overlap_begin_line - scale_start) / 1.0 / scale_pitch));
    irt_int begin_scale_line = scale_start + begin_scale_idx * scale_pitch;
    for (irt_int scale_line = begin_scale_line; scale_line <= overlap_end_line; scale_line += scale_pitch) {
      if ((!lb_boundary && scale_line == begin_line) || (!rt_boundary && scale_line == end_line)) {
        continue;
      }
      scale_line_list.push_back(scale_line);
    }
    return scale_line_list;
  }

  static std::vector<irt_int> getOpenEnlargedScaleList(irt_int begin_line, irt_int end_line, std::vector<ScaleGrid>& scale_grid_list)
  {
    if (scale_grid_list.empty()) {
      LOG_INST.error(Loc::current(), "The scale grid list is empty!");
    }
    begin_line = std::min(begin_line + 1, scale_grid_list.back().get_end_line());
    end_line = std::max(end_line - 1, scale_grid_list.front().get_start_line());
    return getClosedEnlargedScaleList(begin_line, end_line, scale_grid_list);
  }

  // 计算刻度，原有基础上扩大一个scale
  static std::vector<irt_int> getClosedEnlargedScaleList(irt_int begin_line, irt_int end_line, std::vector<ScaleGrid>& scale_grid_list)
  {
    std::vector<irt_int> scale_list;
    std::vector<irt_int> track_scale_list = getTrackScaleList(scale_grid_list);
    for (size_t i = 0; i < track_scale_list.size(); i++) {
      irt_int curr_scale = track_scale_list[i];
      if (curr_scale < begin_line) {
        continue;
      }
      if (curr_scale > end_line) {
        break;
      }
      if (i != 0 && track_scale_list[i - 1] < begin_line) {
        scale_list.push_back(track_scale_list[i - 1]);
      }
      scale_list.push_back(curr_scale);
      if ((i + 1) != track_scale_list.size() && track_scale_list[i + 1] > end_line) {
        scale_list.push_back(track_scale_list[i + 1]);
        break;
      }
    }
    return scale_list;
  }

  // 查找curr_scale左右邻居，若curr_scale在list中，则返回本身；否则返回左右邻居；
  static std::pair<irt_int, irt_int> getAdjacentScale(irt_int curr_scale, std::vector<irt_int>& scale_list)
  {
    irt_int begin_scale = curr_scale;
    irt_int end_scale = curr_scale;
    if (std::find(scale_list.begin(), scale_list.end(), curr_scale) == scale_list.end()) {
      if (curr_scale < scale_list.front()) {
        end_scale = scale_list.front();
      } else if (curr_scale > scale_list.back()) {
        begin_scale = scale_list.back();
      } else {
        auto upper_iter = std::upper_bound(scale_list.begin(), scale_list.end(), curr_scale);
        begin_scale = *(upper_iter - 1);
        end_scale = *upper_iter;
      }
    }
    return {begin_scale, end_scale};
  }
#endif

#if 1  // irt数据结构工具函数

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
      LOG_INST.warning(Loc::current(), "The coord list size is empty!");
    } else {
      irt_int lb_x = INT32_MAX;
      irt_int lb_y = INT32_MAX;
      irt_int rt_x = INT32_MIN;
      irt_int rt_y = INT32_MIN;
      for (size_t i = 0; i < coord_list.size(); i++) {
        const PlanarCoord& coord = coord_list[i];

        lb_x = std::min(lb_x, coord.get_x());
        lb_y = std::min(lb_y, coord.get_y());
        rt_x = std::max(rt_x, coord.get_x());
        rt_y = std::max(rt_y, coord.get_y());
      }
      bounding_box.set_lb(lb_x, lb_y);
      bounding_box.set_rt(rt_x, rt_y);
    }
    return bounding_box;
  }

  // 获得多个矩形的外接矩形
  static PlanarRect getBoundingBox(const std::vector<PlanarRect>& rect_list, PlanarRect border)
  {
    PlanarRect bounding_box = getBoundingBox(rect_list);
    bounding_box.set_lb(std::max(bounding_box.get_lb_x(), border.get_lb_x()), std::max(bounding_box.get_lb_y(), border.get_lb_y()));
    bounding_box.set_rt(std::min(bounding_box.get_rt_x(), border.get_rt_x()), std::min(bounding_box.get_rt_y(), border.get_rt_y()));
    return bounding_box;
  }

  static PlanarRect getBoundingBox(const std::vector<PlanarRect>& rect_list)
  {
    irt_int lb_x = INT32_MAX;
    irt_int lb_y = INT32_MAX;
    irt_int rt_x = INT32_MIN;
    irt_int rt_y = INT32_MIN;

    for (size_t i = 0; i < rect_list.size(); i++) {
      lb_x = std::min(lb_x, rect_list[i].get_lb_x());
      lb_y = std::min(lb_y, rect_list[i].get_lb_y());
      rt_x = std::max(rt_x, rect_list[i].get_rt_x());
      rt_y = std::max(rt_y, rect_list[i].get_rt_y());
    }
    return PlanarRect(lb_x, lb_y, rt_x, rt_y);
  }

  // 获得坐标集合的重心
  static LayerCoord getBalanceCoord(const std::vector<LayerCoord>& coord_list)
  {
    if (coord_list.empty()) {
      return LayerCoord(-1, -1, -1);
    }
    std::vector<irt_int> x_list;
    std::vector<irt_int> y_list;
    std::vector<irt_int> layer_idx_list;
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
    std::vector<irt_int> x_value_list;
    std::vector<irt_int> y_value_list;
    x_value_list.reserve(coord_list.size());
    y_value_list.reserve(coord_list.size());
    for (const PlanarCoord& coord : coord_list) {
      x_value_list.push_back(coord.get_x());
      y_value_list.push_back(coord.get_y());
    }

    return PlanarCoord(getAverage(x_value_list), getAverage(y_value_list));
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
    irt_int coord_x = coord.get_x();
    irt_int coord_y = coord.get_y();
    irt_int first_coord_x = seg.get_first().get_x();
    irt_int first_coord_y = seg.get_first().get_y();
    irt_int second_coord_x = seg.get_second().get_x();
    irt_int second_coord_y = seg.get_second().get_y();

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

  // 获得配置的值
  template <typename T>
  static T getConfigValue(std::map<std::string, std::any>& config_map, const std::string& config_name, const T& defalut_value)
  {
    T value;
    if (exist(config_map, config_name)) {
      value = std::any_cast<T>(config_map[config_name]);
    } else {
      LOG_INST.warning(Loc::current(), "The config '", config_name, "' uses the default value!");
      value = defalut_value;
    }
    return value;
  }

  // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
  static MTree<LayerCoord> getTreeByFullFlow(LayerCoord& root_coord, std::vector<Segment<LayerCoord>>& segment_list,
                                             std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::vector<LayerCoord> candidate_root_coord_list{root_coord};
    return getTreeByFullFlow(candidate_root_coord_list, segment_list, key_coord_pin_map);
  }

  // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
  static MTree<LayerCoord> getTreeByFullFlow(std::vector<LayerCoord>& candidate_root_coord_list,
                                             std::vector<Segment<LayerCoord>>& segment_list,
                                             std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    // 判断是否有斜线段
    if (!passCheckingOblique(segment_list)) {
      LOG_INST.error(Loc::current(), "There are oblique segments in segment_list!");
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
      LOG_INST.error(Loc::current(), "There are oblique segments in tree!");
    }
    // 检查树是否到达所有的关键坐标
    if (!passCheckingReachable(coord_tree, key_coord_pin_map)) {
      LOG_INST.error(Loc::current(), "The key points unreachable!");
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
                                  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
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
      irt_int first_layer_idx = p_segment.get_first().get_layer_idx();
      irt_int second_layer_idx = p_segment.get_second().get_layer_idx();
      swapASC(first_layer_idx, second_layer_idx);
      for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        p_segment_list_temp.emplace_back(LayerCoord(planar_coord, layer_idx), LayerCoord(planar_coord, layer_idx + 1));
      }
    }
    p_segment_list = p_segment_list_temp;

    // 初始化平面切点
    std::map<irt_int, std::set<irt_int>> x_cut_list_map;
    std::map<irt_int, std::set<irt_int>> y_cut_list_map;

    for (Segment<LayerCoord>& h_segment : h_segment_list) {
      LayerCoord& first_coord = h_segment.get_first();
      LayerCoord& second_coord = h_segment.get_second();
      irt_int layer_idx = first_coord.get_layer_idx();

      x_cut_list_map[layer_idx].insert(first_coord.get_x());
      x_cut_list_map[layer_idx].insert(second_coord.get_x());
      y_cut_list_map[layer_idx].insert(first_coord.get_y());
    }
    for (Segment<LayerCoord>& v_segment : v_segment_list) {
      LayerCoord& first_coord = v_segment.get_first();
      LayerCoord& second_coord = v_segment.get_second();
      irt_int layer_idx = first_coord.get_layer_idx();

      y_cut_list_map[layer_idx].insert(first_coord.get_y());
      y_cut_list_map[layer_idx].insert(second_coord.get_y());
      x_cut_list_map[layer_idx].insert(first_coord.get_x());
    }
    for (Segment<LayerCoord>& p_segment : p_segment_list) {
      LayerCoord& first_coord = p_segment.get_first();
      irt_int first_layer_idx = first_coord.get_layer_idx();

      LayerCoord& second_coord = p_segment.get_second();
      irt_int second_layer_idx = second_coord.get_layer_idx();

      x_cut_list_map[first_layer_idx].insert(first_coord.get_x());
      y_cut_list_map[first_layer_idx].insert(first_coord.get_y());
      x_cut_list_map[second_layer_idx].insert(second_coord.get_x());
      y_cut_list_map[second_layer_idx].insert(second_coord.get_y());
    }
    for (auto& [key_coord, pin_idx] : key_coord_pin_map) {
      irt_int layer_idx = key_coord.get_layer_idx();
      x_cut_list_map[layer_idx].insert(key_coord.get_x());
      y_cut_list_map[layer_idx].insert(key_coord.get_y());
    }

    // 切割平面的h线
    std::vector<Segment<LayerCoord>> h_segment_list_temp;
    for (Segment<LayerCoord>& h_segment : h_segment_list) {
      irt_int first_x = h_segment.get_first().get_x();
      irt_int second_x = h_segment.get_second().get_x();
      irt_int y = h_segment.get_first().get_y();
      irt_int layer_idx = h_segment.get_first().get_layer_idx();

      swapASC(first_x, second_x);
      std::vector<irt_int> x_list;
      for (irt_int x_cut : x_cut_list_map[layer_idx]) {
        if (first_x <= x_cut && x_cut <= second_x) {
          x_list.push_back(x_cut);
        }
      }
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
      irt_int first_y = v_segment.get_first().get_y();
      irt_int second_y = v_segment.get_second().get_y();
      irt_int x = v_segment.get_first().get_x();
      irt_int layer_idx = v_segment.get_first().get_layer_idx();

      swapASC(first_y, second_y);
      std::vector<irt_int> y_list;
      for (irt_int y_cut : y_cut_list_map[layer_idx]) {
        if (first_y <= y_cut && y_cut <= second_y) {
          y_list.push_back(y_cut);
        }
      }
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
      RTUtil::merge(segment_list, [](Segment<LayerCoord>& sentry, Segment<LayerCoord>& soldier) {
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
                                 std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
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
      LOG_INST.error(Loc::current(), "The segment_list not covered driving_pin!");
    }
    irt_int max_pin_num = INT32_MIN;
    for (auto& [key_coord, pin_idx_set] : key_coord_pin_map) {
      irt_int pin_num = static_cast<irt_int>(pin_idx_set.size());
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
  static void eraseInvalidNode(MTree<LayerCoord>& coord_tree,
                               std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
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
  static void mergeMiddleNode(MTree<LayerCoord>& coord_tree,
                              std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::vector<TNode<LayerCoord>*> merge_node_list;
    std::map<TNode<LayerCoord>*, TNode<LayerCoord>*> middle_to_start_node_map;
    std::queue<TNode<LayerCoord>*> node_queue = initQueue(coord_tree.get_root());
    while (!node_queue.empty()) {
      TNode<LayerCoord>* node = getFrontAndPop(node_queue);
      addListToQueue(node_queue, node->get_child_list());
      irt_int node_layer_idx = node->value().get_layer_idx();
      PlanarCoord& node_coord = node->value().get_planar_coord();

      for (TNode<LayerCoord>* child_node : node->get_child_list()) {
        irt_int child_node_layer_idx = child_node->value().get_layer_idx();
        PlanarCoord& child_node_coord = child_node->value().get_planar_coord();

        if (node_layer_idx == child_node_layer_idx && node_coord != child_node_coord) {
          middle_to_start_node_map[child_node] = node;
          if (!exist(middle_to_start_node_map, node)) {
            continue;
          }
          TNode<LayerCoord>* parent_node = middle_to_start_node_map[node];
          if (getDirection(parent_node->value().get_planar_coord(), node_coord) == getDirection(node_coord, child_node_coord)
              && node->getChildrenNum() == 1 && !exist(key_coord_pin_map, node->value())) {
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
      irt_int first_layer_idx = coord.get_layer_idx();
      PlanarCoord& second_planar_coord = coord.get_planar_coord();
      irt_int second_layer_idx = coord.get_layer_idx();

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
  static bool passCheckingReachable(MTree<LayerCoord>& coord_tree,
                                    std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
  {
    std::map<irt_int, bool> visited_map;
    for (auto& [key_coord, pin_idx_list] : key_coord_pin_map) {
      for (irt_int pin_idx : pin_idx_list) {
        visited_map[pin_idx] = false;
      }
    }
    for (TNode<LayerCoord>* coord_node : getNodeList(coord_tree)) {
      LayerCoord coord = coord_node->value();
      if (!exist(key_coord_pin_map, coord)) {
        continue;
      }
      for (irt_int pin_idx : key_coord_pin_map[coord]) {
        visited_map[pin_idx] = true;
      }
    }
    for (auto [pin_idx, is_visited] : visited_map) {
      if (is_visited == false) {
        LOG_INST.warning(Loc::current(), "The pin idx ", pin_idx, " unreachable!");
        return false;
      }
    }
    return true;
  }

  /**
   * 返回多级层idx
   * eg. curr_layer_idx: 5
   *     layer_idx_list: [1 2 3 4 5 6 7 8]
   *     return: [4 3 2 1]
   *             [5 6 7 8]
   */
  static std::vector<std::vector<irt_int>> getLevelViaBelowLayerIdxList(irt_int curr_layer_idx,
                                                                        std::vector<irt_int> via_below_layer_idx_list)
  {
    std::vector<std::vector<irt_int>> level_layer_idx_list;

    std::vector<irt_int> down_via_below_layer_idx_list;
    for (irt_int layer_idx : via_below_layer_idx_list) {
      if (layer_idx < curr_layer_idx) {
        down_via_below_layer_idx_list.push_back(layer_idx);
      }
    }
    std::sort(down_via_below_layer_idx_list.begin(), down_via_below_layer_idx_list.end(), std::greater());
    if (!down_via_below_layer_idx_list.empty()) {
      level_layer_idx_list.push_back(down_via_below_layer_idx_list);
    }
    std::vector<irt_int> up_via_below_layer_idx_list;
    for (irt_int layer_idx : via_below_layer_idx_list) {
      if (curr_layer_idx <= layer_idx) {
        up_via_below_layer_idx_list.push_back(layer_idx);
      }
    }
    std::sort(up_via_below_layer_idx_list.begin(), up_via_below_layer_idx_list.end(), std::less());
    if (!up_via_below_layer_idx_list.empty()) {
      level_layer_idx_list.push_back(up_via_below_layer_idx_list);
    }
    return level_layer_idx_list;
  }

  // // 考虑的全部via below层
  // static std::vector<irt_int> getAllViaBelowLayerIdxList(irt_int curr_layer_idx, irt_int bottom_layer_idx, irt_int top_layer_idx)
  // {
  //   if (bottom_layer_idx > top_layer_idx) {
  //     LOG_INST.error(Loc::current(), "The bottom_layer_idx > top_layer_idx!");
  //   }
  //   std::vector<irt_int> layer_idx_list;
  //   if (bottom_layer_idx < curr_layer_idx && curr_layer_idx < top_layer_idx) {
  //     layer_idx_list.push_back(curr_layer_idx - 1);
  //     layer_idx_list.push_back(curr_layer_idx);
  //   } else if (curr_layer_idx <= bottom_layer_idx) {
  //     for (irt_int layer_idx = curr_layer_idx; layer_idx <= std::min(bottom_layer_idx + 1, top_layer_idx); layer_idx++) {
  //       layer_idx_list.push_back(layer_idx);
  //     }
  //   } else if (top_layer_idx <= curr_layer_idx) {
  //     for (irt_int layer_idx = std::max(top_layer_idx - 2, bottom_layer_idx); layer_idx <= (curr_layer_idx - 1); layer_idx++) {
  //       layer_idx_list.push_back(layer_idx);
  //     }
  //   }
  //   std::sort(layer_idx_list.begin(), layer_idx_list.end());
  //   layer_idx_list.erase(std::unique(layer_idx_list.begin(), layer_idx_list.end()), layer_idx_list.end());
  //   return layer_idx_list;
  // }

  // // 考虑的相邻via below层
  // static std::vector<irt_int> getAdjViaBelowLayerIdxList(irt_int curr_layer_idx, irt_int bottom_layer_idx, irt_int top_layer_idx)
  // {
  //   if (bottom_layer_idx > top_layer_idx) {
  //     LOG_INST.error(Loc::current(), "The bottom_layer_idx > top_layer_idx!");
  //   }
  //   std::vector<irt_int> layer_idx_list;
  //   layer_idx_list.push_back(std::max(curr_layer_idx - 1, bottom_layer_idx));
  //   layer_idx_list.push_back(std::min(curr_layer_idx, top_layer_idx - 1));

  //   std::sort(layer_idx_list.begin(), layer_idx_list.end());
  //   layer_idx_list.erase(std::unique(layer_idx_list.begin(), layer_idx_list.end()), layer_idx_list.end());
  //   return layer_idx_list;
  // }

  /**
   * curr_layer_idx在可布线层内
   *    如果不是最高可布线层，向上打孔
   *    是最高可布线层，向下打孔
   *
   * curr_layer_idx在可布线层外
   *    打孔到最近的可布线层
   */
  static std::vector<irt_int> getReservedViaBelowLayerIdxList(irt_int curr_layer_idx, irt_int bottom_layer_idx, irt_int top_layer_idx)
  {
    if (bottom_layer_idx > top_layer_idx) {
      LOG_INST.error(Loc::current(), "The bottom_layer_idx > top_layer_idx!");
    }
    std::vector<irt_int> reserved_via_below_layer_idx_list;
    if (curr_layer_idx <= bottom_layer_idx) {
      for (int layer_idx = curr_layer_idx; layer_idx <= bottom_layer_idx && layer_idx < top_layer_idx; layer_idx++) {
        reserved_via_below_layer_idx_list.push_back(layer_idx);
      }
    } else if (bottom_layer_idx < curr_layer_idx && curr_layer_idx < top_layer_idx) {
      reserved_via_below_layer_idx_list.push_back(curr_layer_idx);
    } else if (top_layer_idx <= curr_layer_idx) {
      for (irt_int layer_idx = std::max(bottom_layer_idx, top_layer_idx - 1); layer_idx < curr_layer_idx; layer_idx++) {
        reserved_via_below_layer_idx_list.push_back(layer_idx);
      }
    }
    return reserved_via_below_layer_idx_list;
  }

  static std::vector<ScaleGrid> makeScaleGridList(std::vector<irt_int>& scale_list)
  {
    std::vector<ScaleGrid> scale_grid_list;

    for (size_t i = 1; i < scale_list.size(); i++) {
      irt_int pre_scale = scale_list[i - 1];
      irt_int curr_scale = scale_list[i];

      ScaleGrid scale_grid;
      scale_grid.set_start_line(pre_scale);
      scale_grid.set_step_length(curr_scale - pre_scale);
      scale_grid.set_step_num(1);
      scale_grid.set_end_line(curr_scale);
      scale_grid_list.push_back(scale_grid);
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

  /**
   * 计算overflow
   */
  static double calcCost(irt_int demand, irt_int supply)
  {
    double cost = 0;
    if (supply != 0) {
      cost = static_cast<double>(demand) / supply;
    } else {
      cost = static_cast<double>(demand);
    }
    cost = std::max(static_cast<double>(0), 1 + std::log10(cost));
    return cost;
  }

#endif

#if 1  // boost数据结构工具函数

  static PlanarRect convertToPlanarRect(gtl::rectangle_data<irt_int>& gtl_rect)
  {
    return PlanarRect(gtl::xl(gtl_rect), gtl::yl(gtl_rect), gtl::xh(gtl_rect), gtl::yh(gtl_rect));
  }

  static PlanarRect convertToPlanarRect(BoostBox& boost_box)
  {
    return PlanarRect(boost_box.min_corner().x(), boost_box.min_corner().y(), boost_box.max_corner().x(), boost_box.max_corner().y());
  }

  static BoostBox convertToBoostBox(const PlanarRect& rect)
  {
    return BoostBox(BoostPoint(rect.get_lb_x(), rect.get_lb_y()), BoostPoint(rect.get_rt_x(), rect.get_rt_y()));
  }

  static BoostBox convertToBoostBox(gtl::rectangle_data<irt_int>& gtl_rect)
  {
    return BoostBox(BoostPoint(gtl::xl(gtl_rect), gtl::yl(gtl_rect)), BoostPoint(gtl::xh(gtl_rect), gtl::yh(gtl_rect)));
  }

  static gtl::rectangle_data<irt_int> convertToGTLRect(const PlanarRect& rect)
  {
    return gtl::rectangle_data<irt_int>(rect.get_lb_x(), rect.get_lb_y(), rect.get_rt_x(), rect.get_rt_y());
  }

  static gtl::rectangle_data<irt_int> convertToGTLRect(BoostBox& boost_box)
  {
    return gtl::rectangle_data<irt_int>(boost_box.min_corner().x(), boost_box.min_corner().y(), boost_box.max_corner().x(),
                                        boost_box.max_corner().y());
  }

  static irt_int getLength(BoostBox& a) { return std::abs(a.max_corner().x() - a.min_corner().x()); }

  static irt_int getWidth(BoostBox& a) { return std::abs(a.max_corner().y() - a.min_corner().y()); }

  static PlanarCoord getCenter(BoostBox& a)
  {
    irt_int center_x = std::abs(a.max_corner().x() + a.min_corner().x()) / 2;
    irt_int center_y = std::abs(a.max_corner().y() + a.min_corner().y()) / 2;
    return PlanarCoord(center_x, center_y);
  }

  static BoostBox enlargeBoostBox(BoostBox& a, irt_int enlarge_size)
  {
    return BoostBox(BoostPoint(a.min_corner().x() - enlarge_size, a.min_corner().y() - enlarge_size),
                    BoostPoint(a.max_corner().x() + enlarge_size, a.max_corner().y() + enlarge_size));
  }

  static void offsetBoostBox(BoostBox& boost_box, PlanarCoord& coord)
  {
    boost_box.min_corner().set<0>(boost_box.min_corner().x() + coord.get_x());
    boost_box.min_corner().set<1>(boost_box.min_corner().y() + coord.get_y());

    boost_box.max_corner().set<0>(boost_box.max_corner().x() + coord.get_x());
    boost_box.max_corner().set<1>(boost_box.max_corner().y() + coord.get_y());
  }

  static bool isOverlap(BoostBox& a, BoostBox& b, bool consider_edge = true)
  {
    irt_int a_lb_x = a.min_corner().x(), a_lb_y = a.min_corner().y();
    irt_int a_rt_x = a.max_corner().x(), a_rt_y = a.max_corner().y();

    irt_int b_lb_x = b.min_corner().x(), b_lb_y = b.min_corner().y();
    irt_int b_rt_x = b.max_corner().x(), b_rt_y = b.max_corner().y();

    irt_int x_spacing = std::max(b_lb_x - a_rt_x, a_lb_x - b_rt_x);
    irt_int y_spacing = std::max(b_lb_y - a_rt_y, a_lb_y - b_rt_y);

    if (x_spacing == 0 || y_spacing == 0) {
      return consider_edge;
    } else {
      return (x_spacing < 0 && y_spacing < 0);
    }
  }

  static BoostBox getOverlap(BoostBox& a, BoostBox& b)
  {
    irt_int overlap_lb_x = std::max(a.min_corner().x(), b.min_corner().x());
    irt_int overlap_lb_y = std::max(a.min_corner().y(), b.min_corner().y());
    irt_int overlap_rt_x = std::min(a.max_corner().x(), b.max_corner().x());
    irt_int overlap_rt_y = std::min(a.max_corner().y(), b.max_corner().y());

    if (overlap_lb_x > overlap_rt_x || overlap_lb_y > overlap_rt_y) {
      return BoostBox(BoostPoint(0, 0), BoostPoint(0, 0));
    } else {
      return BoostBox(BoostPoint(overlap_lb_x, overlap_lb_y), BoostPoint(overlap_rt_x, overlap_rt_y));
    }
  }

  static bool isHorizontal(BoostBox a) { return (a.max_corner().x() - a.min_corner().x()) >= (a.max_corner().y() - a.min_corner().y()); }

  static irt_int getDiagonalLength(BoostBox& a)
  {
    irt_int length = getLength(a);
    irt_int width = getWidth(a);
    return (irt_int) std::sqrt((length * length + width * width));
  }

  static irt_int getEuclideanDistance(BoostBox& a, BoostBox& b)
  {
    irt_int a_lb_x = a.min_corner().x(), a_lb_y = a.min_corner().y();
    irt_int a_rt_x = a.max_corner().x(), a_rt_y = a.max_corner().y();

    irt_int b_lb_x = b.min_corner().x(), b_lb_y = b.min_corner().y();
    irt_int b_rt_x = b.max_corner().x(), b_rt_y = b.max_corner().y();

    irt_int x_spacing = std::max(b_lb_x - a_rt_x, a_lb_x - b_rt_x);
    irt_int y_spacing = std::max(b_lb_y - a_rt_y, a_lb_y - b_rt_y);

    if (x_spacing > 0 && y_spacing > 0) {
      return (irt_int) std::sqrt((x_spacing * x_spacing + y_spacing * y_spacing));
    } else {
      return std::max(std::max(x_spacing, y_spacing), 0);
    }
  }

#endif

#if 1  // std数据结构工具函数

  template <typename Key, typename Value>
  static Value getValueByAny(std::map<Key, std::any>& map, const Key& key, const Value& defalut_value)
  {
    Value value;
    if (exist(map, key)) {
      value = std::any_cast<Value>(map[key]);
    } else {
      value = defalut_value;
    }
    return value;
  }

  template <typename Key, typename Value>
  static Value getValue(std::map<Key, Value>& map, const Key& key, const Value& defalut_value)
  {
    Value value;
    if (exist(map, key)) {
      value = map[key];
    } else {
      value = defalut_value;
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
    average /= static_cast<irt_int>(value_list.size());
    if constexpr (std::is_same<T, irt_int>::value) {
      average = std::round(average);
    }
    return T(average);
  }

  template <typename T>
  static void merge(std::vector<T>& list, const std::function<bool(T&, T&)>& mergeIf)
  {
    merge(list, mergeIf);
  }

  template <typename T, typename MergeIf>
  static void merge(std::vector<T>& list, MergeIf mergeIf)
  {
    size_t save_id = 0;
    size_t sentry_id = 0;
    size_t soldier_id = sentry_id + 1;
    while (sentry_id < list.size()) {
      T& sentry = list[sentry_id];
      while (soldier_id < list.size()) {
        T& soldier = list[soldier_id];
        if (!mergeIf(sentry, soldier)) {
          break;
        }
        ++soldier_id;
      }
      list[save_id] = std::move(sentry);
      ++save_id;
      if (!(soldier_id < list.size())) {
        break;
      }
      sentry_id = soldier_id;
      soldier_id = sentry_id + 1;
    }
    list.erase(list.begin() + save_id, list.end());
  }

  template <typename T>
  static bool isDifferentSign(T a, T b)
  {
    return a & b ? (a ^ b) < 0 : false;
  }

  static irt_int getFirstDigit(irt_int n)
  {
    n = n >= 100000000 ? (n / 100000000) : n;
    n = n >= 10000 ? (n / 10000) : n;
    n = n >= 100 ? (n / 100) : n;
    n = n >= 10 ? (n / 10) : n;
    return n;
  }

  static irt_int getDigitNum(irt_int n)
  {
    irt_int count = 0;

    while (n != 0) {
      n /= 10;
      count++;
    }
    return count;
  }

  static irt_int getBatchSize(size_t total_size) { return getBatchSize(static_cast<irt_int>(total_size)); }

  static irt_int getBatchSize(irt_int total_size)
  {
    irt_int batch_size = 10000;

    if (total_size < 0) {
      LOG_INST.error(Loc::current(), "The total of size < 0!");
    } else if (total_size <= 10) {
      batch_size = 5;
    } else if (total_size < 100000) {
      batch_size = std::max(5, total_size / 10);
      irt_int factor = static_cast<irt_int>(std::pow(10, getDigitNum(batch_size) - 1));
      batch_size = batch_size / factor * factor;
    }
    return batch_size;
  }

  static bool isDivisible(irt_int dividend, irt_int divisor)
  {
    if (dividend % divisor == 0) {
      return true;
    }
    return false;
  }

  static bool isDivisible(double dividend, double divisor)
  {
    double merchant = dividend / divisor;
    return equalDoubleByError(merchant, static_cast<irt_int>(merchant), DBL_ERROR);
  }

  template <typename T, typename Compare>
  static void swapByCMP(T& a, T& b, Compare cmp)
  {
    if (!cmp(a, b)) {
      std::swap(a, b);
    }
  }

  template <typename T>
  static void swapASC(T& a, T& b)
  {
    swapByCMP(a, b, std::less<T>());
  }

  static void addOffset(PlanarCoord& coord, PlanarCoord& offset_coord) { addOffset(coord, offset_coord.get_x(), offset_coord.get_y()); }

  static void addOffset(PlanarCoord& coord, irt_int x_offset, irt_int y_offset)
  {
    coord.set_x(coord.get_x() + x_offset);
    coord.set_y(coord.get_y() + y_offset);
  }

  static void minusOffset(PlanarCoord& coord, PlanarCoord& offset_coord) { minusOffset(coord, offset_coord.get_x(), offset_coord.get_y()); }

  static void minusOffset(PlanarCoord& coord, irt_int x_offset, irt_int y_offset)
  {
    coord.set_x((coord.get_x() - x_offset) < 0 ? 0 : (coord.get_x() - x_offset));
    coord.set_y((coord.get_y() - y_offset) < 0 ? 0 : (coord.get_y() - y_offset));
  }

  static irt_int getOffset(const irt_int start, const irt_int end)
  {
    irt_int offset = 0;
    if (start < end) {
      offset = 1;
    } else if (start > end) {
      offset = -1;
    } else {
      LOG_INST.warning(Loc::current(), "The step == 0!");
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

  // 将队列中的元素输入到list中，参数不能使用引用
  template <typename T>
  static std::vector<T> getListByQueue(std::queue<T> queue)
  {
    std::vector<T> list;
    while (!queue.empty()) {
      list.push_back(getFrontAndPop(queue));
    }
    return list;
  }

  template <typename T>
  static void reverseList(std::vector<T>& list)
  {
    reverseList(list, 0, static_cast<irt_int>(list.size()) - 1);
  }

  template <typename T>
  static void reverseList(std::vector<T>& list, irt_int start_idx, irt_int end_idx)
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

  // 保留小数点后前n位不为0的数，并进行四舍五入
  template <typename T>
  static T retainPlaces(T a, irt_int n = 1)
  {
    if (isInteger(a)) {
      return a;
    }

    if (n <= 0) {
      return static_cast<T>(std::round(a));
    }

    T value = a;
    irt_int digit_num = 0;
    while (value < 1) {
      value *= 10;
      ++digit_num;
    }

    value *= static_cast<T>(std::pow(10, n - 1));
    return static_cast<T>(std::round(value) / std::pow(10, digit_num + n - 1));
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
      LOG_INST.error(Loc::current(), "The flag list is empty!");
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
    LOG_INST.error(Loc::current(), "The configuration file key '", key, "' does not exist!");
    return value;
  }

  template <typename T = nlohmann::json>
  static T getData(nlohmann::json value, std::string flag)
  {
    if (flag.empty()) {
      LOG_INST.error(Loc::current(), "The flag is empty!");
    }
    value = value[flag];
    if (!value.is_null()) {
      return value;
    }
    LOG_INST.error(Loc::current(), "The configuration file key '", flag, "' does not exist!");
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
      LOG_INST.error(Loc::current(), "The value is nan!");
    }
    return result;
  }

  template <typename T>
  static double getRatio(T a, T b)
  {
    return (b > 0 ? static_cast<double>(a) / static_cast<double>(b) : 0.0);
  }

  template <typename T>
  static double getPercentage(T a, T b)
  {
    return getRatio(a, b) * 100.0;
  }

  static std::ifstream* getInputFileStream(std::string file_path) { return getFileStream<std::ifstream>(file_path); }

  static std::ofstream* getOutputFileStream(std::string file_path) { return getFileStream<std::ofstream>(file_path); }

  template <typename T>
  static T* getFileStream(std::string file_path)
  {
    T* file = new T(file_path);
    if (!file->is_open()) {
      LOG_INST.error(Loc::current(), "Failed to open file '", file_path, "'!");
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

  static bool isInteger(double a) { return equalDoubleByError(a, static_cast<irt_int>(a), DBL_ERROR); }

  static void checkFile(std::string file_path)
  {
    if (0 != access(file_path.c_str(), F_OK)) {
      LOG_INST.error(Loc::current(), "The file ", file_path, " does not exist!");
    }
  }

  static void createDirByFile(std::string file_path) { createDir(dirname((char*) file_path.c_str())); }

  static void createDir(std::string dir_path)
  {
    if (0 != access(dir_path.c_str(), F_OK)) {
      LOG_INST.info(Loc::current(), "Create directory ", dir_path);
      std::error_code system_error;
      if (!std::filesystem::create_directories(dir_path, system_error)) {
        if (!std::filesystem::exists(dir_path)) {
          LOG_INST.error(Loc::current(), "Failed to create directory '", dir_path, "', system_error:", system_error.message());
        }
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

  static std::string getSpaceByTabNum(irt_int tab_num)
  {
    std::string all = "";
    for (irt_int i = 0; i < tab_num; i++) {
      all += "    ";
    }
    return all;
  }

  static std::string getHex(irt_int number)
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
    while (getline(ss, result_token, tok)) {
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
    for (irt_int i = static_cast<irt_int>(origin.size()) - 1; i >= 0; i--) {
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
    for (irt_int i = static_cast<irt_int>(origin.size()) - 1; i >= 0; i--) {
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

    irt_int integer_sec = static_cast<irt_int>(sec);
    irt_int h = integer_sec / 3600;
    irt_int m = (integer_sec % 3600) / 60;
    irt_int s = (integer_sec % 3600) % 60;
    char* buffer = new char[32];
    sprintf(buffer, "%02d:%02d:%02d", h, m, s);
    sec_string = buffer;
    delete[] buffer;
    buffer = nullptr;

    return sec_string;
  }

  static std::string formatMem(double mem)
  {
    std::string mem_string;

    char* buffer = new char[32];
    sprintf(buffer, "%02.2f", mem);
    mem_string = buffer;
    delete[] buffer;
    buffer = nullptr;

    return mem_string;
  }

  template <typename T>
  static std::set<T> getDifference(std::set<T>& master, std::set<T>& set)
  {
    std::vector<T> master_list;
    master_list.assign(master.begin(), master.end());
    std::vector<T> set_list;
    set_list.assign(set.begin(), set.end());

    std::vector<T> result;
    std::set_difference(master_list.begin(), master_list.end(), set_list.begin(), set_list.end(), std::back_inserter(result));

    return std::set<T>(result.begin(), result.end());
  }

#endif

#if 1  // report数据结构

  template <typename T>
  static GridMap<double> getRangeRatioMap(std::vector<T> value_list)
  {
    GridMap<double> value_map;
    std::map<T, irt_int> range_num_map = getRangeNumMap(value_list);
    value_map.init(4, static_cast<irt_int>(range_num_map.size()));

    irt_int idx = 0;
    T range = getScaleRange(value_list);
    for (auto& [left, num] : range_num_map) {
      double ratio_value = num / 1.0 / static_cast<T>(value_list.size());
      double ratio = retainPlaces(ratio_value, 3);
      T right = left + range;
      if (equalDoubleByError(right, 0, DBL_ERROR)) {
        right = 0;
      }
      value_map[0][idx] = left;
      value_map[1][idx] = right;
      value_map[2][idx] = num;
      value_map[3][idx] = ratio * 100;
      ++idx;
    }
    return value_map;
  }

  template <typename T>
  static GridMap<std::string> getRangeRatioMap(std::vector<T> value_list, std::vector<T> scale_list)
  {
    // 数据按区间分类
    T range = getScaleRange(value_list);

    T max_value = INT32_MIN;
    T min_value = INT32_MAX;
    for (T& value : value_list) {
      max_value = std::max(max_value, value);
      min_value = std::min(min_value, value);
    }

    std::vector<T> total_scale_list(scale_list.begin(), scale_list.end());
    for (T scale = min_value; scale <= max_value; scale += range) {
      total_scale_list.push_back(scale);
    }
    total_scale_list.push_back(max_value);
    std::sort(total_scale_list.begin(), total_scale_list.end());
    merge(total_scale_list, [](T a, T b) { return equalDoubleByError(a, b, DBL_ERROR); });
    total_scale_list.erase(std::unique(total_scale_list.begin(), total_scale_list.end()), total_scale_list.end());

    std::vector<std::pair<T, T>> scale_range_list;
    for (size_t i = 1; i < total_scale_list.size(); i++) {
      scale_range_list.emplace_back(total_scale_list[i - 1], total_scale_list[i]);
    }

    std::map<std::pair<T, T>, irt_int> range_num_map;
    for (T& value : value_list) {
      for (size_t i = 0; i < scale_range_list.size(); i++) {
        T left = scale_range_list[i].first;
        T right = scale_range_list[i].second;
        if (left <= value && value < right) {
          range_num_map[scale_range_list[i]] += 1;
          break;
        }
        if (i + 1 == scale_range_list.size() && equalDoubleByError(value, right, DBL_ERROR)) {
          range_num_map[scale_range_list[i]] += 1;
          break;
        }
      }
    }

    // 生成字符串信息
    GridMap<std::string> value_map;
    value_map.init(2, static_cast<irt_int>(range_num_map.size()));

    irt_int idx = 0;
    for (auto& [range, num] : range_num_map) {
      double ratio_value = num / 1.0 / static_cast<T>(value_list.size());
      double ratio = retainPlaces(ratio_value, 3);

      std::string range_str = getString("[", range.first, ",", range.second);
      if (idx == static_cast<irt_int>(range_num_map.size()) - 1) {
        range_str += "]";
      } else {
        range_str += ")";
      }

      std::string ratio_str = RTUtil::getString(num, "(", ratio * 100, "%)");

      value_map[0][idx] = range_str;
      value_map[1][idx] = ratio_str;
      ++idx;
    }
    return value_map;
  }

  template <typename T>
  static std::map<irt_int, std::map<std::pair<T, T>, irt_int>> getLayerRangeNumMap(std::map<irt_int, std::vector<T>> layer_value_map,
                                                                                   std::vector<T> scale_list)
  {
    std::map<irt_int, std::map<std::pair<T, T>, irt_int>> layer_range_num_map;

    // 计算数据区间间距
    std::vector<T> total_value_list;
    for (auto& [layer_idx, value_list] : layer_value_map) {
      total_value_list.insert(total_value_list.end(), value_list.begin(), value_list.end());
    }
    if (total_value_list.empty()) {
      return layer_range_num_map;
    }

    T range = getScaleRange(total_value_list);

    // 生成数据区间
    T min_value = INT32_MAX;
    T max_value = INT32_MIN;
    for (auto& [layer_idx, value_list] : layer_value_map) {
      for (T value : value_list) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
      }
    }
    if (!scale_list.empty()) {
      std::sort(scale_list.begin(), scale_list.end());
      min_value = std::max(min_value, scale_list.front());
      max_value = std::max(max_value, scale_list.back());
    }

    std::vector<T> total_scale_list(scale_list.begin(), scale_list.end());
    for (T scale = min_value; equalDoubleByError(scale, max_value, 0.001) || scale < max_value; scale += range) {
      total_scale_list.push_back(scale);
    }
    std::sort(total_scale_list.begin(), total_scale_list.end());
    merge(total_scale_list, [](T a, T b) { return equalDoubleByError(a, b, 0.001); });

    // 生成区间
    std::vector<std::pair<T, T>> scale_range_list;
    if (total_scale_list.size() == 1) {  // 当锚点只有一个且所有元素都比锚点小时，生成锚点闭区间
      scale_range_list.emplace_back(total_scale_list.front(), total_scale_list.front());
    } else if (total_scale_list.size() > 1) {
      for (size_t i = 1; i < total_scale_list.size(); i++) {
        scale_range_list.emplace_back(total_scale_list[i - 1], total_scale_list[i]);
      }
    }
    // 生成各个区间的数据
    for (auto& [layer_idx, value_list] : layer_value_map) {
      std::map<std::pair<T, T>, irt_int>& range_num_map = layer_range_num_map[layer_idx];
      for (auto& scale_range : scale_range_list) {
        range_num_map[scale_range] = 0;
      }
    }

    for (auto& [layer_idx, value_list] : layer_value_map) {
      for (T& value : value_list) {
        for (size_t i = 0; i < scale_range_list.size(); i++) {
          T left = scale_range_list[i].first;
          T right = scale_range_list[i].second;
          if (left <= value && value < right) {
            ++layer_range_num_map[layer_idx][scale_range_list[i]];
            break;
          }
          if (i + 1 == scale_range_list.size() && equalDoubleByError(value, right, 0.001)) {
            ++layer_range_num_map[layer_idx][scale_range_list[i]];
            break;
          }
        }
      }
    }
    return layer_range_num_map;
  }

  template <typename T>
  static std::map<T, irt_int> getRangeNumMap(std::vector<T> value_list)
  {
    std::map<T, irt_int> scale_num_map;
    T range = getScaleRange(value_list);

    T max_value = INT32_MIN;
    T min_value = INT32_MAX;
    for (T& value : value_list) {
      max_value = std::max(max_value, value);
      min_value = std::min(min_value, value);
    }

    for (T left = min_value; left < max_value; left += range) {
      if (equalDoubleByError(left, max_value, DBL_ERROR)) {
        break;
      }
      scale_num_map[left] = 0;
    }

    for (T& value : value_list) {
      for (auto iter = scale_num_map.rbegin(); iter != scale_num_map.rend(); iter++) {
        if (value >= iter->first) {
          (iter->second)++;
          break;
        }
      }
    }
    return scale_num_map;
  }

  template <typename T>
  static T getScaleRange(std::vector<T> value_list, irt_int digit = 1)
  {
    T max_value = INT32_MIN;
    T min_value = INT32_MAX;
    for (T& value : value_list) {
      max_value = std::max(max_value, value);
      min_value = std::min(min_value, value);
    }
    T range = std::max(0.001, (max_value - min_value) / 10);
    return retainPlaces(range, digit);
  }

  static void check(std::vector<PlanarRect>& rect_list1, std::vector<PlanarRect>& rect_list2)
  {
    if (rect_list1.size() != rect_list2.size()) {
      LOG_INST.error(Loc::current(), "number is different!");
    }

    double area1 = 0;
    for (PlanarRect& rect : rect_list1) {
      area1 += rect.getArea();
    }

    double area2 = 0;
    for (PlanarRect& rect : rect_list2) {
      area2 += rect.getArea();
    }

    if (area1 != area2) {
      LOG_INST.error(Loc::current(), "area is different!");
    }
  }

#endif
};  // namespace irt

}  // namespace irt
