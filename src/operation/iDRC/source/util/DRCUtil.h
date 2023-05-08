#ifndef IDRC_SRC_UTIL_DRC_H_
#define IDRC_SRC_UTIL_DRC_H_

#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <vector>

#include "BoostType.h"
#include "DrcDesign.h"
#include "DrcEnum.h"

namespace idrc {

class DRCUtil
{
 public:
  // function for rect
  //------------------------------------------------------------
  //------------------------------------------------------------

  static std::vector<DrcCoordinate<int>> getRectangleCoordinateList(const DrcRectangle<int>& rect)
  {
    std::vector<DrcCoordinate<int>> coordinate_list;
    coordinate_list.emplace_back(DrcCoordinate<int>(rect.get_lb_x(), rect.get_lb_y()));
    coordinate_list.emplace_back(DrcCoordinate<int>(rect.get_lb_x(), rect.get_rt_y()));
    coordinate_list.emplace_back(DrcCoordinate<int>(rect.get_rt_x(), rect.get_rt_y()));
    coordinate_list.emplace_back(DrcCoordinate<int>(rect.get_rt_x(), rect.get_lb_y()));
    return coordinate_list;
  }

  static bool isRectContainCoordinate(const DrcRect* rect, const DrcCoordinate<int>& coordinate, bool edge = false)
  {
    int x = coordinate.get_x();
    int y = coordinate.get_y();
    if (!edge) {
      return x > rect->get_left() && x < rect->get_right() && y > rect->get_bottom() && y < rect->get_top();
    }
    return x >= rect->get_left() && x <= rect->get_right() && y >= rect->get_bottom() && y <= rect->get_top();
  }

  static bool intersection(const DrcRect* rect1, const DrcRect* rect2, bool edge = false)
  {
    if (!edge) {
      return !(rect1->get_left() >= rect2->get_right() || rect1->get_bottom() >= rect2->get_top() || rect1->get_right() <= rect2->get_left()
               || rect1->get_top() <= rect2->get_bottom());
    } else {
      return !(rect1->get_left() > rect2->get_right() || rect1->get_bottom() > rect2->get_top() || rect1->get_right() < rect2->get_left()
               || rect1->get_top() < rect2->get_bottom());
    }
  }

  static bool intersection(const DrcEdge* check_edge, const RTreeBox& box, bool edge = false)
  {
    if (!edge) {
      return !(check_edge->get_min_x() >= box.max_corner().x() || check_edge->get_min_y() >= box.max_corner().y()
               || check_edge->get_max_x() <= box.min_corner().x() || check_edge->get_max_y() <= box.min_corner().y());
    } else {
      return !(check_edge->get_min_x() > box.max_corner().x() || check_edge->get_min_y() > box.max_corner().y()
               || check_edge->get_max_x() < box.min_corner().x() || check_edge->get_max_y() < box.min_corner().y());
    }
  }

  static bool intersection(const RTreeBox* rect1, const DrcRect* rect2, bool edge = false)
  {
    if (!edge) {
      return !(rect1->min_corner().x() >= rect2->get_right() || rect1->min_corner().y() >= rect2->get_top()
               || rect1->max_corner().x() <= rect2->get_left() || rect1->max_corner().y() <= rect2->get_bottom());
    } else {
      return !(rect1->min_corner().x() > rect2->get_right() || rect1->min_corner().y() > rect2->get_top()
               || rect1->max_corner().x() < rect2->get_left() || rect1->max_corner().y() < rect2->get_bottom());
    }
  }

  static bool intersectionWithEdge(const DrcRect* rect, const DrcEdge* edge)
  {
    DrcCoordinate<int> bg(edge->get_begin_x(), edge->get_begin_y());
    DrcCoordinate<int> ed(edge->get_end_x(), edge->get_end_y());

    if (!isRectContainCoordinate(rect, bg, false) && !isRectContainCoordinate(rect, ed, false)) {
      if (edge->isHorizontal()) {
        return (edge->get_max_y() > rect->get_bottom() && edge->get_max_y() < rect->get_top());
      } else if (edge->isVertical()) {
        return (edge->get_max_x() > rect->get_left() && edge->get_max_x() < rect->get_right());
      }
    }
    return true;
  }

  static bool intersection(const DrcRectangle<int>& rect1, const DrcRect* rect2, bool edge = false)
  {
    if (!edge) {
      return !(rect1.get_lb_x() >= rect2->get_right() || rect1.get_lb_y() >= rect2->get_top() || rect1.get_rt_x() <= rect2->get_left()
               || rect1.get_rt_y() <= rect2->get_bottom());
    } else {
      return !(rect1.get_lb_x() > rect2->get_right() || rect1.get_lb_y() > rect2->get_top() || rect1.get_rt_x() < rect2->get_left()
               || rect1.get_rt_y() < rect2->get_bottom());
    }
  }

  static bool isSameRect(const DrcRect* rect1, const DrcRect* rect2)
  {
    return (rect1->get_left() == rect2->get_left() && rect1->get_bottom() == rect2->get_bottom() && rect1->get_right() == rect2->get_right()
            && rect1->get_top() == rect2->get_top());
  }

  static bool isContainedBy(const DrcRect* rect1, const DrcRect* rect2)
  {
    return (rect1->get_left() >= rect2->get_left() && rect1->get_bottom() >= rect2->get_bottom() && rect1->get_right() <= rect2->get_right()
            && rect1->get_top() <= rect2->get_top());
  }

  static bool isContainedBy(const DrcRectangle<int>& rect1, const DrcRect* rect2)
  {
    return (rect1.get_lb_x() >= rect2->get_left() && rect1.get_lb_y() >= rect2->get_bottom() && rect1.get_rt_x() <= rect2->get_right()
            && rect1.get_rt_y() <= rect2->get_top());
  }

  static bool isParallelOverlap(const DrcRect* rect1, const DrcRect* rect2)
  {
    return !((rect1->get_left() > rect2->get_right() || rect1->get_right() < rect2->get_left())
             && (rect1->get_bottom() > rect2->get_top() || rect1->get_top() < rect2->get_bottom()));
  }

  static bool isHorizontalParallelOverlap(const DrcRect* rect1, const DrcRect* rect2)
  {
    return !(rect1->get_left() > rect2->get_right() || rect1->get_right() < rect2->get_left());
  }

  static RTreeBox getPolyBoundingBox(DrcPoly* target_poly)
  {
    int lb_x = 1e9;
    int lb_y = 1e9;
    int rt_x = 0;
    int rt_y = 0;
    for (auto& edges : target_poly->getEdges()) {
      for (auto& edge : edges) {
        lb_x = std::min(lb_x, edge->get_min_x());
        lb_y = std::min(lb_y, edge->get_min_y());
        rt_x = std::max(rt_x, edge->get_max_x());
        rt_y = std::max(rt_y, edge->get_max_y());
      }
    }
    return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
  }

  static RTreeBox getSpanBoxBetweenEdgeAndRect(DrcEdge* edge, DrcRect* rect)
  {
    int x_values[4] = {edge->get_min_x(), edge->get_max_x(), rect->get_left(), rect->get_right()};
    int y_values[4] = {edge->get_min_y(), edge->get_max_y(), rect->get_bottom(), rect->get_top()};
    std::sort(x_values, x_values + 4);
    std::sort(y_values, y_values + 4);
    int lb_x = x_values[1];
    int lb_y = y_values[1];
    int rt_x = x_values[2];
    int rt_y = y_values[2];
    return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
  }

  static RTreeBox getSpanBoxBetweenTwoRects(DrcRect* rect1, DrcRect* rect2)
  {
    int x_values[4] = {rect1->get_left(), rect1->get_right(), rect2->get_left(), rect2->get_right()};
    int y_values[4] = {rect1->get_bottom(), rect1->get_top(), rect2->get_bottom(), rect2->get_top()};
    std::sort(x_values, x_values + 4);
    std::sort(y_values, y_values + 4);
    int lb_x = x_values[1];
    int lb_y = y_values[1];
    int rt_x = x_values[2];
    int rt_y = y_values[2];
    return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
  }
  /**
   * @brief Get the Span Rect Between Two Rects object
   * 竖直平行交叠下span_box   水平平行交叠下的span_box
   *             ___         ____________
   *  __________|   |       |__1_________|
   * |   |span  |   |           |span box|
   * | 1 |box   | 2 |           |________|____
   * |   |______|___|           |__2__________|
   * |___|
   * @param rect1
   * @param rect2
   * @return DrcRectangle<int>
   */
  static DrcRectangle<int> getSpanRectBetweenTwoRects(DrcRect* rect1, DrcRect* rect2)
  {
    int x_values[4] = {rect1->get_left(), rect1->get_right(), rect2->get_left(), rect2->get_right()};
    int y_values[4] = {rect1->get_bottom(), rect1->get_top(), rect2->get_bottom(), rect2->get_top()};
    std::sort(x_values, x_values + 4);
    std::sort(y_values, y_values + 4);
    int lb_x = x_values[1];
    int lb_y = y_values[1];
    int rt_x = x_values[2];
    int rt_y = y_values[2];
    return DrcRectangle<int>(lb_x, lb_y, rt_x, rt_y);
  }

  static std::pair<DrcCoordinate<int>, DrcCoordinate<int>> getSpanCornerPairBetweenTwoRects(DrcRect* rect1, DrcRect* rect2)
  {
    // DrcRectangle<int> span_rect = getSpanRectBetweenTwoRects(rect1, rect2);
    // std::vector<DrcCoordinate<int>> cornor_pair;
    // std::vector<DrcCoordinate<int>> coord_list = getRectangleCoordinateList(span_rect);
    // for (auto& coord : coord_list) {
    //   if (isRectContainCoordinate(rect1, coord, true) || isRectContainCoordinate(rect2, coord, true)) {
    //     cornor_pair.push_back(coord);
    //   }
    //   if (cornor_pair.size() == 2) {
    //     break;
    //   }
    // }
    // return cornor_pair;
    std::pair<DrcCoordinate<int>, DrcCoordinate<int>> cornor_pair;
    int x1 = 0;
    int x2 = 0;
    int y1 = 0;
    int y2 = 0;
    if (std::abs(rect1->get_left() - rect2->get_right()) < std::abs(rect1->get_right() - rect2->get_left())) {
      x1 = rect1->get_left();
      x2 = rect2->get_right();
    } else {
      x1 = rect1->get_right();
      x2 = rect2->get_left();
    }
    if (std::abs(rect1->get_bottom() - rect2->get_top()) < std::abs(rect1->get_top() - rect2->get_bottom())) {
      y1 = rect1->get_bottom();
      y2 = rect2->get_top();
    } else {
      y1 = rect1->get_top();
      y2 = rect2->get_bottom();
    }
    cornor_pair.first = DrcCoordinate<int>(x1, y1);
    cornor_pair.second = DrcCoordinate<int>(x2, y2);
    return cornor_pair;
  }

  static double getRectDiagonalLength(const DrcRectangle<int>& rect)
  {
    int x_length = std::abs(rect.get_rt_x() - rect.get_lb_x());
    int y_length = std::abs(rect.get_rt_y() - rect.get_lb_y());
    double diag_length = std::sqrt(x_length * x_length + y_length * y_length);
    return diag_length;
  }

  static double getRectDiagonalLength(const BoostRect& rect)
  {
    int x_length = std::abs(xh(rect) - xl(rect));
    int y_length = std::abs(yh(rect) - yl(rect));
    double diag_length = std::sqrt(x_length * x_length + y_length * y_length);
    return diag_length;
  }

  static double getRectDiagonalLength(const RTreeBox& box)
  {
    int x_length = std::abs(box.max_corner().get<0>() - box.min_corner().get<0>());
    int y_length = std::abs(box.max_corner().get<1>() - box.min_corner().get<1>());
    double diag_length = std::sqrt(x_length * x_length + y_length * y_length);
    return diag_length;
  }

  static DrcRect enlargeRect(const DrcRect* rect, int value)
  {
    int lb_x = rect->get_left() - value;
    int lb_y = rect->get_bottom() - value;
    int rt_x = rect->get_right() + value;
    int rt_y = rect->get_top() + value;

    DrcRect enlarge_rect;
    enlarge_rect.set_lb(lb_x, lb_y);
    enlarge_rect.set_rt(rt_x, rt_y);
    return enlarge_rect;
  }

  static RTreeSegment getRTreeSegment(DrcEdge* drcEdge)
  {
    RTreePoint point1(drcEdge->get_min_x(), drcEdge->get_min_y());
    RTreePoint point2(drcEdge->get_max_x(), drcEdge->get_max_y());
    return RTreeSegment(point1, point2);
  }

  static BoostRect getBoostRect(DrcRect* rect)
  {
    int lb_x = rect->get_left();
    int lb_y = rect->get_bottom();
    int rt_x = rect->get_right();
    int rt_y = rect->get_top();
    return BoostRect(lb_x, lb_y, rt_x, rt_y);
  }

  static DrcRectangle<int> getRectangleFromBoostRect(const BoostRect& boost_rect)
  {
    int lb_x = xl(boost_rect);
    int lb_y = yl(boost_rect);
    int rt_x = xh(boost_rect);
    int rt_y = yh(boost_rect);
    return DrcRectangle<int>(lb_x, lb_y, rt_x, rt_y);
  }

  static DrcRectangle<int> getRectangleFromRTreeBox(const RTreeBox& box)
  {
    int lb_x = box.min_corner().get<0>();
    int lb_y = box.min_corner().get<1>();
    int rt_x = box.max_corner().get<0>();
    int rt_y = box.max_corner().get<1>();
    return DrcRectangle<int>(lb_x, lb_y, rt_x, rt_y);
  }

  static BoostRect getBoostRect(const RTreeBox& box)
  {
    int lb_x = box.min_corner().get<0>();
    int lb_y = box.min_corner().get<1>();
    int rt_x = box.max_corner().get<0>();
    int rt_y = box.max_corner().get<1>();
    return BoostRect(lb_x, lb_y, rt_x, rt_y);
  }

  static RTreeBox getRTreeBox(const BoostRect& rect)
  {
    RTreePoint leftBottom(xl(rect), yl(rect));
    RTreePoint rightTop(xh(rect), yh(rect));
    return RTreeBox(leftBottom, rightTop);
  }

  static RTreeBox getRTreeBox(const DrcRectangle<int>& rect)
  {
    RTreePoint leftBottom(rect.get_lb_x(), rect.get_lb_y());
    RTreePoint rightTop(rect.get_rt_x(), rect.get_rt_y());
    return RTreeBox(leftBottom, rightTop);
  }

  static RTreeBox getRTreeBox(DrcRect* rect)
  {
    RTreePoint leftBottom(rect->get_left(), rect->get_bottom());
    RTreePoint rightTop(rect->get_right(), rect->get_top());
    return RTreeBox(leftBottom, rightTop);
  }

  static BoostSegment getBoostSegment(DrcEdge* edge)
  {
    BoostPoint begin(edge->get_begin_x(), edge->get_begin_y());
    BoostPoint end(edge->get_end_x(), edge->get_end_y());
    return BoostSegment(begin, end);
  }

  static bool areTwoEqualRectangles(const DrcRectangle<int>& rectangle, DrcRect* drc_rect)
  {
    return rectangle.get_lb_x() == drc_rect->get_left() && rectangle.get_lb_y() == drc_rect->get_bottom()
           && rectangle.get_rt_x() == drc_rect->get_right() && rectangle.get_rt_y() == drc_rect->get_top();
  }

  static bool isBoxIntersectedByRect(const RTreeBox& box, DrcRect* rect)
  {
    int lb_x = box.min_corner().get<0>();
    int lb_y = box.min_corner().get<1>();
    int rt_x = box.max_corner().get<0>();
    int rt_y = box.max_corner().get<1>();
    return !(rect->get_left() >= rt_x || rect->get_bottom() >= rt_y || rect->get_right() <= lb_x || rect->get_top() <= lb_y);
  }

  static bool isBoxPenetratedByRect(const RTreeBox& box, DrcRect* rect, bool isHorizontalParallelOverlap)
  {
    int lb_x = box.min_corner().get<0>();
    int lb_y = box.min_corner().get<1>();
    int rt_x = box.max_corner().get<0>();
    int rt_y = box.max_corner().get<1>();
    if (isBoxIntersectedByRect(box, rect)) {
      if (isHorizontalParallelOverlap) {
        return rect->get_left() <= lb_x && rect->get_right() >= rt_x;
      } else {
        return rect->get_bottom() <= lb_y && rect->get_top() >= rt_y;
      }
    }
    return false;
  }

  static bool isPenetratedIntersected(const RTreeBox& box1, const RTreeBox& box2)
  {
    int lb_x1 = box1.min_corner().get<0>();
    int lb_y1 = box1.min_corner().get<1>();
    int rt_x1 = box1.max_corner().get<0>();
    int rt_y1 = box1.max_corner().get<1>();
    //////////////////////////////////////
    int lb_x2 = box2.min_corner().get<0>();
    int lb_y2 = box2.min_corner().get<1>();
    int rt_x2 = box2.max_corner().get<0>();
    int rt_y2 = box2.max_corner().get<1>();
    //////////////////////////////////////
    if ((lb_x1 == lb_x2) && (rt_x1 == rt_x2)) {
      return !((lb_y1 > rt_y2) || (rt_y1 < lb_y2));
    }
    if ((lb_y1 == lb_y2) && (rt_y1 == rt_y2)) {
      return !((lb_x1 > rt_x2) || (rt_x1 < lb_x2));
    }
    return false;
  }
  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////

  // function for edge
  //------------------------------------------------------------
  //------------------------------------------------------------

  static bool isTwoEdgeParallel(DrcEdge* edge1, DrcEdge* edge2)
  {
    if (edge1->isHorizontal() && edge2->isHorizontal()) {
      return true;
    } else if (edge1->isVertical() && edge2->isVertical()) {
      return true;
    }
    return false;
  }

  static int getPRLRunLength(DrcRect* target_rect, DrcRect* result_rect)
  {
    if (!isParallelOverlap(target_rect, result_rect) || intersection(target_rect, result_rect)) {
      return 0;
    } else {
      if (((target_rect->get_bottom() > result_rect->get_top())) || ((target_rect->get_top() < result_rect->get_bottom()))) {
        return std::min(target_rect->get_right(), result_rect->get_right()) - std::max(target_rect->get_left(), result_rect->get_left());
      }
      if ((!(target_rect->get_left() > result_rect->get_right())) || (!(target_rect->get_right() < result_rect->get_left()))) {
        return std::min(target_rect->get_top(), result_rect->get_top()) - std::max(target_rect->get_bottom(), result_rect->get_bottom());
      }
    }
    return 0;
  }

  static bool isParallelOverlap(DrcEdge* edge1, DrcEdge* edge2)
  {
    if (edge1->isHorizontal() && edge2->isHorizontal()) {
      int bx1 = edge1->get_min_x();
      int ex1 = edge1->get_max_x();
      int bx2 = edge2->get_min_x();
      int ex2 = edge2->get_max_x();
      return !((ex1 < bx2) || (bx1 > ex2));
    } else if (edge1->isVertical() && edge2->isVertical()) {
      int by1 = edge1->get_min_y();
      int ey1 = edge1->get_max_y();
      int by2 = edge2->get_min_y();
      int ey2 = edge2->get_max_y();
      return !((ey1 < by2) || (by1 > ey2));
    } else {
      std::cout << "[DRCUtil Error]:The two edges are not parallel" << std::endl;
    }
  }

  static bool isOppositeEdgeDir(DrcEdge* edge1, DrcEdge* edge2)
  {
    EdgeDirection edge_dir1 = edge1->get_edge_dir();
    EdgeDirection edge_dir2 = edge2->get_edge_dir();
    return (edge_dir1 == EdgeDirection::kWest && edge_dir2 == EdgeDirection::kEast)
           || (edge_dir1 == EdgeDirection::kEast && edge_dir2 == EdgeDirection::kWest)
           || (edge_dir1 == EdgeDirection::kNorth && edge_dir2 == EdgeDirection::kSouth)
           || (edge_dir1 == EdgeDirection::kSouth && edge_dir2 == EdgeDirection::kNorth);
  }

  static RTreeBox getSpanBoxBetweenTwoEdges(DrcEdge* edge1, DrcEdge* edge2)
  {
    int x_values[4] = {edge1->get_begin_x(), edge1->get_end_x(), edge2->get_begin_x(), edge2->get_end_x()};
    int y_values[4] = {edge1->get_begin_y(), edge1->get_end_y(), edge2->get_begin_y(), edge2->get_end_y()};
    std::sort(x_values, x_values + 4);
    std::sort(y_values, y_values + 4);
    int lb_x = x_values[1];
    int lb_y = y_values[1];
    int rt_x = x_values[2];
    int rt_y = y_values[2];
    return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
  }

  static RTreeBox getEdgeSpacingQueryBox(DrcEdge* edge, int spacing)
  {
    int lb_x = 0;
    int lb_y = 0;
    int rt_x = 0;
    int rt_y = 0;
    EdgeDirection edge_dir = edge->get_edge_dir();
    if (edge_dir == EdgeDirection::kNorth) {
      lb_x = edge->get_min_x();
      lb_y = edge->get_min_y() - spacing;
      rt_x = edge->get_max_x() + spacing;
      rt_y = edge->get_max_y() + spacing;
    } else if (edge_dir == EdgeDirection::kSouth) {
      lb_x = edge->get_min_x() - spacing;
      lb_y = edge->get_min_y() - spacing;
      rt_x = edge->get_max_x();
      rt_y = edge->get_max_y() + spacing;
    } else if (edge_dir == EdgeDirection::kWest) {
      lb_x = edge->get_min_x() - spacing;
      lb_y = edge->get_min_y();
      rt_x = edge->get_max_x() + spacing;
      rt_y = edge->get_max_y() + spacing;
    } else if (edge_dir == EdgeDirection::kEast) {
      lb_x = edge->get_min_x() - spacing;
      lb_y = edge->get_min_y() - spacing;
      rt_x = edge->get_max_x() + spacing;
      rt_y = edge->get_max_y();
    } else {
      std::cout << "Error: unknown edge direction in getEdgeSpacingQueryBox" << std::endl;
    }
    return RTreeBox(RTreePoint(lb_x, lb_y), RTreePoint(rt_x, rt_y));
  }

  // function for corner
  //-----------------------------------------------
  //-----------------------------------------------
  static bool isCornerConCave(DrcEdge* edge)
  {
    auto next_edge_dir = edge->getNextEdge()->get_edge_dir();
    switch (edge->get_edge_dir()) {
      case EdgeDirection::kEast:
        return next_edge_dir == EdgeDirection::kSouth;
      case EdgeDirection::kWest:
        return next_edge_dir == EdgeDirection::kNorth;
      case EdgeDirection::kNorth:
        return next_edge_dir == EdgeDirection::kEast;
      case EdgeDirection::kSouth:
        return next_edge_dir == EdgeDirection::kWest;
      case EdgeDirection::kNone:
        return false;
    }
    return false;
  }

  static bool isCornerConVex(DrcEdge* edge)
  {
    auto next_edge_dir = edge->getNextEdge()->get_edge_dir();
    switch (edge->get_edge_dir()) {
      case EdgeDirection::kEast:
        return next_edge_dir == EdgeDirection::kNorth;
      case EdgeDirection::kWest:
        return next_edge_dir == EdgeDirection::kSouth;
      case EdgeDirection::kNorth:
        return next_edge_dir == EdgeDirection::kWest;
      case EdgeDirection::kSouth:
        return next_edge_dir == EdgeDirection::kEast;
      case EdgeDirection::kNone:
        return false;
    }
    return false;
  }
};
}  // namespace idrc

#endif