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

#include "DRCHeader.hpp"
#include "Logger.hpp"
#include "Orientation.hpp"
#include "PlanarRect.hpp"
#include "Rotation.hpp"
#include "json.hpp"

namespace idrc {

#define DRCUTIL (idrc::Utility::getInst())

class Utility
{
 public:
  static void initInst();
  static Utility& getInst();
  static void destroyInst();
  // function

#if 1  // 距离线长计算

  // 获得两坐标的曼哈顿距离
  static int32_t getManhattanDistance(PlanarCoord start_coord, PlanarCoord end_coord)
  {
    return std::abs(start_coord.get_x() - end_coord.get_x()) + std::abs(start_coord.get_y() - end_coord.get_y());
  }

  // 获得两个矩形的欧式距离
  static double getEuclideanDistance(const PlanarRect& a, const PlanarRect& b)
  {
    int32_t x_spacing = std::max(b.get_ll_x() - a.get_ur_x(), a.get_ll_x() - b.get_ur_x());
    int32_t y_spacing = std::max(b.get_ll_y() - a.get_ur_y(), a.get_ll_y() - b.get_ur_y());

    if (x_spacing > 0 && y_spacing > 0) {
      return std::sqrt((double) (x_spacing * x_spacing + y_spacing * y_spacing));
    } else {
      return std::max(std::max(x_spacing, y_spacing), 0);
    }
  }

  static int32_t getParallelLength(PlanarRect& a, PlanarRect& b)
  {
    int32_t x_parallel_length = std::max(0, std::min(a.get_ur_x(), b.get_ur_x()) - std::max(a.get_ll_x(), b.get_ll_x()));
    int32_t y_parallel_length = std::max(0, std::min(a.get_ur_y(), b.get_ur_y()) - std::max(a.get_ll_y(), b.get_ll_y()));
    return std::max(x_parallel_length, y_parallel_length);
  }

#endif

#if 1  // 方向方位计算

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

#endif

#if 1  // 位置关系计算

  // 三个坐标是否共线
  static bool isCollinear(PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    return getDirection(first_coord, second_coord) == getDirection(second_coord, third_coord);
  }

  // 是否是凸角位置
  static bool isConvexCorner(Rotation rotation, PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    if (isCollinear(first_coord, second_coord, third_coord)) {
      return false;
    }

    return crossProduct(rotation, first_coord, second_coord, third_coord) < 0;
  }

  // 是否是凹角位置
  static bool isConcaveCorner(Rotation rotation, PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    if (isCollinear(first_coord, second_coord, third_coord)) {
      return false;
    }

    return crossProduct(rotation, first_coord, second_coord, third_coord) > 0;
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

  // 线段在线段内
  static bool isInside(const Segment<PlanarCoord>& master, const Segment<PlanarCoord>& seg)
  {
    if (!isRightAngled(master.get_first(), master.get_second()) || !isRightAngled(seg.get_first(), seg.get_second())) {
      DRCLOG.error(Loc::current(), "The segment is error!");
    }
    PlanarRect rect = getRect(master.get_first(), master.get_second());
    return isInside(rect, seg.get_first()) && isInside(rect, seg.get_second());
  }

  /**
   *  ！在检测DRC中
   *  如果a与b中有膨胀矩形,那么则用isOpenOverlap
   *  如果a与b中都是真实矩形,那么用isClosedOverlap
   *
   *  isOpenOverlap:不考虑边的overlap
   */
  static bool isOpenOverlap(const PlanarRect& a, const PlanarRect& b) { return isOverlap(a, b, false); }

  static bool isOpenOverlap(const PlanarCoord& start_coord, const PlanarCoord& end_coord, const PlanarRect& rect)
  {
    return isOverlap(getRect(start_coord, end_coord), rect, false);
  }

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

#endif

#if 1  // boost数据结构工具函数

#if 1  // int类型

  static Rotation getRotation(GTLPolyInt& gtl_poly)
  {
    gtl::direction_1d gtl_rotation = gtl::winding(gtl_poly);
    if (gtl::direction_1d(gtl::direction_1d_enum::CLOCKWISE) == gtl_rotation) {
      return Rotation::kClockwise;
    } else if (gtl::direction_1d(gtl::direction_1d_enum::COUNTERCLOCKWISE) == gtl_rotation) {
      return Rotation::kCounterclockwise;
    } else {
      return Rotation::kNone;
    }
  }

  static Rotation getRotation(GTLHolePolyInt& gtl_holy_poly)
  {
    gtl::direction_1d gtl_rotation = gtl::winding(gtl_holy_poly);
    if (gtl::direction_1d(gtl::direction_1d_enum::CLOCKWISE) == gtl_rotation) {
      return Rotation::kClockwise;
    } else if (gtl::direction_1d(gtl::direction_1d_enum::COUNTERCLOCKWISE) == gtl_rotation) {
      return Rotation::kCounterclockwise;
    } else {
      return Rotation::kNone;
    }
  }

  static PlanarCoord convertToPlanarCoord(GTLPointInt gtl_point) { return PlanarCoord(gtl_point.x(), gtl_point.y()); }

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

  static bool isOverlap(GTLPolySetInt a, GTLRectInt b, bool consider_edge = true)
  {
    GTLPolySetInt gtl_poly_set;
    gtl_poly_set += b;
    if (consider_edge) {
      a.interact(gtl_poly_set);
    } else {
      a &= gtl_poly_set;
    }
    return gtl::area(a) > 0;
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

#endif

#endif

#if 1  // idrc数据结构工具函数

  // 获得配置的值
  template <typename T>
  static T getConfigValue(std::map<std::string, std::any>& config_map, const std::string& config_name, const T& default_value)
  {
    T value;
    if (exist(config_map, config_name)) {
      value = std::any_cast<T>(config_map[config_name]);
    } else {
      DRCLOG.warn(Loc::current(), "The config '", config_name, "' uses the default value!");
      value = default_value;
    }
    return value;
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
      DRCLOG.info(Loc::current(), table_str);
    }
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
  static int32_t crossProduct(Rotation rotation, PlanarCoord& first_coord, PlanarCoord& second_coord, PlanarCoord& third_coord)
  {
    int32_t cross_product = 0;
    if (rotation == Rotation::kClockwise) {
      cross_product = ((second_coord.get_x() - first_coord.get_x()) * (third_coord.get_y() - first_coord.get_y())
                       - (second_coord.get_y() - first_coord.get_y()) * (third_coord.get_x() - first_coord.get_x()));
    } else if (rotation == Rotation::kCounterclockwise) {
      cross_product = ((second_coord.get_x() - third_coord.get_x()) * (first_coord.get_y() - third_coord.get_y())
                       - (second_coord.get_y() - third_coord.get_y()) * (first_coord.get_x() - third_coord.get_x()));
    } else {
      DRCLOG.error(Loc::current(), "The rotation is error!");
    }
    return cross_product;
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

  static PlanarRect getEnlargedRect(PlanarCoord start_coord, PlanarCoord end_coord, int32_t enlarge_size)
  {
    if (!CmpPlanarCoordByXASC()(start_coord, end_coord)) {
      std::swap(start_coord, end_coord);
    }
    PlanarRect rect(start_coord, end_coord);

    if (isRightAngled(start_coord, end_coord)) {
      rect = getEnlargedRect(rect, enlarge_size);
    } else {
      DRCLOG.error(Loc::current(), "The segment is oblique!");
    }
    return rect;
  }

  static PlanarRect getEnlargedRect(PlanarCoord start_coord, PlanarCoord end_coord, int32_t ll_x_minus_offset, int32_t ll_y_minus_offset,
                                    int32_t ur_x_add_offset, int32_t ur_y_add_offset)
  {
    if (!CmpPlanarCoordByXASC()(start_coord, end_coord)) {
      std::swap(start_coord, end_coord);
    }
    PlanarRect rect(start_coord, end_coord);

    if (isRightAngled(start_coord, end_coord)) {
      rect = getEnlargedRect(rect, ll_x_minus_offset, ll_y_minus_offset, ur_x_add_offset, ur_y_add_offset);
    } else {
      DRCLOG.error(Loc::current(), "The segment is oblique!");
    }
    return rect;
  }

  static PlanarRect getRegularRect(PlanarRect rect, PlanarRect border)
  {
    PlanarRect regular_rect;
    regular_rect.set_ll(std::max(rect.get_ll_x(), border.get_ll_x()), std::max(rect.get_ll_y(), border.get_ll_y()));
    regular_rect.set_ur(std::min(rect.get_ur_x(), border.get_ur_x()), std::min(rect.get_ur_y(), border.get_ur_y()));
    return regular_rect;
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

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, int32_t enlarge_size)
  {
    return getEnlargedRect(rect, enlarge_size, enlarge_size, enlarge_size, enlarge_size);
  }

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, int32_t x_enlarge_size, int32_t y_enlarge_size)
  {
    return getEnlargedRect(rect, x_enlarge_size, y_enlarge_size, x_enlarge_size, y_enlarge_size);
  }

  // 扩大矩形
  static PlanarRect getEnlargedRect(PlanarRect rect, int32_t ll_x_minus_offset, int32_t ll_y_minus_offset, int32_t ur_x_add_offset, int32_t ur_y_add_offset)
  {
    minusOffset(rect.get_ll(), ll_x_minus_offset, ll_y_minus_offset);
    addOffset(rect.get_ur(), ur_x_add_offset, ur_y_add_offset);
    return rect;
  }

  static void minusOffset(PlanarCoord& coord, int32_t x_offset, int32_t y_offset)
  {
    coord.set_x((coord.get_x() - x_offset) < 0 ? 0 : (coord.get_x() - x_offset));
    coord.set_y((coord.get_y() - y_offset) < 0 ? 0 : (coord.get_y() - y_offset));
  }

  static void addOffset(PlanarCoord& coord, int32_t x_offset, int32_t y_offset)
  {
    coord.set_x(coord.get_x() + x_offset);
    coord.set_y(coord.get_y() + y_offset);
  }

  static PlanarRect getSpacingRect(PlanarRect a, PlanarRect b)
  {
    int32_t x_spacing = std::max(0, std::max(a.get_ll_x() - b.get_ur_x(), b.get_ll_x() - a.get_ur_x()));
    int32_t y_spacing = std::max(0, std::max(a.get_ll_y() - b.get_ur_y(), b.get_ll_y() - a.get_ur_y()));
    PlanarRect spacing_rect;
    if (x_spacing > 0 && y_spacing > 0) {
      if (a.get_ur_x() < b.get_ll_x()) {
        spacing_rect.set_ll_x(a.get_ur_x());
        spacing_rect.set_ur_x(b.get_ll_x());
      } else {
        spacing_rect.set_ll_x(b.get_ur_x());
        spacing_rect.set_ur_x(a.get_ll_x());
      }
      if (a.get_ur_y() < b.get_ll_y()) {
        spacing_rect.set_ll_y(a.get_ur_y());
        spacing_rect.set_ur_y(b.get_ll_y());
      } else {
        spacing_rect.set_ll_y(b.get_ur_y());
        spacing_rect.set_ur_y(a.get_ll_y());
      }
    } else {
      if (x_spacing == 0) {
        spacing_rect = getOverlap(getEnlargedRect(a, 0, y_spacing), getEnlargedRect(b, 0, y_spacing));
      } else if (y_spacing == 0) {
        spacing_rect = getOverlap(getEnlargedRect(a, x_spacing, 0), getEnlargedRect(b, x_spacing, 0));
      }
    }
    return spacing_rect;
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

  static bool hasShrinkedRect(PlanarRect rect, int32_t shrinked_size)
  {
    addOffset(rect.get_ll(), shrinked_size, shrinked_size);
    minusOffset(rect.get_ur(), shrinked_size, shrinked_size);

    return rect.get_ll_x() <= rect.get_ur_x() && rect.get_ll_y() <= rect.get_ur_y();
  }

  static PlanarRect getShrinkedRect(PlanarRect rect, int32_t shrinked_size)
  {
    return getShrinkedRect(rect, shrinked_size, shrinked_size, shrinked_size, shrinked_size);
  }

  static PlanarRect getShrinkedRect(PlanarRect rect, int32_t ll_x_add_offset, int32_t ll_y_add_offset, int32_t ur_x_minus_offset, int32_t ur_y_minus_offset)
  {
    addOffset(rect.get_ll(), ll_x_add_offset, ll_y_add_offset);
    minusOffset(rect.get_ur(), ur_x_minus_offset, ur_y_minus_offset);
    return rect;
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

  // 获得坐标集合的外接矩形
  static PlanarRect getBoundingBox(const std::vector<PlanarCoord>& coord_list)
  {
    PlanarRect bounding_box;
    if (coord_list.empty()) {
      DRCLOG.warn(Loc::current(), "The coord list size is empty!");
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

#endif

#if 1  // std数据结构工具函数

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
      DRCLOG.error(Loc::current(), "The total of size < 0!");
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
    return equalDoubleByError(merchant, static_cast<int32_t>(merchant), DRC_ERROR);
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
      DRCLOG.warn(Loc::current(), "The step == 0!");
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
      DRCLOG.error(Loc::current(), "The flag list is empty!");
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
    DRCLOG.error(Loc::current(), "The configuration file key '", key, "' does not exist!");
    return value;
  }

  template <typename T = nlohmann::json>
  static T getData(nlohmann::json value, std::string flag)
  {
    if (flag.empty()) {
      DRCLOG.error(Loc::current(), "The flag is empty!");
    }
    value = value[flag];
    if (!value.is_null()) {
      return value;
    }
    DRCLOG.error(Loc::current(), "The configuration file key '", flag, "' does not exist!");
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
      DRCLOG.error(Loc::current(), "The value is nan!");
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
      DRCLOG.error(Loc::current(), "Failed to open file '", file_path, "'!");
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

  static bool isInteger(double a) { return equalDoubleByError(a, static_cast<int32_t>(a), DRC_ERROR); }

  static void checkFile(std::string file_path)
  {
    if (!std::filesystem::exists(file_path)) {
      DRCLOG.error(Loc::current(), "The file ", file_path, " does not exist!");
    }
  }

  static void createDirByFile(std::string file_path) { createDir(dirname((char*) file_path.c_str())); }

  static void createDir(std::string dir_path)
  {
    if (!std::filesystem::exists(dir_path)) {
      std::error_code system_error;
      if (!std::filesystem::create_directories(dir_path, system_error)) {
        DRCLOG.error(Loc::current(), "Failed to create directory '", dir_path, "', system_error:", system_error.message());
      }
    }
  }

  static bool existFile(const std::string& file_path) { return std::filesystem::exists(file_path); }

  static void changePermissions(const std::string& dir_path, std::filesystem::perms permissions)
  {
    std::error_code system_error;
    std::filesystem::permissions(dir_path, permissions);
    if (system_error) {
      DRCLOG.error(Loc::current(), "Failed to change permissions for '", dir_path, "', system_error: ", system_error.message());
    }
  }

  static void removeDir(const std::string& dir_path)
  {
    std::error_code system_error;

    // 检查文件夹是否存在
    if (std::filesystem::exists(dir_path, system_error)) {
      // 尝试删除文件夹
      if (!std::filesystem::remove_all(dir_path, system_error)) {
        DRCLOG.error(Loc::current(), "Failed to remove directory '", dir_path, "'. Error: ", system_error.message());
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

}  // namespace idrc
