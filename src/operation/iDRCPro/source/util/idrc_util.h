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

#include <vector>

#include "drc_basic_point.h"
#include "idrc_engine.h"

namespace idrc {

class DrcUtil
{
 public:
  // 叉乘
  template <typename T>
  static int crossProduct(T& first_coord, T& second_coord, T& third_coord)
  {
    return (second_coord.get_x() - first_coord.get_x()) * (third_coord.get_y() - first_coord.get_y())
           - (second_coord.get_y() - first_coord.get_y()) * (third_coord.get_x() - first_coord.get_x());
  }

  // 用叉乘判断是否是凸角
  template <typename T>
  static bool isConvexCorner(T& first_coord, T& second_coord, T& third_coord)
  {
    return crossProduct(first_coord, second_coord, third_coord) < 0;
  }

  // 用叉乘判断是否是凹角
  template <typename T>
  static bool isConcaveCorner(T& first_coord, T& second_coord, T& third_coord)
  {
    return crossProduct(first_coord, second_coord, third_coord) > 0;
  }

  // 获得 polygon 的所有点，按顺时针组织
  static std::vector<ieda_solver::GtlPoint> getPolygonPoints(DrcBasicPoint* point)
  {
    std::vector<ieda_solver::GtlPoint> point_list{{point->get_x(), point->get_y()}};
    auto* iter_pt = point->get_next();
    while (nullptr != iter_pt && iter_pt != point) {
      point_list.emplace_back(iter_pt->get_x(), iter_pt->get_y());

      iter_pt = iter_pt->get_next();
    }

    return point_list;
  }

  // 方向取反
  static DrcDirection oppositeDirection(DrcDirection direction)
  {
    switch (direction) {
      case DrcDirection::kUp:
        return DrcDirection::kDown;
      case DrcDirection::kDown:
        return DrcDirection::kUp;
      case DrcDirection::kLeft:
        return DrcDirection::kRight;
      case DrcDirection::kRight:
        return DrcDirection::kLeft;
      default:
        return DrcDirection::kNone;
    }
  }

  // 线段左边为多边形外部
  static DrcDirection outsidePolygonDirection(DrcBasicPoint* point_prev, DrcBasicPoint* point_next)
  {
    DrcDirection direction = point_prev->direction(point_next);
    switch (direction) {
      case DrcDirection::kUp:
        return DrcDirection::kLeft;
      case DrcDirection::kDown:
        return DrcDirection::kRight;
      case DrcDirection::kLeft:
        return DrcDirection::kDown;
      case DrcDirection::kRight:
        return DrcDirection::kUp;
      default:
        return DrcDirection::kNone;
    }
  }

  // 获得正交方向
  static std::pair<DrcDirection, DrcDirection> getOrthogonalDirection(DrcDirection direction)
  {
    switch (direction) {
      case DrcDirection::kUp:
      case DrcDirection::kDown:
        return std::make_pair(DrcDirection::kLeft, DrcDirection::kRight);
      case DrcDirection::kLeft:
      case DrcDirection::kRight:
        return std::make_pair(DrcDirection::kUp, DrcDirection::kDown);
      default:
        return std::make_pair(DrcDirection::kNone, DrcDirection::kNone);
    }
  }

  // 使用扫描线结果判断角的类型
  // static DrcCornerType cornerType(DrcBasicPoint* point)
  // {
  //   if (!point->is_endpoint()) {
  //     return DrcCornerType::kNone;
  //   }
  //   int corner_value = 0;
  //   // for (auto type : point->getTypesClockwise()) {
  //   //   if (type == ScanlineDataType::kWidth) {
  //   //     corner_value <<= 1;
  //   //     corner_value += 0;
  //   //   } else if (type == ScanlineDataType::kSpacing || type == ScanlineDataType::kNone) {
  //   //     corner_value <<= 1;
  //   //     corner_value += 1;
  //   //   }
  //   // }

  //   bool is_convex = corner_value == 0b0011 || corner_value == 0b0110 || corner_value == 0b1100 || corner_value == 0b1001;
  //   bool is_concave = corner_value == 0b0000;

  //   if (is_concave ^ is_convex) {
  //     return is_concave ? DrcCornerType::kConcave : DrcCornerType::kConvex;
  //   } else {
  //     return DrcCornerType::kNone;
  //   }
  // }

  // 摆正两个点的顺序
  // static bool sort2PointsByPolygonOrder(DrcBasicPoint*& point1, DrcBasicPoint*& point2)
  // {
  //   if (point2->nextEndpoint() == point1) {
  //     std::swap(point1, point2);
  //   } else if (point1->nextEndpoint() != point2) {
  //     return false;
  //   }
  //   return true;
  // }

  // 判断违例的函数
  // 思路都是一样的
  // 1. 摆正两个点的位置，方便后续使用几何信息
  // 2. 获得前后点，用来判断形状
  // 3. 计算获得形状或距离 condition
  // 4. 组合 condition 获得结果
  //
  // 后续检测时，每条规则会有一个先决的 spacing 或者 width 进入到 map 里面
  // 接着我们遍历 map 的所有 key，如果与当前规则的先决条件相同，则将对应线段的两个端点输入到下面这些方法中
  // 如果返回非空，则指明了违例区域
  // std::vector<DrcBasicPoint*> isStepViolation(DrcBasicPoint* point1, DrcBasicPoint* point2, int step_width_interval)
  // {
  //   // 1.
  //   if (!sort2PointsByPolygonOrder(point1, point2)) {
  //     return {};
  //   }

  //   // 2.
  //   auto* point_prev = point1->prevEndpoint();
  //   auto* point_next = point2->nextEndpoint();

  //   // 3.
  //   int distance_prev = point1->distance(point_prev);
  //   int distance_next = point2->distance(point_next);

  //   // 4.
  //   if (distance_prev < step_width_interval) {
  //     return {point_prev, point1, point2};
  //   } else if (distance_next < step_width_interval) {
  //     return {point1, point2, point_next};
  //   }

  //   return {};
  // }

  // std::vector<DrcBasicPoint*> isLef58StepViolation(DrcBasicPoint* point1, DrcBasicPoint* point2, int step_adjacent_width)
  // {
  //   if (!sort2PointsByPolygonOrder(point1, point2)) {
  //     return {};
  //   }
  //   auto* point_prev = point1->prevEndpoint();
  //   auto* point_next = point2->nextEndpoint();

  //   int distance_prev = point1->distance(point_prev);
  //   int distance_next = point2->distance(point_next);
  //   auto corner_point1 = cornerType(point1);
  //   auto corner_point2 = cornerType(point2);
  //   auto corner_prev = cornerType(point_prev);
  //   auto corner_next = cornerType(point_next);

  //   if (corner_prev == DrcCornerType::kConcave && corner_point1 == DrcCornerType::kConvex && corner_point2 == DrcCornerType::kConcave
  //       && distance_prev <= step_adjacent_width) {
  //     return {point_prev, point1, point2};
  //   } else if (corner_point1 == DrcCornerType::kConcave && corner_point2 == DrcCornerType::kConvex && corner_next ==
  //   DrcCornerType::kConcave
  //              && distance_next <= step_adjacent_width) {
  //     return {point1, point2, point_next};
  //   }

  //   return {};
  // }

  // std::vector<DrcBasicPoint*> isNotch(DrcBasicPoint* point1, DrcBasicPoint* point2, int notch_side_length)
  // {
  //   if (!sort2PointsByPolygonOrder(point1, point2)) {
  //     return {};
  //   }
  //   auto* point_prev = point1->prevEndpoint();
  //   auto* point_next = point2->nextEndpoint();

  //   int distance_prev = point1->distance(point_prev);
  //   int distance_next = point2->distance(point_next);
  //   auto corner_point1 = cornerType(point1);
  //   auto corner_point2 = cornerType(point2);
  //   auto corner_prev = cornerType(point_prev);
  //   auto corner_next = cornerType(point_next);

  //   if (corner_point1 == DrcCornerType::kConcave && corner_point2 == DrcCornerType::kConcave) {
  //     if (distance_prev < notch_side_length && corner_prev == DrcCornerType::kConcave && distance_next > notch_side_length
  //         || distance_next < notch_side_length && corner_next == DrcCornerType::kConcave && distance_prev > notch_side_length) {
  //       return {point_prev, point1, point2, point_next};
  //     }
  //   }
  //   return {};
  // }

  // static bool isOrthogonality(DrcDirection dir1, DrcDirection dir2)
  // {
  //   if (dir1 == DrcDirection::kNone || dir2 == DrcDirection::kNone) {
  //     return false;
  //   }

  //   if (dir1 == DrcDirection::kUp || dir1 == DrcDirection::kDown) {
  //     return dir2 == DrcDirection::kLeft || dir2 == DrcDirection::kRight;
  //   } else {
  //     return dir2 == DrcDirection::kUp || dir2 == DrcDirection::kDown;
  //   }
  // }

  // static DrcDirection directionPtToPt(DrcBasicPoint* point1, DrcBasicPoint* point2)
  // {
  //   if (point1->get_x() == point2->get_x()) {
  //     if (point1->get_y() < point2->get_y()) {
  //       return DrcDirection::kUp;
  //     } else {
  //       return DrcDirection::kDown;
  //     }
  //   } else if (point1->get_y() == point2->get_y()) {
  //     if (point1->get_x() < point2->get_x()) {
  //       return DrcDirection::kRight;
  //     } else {
  //       return DrcDirection::kLeft;
  //     }
  //   } else {
  //     return DrcDirection::kNone;
  //   }
  // }

  // std::pair<DrcBasicPoint*, DrcBasicPoint*> getOrthogonalEdge(DrcBasicPoint* point, DrcDirection dir)
  // {
  //   if (point->is_endpoint()) {
  //     if (isOrthogonality(dir, directionPtToPt(point, point->get_prev()))) {
  //       return std::make_pair(point->prevEndpoint(), point);
  //     } else {
  //       return std::make_pair(point, point->nextEndpoint());
  //     }
  //   } else {
  //     return std::make_pair(point->prevEndpoint(), point->nextEndpoint());
  //   }
  // }

  // static int getPRLBySpacing(DrcBasicPoint* point1, DrcBasicPoint* point2)
  // {
  //   DrcDirection dir = directionPtToPt(point1, point2);
  //   if (dir == DrcDirection::kNone) {
  //     return 0;
  //   }

  //   std::pair<DrcBasicPoint*, DrcBasicPoint*> segment1 = getOrthogonalEdge(point1, dir);
  //   std::pair<DrcBasicPoint*, DrcBasicPoint*> segment2 = getOrthogonalEdge(point2, dir);

  //   std::array<int, 4> intervals;

  //   if (segment1.first->get_x() == segment1.second->get_x()) {
  //     intervals[0] = segment1.first->get_y();
  //     intervals[1] = segment1.second->get_y();
  //     intervals[2] = segment2.first->get_y();
  //     intervals[3] = segment2.second->get_y();
  //   } else {
  //     intervals[0] = segment1.first->get_x();
  //     intervals[1] = segment1.second->get_x();
  //     intervals[2] = segment2.first->get_x();
  //     intervals[3] = segment2.second->get_x();
  //   }
  //   std::sort(intervals.begin(), intervals.end());

  //   return intervals[2] - intervals[1];
  // }

  // static bool isSameSide(DrcBasicPoint* point, DrcDirection direction, int target_id, int within = -1)
  // {
  //   ScanlineNeighbour* neighbour1 = point->get_neighbour(direction);
  //   ScanlineNeighbour* neighbour2 = nullptr;
  //   if (neighbour1) {
  //     neighbour2 = neighbour1->get_point()->get_neighbour(direction);

  //     if (neighbour1->is_spacing()) {
  //       if ((neighbour1->get_point()->get_id() != target_id) || (within > 0 && point->distance(neighbour1->get_point()) > within)) {
  //         return false;
  //       }
  //     } else if (neighbour1->is_width() && neighbour2 && neighbour2->is_spacing()) {
  //       if ((neighbour2->get_point()->get_id() != target_id) || (within > 0 && point->distance(neighbour2->get_point()) > within)) {
  //         return false;
  //       }
  //     } else {
  //       return false;
  //     }
  //   } else {
  //     return false;
  //   }
  //   return true;
  // }

  // static std::pair<int, int> walkToCliff(DrcBasicPoint* point, DrcDirection spacing_direction, int target_id, int within)
  // {
  //   std::function<int(DrcBasicPoint*)> get_coord;
  //   if (spacing_direction == DrcDirection::kUp || spacing_direction == DrcDirection::kDown) {
  //     get_coord = [](DrcBasicPoint* p) { return p->get_x(); };
  //   } else {
  //     get_coord = [](DrcBasicPoint* p) { return p->get_y(); };
  //   }

  //   std::pair<int, int> cliff{get_coord(point), get_coord(point)};
  //   auto walk_one_side = [&](std::function<DrcBasicPoint*(DrcBasicPoint*)> walk_method) {
  //     DrcBasicPoint* current_point = point;
  //     DrcBasicPoint* next_point = walk_method(current_point);
  //     while (next_point && next_point != point) {
  //       if (!isSameSide(next_point, spacing_direction, target_id, within)) {
  //         break;
  //       }

  //       cliff.first = std::min(cliff.first, get_coord(next_point));
  //       cliff.second = std::max(cliff.second, get_coord(next_point));

  //       current_point = next_point;
  //       next_point = walk_method(current_point);
  //     }
  //   };

  //   walk_one_side([](DrcBasicPoint* point) { return point->get_next(); });
  //   walk_one_side([](DrcBasicPoint* point) { return point->get_prev(); });

  //   return cliff;
  // }

  // static int getPRLByWithin(DrcBasicPoint* point1, DrcBasicPoint* point2, int within)
  // {
  //   DrcDirection dir = directionPtToPt(point1, point2);
  //   if (dir == DrcDirection::kNone) {
  //     return 0;
  //   }

  //   auto [low, high] = walkToCliff(point1, dir, point2->get_id(), within);

  //   return high - low;
  // }

 private:
};

}  // namespace idrc