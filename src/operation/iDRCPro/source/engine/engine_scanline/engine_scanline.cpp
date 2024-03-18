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

#include "engine_scanline.h"

#include "condition_manager.h"
#include "geometry_boost.h"
#include "idrc_data.h"
#include "idrc_engine_manager.h"
#include "idrc_util.h"

namespace idrc {

DrcEngineScanline::~DrcEngineScanline()
{
  if (_preprocess != nullptr) {
    delete _preprocess;
    _preprocess = nullptr;
  }
}

/// @brief process scanline in both of the directions
void DrcEngineScanline::doScanline()
{
  scan(ScanlineTravelDirection::kHorizontal);
  scan(ScanlineTravelDirection::kVertical);
}

/// @brief process scanline in one direction
/// @param direction scanline travel direction
void DrcEngineScanline::scan(ScanlineTravelDirection direction)
{
  ScanlineStatus scanline_status(direction, _preprocess);

  while (scanline_status.endpoints_it != scanline_status.endpoints_end) {
    // add all endpoints in current bucket to scanline status
    addCurrentBucketToScanline(scanline_status);

    // process scanline status once
    processScanlineStatus(scanline_status);

    // remove ending points
    removeEndingPoints(scanline_status);
  }
}

// TODO: deal with overlap, one direction is enough
ScanlinePoint* DrcEngineScanline::recordOverlap(ScanlinePoint* point, ScanlinePoint* current_activate_point)
{
  // auto get_polygon = [&](ScanlinePoint* point) -> ieda_solver::GtlPolygon& {
  //   auto* geometry_engine
  //       = _engine_manager->get_layout(_preprocess->get_layer())->get_sub_layout(point->get_point()->get_net_id())->get_engine();
  //   auto* boost_engine = static_cast<ieda_solver::GeometryBoost*>(geometry_engine);
  //   auto& polygons = boost_engine->get_polygon_list();
  //   auto& polygon = polygons[point->get_point()->get_net_polygon_id()];
  //   return polygon;
  // };

  if (!current_activate_point && !point->get_is_forward()) {
    current_activate_point = point;
  } else if (current_activate_point && (point)->get_is_forward() && current_activate_point->get_id() == point->get_id()) {
    current_activate_point = nullptr;
  } else if (current_activate_point && current_activate_point->get_id() != point->get_id()) {
    // TODO: overlap
    // auto& polygon_1 = get_polygon(current_activate_point);
    // auto& polygon_2 = get_polygon(point);
    // std::vector<ieda_solver::GtlPolygon> overlap_list;
    // ieda_solver::GtlPolygonSet intersection = polygon_1 & polygon_2;
    // intersection.get(overlap_list);
    // int a = 0;
  }
  return current_activate_point;
}

/// @brief add all points with same travel direction coordinate to scanline status
/// @param status scanline status
void DrcEngineScanline::addCurrentBucketToScanline(ScanlineStatus& status)
{
  status.prepareNewBucket();

  // add new points to scanline status
  auto bucket_end = status.nextBucketEnd();
  auto scanline_status_it = status.status_points.begin();
  bool is_first_insert = true;
  ScanlinePoint* current_activate_point = nullptr;

  for (; status.endpoints_it != bucket_end; ++status.endpoints_it) {
    ScanlinePoint* current_point = *status.endpoints_it;

    if (current_point->get_is_start()) {
      // find correct position to insert
      while (scanline_status_it != status.status_points.end() && (*status.compare_scanline_point)(*scanline_status_it, current_point)) {
        current_activate_point = recordOverlap(*scanline_status_it, current_activate_point);
        ++scanline_status_it;
      }
      scanline_status_it = status.status_points.insert(scanline_status_it, current_point);
    } else {
      // point is ending point, replace pair starting point in scanline status
      auto* current_pair_point = current_point->get_pair();
      while (scanline_status_it != status.status_points.end() && *scanline_status_it != current_pair_point) {
        current_activate_point = recordOverlap(*scanline_status_it, current_activate_point);
        ++scanline_status_it;
      }
      *scanline_status_it = current_point;
    }

    // first insert, mark range begin
    if (is_first_insert) {
      status.insert_begin = scanline_status_it;
      is_first_insert = false;
      for (int i = 0; i < 2 && status.insert_begin != status.status_points.begin(); ++i) {
        --status.insert_begin;
      }
    }
  }

  // mark range end
  status.insert_end = scanline_status_it;
  for (int i = 0; i < 3 && status.insert_end != status.status_points.end(); ++i) {
    ++status.insert_end;
  }
}

/// @brief determine segment type while scanline process
/// @param status scanline status
/// @param point_forward forward point of current segment
/// @param point_backward backward point of current segment
/// @return segment type
DrcSegmentType DrcEngineScanline::judgeSegmentType(ScanlineStatus& status, ScanlinePoint* point_forward, ScanlinePoint* point_backward)
{
  if (point_forward->get_is_new() && point_backward->get_is_new()
      && (point_forward->get_point()->get_next() == point_backward->get_point()
          || point_backward->get_point()->get_next() == point_forward->get_point())) {
    if (point_forward->get_is_start() && point_backward->get_is_start() && point_forward->get_is_forward()
        && !point_backward->get_is_forward()) {
      return DrcSegmentType::kConvexStartEdge;
    } else if (point_forward->get_is_start() && point_backward->get_is_start() && !point_forward->get_is_forward()
               && point_backward->get_is_forward()) {
      return DrcSegmentType::kConcaveStartEdge;
    } else if (!point_forward->get_is_start() && !point_backward->get_is_start() && point_forward->get_is_forward()
               && !point_backward->get_is_forward()) {
      return DrcSegmentType::kConvexEndEdge;
    } else if (!point_forward->get_is_start() && !point_backward->get_is_start() && !point_forward->get_is_forward()
               && point_backward->get_is_forward()) {
      return DrcSegmentType::kConcaveEndEdge;
    } else if ((point_forward->get_is_forward() && point_forward->get_is_start())
               || (!point_forward->get_is_forward() && !point_forward->get_is_start())) {
      return DrcSegmentType::kTurnOutEdge;
    } else {
      return DrcSegmentType::kTurnInEdge;
    }
  } else if (!point_forward->get_is_forward() && point_backward->get_is_forward()) {
    if (point_forward->get_id() != point_backward->get_id()) {
      return DrcSegmentType::kMutualSpacing;
    } else {
      return DrcSegmentType::kSlefSpacing;
    }
  } else if (point_forward->get_is_forward() && !point_backward->get_is_forward()) {
    return DrcSegmentType::kWidth;
  }
  return DrcSegmentType::kNone;
}

/// @brief hash two side ids to one
/// @param id1 one side id
/// @param id2 another side id
/// @return hashed side id
uint64_t DrcEngineScanline::hash2SideIds(int id1, int id2)
{
  // TODO: 碰撞
  return DrcUtil::hash(id1, id2);
}

/// @brief combine sequence to one enum
/// @tparam T
/// @param sequence
/// @param segment_types
// template <typename T>
// inline void DrcEngineScanline::combineSequence(T& sequence, std::deque<DrcSegmentType>& segment_types)
// {
//   if (segment_types[1] == DrcSegmentType::kMutualSpacing) {
//     if ((segment_types[0] == DrcSegmentType::kStartingEdge && segment_types[2] == DrcSegmentType::kWidth)
//         || (segment_types[0] == DrcSegmentType::kWidth && segment_types[2] == DrcSegmentType::kStartingEdge)) {
//       sequence = ConditionSequence::SequenceType::kSE_MS_W;
//     } else if (segment_types[0] == DrcSegmentType::kStartingEdge && segment_types[2] == DrcSegmentType::kStartingEdge) {
//       sequence = ConditionSequence::SequenceType::kSE_MS_SE;
//     } else if ((segment_types[0] == DrcSegmentType::kStartingEdge && segment_types[2] == DrcSegmentType::kTurningEdge)
//                || (segment_types[0] == DrcSegmentType::kTurningEdge && segment_types[2] == DrcSegmentType::kStartingEdge)) {
//       sequence = ConditionSequence::SequenceType::kSE_MS_TE;
//     } else if ((segment_types[0] == DrcSegmentType::kTurningEdge && segment_types[2] == DrcSegmentType::kWidth)
//                || (segment_types[0] == DrcSegmentType::kWidth && segment_types[2] == DrcSegmentType::kTurningEdge)) {
//       sequence = ConditionSequence::SequenceType::kTE_MS_W;
//     } else if (segment_types[0] == DrcSegmentType::kTurningEdge && segment_types[2] == DrcSegmentType::kTurningEdge) {
//       sequence = ConditionSequence::SequenceType::kTE_MS_TE;
//     } else if ((segment_types[0] == DrcSegmentType::kEndingEdge && segment_types[2] == DrcSegmentType::kWidth)
//                || (segment_types[0] == DrcSegmentType::kWidth && segment_types[2] == DrcSegmentType::kEndingEdge)) {
//       sequence = ConditionSequence::SequenceType::kEE_MS_W;
//     } else if (segment_types[0] == DrcSegmentType::kEndingEdge && segment_types[2] == DrcSegmentType::kEndingEdge) {
//       sequence = ConditionSequence::SequenceType::kEE_MS_EE;
//     } else if ((segment_types[0] == DrcSegmentType::kEndingEdge && segment_types[2] == DrcSegmentType::kTurningEdge)
//                || (segment_types[0] == DrcSegmentType::kTurningEdge && segment_types[2] == DrcSegmentType::kEndingEdge)) {
//       sequence = ConditionSequence::SequenceType::kEE_MS_TE;
//     }
//   }
// }

/// @brief scanline status updated, process current status
/// @param status scanline status
void DrcEngineScanline::processScanlineStatus(ScanlineStatus& status)
{
  // std::deque<ScanlinePoint*> activate_points;
  // std::deque<DrcSegmentType> activate_types;
  auto scanline_status_it = status.insert_begin;
  // if (scanline_status_it != status.status_points.end()) {
  //   activate_points.push_back(*scanline_status_it);
  // }

  while (scanline_status_it != status.insert_end) {
    ScanlinePoint* point_backward = *scanline_status_it;
    if (++scanline_status_it != status.insert_end) {
      ScanlinePoint* point_forward = *scanline_status_it;

      // get current segment type
      auto type = judgeSegmentType(status, point_forward, point_backward);

      // activate_types.push_back(type);
      // activate_points.push_back(point_forward);
      // if (type.isEdge()) {
      //   _condition_manager->recordEdge(_preprocess->get_layer(), point_forward->get_id(), type, point_forward->get_point(),
      //                                  point_backward->get_point());
      // }
    }
  }
}

/// @brief remove ending points
/// @param status scanline status
void DrcEngineScanline::removeEndingPoints(ScanlineStatus& status)
{
  status.status_points.erase(
      std::remove_if(status.status_points.begin(), status.status_points.end(), [](ScanlinePoint* point) { return !point->get_is_start(); }),
      status.status_points.end());
}

}  // namespace idrc