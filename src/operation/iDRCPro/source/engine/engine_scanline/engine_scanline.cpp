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

#include "idrc_data.h"

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

/// @brief add all points with same travel direction coordinate to scanline status
/// @param status scanline status
void DrcEngineScanline::addCurrentBucketToScanline(ScanlineStatus& status)  // TODO: deal with overlap
{
  // mark old points in scanline status
  for (auto* point : status.status_points) {
    point->set_is_new(false);
  }

  // add new points to scanline status
  auto bucket_end = status.bucketEnd();
  auto scanline_status_it = status.status_points.begin();
  status.insert_begin = scanline_status_it;
  status.insert_end = scanline_status_it;
  bool is_first_insert = true;
  for (; status.endpoints_it != bucket_end; ++status.endpoints_it) {
    ScanlinePoint* current_point = *status.endpoints_it;
    int current_point_coord = status.get_orthogonal_coord((current_point)->get_point());
    if (current_point->get_is_start()) {
      // find correct position to insert
      while (scanline_status_it != status.status_points.end()
             && status.get_orthogonal_coord((*scanline_status_it)->get_point()) < current_point_coord) {
        ++scanline_status_it;
      }
      while (scanline_status_it != status.status_points.end()
             && status.get_orthogonal_coord((*scanline_status_it)->get_point()) == current_point_coord
             && (*scanline_status_it)->get_is_forward() > current_point->get_is_forward()) {
        ++scanline_status_it;
      }
      while (scanline_status_it != status.status_points.end()
             && status.get_orthogonal_coord((*scanline_status_it)->get_point()) == current_point_coord
             && (*scanline_status_it)->get_is_forward() == current_point->get_is_forward()
             && (*scanline_status_it)->get_id() < current_point->get_id()) {
        ++scanline_status_it;
      }
      status.status_points.insert(scanline_status_it, current_point);
    } else {
      // change point to ending endpoint in scanline status
      while (scanline_status_it != status.status_points.begin()
             && status.get_orthogonal_coord((*scanline_status_it)->get_point()) == current_point_coord) {
        --scanline_status_it;
      }  // TODO: 归并不应该往前找，排序没有统一吗
      auto* current_pair_point = current_point->get_pair();
      scanline_status_it = std::find(scanline_status_it, status.status_points.end(), current_pair_point);
      *scanline_status_it = current_point;
    }

    // first insert, mark range begin
    if (is_first_insert) {
      status.insert_begin = scanline_status_it;
      is_first_insert = false;
      for (int i = 0; i < 1 && status.insert_begin != status.status_points.begin(); ++i) {
        --status.insert_begin;
      }
    }
  }

  // mark range end
  status.insert_end = scanline_status_it;
  for (int i = 0; i < 1 && status.insert_end != status.status_points.end(); ++i) {
    ++status.insert_end;
  }
}

/// @brief determine the type of current segment
/// @param status scanline status
/// @param activate_polygons active polygons in current bucket
/// @param point_forward segment endpoint with bigger orthogonal coordinate
/// @param point_backward segment endpoint with smaller orthogonal coordinate
/// @return type of current segment
// ScanlineDataType DrcEngineScanline::judgeSegmentType(ScanlineStatus& status, std::map<int, ScanlinePoint*>& activate_polygons,
//                                                      ScanlinePoint* point_forward, ScanlinePoint* point_backward)
// {
//   ScanlineDataType result_type = ScanlineDataType::kNone;
//   if (activate_polygons.size() >= 2
//       || (!activate_polygons.empty() && point_forward->get_id() != point_backward->get_id()
//           && status.get_orthogonal_coord(point_forward->get_point()) == status.get_orthogonal_coord(point_backward->get_point()))) {
//     result_type += ScanlineDataType::kOverlap;
//   }
//   for (auto& activate_polygon : activate_polygons) {
//     if (status.isEndpointInCurrentBucket(activate_polygon.second) && activate_polygon.second->get_orthogonal_point()->get_is_start()) {
//       result_type += ScanlineDataType::kEdge;
//       result_type += ScanlineDataType::kWidth;
//     }
//   }
//   if ((status.isEndpointInCurrentBucket(point_forward) && !point_forward->get_orthogonal_point()->get_is_start())
//       || (status.isEndpointInCurrentBucket(point_backward) && point_backward->get_orthogonal_point()->get_is_start())) {
//     result_type += ScanlineDataType::kEdge;
//     result_type += ScanlineDataType::kWidth;
//   }
//   if (!activate_polygons.empty()
//       && ((!point_forward->get_is_forward() && !point_backward->get_is_forward())
//           || (point_forward->get_is_forward() && point_backward->get_is_forward()))
//       && point_forward->get_id() != point_backward->get_id()) {
//     result_type += ScanlineDataType::kInterSpacing;
//   }
//   if (!activate_polygons.empty() && point_forward->get_is_forward() && !point_backward->get_is_forward()) {
//     result_type += ScanlineDataType::kWidth;
//   }
//   if ((!point_forward->get_is_forward() && point_backward->get_is_forward())
//       || (point_forward->get_point()->get_id() != point_backward->get_point()->get_id())) {
//     result_type += ScanlineDataType::kSpacing;
//   }
//   return result_type;
// }

ScanlineSegmentType DrcEngineScanline::judgeSegmentType(ScanlineStatus& status, ScanlinePoint* point_forward, ScanlinePoint* point_backward)
{
  ScanlineSegmentType result_type = ScanlineSegmentType::kNone;
  // TODO: judge segment type
  return result_type;
}

/// @brief create neighbour for two basic points and fill result to basic point
/// @param status scanline status
/// @param basepoint_forward basepoint with bigger orthogonal coordinate
/// @param basepoint_backward basepoint with smaller orthogonal coordinate
/// @param result_type type of current segment
// void DrcEngineScanline::fillResultToBasicPoint(ScanlineStatus& status, DrcBasicPoint* basepoint_forward, DrcBasicPoint*
// basepoint_backward,
//                                                ScanlineDataType result_type)
// {
//   DrcDirection direction_forward;
//   DrcDirection direction_backward;

//   switch (status.direction) {
//     case ScanlineTravelDirection::kHorizontal:
//       direction_forward = DrcDirection::kDown;
//       direction_backward = DrcDirection::kUp;
//       break;
//     case ScanlineTravelDirection::kVertical:
//       direction_forward = DrcDirection::kLeft;
//       direction_backward = DrcDirection::kRight;
//       break;
//   }

//   auto* neighbour_forward = basepoint_forward->get_neighbour(direction_forward);
//   auto* neighbour_backward = basepoint_backward->get_neighbour(direction_backward);

//   if (neighbour_forward && neighbour_backward) {
//     neighbour_forward->add_type(result_type);
//     neighbour_backward->add_type(result_type);
//   } else {
//     neighbour_forward = new ScanlineNeighbour(result_type, basepoint_backward);
//     neighbour_backward = new ScanlineNeighbour(result_type, basepoint_forward);

//     basepoint_forward->set_neighbour(direction_forward, neighbour_forward);
//     basepoint_backward->set_neighbour(direction_backward, neighbour_backward);
//   }
// }

/// @brief create new non-endpoint if scanline shifted into an bucket with other traver direction coordinate
/// @param status scanline status
/// @param point current point possibliy has coordinate different from scanline coordinate
/// @return whether create new non-endpoint
bool DrcEngineScanline::tryCreateNonEndpoint(ScanlineStatus& status, ScanlinePoint* point)
{
  if (!status.isPointInCurrentBucket(point)) {
    int x = point->get_x();
    int y = point->get_y();
    DrcBasicPoint* first_point = nullptr;
    DrcBasicPoint* second_point = nullptr;

    switch (status.direction) {
      case ScanlineTravelDirection::kHorizontal:
        x = status.current_bucket_coord;
        if (point->get_is_forward()) {
          first_point = point->get_point();
          second_point = point->get_point()->get_next();
        } else {
          first_point = point->get_point()->get_prev();
          second_point = point->get_point();
        }
        break;
      case ScanlineTravelDirection::kVertical:
        y = status.current_bucket_coord;
        if (!point->get_is_forward()) {
          first_point = point->get_point();
          second_point = point->get_point()->get_next();
        } else {
          first_point = point->get_point()->get_prev();
          second_point = point->get_point();
        }
        break;
      default:
        break;
    }

    DrcBasicPoint* new_point
        = new DrcBasicPoint(x, y, point->get_id(), point->get_point()->get_polygon_id(), false, first_point, second_point);
    first_point->set_next(new_point);
    second_point->set_prev(new_point);

    _preprocess->addBasicPoint(new_point);
    point->set_point(new_point);
    point->add_created_point(new_point);

    return true;
  }
  return false;
}

/// @brief scanline status updated, process current status
/// @param status scanline status
void DrcEngineScanline::processScanlineStatus(ScanlineStatus& status)
{
  std::deque<ScanlinePoint*> activate_points;
  auto scanline_status_it = status.insert_begin;
  while (scanline_status_it != status.insert_end) {
    ScanlinePoint* point_backward = *scanline_status_it;
    if (++scanline_status_it != status.insert_end) {
      ScanlinePoint* point_forward = *scanline_status_it;
      // TODO: judge segment type
      // TODO: use type history to make sequence
      // TODO: put sequence to condition manager
    }
  }
  // std::map<int, ScanlinePoint*> activate_id_set;

  // // auto scanline_status_it = status.insert_begin;
  // // while (scanline_status_it != status.insert_end) {
  // //   ScanlinePoint* point_backward = *scanline_status_it;
  // //   if (++scanline_status_it != status.insert_end) {
  // // TODO: 像上面注释一样局部处理扫描线状态，目前局部处理会出现激活条件太远导致 overlap 不触发
  // auto scanline_status_it = status.status_points.begin();
  // while (scanline_status_it != status.status_points.end()) {
  //   ScanlinePoint* point_backward = *scanline_status_it;
  //   if (++scanline_status_it != status.status_points.end()) {
  //     ScanlinePoint* point_forward = *scanline_status_it;

  //     // refresh active polygon set
  //     bool is_ortho_start = point_backward->get_orthogonal_point()->get_is_start();
  //     bool is_forward = point_backward->get_is_forward();
  //     if (!is_forward || (is_ortho_start && status.isEndpointInCurrentBucket(point_backward))) {
  //       activate_id_set[point_backward->get_point()->get_id()] = point_backward;
  //     } else {
  //       activate_id_set.erase(point_backward->get_point()->get_id());
  //     }

  //     // bool is_near_new_points = false;
  //     // auto near_prev_it = std::prev(scanline_status_it);
  //     // auto near_next_it = scanline_status_it;
  //     // if (near_prev_it != status.status_points.begin() && --near_prev_it != status.status_points.begin()) {
  //     //   is_near_new_points = (*near_prev_it)->get_is_new();
  //     // }
  //     // if (++near_next_it != status.status_points.end()) {
  //     //   is_near_new_points = (*near_next_it)->get_is_new();
  //     // }

  //     bool at_least_one_new = point_forward->get_is_new() || point_backward->get_is_new();
  //     ScanlineDataType segment_type = judgeSegmentType(status, activate_id_set, point_forward, point_backward);

  //     if (segment_type.hasType(ScanlineDataType::kEdge) /*|| is_near_new_points*/ || at_least_one_new) {
  //       tryCreateNonEndpoint(status, point_forward);
  //       tryCreateNonEndpoint(status, point_backward);
  //       fillResultToBasicPoint(status, point_forward->get_point(), point_backward->get_point(), segment_type);

  //       // TODO: 重构连接功能，舍弃 ScanlinePoint 里面的 non endpoint 存储
  //       if (segment_type.hasType(ScanlineDataType::kOverlap)) {
  //         auto connect_overlap = [](DrcDirection dir, DrcBasicPoint* point1, DrcBasicPoint* point2) {
  //           auto point2_neighbour = point2->get_neighbour(dir);
  //           if (!point1->get_neighbour(dir) && point2_neighbour && point2_neighbour->is_overlap()) {
  //             auto neighbour = new ScanlineNeighbour(point2_neighbour->get_type(), point2_neighbour->get_point());
  //             point1->set_neighbour(dir, neighbour);
  //           }
  //         };

  //         std::vector<DrcDirection> directions{DrcDirection::kUp, DrcDirection::kDown, DrcDirection::kLeft, DrcDirection::kRight};
  //         for (auto& activate_polygon : activate_id_set) {
  //           if (status.isEndpointInCurrentBucket(activate_polygon.second)) {
  //             auto same_position_point_forward
  //                 = activate_polygon.second->get_orthogonal_point()->get_created_point(point_forward->get_point());
  //             auto same_position_point_backward
  //                 = activate_polygon.second->get_orthogonal_point()->get_created_point(point_backward->get_point());
  //             for (auto dir : directions) {
  //               if (same_position_point_forward) {
  //                 connect_overlap(dir, point_forward->get_point(), same_position_point_forward);
  //                 connect_overlap(dir, same_position_point_forward, point_forward->get_point());
  //               }
  //               if (same_position_point_backward) {
  //                 connect_overlap(dir, point_backward->get_point(), same_position_point_backward);
  //                 connect_overlap(dir, same_position_point_backward, point_backward->get_point());
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
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