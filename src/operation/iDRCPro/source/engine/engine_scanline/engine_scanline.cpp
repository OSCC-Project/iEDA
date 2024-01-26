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
#include "idrc_data.h"
#include "idrc_util.h"

namespace std {
template <typename T>
inline void hash_combine(std::size_t& seed, T const& v)
{
  seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename U, typename V>
struct hash<std::pair<U, V>>
{
  size_t operator()(std::pair<U, V> const& tt) const
  {
    size_t seed = 0;
    hash_combine(seed, tt.first);
    hash_combine(seed, tt.second);
    return seed;
  }
};
}  // namespace std

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
// TODO: deal with overlap, one direction is enough
void DrcEngineScanline::addCurrentBucketToScanline(ScanlineStatus& status)
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
      for (int i = 0; i < 2 && status.insert_begin != status.status_points.begin(); ++i) {
        --status.insert_begin;
      }
    }
  }

  // mark range end
  status.insert_end = scanline_status_it;
  for (int i = 0; i < 2 && status.insert_end != status.status_points.end(); ++i) {
    ++status.insert_end;
  }
}

/// @brief determine segment type while scanline process
/// @param status scanline status
/// @param point_forward forward point of current segment
/// @param point_backward backward point of current segment
/// @return segment type
ScanlineSegmentType DrcEngineScanline::judgeSegmentType(ScanlineStatus& status, ScanlinePoint* point_forward, ScanlinePoint* point_backward)
{
  ScanlineSegmentType result_type = ScanlineSegmentType::kNone;
  // TODO: judge segment type
  if (point_forward->get_point()->is_endpoint() && point_backward->get_point()->is_endpoint()
      && !point_forward->get_orthogonal_point()->get_is_start() && point_backward->get_orthogonal_point()->get_is_start()) {
    result_type = ScanlineSegmentType::kEdge;
  } else if (!point_forward->get_is_forward() && point_backward->get_is_forward()) {
    result_type = ScanlineSegmentType::kSpacing;
  } else if (point_forward->get_is_forward() && !point_backward->get_is_forward()) {
    result_type = ScanlineSegmentType::kWidth;
  }
  return result_type;
}

/// @brief hash two side ids to one
/// @param id1 one side id
/// @param id2 another side id
/// @return hashed side id
uint64_t DrcEngineScanline::hash2SideIds(int id1, int id2)
{
  std::hash<std::pair<int, int>> hasher;
  return hasher(std::make_pair(id1, id2));
}

/// @brief scanline status updated, process current status
/// @param status scanline status
void DrcEngineScanline::processScanlineStatus(ScanlineStatus& status)
{
  std::deque<ScanlinePoint*> activate_points{nullptr, nullptr, nullptr};
  std::deque<ScanlineSegmentType> activate_types{ScanlineSegmentType::kNone, ScanlineSegmentType::kNone, ScanlineSegmentType::kNone,
                                                 ScanlineSegmentType::kNone};
  auto scanline_status_it = status.insert_begin;
  if (scanline_status_it != status.status_points.end()) {
    activate_points.push_back(*scanline_status_it);
  }
  while (scanline_status_it != status.insert_end) {
    ScanlinePoint* point_backward = *scanline_status_it;
    if (++scanline_status_it != status.insert_end) {
      ScanlinePoint* point_forward = *scanline_status_it;

      // get current segment type
      auto type = judgeSegmentType(status, point_forward, point_backward);

      // refresh active segments
      activate_points.pop_front();
      activate_types.pop_front();
      activate_points.push_back(point_forward);
      activate_types.push_back(type);

      // skip both old points
      if (!point_forward->get_is_new() && !point_backward->get_is_new()) {
        continue;
      }

      // make sequence, make edge ids hash
      ConditionSequence::SequenceType sequence = ConditionSequence::SequenceType::kNone;
      // TODO: use activate types to determine sequence refactor
      if ((activate_types[0] == ScanlineSegmentType::kWidth && activate_types[1] == ScanlineSegmentType::kSpacing
           && activate_types[2] == ScanlineSegmentType::kEdge && activate_types[3] == ScanlineSegmentType::kWidth)
          || (activate_types[0] == ScanlineSegmentType::kWidth && activate_types[1] == ScanlineSegmentType::kEdge
              && activate_types[2] == ScanlineSegmentType::kSpacing && activate_types[3] == ScanlineSegmentType::kWidth)) {
        sequence = ConditionSequence::SequenceType::kWSEW_WESW;
      } else if (activate_types[1] == ScanlineSegmentType::kEdge && activate_types[2] == ScanlineSegmentType::kSpacing
                 && activate_types[3] == ScanlineSegmentType::kEdge) {
        sequence = ConditionSequence::SequenceType::kESE;
      } else if ((activate_types[1] == ScanlineSegmentType::kEdge && activate_types[2] == ScanlineSegmentType::kSpacing
                  && activate_types[3] == ScanlineSegmentType::kWidth)
                 || (activate_types[1] == ScanlineSegmentType::kWidth && activate_types[2] == ScanlineSegmentType::kSpacing
                     && activate_types[3] == ScanlineSegmentType::kEdge)) {
        sequence = ConditionSequence::SequenceType::kESW_WSE;
      }

      uint64_t recognize_code = 0;
      for (int i = 1; i < (int) activate_types.size(); ++i) {
        if (activate_types[i] == ScanlineSegmentType::kSpacing) {
          auto gtl_pts_1 = DrcUtil::getPolygonPoints(activate_points[i - 1]->get_point());
          auto gtl_pts_2 = DrcUtil::getPolygonPoints(activate_points[i]->get_point());
          recognize_code = hash2SideIds(activate_points[i - 1]->get_side_id(), activate_points[i]->get_side_id());
          break;
        }
      }

      // put sequence to condition manager
      if (_condition_manager->isSequenceNeedDeliver(_preprocess->get_layer(), recognize_code, sequence)) {
        std::vector<DrcBasicPoint*> point_list(activate_points.size(), nullptr);
        for (int i = 0; i < (int) activate_points.size(); ++i) {
          point_list[i] = activate_points[i] ? activate_points[i]->get_point() : nullptr;
        }

        _condition_manager->recordRegion(_preprocess->get_layer(), recognize_code, sequence, point_list);
      }
    }
  }
  // TODO: 处理边缘情况，即插入点在最边缘时应多处理一段，保证处理到边缘的 NEN 或者其他情况
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