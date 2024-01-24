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
#include "scanline_preprocess.h"

namespace idrc {

enum class ScanlineSegmentType
{
  kNone,
  kOverlap,
  kEdge,
  kSpacing,
  kInterSpacing,
  kWidth
};

struct ScanlineStatus
{
  ScanlineTravelDirection direction;

  std::vector<ScanlinePoint*>::iterator endpoints_it;
  std::vector<ScanlinePoint*>::iterator endpoints_end;

  int current_bucket_coord = 0;

  std::list<ScanlinePoint*> status_points;
  std::list<ScanlinePoint*>::iterator insert_begin = status_points.end();
  std::list<ScanlinePoint*>::iterator insert_end = status_points.end();

  std::function<int(DrcBasicPoint*)> get_travel_direction_coord;  // TODO: 模板 lambda
  std::function<int(DrcBasicPoint*)> get_orthogonal_coord;

  ScanlineStatus(ScanlineTravelDirection travel_direction, ScanlinePreprocess* preprocess) : direction(travel_direction)
  {
    switch (direction) {
      case ScanlineTravelDirection::kHorizontal:
        get_travel_direction_coord = []<typename T>(T* point) { return point->get_x(); };
        get_orthogonal_coord = []<typename T>(T* point) { return point->get_y(); };
        endpoints_it = preprocess->get_scanline_points_horizontal().begin();
        endpoints_end = preprocess->get_scanline_points_horizontal().end();
        break;
      case ScanlineTravelDirection::kVertical:
        get_travel_direction_coord = []<typename T>(T* point) { return point->get_y(); };
        get_orthogonal_coord = []<typename T>(T* point) { return point->get_x(); };
        endpoints_it = preprocess->get_scanline_points_vertical().begin();
        endpoints_end = preprocess->get_scanline_points_vertical().end();
        break;
      default:
        //   std::cout << "scanline error: direction" << std::endl;
        break;
    }
  }

  std::vector<ScanlinePoint*>::iterator bucketEnd()
  {
    current_bucket_coord = get_travel_direction_coord((*endpoints_it)->get_point());
    auto it = endpoints_it;
    int starting_coord = get_travel_direction_coord((*it)->get_point());
    while ((++it) != endpoints_end && get_travel_direction_coord((*it)->get_point()) == starting_coord) {
    }
    return it;
  }

  bool isPointInCurrentBucket(ScanlinePoint* point) { return get_travel_direction_coord(point->get_point()) == current_bucket_coord; }
  bool isEndpointInCurrentBucket(ScanlinePoint* point) { return point->get_point()->is_endpoint() && isPointInCurrentBucket(point); }
};

class DrcEngineScanline
{
 public:
  DrcEngineScanline(idb::IdbLayer* layer, DrcDataManager* data_manager) : _data_manager(data_manager)
  {
    _preprocess = new ScanlinePreprocess(layer, data_manager);
  }
  ~DrcEngineScanline();

  ScanlinePreprocess* get_preprocess() { return _preprocess; }

  void doScanline();

 private:
  ScanlinePreprocess* _preprocess;
  DrcDataManager* _data_manager;

  void scan(ScanlineTravelDirection direction);
  void addCurrentBucketToScanline(ScanlineStatus& status);
  ScanlineSegmentType judgeSegmentType(ScanlineStatus& status, ScanlinePoint* point_forward, ScanlinePoint* point_backward);
  // ScanlineDataType judgeSegmentType(ScanlineStatus& status, std::map<int, ScanlinePoint*>& activate_polygons, ScanlinePoint*
  // point_forward,
  //                                   ScanlinePoint* point_backward);
  // void fillResultToBasicPoint(ScanlineStatus& status, DrcBasicPoint* basepoint_forward, DrcBasicPoint* basepoint_backward,
  //                             ScanlineDataType result_type);
  bool tryCreateNonEndpoint(ScanlineStatus& status, ScanlinePoint* point);
  void processScanlineStatus(ScanlineStatus& status);
  void removeEndingPoints(ScanlineStatus& status);
};

}  // namespace idrc