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

#include "drc_basic_segment.h"
#include "scanline_preprocess.h"

namespace idrc {

class DrcConditionManager;
class DrcEngineManager;

struct ScanlineStatus
{
  ScanlineTravelDirection direction;

  std::vector<ScanlinePoint*>::iterator endpoints_it;
  std::vector<ScanlinePoint*>::iterator endpoints_end;

  std::unique_ptr<CompareScanlinePoint> compare_scanline_point = nullptr;

  int current_bucket_coord = 0;

  std::list<ScanlinePoint*> status_points;
  std::list<ScanlinePoint*>::iterator insert_begin = status_points.begin();
  std::list<ScanlinePoint*>::iterator insert_end = status_points.begin();

  std::function<int(DrcBasicPoint*)> get_travel_direction_coord;  // TODO: 模板 lambda
  std::function<int(DrcBasicPoint*)> get_orthogonal_coord;

  ScanlineStatus(ScanlineTravelDirection travel_direction, ScanlinePreprocess* preprocess) : direction(travel_direction)
  {
    switch (direction) {
      case ScanlineTravelDirection::kHorizontal:
        compare_scanline_point = std::make_unique<CompareScanlinePointHorizontal>();
        get_travel_direction_coord = []<typename T>(T* point) { return point->get_x(); };
        get_orthogonal_coord = []<typename T>(T* point) { return point->get_y(); };
        endpoints_it = preprocess->get_scanline_points_horizontal().begin();
        endpoints_end = preprocess->get_scanline_points_horizontal().end();
        break;
      case ScanlineTravelDirection::kVertical:
        compare_scanline_point = std::make_unique<CompareScanlinePointVertical>();
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

  std::vector<ScanlinePoint*>::iterator nextBucketEnd()
  {
    current_bucket_coord = get_travel_direction_coord((*endpoints_it)->get_point());
    auto it = endpoints_it;
    int starting_coord = get_travel_direction_coord((*it)->get_point());
    while ((++it) != endpoints_end && get_travel_direction_coord((*it)->get_point()) == starting_coord) {
    }
    return it;
  }

  void prepareNewBucket()
  {
    // mark old points in scanline status
    for (auto* point : status_points) {
      point->set_is_new(false);
    }

    insert_begin = status_points.begin();
    insert_end = status_points.begin();
  }
};

class DrcEngineScanline
{
 public:
  DrcEngineScanline(std::string layer, DrcEngineManager* engine_manager, DrcConditionManager* condition_manager)
      : _engine_manager(engine_manager), _condition_manager(condition_manager)
  {
    _preprocess = new ScanlinePreprocess(layer);
  }
  ~DrcEngineScanline();

  ScanlinePreprocess* get_preprocess() { return _preprocess; }

  void doScanline();

 private:
  ScanlinePreprocess* _preprocess;
  DrcEngineManager* _engine_manager;
  DrcConditionManager* _condition_manager;

  void scan(ScanlineTravelDirection direction);
  ScanlinePoint* recordOverlap(ScanlinePoint* point, ScanlinePoint* current_activate_point);
  void addCurrentBucketToScanline(ScanlineStatus& status);
  DrcSegmentType judgeSegmentType(ScanlineStatus& status, ScanlinePoint* point_forward, ScanlinePoint* point_backward);
  uint64_t hash2SideIds(int id1, int id2);
  template <typename T>
  void combineSequence(T& sequence, std::deque<DrcSegmentType>& segment_types);
  void processScanlineStatus(ScanlineStatus& status);
  void removeEndingPoints(ScanlineStatus& status);
};

}  // namespace idrc