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

#include "boost_definition.h"
#include "idrc_dm.h"
#include "scanline_point.h"

enum class ScanlineTravelDirection
{
  kVertical,
  kHorizontal
};

namespace idb {
class IdbLayer;
}  // namespace idb

namespace idrc {

class ScanlinePreprocess
{
 public:
  ScanlinePreprocess(std::string layer) : _layer(layer) {}
  ~ScanlinePreprocess();

  std::string get_layer() { return _layer; }

  void addData(std::vector<std::vector<ieda_solver::GtlPoint>>& polygons_points, int net_id);
  void addPolygon(std::vector<ieda_solver::GtlPoint>& polygon_points, int net_id, int net_polygon_id);
  void sortEndpoints();

  // getter
  std::vector<ScanlinePoint*>& get_scanline_points_vertical() { return _scanline_points_vertical; }
  std::vector<ScanlinePoint*>& get_scanline_points_horizontal() { return _scanline_points_horizontal; }

  std::vector<DrcBasicPoint*>& get_basic_points() { return _basic_points; }

  void reserveSpace(int n)
  {
    _basic_points.reserve(_basic_points.size() + n);
    _scanline_points_horizontal.reserve(_scanline_points_horizontal.size() + n);
    _scanline_points_vertical.reserve(_scanline_points_vertical.size() + n);
  }

  // setter
  void addBasicPoint(DrcBasicPoint* point) { _basic_points.push_back(point); };

 private:
  std::string _layer = nullptr;

  std::vector<DrcBasicPoint*> _basic_points;
  std::vector<ScanlinePoint*> _scanline_points_vertical;
  std::vector<ScanlinePoint*> _scanline_points_horizontal;

  int _polygon_count = 0;
  int _side_count = 0;

  template <typename T>
  void deleteVectorElements(T& v);

  std::pair<DrcBasicPoint*, DrcBasicPoint*> createPolygonEndpoints(std::vector<ieda_solver::GtlPoint>& polygon_points, int net_id,
                                                                   int net_polygon_id);
  void createScanlinePoints(DrcBasicPoint* start_point, std::function<std::pair<bool, bool>(DrcBasicPoint*, DrcBasicPoint*)> compare,
                            std::vector<ScanlinePoint*>& scanline_points);
};

}  // namespace idrc