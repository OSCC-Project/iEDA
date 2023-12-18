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
#include "scanline_point.h"

enum class ScanlineTravelDirection
{
  kVertical,
  kHorizontal
};

namespace idrc {

class ScanlineDataManager
{
 public:
  ScanlineDataManager(int layer_id) : _layer_id(layer_id) {}
  ~ScanlineDataManager();

  int get_layer_id() { return _layer_id; }

  void addData(std::vector<std::vector<ieda_solver::GtlPoint>>& polygons_points, int net_id);
  void addPolygon(std::vector<ieda_solver::GtlPoint>& polygon_points, int net_id);
  void sortEndpoints();

  // getter
  std::vector<ScanlinePoint*>& get_scanline_points_vertical() { return _scanline_points_vertical; }
  std::vector<ScanlinePoint*>& get_scanline_points_horizontal() { return _scanline_points_horizontal; }

  std::vector<DrcBasicPoint*>& get_basic_points() { return _basic_points; }

  /// @brief  debug
  /// @return
  std::vector<ieda_solver::BgPoint> get_boost_points()
  {
    std::vector<ieda_solver::BgPoint> boost_pts;
    boost_pts.reserve(_basic_points.size());
    for (auto& pt : _basic_points) {
      ieda_solver::BgPoint boost_pt(pt->get_x(), pt->get_y());
      boost_pts.emplace_back(boost_pt);
    }

    return boost_pts;
  }

  std::vector<ieda_solver::BgPolygon> get_boost_polygons()
  {
    std::vector<ieda_solver::BgPolygon> boost_polygons;
    for (auto& pt : _basic_points) {
      if (pt->is_start()) {
        std::vector<ieda_solver::BgPoint> boost_pts;
        auto* iter_pt = pt;
        while (iter_pt != nullptr) {
          ieda_solver::BgPoint boost_pt(iter_pt->get_x(), iter_pt->get_y());
          boost_pts.emplace_back(boost_pt);

          iter_pt = iter_pt->get_next();

          if (iter_pt == pt) {
            break;
          }
        }
        ieda_solver::BgPolygon boost_polygon;
        boost::geometry::assign_points(boost_polygon, boost_pts);
        boost_polygons.emplace_back(boost_polygon);
      }
    }

    return boost_polygons;
  }

  // setter
  void addBasicPoint(DrcBasicPoint* point) { _basic_points.push_back(point); };

 private:
  int _layer_id = -1;

  std::vector<DrcBasicPoint*> _basic_points;
  std::vector<ScanlinePoint*> _scanline_points_vertical;
  std::vector<ScanlinePoint*> _scanline_points_horizontal;

  template <typename T>
  void deleteVectorElements(T& v);
};

}  // namespace idrc