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

#include "scanline_preprocess.h"

#include "idrc_util.h"

namespace idrc {

template <typename T>
void ScanlinePreprocess::deleteVectorElements(T& v)
{
  for (auto& element : v) {
    delete element;
    element = nullptr;
  }
}

ScanlinePreprocess::~ScanlinePreprocess()
{
  deleteVectorElements(_basic_points);
  deleteVectorElements(_scanline_points_vertical);
  deleteVectorElements(_scanline_points_horizontal);
}

/**
 * add boost points to scanline data manager
 * @param
 * GtlPoint : boost point
 * std::vector<GtlPoint> : point list for one polygon
 * std::vector<std::vector<GtlPoint>> : define all point list in the polygon list
 * net_id : indicate id
 */
void ScanlinePreprocess::addData(std::vector<std::vector<ieda_solver::GtlPoint>>& polygons_points, int net_id)
{
  for (auto& polygon_points : polygons_points) {
    addPolygon(polygon_points, net_id);
  }
}

/// @brief add boost polygon to scanline data manager
/// @param polygon_points polygon endpoints clockwise
/// @param net_id polygon net_id
void ScanlinePreprocess::addPolygon(std::vector<ieda_solver::GtlPoint>& polygon_points, int net_id)
{
  auto start_points = createPolygonEndpoints(polygon_points, net_id);
  createScanlinePoints(
      start_points.first,
      [](DrcBasicPoint* p1, DrcBasicPoint* p2) {
        return p1->get_x() > p2->get_x() ? std::make_pair(false, false) : std::make_pair(true, true);
      },
      _scanline_points_horizontal);
  createScanlinePoints(
      start_points.second,
      [](DrcBasicPoint* p1, DrcBasicPoint* p2) {
        return p1->get_y() > p2->get_y() ? std::make_pair(true, false) : std::make_pair(false, true);
      },
      _scanline_points_vertical);
}

/// @brief sort scanline points in both horizontal and vertical direction
void ScanlinePreprocess::sortEndpoints()
{
  std::sort(_scanline_points_horizontal.begin(), _scanline_points_horizontal.end(), CompareScanlinePointByX());
  std::sort(_scanline_points_vertical.begin(), _scanline_points_vertical.end(), CompareScanlinePointByY());
}

std::pair<DrcBasicPoint*, DrcBasicPoint*> ScanlinePreprocess::createPolygonEndpoints(std::vector<ieda_solver::GtlPoint>& polygon_points,
                                                                                     int net_id)
{
  // _basic_points.reserve(_basic_points.size() + polygon_points.size());
  ComparePointByX<DrcBasicPoint> compare_by_x;
  ComparePointByY<DrcBasicPoint> compare_by_y;

  DrcBasicPoint* left_bottom_pt = nullptr;
  DrcBasicPoint* bottom_left_pt = nullptr;
  DrcBasicPoint* prev_point = nullptr;
  DrcBasicPoint* first_point = nullptr;
  int current_polygon_id = ++_polygon_count;
  for (auto& vertex : polygon_points) {
    DrcBasicPoint* new_basic_pt = new DrcBasicPoint(vertex.x(), vertex.y(), net_id, current_polygon_id);
    _basic_points.emplace_back(new_basic_pt);

    // find start point
    if (left_bottom_pt == nullptr) {
      left_bottom_pt = new_basic_pt;
    } else {
      if (!left_bottom_pt || compare_by_x(new_basic_pt, left_bottom_pt)) {
        left_bottom_pt = new_basic_pt;
      }
      if (!bottom_left_pt || compare_by_y(new_basic_pt, bottom_left_pt)) {
        bottom_left_pt = new_basic_pt;
      }
    }

    // link points
    if (prev_point != nullptr) {
      prev_point->set_next(new_basic_pt);
      new_basic_pt->set_prev(prev_point);
    } else {
      first_point = new_basic_pt;
    }

    prev_point = new_basic_pt;
  }
  first_point->set_prev(prev_point);
  prev_point->set_next(first_point);

  return std::make_pair(left_bottom_pt, bottom_left_pt);
}

void ScanlinePreprocess::createScanlinePoints(DrcBasicPoint* start_point,
                                              std::function<std::pair<bool, bool>(DrcBasicPoint*, DrcBasicPoint*)> compare,
                                              std::vector<ScanlinePoint*>& scanline_points)
{
  auto* endpoint1 = start_point;
  auto* endpoint2 = endpoint1->get_next();
  int side_id = ++_side_count;
  bool side_state = !compare(endpoint1, endpoint2).first;

  do {
    auto edge_state = compare(endpoint1, endpoint2);
    if (edge_state.first != side_state) {
      side_id = ++_side_count;
      side_state = edge_state.first;
    }
    ScanlinePoint* starting_point = new ScanlinePoint(endpoint1, side_id, edge_state.first, edge_state.second);
    ScanlinePoint* ending_point = new ScanlinePoint(endpoint2, side_id, edge_state.first, !edge_state.second);
    scanline_points.emplace_back(starting_point);
    scanline_points.emplace_back(ending_point);
    starting_point->set_pair(ending_point);
    ending_point->set_pair(starting_point);

    endpoint1 = endpoint2->get_next();
    endpoint2 = endpoint1->get_next();
  } while (endpoint1 != start_point);
}

}  // namespace idrc