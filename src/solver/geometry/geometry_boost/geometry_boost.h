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

#include <map>
#include <vector>

#include "../engine_geometry.h"
#include "../geometry_polygon.h"
#include "../geometry_polygon_set.h"
#include "../geometry_rectangle.h"
#include "boost_definition.h"

namespace ieda_solver {

/**
 * GeometryBoost describes the geometry data for a region in only one layer
 */

class PolygonProperty;

class GeometryBoost : public EngineGeometry
{
 public:
  GeometryBoost();
  ~GeometryBoost();

  GeometryPolygonSet& get_polyset() { return _polyset; }
  void addRect(int llx, int lly, int urx, int ury) override;
  // std::pair<uint64_t, std::vector<std::vector<GtlPoint>>> get_boost_polygons_points();
  // std::vector<std::vector<std::pair<int, int>>> get_polygons_points() override;

  void addGeometry(EngineGeometry* geometry) override;

  std::vector<GeometryPolygon>& getPolygons() override;
  std::vector<GeometryRect>& getOverlap() override;
  std::vector<GeometryRect>& getWires() override;

 private:
  // GeometryPolygonSet _overlap_set;
};

class PolygonProperty
{
 public:
  PolygonProperty() = default;
  ~PolygonProperty() = default;

 private:
};

}  // namespace ieda_solver