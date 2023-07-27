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

#include "../../../../database/interaction/ids.hpp"
#include "LayerRect.hpp"
#include "RQShape.hpp"
#include "RTU.hpp"

namespace irt {

using StaticBox = std::pair<BoostBox, RQShape*>;

class RegionQuery
{
 public:
  RegionQuery() { init(); }
  ~RegionQuery() {}
  // getter
  // setters
  // function
  void init();
  void addEnvRectList(const ids::DRCRect& env_rect);
  void addEnvRectList(const std::vector<ids::DRCRect>& env_rect_list);
  std::vector<RQShape> getRQShapeList(const std::vector<ids::DRCRect>& env_rect_list);
  BoostBox convertBoostBox(ids::DRCRect ids_rect);
  void delEnvRectList(const ids::DRCRect& env_rect);
  void delEnvRectList(const std::vector<ids::DRCRect>& env_rect_list);
  bool hasViolation(const ids::DRCRect& drc_rect);
  bool hasViolation(const std::vector<ids::DRCRect>& drc_rect_list);
  std::map<std::string, int> getViolation(const std::vector<ids::DRCRect>& drc_rect_list);
  std::map<std::string, int> getViolation();
  std::map<std::string, int> checkByOther(std::vector<RQShape>& drc_shape_list);
  std::map<std::string, int> checkBySelf(std::vector<RQShape>& drc_shape_list);
  bool checkMinSpacing(RQShape& net_shape1, RQShape& net_shape2, std::vector<RQShape>& net_shape_list);
  std::vector<LayerRect> getMaxScope(const ids::DRCRect& drc_rect);
  std::vector<LayerRect> getMaxScope(const std::vector<ids::DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMinScope(const ids::DRCRect& drc_rect);
  std::vector<LayerRect> getMinScope(const std::vector<ids::DRCRect>& drc_rect_list);
  LayerRect convertToLayerRect(ids::DRCRect ids_rect);
  void plotRegionQuery(const std::vector<ids::DRCRect>& drc_rect_list);

 private:
  std::map<irt_int, std::vector<RQShape>> _obj_id_shape_map;
  std::vector<bgi::rtree<std::pair<BoostBox, RQShape*>, bgi::quadratic<16>>> _region_map;
};

}  // namespace irt
