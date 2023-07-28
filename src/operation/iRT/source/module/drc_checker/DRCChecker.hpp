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

#include "DRCRect.hpp"
#include "DataManager.hpp"

namespace irt {

#define DC_INST (irt::DRCChecker::getInst())

class DRCChecker
{
 public:
  static void initInst();
  static DRCChecker& getInst();
  static void destroyInst();
  // function
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, MTree<PHYNode>& phy_node_tree);
  void* initRegionQuery();
  void addEnvRectList(void* region_query, const DRCRect& env_rect);
  void addEnvRectList(void* region_query, const std::vector<DRCRect>& drc_rect_list);
  void delEnvRectList(void* region_query, const DRCRect& env_rect);
  void delEnvRectList(void* region_query, const std::vector<DRCRect>& drc_rect_list);
  bool hasViolation(void* region_query, const DRCRect& drc_rect);
  bool hasViolation(void* region_query, const std::vector<DRCRect>& drc_rect_list);
  std::map<std::string, int> getViolation(void* region_query);
  std::map<std::string, int> getViolation(void* region_query, const std::vector<DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMaxScope(const std::vector<DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMinScope(const std::vector<DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMaxScope(const DRCRect& drc_rect);
  std::vector<LayerRect> getMinScope(const DRCRect& drc_rect);
  void plotRegionQuery(void* region_query, const std::vector<DRCRect>& drc_rect_list);

 private:
  // self
  static DRCChecker* _dc_instance;

  DRCChecker() = default;
  DRCChecker(const DRCChecker& other) = delete;
  DRCChecker(DRCChecker&& other) = delete;
  ~DRCChecker() = default;
  DRCChecker& operator=(const DRCChecker& other) = delete;
  DRCChecker& operator=(DRCChecker&& other) = delete;
};
}  // namespace irt
