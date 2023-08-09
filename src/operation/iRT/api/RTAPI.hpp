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

#include <any>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../../database/interaction/ids.hpp"

namespace irt {

#define RTAPI_INST (irt::RTAPI::getInst())

enum class Tool
{
  kDetailedRouter,
  kGlobalRouter,
  kPinAccessor,
  kResourceAllocator,
  kTrackAssigner,
  kViolationRepairer
};

class RTAPI
{
 public:
  static RTAPI& getInst();
  static void destroyInst();

  // RT
  void initRT(std::map<std::string, std::any> config_map);
  void runRT(std::vector<Tool> tool_list);
  Stage convertToStage(Tool tool);
  void destroyRT();

  // EGR
  void runEGR(std::map<std::string, std::any> config_map);

  // EVAL
  eval::TileGrid* getCongestonMap(std::map<std::string, std::any> config_map);
  std::vector<double> getWireLengthAndViaNum(std::map<std::string, std::any> config_map);

  // DRC
  void* initRegionQuery();
  void destroyRegionQuery(void* region_query);
  void addEnvRectList(void* region_query, const ids::DRCRect& env_rect);
  void addEnvRectList(void* region_query, const std::vector<ids::DRCRect>& env_rect_list);
  void delEnvRectList(void* region_query, const ids::DRCRect& env_rect);
  void delEnvRectList(void* region_query, const std::vector<ids::DRCRect>& env_rect_list);
  bool hasViolation(void* region_query, const ids::DRCRect& drc_rect);
  bool hasViolation(void* region_query, const std::vector<ids::DRCRect>& drc_rect_list);
  std::map<std::string, int> getViolation(void* region_query);
  std::map<std::string, int> getViolation(void* region_query, const std::vector<ids::DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMaxScope(const std::vector<ids::DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMinScope(const std::vector<ids::DRCRect>& drc_rect_list);
  std::vector<LayerRect> getMaxScope(const ids::DRCRect& drc_rect);
  std::vector<LayerRect> getMinScope(const ids::DRCRect& drc_rect);
  LayerRect convertToLayerRect(ids::DRCRect ids_rect);
  ids::DRCRect convertToIDSRect(int net_idx, LayerRect rt_rect, bool is_routing);
  // void plotRegionQuery(void* region_query, const std::vector<ids::DRCRect>& drc_rect_list);

  // CTS
  std::vector<ids::PHYNode> getPHYNodeList(std::vector<ids::Segment> segment_list);

 private:
  static RTAPI* _rt_api_instance;

  RTAPI() = default;
  RTAPI(const RTAPI& other) = delete;
  RTAPI(RTAPI&& other) = delete;
  ~RTAPI() = default;
  RTAPI& operator=(const RTAPI& other) = delete;
  RTAPI& operator=(RTAPI&& other) = delete;
  // function
};

}  // namespace irt
