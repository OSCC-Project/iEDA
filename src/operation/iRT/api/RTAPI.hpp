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

  // AI
  void runGRToAI(std::string ai_json_file_path, int lower_bound_value, int upper_bound_value);

  // EVAL
  eval::TileGrid* getCongestonMap(std::map<std::string, std::any> config_map);
  std::vector<double> getWireLengthAndViaNum(std::map<std::string, std::any> config_map);

  // DRC
  bool hasViolation(std::vector<LayerRect> environment, const LayerRect& drc_rect);
  bool hasViolation(std::vector<LayerRect> environment, const std::vector<LayerRect>& drc_rect_list);
  void* initRegionQuery();
  void addEnvRectList(void* region_query, const std::vector<LayerRect>& env_rect_list);
  bool hasViolation(void* region_query, const std::vector<LayerRect>& drc_rect_list);
  std::vector<LayerRect> getMaxScope(const std::vector<LayerRect>& drc_rect_list);
  std::vector<LayerRect> getMinScope(const std::vector<LayerRect>& drc_rect_list);
  std::vector<LayerRect> getMaxScope(const LayerRect& drc_rect);
  std::vector<LayerRect> getMinScope(const LayerRect& drc_rect);
  LayerRect convertToRTRect(ids::DRCRect ids_rect);
  ids::DRCRect covertToIDSRect(LayerRect rt_rect);

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
