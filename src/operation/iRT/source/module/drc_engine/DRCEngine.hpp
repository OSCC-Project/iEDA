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

#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Monitor.hpp"

namespace irt {

#define RTDE (irt::DRCEngine::getInst())

class DRCEngine
{
 public:
  static void initInst();
  static DRCEngine& getInst();
  static void destroyInst();
  // function
  std::vector<Violation> getViolationList(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                          std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                          std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_fixed_result_map,
                                          std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_result_map, std::string stage);

 private:
  // self
  static DRCEngine* _de_instance;

  DRCEngine() = default;
  DRCEngine(const DRCEngine& other) = delete;
  DRCEngine(DRCEngine&& other) = delete;
  ~DRCEngine() = default;
  DRCEngine& operator=(const DRCEngine& other) = delete;
  DRCEngine& operator=(DRCEngine&& other) = delete;
  // function
  std::vector<Violation> getViolationListBySelf(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_fixed_result_map,
                                                std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_result_map,
                                                std::string stage);
  std::vector<Violation> getViolationListByOther(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_fixed_result_map,
                                                std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_result_map,
                                                std::string stage);
};

}  // namespace irt
