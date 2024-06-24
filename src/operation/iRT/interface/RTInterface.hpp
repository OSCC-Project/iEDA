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

namespace ieda_feature {
class RTSummary;
}  // namespace ieda_feature

namespace irt {

#define RTI (irt::RTInterface::getInst())

class RTInterface
{
 public:
  static RTInterface& getInst();
  static void destroyInst();

#if 1  // 外部调用RT的API
  // RT主要函数
  void initRT(std::map<std::string, std::any> config_map);
  void runEGR();
  void runRT();
  void destroyRT();
  // 清理def
  void clearDef();
#endif

#if 1  // RT调用外部的API
  // 调用iDRC 计算版图的DRC违例
  std::vector<Violation> getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                          std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
                                          std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_wire_via_map, std::string stage);
  // 调用iSTA 计算时序
  std::map<std::string, std::vector<double>> getTiming(std::vector<std::map<std::string, std::vector<LayerCoord>>>& real_pin_coord_map_list,
                                                       std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list);
  // 输出summary
  ieda_feature::RTSummary outputSummary();
#endif

 private:
  static RTInterface* _rt_interface_instance;

  RTInterface() = default;
  RTInterface(const RTInterface& other) = delete;
  RTInterface(RTInterface&& other) = delete;
  ~RTInterface() = default;
  RTInterface& operator=(const RTInterface& other) = delete;
  RTInterface& operator=(RTInterface&& other) = delete;
  // function
};

}  // namespace irt
