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

class RTAPI
{
 public:
  static RTAPI& getInst();
  static void destroyInst();

#if 1  // 外部调用RT的API
  // RT主要函数
  void initRT(std::map<std::string, std::any> config_map);
  void runRT();
  void destroyRT();
  // 清理def
  void clearDef();
  // 拥塞驱动
  eval::TileGrid* getCongestionMap(std::map<std::string, std::any> config_map, double& wire_length);
  std::vector<double> getWireLengthAndViaNum(std::map<std::string, std::any> config_map);
#endif

#if 1  // RT调用外部的API
  // 调用iDRC 计算版图的DRC违例
  std::vector<Violation> getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                          std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
                                          std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_wire_via_map);
  // 调用iSTA 计算时序
  std::map<std::string, std::vector<double>> getTiming(
      std::map<int32_t, std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>>& net_pin_coord_map,
      std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_segment_map);
  // 输出def
  void outputDef(std::string output_def_file_path);
  // 输出summary
  void outputSummary();
#endif

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
