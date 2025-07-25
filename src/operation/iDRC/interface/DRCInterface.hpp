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

#include "../../../database/interaction/RT_DRC/ids.hpp"

#if 1  // 前向声明

namespace idb {
class IdbNet;
class IdbLayerRouting;
class IdbLayerCut;
enum class IdbLayerDirection : uint8_t;
}  // namespace idb

namespace idrc {
class RoutingLayer;
class CutLayer;
class DRCShape;
enum class Direction;
}  // namespace idrc

#endif

namespace idrc {

#define DRCI (idrc::DRCInterface::getInst())

class DRCInterface
{
 public:
  static DRCInterface& getInst();
  static void destroyInst();

#if 1  // 外部调用DRC的API

#if 1  // iDRC
  void initDRC(std::map<std::string, std::any> config_map, bool enable_quiet);
  void checkDef();
  void destroyDRC();
  std::vector<ids::Violation> getViolationList(const std::vector<ids::Shape>& ids_env_shape_list, const std::vector<ids::Shape>& ids_result_shape_list,const std::string option = "");
#endif

#endif

#if 1  // DRC调用外部的API

#if 1  // TopData

#if 1  // input
  void input(std::map<std::string, std::any>& config_map);
  void wrapConfig(std::map<std::string, std::any>& config_map);
  void wrapDatabase();
  void wrapMicronDBU();
  void wrapDie();
  void wrapDesignRule();
  void wrapLayerList();
  void wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapRoutingDesignRule(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapCutDesignRule(CutLayer& cut_layer, idb::IdbLayerCut* idb_layer);
  void wrapLayerInfo();
  Direction getDRCDirectionByDB(idb::IdbLayerDirection idb_direction);
#endif

#if 1  // output
  void output();
#endif

#endif

#if 1  // check
  std::vector<ids::Shape> buildEnvShapeList();
  bool isSkipping(idb::IdbNet* idb_net);
  std::vector<ids::Shape> buildResultShapeList();
  void printSummary(std::map<std::string, std::vector<ids::Violation>>& type_violation_map);
  void outputViolationJson(std::map<std::string, std::vector<ids::Violation>>& type_violation_map);
  void outputSummary(std::map<std::string, std::vector<ids::Violation>>& type_violation_map);
  DRCShape convertToDRCShape(const ids::Shape& ids_shape);
#endif

#endif

 private:
  static DRCInterface* _drc_interface_instance;

  DRCInterface() = default;
  DRCInterface(const DRCInterface& other) = delete;
  DRCInterface(DRCInterface&& other) = delete;
  ~DRCInterface() = default;
  DRCInterface& operator=(const DRCInterface& other) = delete;
  DRCInterface& operator=(DRCInterface&& other) = delete;
  // function
};

}  // namespace idrc
