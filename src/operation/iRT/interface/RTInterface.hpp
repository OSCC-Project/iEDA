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
class IdbLayerRouting;
class IdbLayerCut;
class IdbNet;
class IdbPin;
enum class IdbLayerDirection : uint8_t;
enum class IdbConnectType : uint8_t;
class IdbRegularWireSegment;
}  // namespace idb

namespace irt {
class RoutingLayer;
class CutLayer;
class Violation;
class LayerCoord;
class LayerRect;
template <typename T>
class Segment;
class Net;
class Pin;
enum class Direction;
enum class ConnectType;
class EXTLayerRect;
class TAPanel;
class PlanarCoord;
}  // namespace irt

namespace ieda_feature {
class RTSummary;
class FeatureManager;
}  // namespace ieda_feature

#endif

namespace irt {

#define RTI (irt::RTInterface::getInst())

class RTInterface
{
 public:
  static RTInterface& getInst();
  static void destroyInst();

#if 1  // 外部调用RT的API

#if 1  // iRT
  void initRT(std::map<std::string, std::any> config_map);
  void runEGR();
  void runRT();
  void destroyRT();
  void clearDef();
  void outputDBJson(std::map<std::string, std::any> config_map);
#endif

#endif

#if 1  // RT调用外部的API

#if 1  // TopData

#if 1  // input
  void input(std::map<std::string, std::any>& config_map);
  void wrapConfig(std::map<std::string, std::any>& config_map);
  void wrapDatabase();
  void wrapDBInfo();
  void wrapMicronDBU();
  void wrapManufactureGrid();
  void wrapDie();
  void wrapRow();
  void wrapLayerList();
  void wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapRoutingDesignRule(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer);
  void wrapCutDesignRule(CutLayer& cut_layer, idb::IdbLayerCut* idb_layer);
  void wrapLayerInfo();
  void wrapLayerViaMasterList();
  void wrapObstacleList();
  void wrapNetInfo();
  void wrapNetList();
  bool isSkipping(idb::IdbNet* idb_net, bool with_log);
  void wrapPinList(Net& net, idb::IdbNet* idb_net);
  void wrapPinShapeList(Pin& pin, idb::IdbPin* idb_pin);
  void wrapDrivenPin(Net& net, idb::IdbNet* idb_net);
  Direction getRTDirectionByDB(idb::IdbLayerDirection idb_direction);
  ConnectType getRTConnectTypeByDB(idb::IdbConnectType idb_connect_type);
#endif

#if 1  // output
  void output();
  void outputTrackGrid();
  void outputGCellGrid();
  void outputNetList();
  void outputSummary();
#endif

#if 1  // convert idb
  idb::IdbRegularWireSegment* getIDBSegmentByNetResult(int32_t net_idx, Segment<LayerCoord>& segment);
  idb::IdbRegularWireSegment* getIDBSegmentByNetPatch(int32_t net_idx, EXTLayerRect& ext_layer_rect);
  idb::IdbRegularWireSegment* getIDBWire(int32_t net_idx, Segment<LayerCoord>& segment);
  idb::IdbRegularWireSegment* getIDBVia(int32_t net_idx, Segment<LayerCoord>& segment);
#endif

#endif

#if 1  // iDRC
  void initIDRC();
  void destroyIDRC();
  std::vector<Violation> getViolationList(std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                          std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                          std::map<int32_t, std::vector<Segment<LayerCoord>*>>& net_result_map,
                                          std::map<int32_t, std::vector<EXTLayerRect*>>& net_patch_map);
  ids::Shape getIDSShape(int32_t net_idx, LayerRect layer_rect, bool is_routing);
#endif

#if 1  // iSTA
  void updateTimingAndPower(std::vector<std::map<std::string, std::vector<LayerCoord>>>& real_pin_coord_map_list,
                            std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list,
                            std::map<std::string, std::map<std::string, double>>& clock_timing, std::map<std::string, double>& power);
#endif

#if 1  // flute
  void initFlute();
  void destroyFlute();
  std::vector<Segment<PlanarCoord>> getPlanarTopoList(std::vector<PlanarCoord> planar_coord_list);
#endif

#if 1  // lsa
  void routeTAPanel(TAPanel& ta_panel);
#endif

#if 1  // ecos
  void sendNotification(std::string stage, std::string json_path);
  void sendNotification(std::string stage, int32_t iter, std::string json_path);
#endif

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
