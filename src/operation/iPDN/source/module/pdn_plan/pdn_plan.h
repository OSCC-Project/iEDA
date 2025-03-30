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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ipdn_basic.h"
#include "pdn_cut_stripe.h"

namespace idb {
class IdbLayer;
class IdbSpecialWireSegment;
class IdbBlockageList;
class IdbInstance;
class IdbRect;
class IdbVia;
class IdbLayerCut;
class IdbPin;
class IdbSpecialNet;
class IdbLayerRouting;
class IdbSpecialWire;

enum class SegmentType : int8_t;
enum class IdbWireShapeType : uint8_t;
enum class IdbOrient : uint8_t;

template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace ipdn {

class PdnPlan
{
 public:
  explicit PdnPlan();
  ~PdnPlan();

  /// operator
  void addIOPin(std::string pin_name, std::string net_name, std::string direction, bool is_power = true);
  void globalConnect(const std::string pdn_net_name, const std::string instance_pdn_pin_name, bool is_power);
  void placePdnPort(std::string pin_name, std::string io_cell_name, int32_t offset_x, int32_t offset_y, int32_t width, int32_t height,
                    std::string layer_name);
  void createGrid(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width, double route_offset);

  void createStripe(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width, double pitchh,
                    double route_offset);
  void connectLayer(std::string layer_name_first, std::string layer_name_second);

  void connectMacroToPdnGrid(std::vector<std::string> power_name, std::vector<std::string> ground_name, std::string pin_layer,
                             std::string pdn_layer, std::string orient);
  void updateRouteMap();

  void connectIOPinToPowerStripe(std::vector<double>& point_list, const std::string layer_name);
  void connectPowerStripe(std::vector<double>& point_list, const std::string& net_name, const std::string& layer_name, int32_t width);
  bool addSegmentStripeList(std::vector<double>& point_list, std::string net_name, std::string layer_name, int32_t width);
  bool addSegmentStripeList(std::vector<idb::IdbCoordinate<int32_t>*> point_list, std::string net_name, std::string layer_name,
                            int32_t width);
  bool addSegmentStripe(int32_t x_start, int32_t y_start, int32_t x_end, int32_t y_end, std::string net_name, std::string layer_name,
                        int32_t width);

 private:
  std::map<std::string, RouteInfo> _layer_power_route_info_map;
  CutStripe* _cut_stripe = nullptr;

  int32_t transUnitDB(double value);
  std::pair<std::string, std::string> orientToStr(idb::IdbOrient);

  idb::IdbSpecialWireSegment* createSpecialWireSegment(idb::IdbLayer* layer, int32_t route_width, idb::IdbWireShapeType wire_shape_type,
                                                       int32_t x_start, int32_t y_start, int32_t x_end, int32_t y_end);
  std::vector<idb::IdbSpecialWireSegment*> createSpecialWireSegmentWithInBlockage(idb::IdbSpecialWireSegment* wire_segment,
                                                                                  idb::IdbBlockageList* blockage_list);
  void addSpecialWireSegmentWithInBlockage(idb::IdbSpecialWire* sp_wire, idb::IdbSpecialWireSegment* sp_wire_segment);

  /// macro
  void connectMacroToPdnGrid(idb::IdbInstance* macro, std::vector<std::string> power_name, std::vector<std::string> ground_name,
                             std::string pin_layer, std::string pdn_layer);
  void initMacroPowerPinShape(idb::IdbInstance* macro, std::vector<std::string> power_name, std::vector<std::string> ground_name,
                              std::map<std::string, std::map<std::string, std::vector<idb::IdbRect>>>& power,
                              std::map<std::string, std::map<std::string, std::vector<idb::IdbRect>>>& ground);
  std::vector<idb::IdbRect> findOverlapbetweenMacroPdnAndStripe(std::vector<idb::IdbRect> rect_list, const std::string& layer,
                                                                idb::IdbSpecialWire* sp_wire);

  std::map<std::string, std::vector<idb::IdbRect>> mergeOverlapRect(idb::IdbPin* pin);
  std::vector<idb::IdbRect> mergeOverlapRect(std::vector<idb::IdbRect*> rect_list);

  void connectTwoLayerForNet(idb::IdbSpecialNet* net, std::vector<idb::IdbVia*>& via_list, idb::IdbLayerRouting* layer_top,
                             idb::IdbLayerRouting* layer_bottom);

  void connectTwoLayerForWire(idb::IdbSpecialWire* wire, std::vector<idb::IdbVia*>& via_list,
                              std::vector<idb::IdbSpecialWireSegment*>& segment_list_top,
                              std::vector<idb::IdbSpecialWireSegment*>& segment_list_bottom);
};

}  // namespace ipdn