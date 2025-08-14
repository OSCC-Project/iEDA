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
/**
 * @file PowerVia.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "PNPGridManager.hh"

namespace idb {
class IdbDesign;
class IdbSpecialNet;
class IdbSpecialNetList;
class IdbSpecialWireList;
class IdbSpecialWire;
class IdbSpecialWireSegment;
class IdbLayer;
class IdbLayerCut;
class IdbLayerRouting;
class IdbVia;
class IdbPin;
class IdbRect;
class IdbInstance;

enum class SegmentType : int8_t;
enum class IdbWireShapeType : uint8_t;
enum class IdbOrient : uint8_t;

template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace ipnp {

class PowerVia
{
 public:
  PowerVia() = default;
  ~PowerVia() = default;

  void connectAllPowerLayers(PNPGridManager& pnp_network);
  void connectM2M1Layer();

 private:
  void connectNetworkLayers(PNPGridManager& pnp_network, PowerType net_type);
  void connectLayers(std::string net_name, std::string top_layer_name, std::string bottom_layer_name);
  void connect_Layer_Row(std::string net_name, std::string top_layer_name, std::string bottom_layer_name);
  void connect_M2_M1(std::string net_name);

  int32_t transUnitDB(double value);

  idb::IdbVia* findVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design);

  idb::IdbVia* createVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, std::string via_name);

  idb::IdbSpecialWireSegment* createSpecialWireVia(idb::IdbLayer* layer, int32_t route_width, idb::IdbWireShapeType wire_shape_type,
                                                   idb::IdbCoordinate<int32_t>* coord, idb::IdbVia* via);

  bool getIntersectCoordinate(idb::IdbSpecialWireSegment* segment_top, idb::IdbSpecialWireSegment* segment_bottom,
                              idb::IdbRect& intersection_rect);

  bool addSingleVia(std::string net_name, std::string top_layer, std::string bottom_layer, double x, double y, int32_t width,
                    int32_t height);
};

}  // namespace ipnp