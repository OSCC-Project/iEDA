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
 * @file PowerRouter.hh
 * @author Jianrong Su
 * @brief
 * @version 0.1
 * @date 2025-03-28
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "GridManager.hh"
#include "iPNPCommon.hh"

namespace idb {
class IdbDesign;
class IdbSpecialNet;
class IdbSpecialNetList;
class IdbSpecialWireList;
class IdbSpecialWire;
class IdbSpecialWireSegment;
class IdbLayer;
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

class PowerRouter
{
public:
  PowerRouter() = default;
  ~PowerRouter() = default;

  void addPowerNets(idb::IdbDesign* idb_design, GridManager pnp_network);

private:

  void addPowerStripesToCore(idb::IdbSpecialNet* power_net, GridManager pnp_network);
  void addPowerStripesToDie(idb::IdbSpecialNet* power_net, GridManager pnp_network);
  void addPowerFollowPin(idb::IdbDesign* idb_design, idb::IdbSpecialNet* power_net);
  void addPowerPort(idb::IdbDesign* idb_design, GridManager pnp_network, std::string pin_name, std::string layer_name);
  
  void addVSSNet(idb::IdbDesign* idb_design, GridManager pnp_network);
  void addVDDNet(idb::IdbDesign* idb_design, GridManager pnp_network);
};

}  // namespace ipnp