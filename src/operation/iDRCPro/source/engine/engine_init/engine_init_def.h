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

#include <stdint.h>

/**
 * check geometry overlap method
 */

namespace idb {
class IdbLayerShape;
class IdbPin;
class IdbVia;
class IdbRect;
class IdbNet;
template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace idrc {

#define NET_ID_ENVIRONMENT -1
#define NET_ID_OBS -2
// #define NET_ID_POWER -3
// #define NET_ID_GROUND -4

class DrcEngineManager;
enum class LayoutType;

class DrcEngineInitDef
{
 public:
  DrcEngineInitDef(DrcEngineManager* engine_manager) : _engine_manager(engine_manager) {}
  ~DrcEngineInitDef() {}

  void init();

 private:
  DrcEngineManager* _engine_manager = nullptr;

  void initDataFromShape(idb::IdbLayerShape* idb_shape, int net_id = -1);
  void initDataFromPoints(idb::IdbCoordinate<int>* point_1, idb::IdbCoordinate<int>* point_2, int routing_width, int layer_id,
                          int net_id = -1, bool b_pdn = false);
  void initDataFromRect(idb::IdbRect* rect, LayoutType type, int layer_id, int net_id = -1);
  void initDataFromPin(idb::IdbPin* idb_pin);
  void initDataFromVia(idb::IdbVia* idb_via, int net_id = -1);

  void initDataFromIOPins();
  void initDataFromInstances();
  void initDataFromPDN();
  void initDataFromNets();
  void initDataFromNet(idb::IdbNet* idb_net);
};

}  // namespace idrc