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
class IdbLayer;
class IdbRegularWireSegment;
template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace idrc {

class DrcEngineManager;
class DrcDataManager;
enum class LayoutType;

class DrcEngineInit
{
 public:
  DrcEngineInit(DrcEngineManager* engine_manager = nullptr, DrcDataManager* data_manager = nullptr)
      : _engine_manager(engine_manager), _data_manager(data_manager)
  {
  }
  ~DrcEngineInit() {}

  virtual void init() = 0;

 protected:
  DrcEngineManager* _engine_manager = nullptr;
  DrcDataManager* _data_manager = nullptr;

  void initDataFromShape(idb::IdbLayerShape* idb_shape, int net_id = -1);
  void initDataFromPoints(idb::IdbCoordinate<int>* point_1, idb::IdbCoordinate<int>* point_2, int routing_width, idb::IdbLayer* layer,
                          int net_id = -1, bool b_pdn = false);
  void initDataFromRect(idb::IdbRect* rect, LayoutType type, idb::IdbLayer* layer, int net_id = -1);
  void initDataFromPin(idb::IdbPin* idb_pin, int default_id = -1);
  void initDataFromVia(idb::IdbVia* idb_via, int net_id = -1);
  void initDataFromNet(idb::IdbNet* idb_net);
};

}  // namespace idrc