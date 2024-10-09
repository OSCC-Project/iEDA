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

#include <map>
#include <vector>

#include "IdbGeometry.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "ieco_data_via.h"

namespace ieco {

class EcoDataManager;

class ECOViaInit
{
 public:
  ECOViaInit(EcoDataManager* data_manager);
  ~ECOViaInit();

  void initData();

 private:
  EcoDataManager* _data_manager;

  void init_via_masters();
  void init_nets();
  void init_segment_rect(EcoDataVia& eco_via, idb::IdbRegularWireSegment* idb_segment, int cut_layer_order);
  void init_segment_via(EcoDataVia& eco_via, idb::IdbRegularWireSegment* idb_segment, int cut_layer_order);
  void init_pin(EcoDataVia& eco_via, idb::IdbNet* idb_net, int cut_layer_order);

  std::map<int, std::vector<EcoDataVia>> get_net_vias(idb::IdbNet* idb_net);
};

}  // namespace ieco