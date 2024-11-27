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
#include <string>

#include "IdbGeometry.h"
#include "IdbLayerShape.h"
#include "IdbPins.h"
#include "IdbTrackGrid.h"
#include "IdbVias.h"
#include "lm_layer_grid.h"
#include "lm_layout.h"
#include "lm_patch.h"

namespace ilm {

class LmLayoutInit
{
 public:
  LmLayoutInit(LmLayout* layout) : _layout(layout) {}
  ~LmLayoutInit() = default;
  void init();

 private:
  LmLayout* _layout;
  int64_t _node_id = 0;
  int error_pin_num = 0;

  void buildConnectedPoints();
  int buildConnectedPointsRoutingLayer();
  int buildConnectedPointsCutLayer();
  void optConnectionsRoutingLayer();

  void initViaIds();
  void initDie();
  void initLayers();
  void initTracks(std::string layername = "M2");
  void initTrackGrid(idb::IdbTrackGrid* idb_track_grid, LmLayerGrid& lm_grid);
  void buildPatchGrid();
  void initPDN();
  void initInstances();
  void initIOPins();
  void initNets(bool init_delta = false);

  void transPin(idb::IdbPin* idb_pin, int net_id, int pin_id = -1, bool b_io = false);
  void transVia(idb::IdbVia* idb_via, int net_id, LmNodeTYpe type);
  void transEnclosure(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id, int via_row, int via_col);
  void transNetRect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id);
  void transNetDelta(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id);
  bool setConnectNode(LmNode& node);
  void set_node_id(LmNode& node) { node.set_node_id(_node_id++); }
  bool is_steiner_node(LmNode& node);
};

}  // namespace ilm