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
/**
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "lm_node.h"

#include "Log.hh"
#include "lm_grid_info.h"

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LmNodeData::set_type(LmNodeTYpe type)
{
  _type = LmNodeTYpe((static_cast<uint8_t>(_type)) | (static_cast<uint8_t>(type)));
}

bool LmNodeData::is_type(LmNodeTYpe type)
{
  return (static_cast<uint8_t>(_type) & static_cast<uint8_t>(type)) > 0 ? true : false;
}

bool LmNodeData::is_connect_type(LmNodeConnectType type)
{
  return (static_cast<uint8_t>(_connect_type) & static_cast<uint8_t>(type)) > 0 ? true : false;
}

void LmNodeData::set_connect_type(LmNodeConnectType type)
{
  _connect_type = LmNodeConnectType((static_cast<uint8_t>(_connect_type)) | (static_cast<uint8_t>(type)));
}

void LmNodeData::set_pin_id(int32_t id)
{
  if (_pin_id == -1) {
    _pin_id = id;
  } else {
    if (_pin_id != id) {
      LOG_INFO << "_pin_id : " << _pin_id << "; id :" << id;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
uint64_t LmNode::get_node_id()
{
  return (((uint64_t) _layer_id) * ((uint64_t) gridInfoInst.node_col_num) * ((uint64_t) gridInfoInst.node_row_num))
         + ((uint64_t) _row_id) * ((uint64_t) gridInfoInst.node_col_num) + ((uint64_t) _col_id);
}

int64_t LmNode::get_x()
{
  return gridInfoInst.calculate_x(_col_id);
}

int64_t LmNode::get_y()
{
  return gridInfoInst.calculate_y(_row_id);
}

LmNodeData* LmNode::get_node_data(int net_id, bool b_create)
{
  //   if (net_id == -1 && _data_map.size() > 0) {
  //     /// get first data
  //     return &_data_map.begin()->second;
  //   }
  //   auto it = _data_map.find(net_id);
  //   if (it != _data_map.end()) {
  //     return &it->second;
  //   } else {
  //     if (b_create) {
  //       LmNodeData data;
  //       data.set_net_id(net_id);
  //       addNodeData(data);
  //     }
  //   }
  //   return &_data_map.find(net_id)->second;
  if (_node_data == nullptr && b_create) {
    _node_data = new LmNodeData();
  }

  return _node_data;
}

}  // namespace ilm
