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
 * @file		patch.h
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "lm_node.h"

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LmNodeData::set_direction(LmNodeDirection direction)
{
  _direction = LmNodeDirection((static_cast<uint8_t>(_direction)) ^ (static_cast<uint8_t>(direction)));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// if node has more than 3 direction in routing layer, this node is steiner point
bool LmNode::isSteinerPoint()
{
  int direction_num = 0;

  uint8_t direction = static_cast<uint8_t>(_node_data.get_direction());
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_left)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_right)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_up)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_down)) > 0 ? direction_num + 1 : direction_num;

  return direction_num > 3 ? true : false;
}

/// if node has top or bottom direction, it is a via connected point
bool LmNode::isVia()
{
  int direction_num = 0;

  uint8_t direction = static_cast<uint8_t>(_node_data.get_direction());
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_top)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_bottom)) > 0 ? direction_num + 1 : direction_num;

  return direction_num > 0 ? true : false;
}

}  // namespace ilm
