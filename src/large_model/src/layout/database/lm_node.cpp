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

namespace ilm {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LmNodeData::set_type(LmNodeTYpe type)
{
  _type = LmNodeTYpe((static_cast<uint8_t>(_type)) ^ (static_cast<uint8_t>(type)));
}

bool LmNodeData::is_type(LmNodeTYpe type)
{
  return (static_cast<uint8_t>(_type) & static_cast<uint8_t>(type)) > 0 ? true : false;
}

void LmNodeData::set_direction(LmNodeDirection direction)
{
  _direction = LmNodeDirection((static_cast<uint8_t>(_direction)) ^ (static_cast<uint8_t>(direction)));
}

bool LmNodeData::is_direction(LmNodeDirection direction)
{
  return (static_cast<uint8_t>(_direction) & static_cast<uint8_t>(direction)) > 0 ? true : false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// if node has more than 3 direction in routing layer, this node is steiner point

bool LmNode::is_steiner_point()
{
  int direction_num = 0;

  uint8_t direction = static_cast<uint8_t>(_node_data.get_direction());
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_left)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_right)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_up)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_down)) > 0 ? direction_num + 1 : direction_num;

  return direction_num > 3 ? true : false;
}
/// @brief corner means has 2 and only 2 directions orthogonal
/// @return
bool LmNode::is_corner()
{
  int direction_num = 0;

  uint8_t direction = static_cast<uint8_t>(_node_data.get_direction());
  int left = direction & static_cast<uint8_t>(LmNodeDirection::lm_left) > 0 ? true : false;
  int right = direction & static_cast<uint8_t>(LmNodeDirection::lm_right) > 0 ? true : false;
  int up = direction & static_cast<uint8_t>(LmNodeDirection::lm_up) > 0 ? true : false;
  int down = direction & static_cast<uint8_t>(LmNodeDirection::lm_down) > 0 ? true : false;

  direction_num = left & up ? direction_num + 1 : direction_num;
  direction_num = left & down ? direction_num + 1 : direction_num;
  direction_num = right & up ? direction_num + 1 : direction_num;
  direction_num = right & down ? direction_num + 1 : direction_num;

  return direction_num == 1 ? true : false;
}

bool LmNode::is_end_point()
{
  int direction_num = 0;

  uint8_t direction = static_cast<uint8_t>(_node_data.get_direction());
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_left)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_right)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_up)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_down)) > 0 ? direction_num + 1 : direction_num;

  return direction_num == 1 ? true : false;
}

/// if node has top or bottom direction, it is a via connected point
bool LmNode::is_via()
{
  int direction_num = 0;

  uint8_t direction = static_cast<uint8_t>(_node_data.get_direction());
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_top)) > 0 ? direction_num + 1 : direction_num;
  direction_num = (direction & static_cast<uint8_t>(LmNodeDirection::lm_bottom)) > 0 ? direction_num + 1 : direction_num;

  return direction_num > 0 ? true : false;
}

}  // namespace ilm
