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
#include "lm_layout.h"

namespace ilm {

int LmLayout::findLayerId(std::string name)
{
  auto it = _layer_id_map.find(name);
  if (it != _layer_id_map.end()) {
    return it->second;
  }

  return -1;
}

int LmLayout::findViaId(std::string name)
{
  auto it = _via_id_map.find(name);
  if (it != _via_id_map.end()) {
    return it->second;
  }

  return -1;
}

int LmLayout::findPdnId(std::string name)
{
  auto it = _pdn_id_map.find(name);
  if (it != _pdn_id_map.end()) {
    return it->second;
  }

  return -1;
}

int LmLayout::findNetId(std::string name)
{
  auto it = _net_id_map.find(name);
  if (it != _net_id_map.end()) {
    return it->second;
  }

  return -1;
}

}  // namespace ilm
