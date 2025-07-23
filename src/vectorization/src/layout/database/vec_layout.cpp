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
 * @project		vectorization
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "vec_layout.h"

namespace ivec {

void VecLayout::add_layer_map(int id, std::string name)
{
  _layer_name_map.insert(std::make_pair(name, id));
  _layer_id_map.insert(std::make_pair(id, name));
}

void VecLayout::add_via_map(int id, std::string name)
{
  _via_name_map.insert(std::make_pair(name, id));
  _via_id_map.insert(std::make_pair(id, name));
}

void VecLayout::add_pdn_map(int id, std::string name)
{
  _pdn_name_map.insert(std::make_pair(name, id));
  _pdn_id_map.insert(std::make_pair(id, name));
}

void VecLayout::add_net_map(int id, std::string name)
{
  _net_name_map.insert(std::make_pair(name, id));
  _net_id_map.insert(std::make_pair(id, name));
}

void VecLayout::add_pin_map(int id, std::string inst_name, std::string pin_name)
{
  auto name_pair = std::make_pair(inst_name, pin_name);
  _pin_name_map.insert(std::make_pair(name_pair, id));
  _pin_id_map.insert(std::make_pair(id, name_pair));
}

void VecLayout::add_instance_map(int id, std::string name)
{
  _inst_name_map.insert(std::make_pair(name, id));
  _inst_id_map.insert(std::make_pair(id, name));
}

int VecLayout::findLayerId(std::string name)
{
  auto it = _layer_name_map.find(name);
  if (it != _layer_name_map.end()) {
    return it->second;
  }

  return -1;
}

std::string VecLayout::findLayerName(int id)
{
  auto it = _layer_id_map.find(id);
  if (it != _layer_id_map.end()) {
    return it->second;
  }

  return "";
}

int VecLayout::findViaId(std::string name)
{
  auto it = _via_name_map.find(name);
  if (it != _via_name_map.end()) {
    return it->second;
  }

  return -1;
}

std::string VecLayout::findViaName(int id)
{
  auto it = _via_id_map.find(id);
  if (it != _via_id_map.end()) {
    return it->second;
  }

  return "";
}

int VecLayout::findPdnId(std::string name)
{
  auto it = _pdn_name_map.find(name);
  if (it != _pdn_name_map.end()) {
    return it->second;
  }

  return -1;
}

std::string VecLayout::findPdnName(int id)
{
  auto it = _pdn_id_map.find(id);
  if (it != _pdn_id_map.end()) {
    return it->second;
  }

  return "";
}

int VecLayout::findNetId(std::string name)
{
  auto it = _net_name_map.find(name);
  if (it != _net_name_map.end()) {
    return it->second;
  }

  return -1;
}

std::string VecLayout::findNetName(int id)
{
  auto it = _net_id_map.find(id);
  if (it != _net_id_map.end()) {
    return it->second;
  }

  return "";
}

int VecLayout::findPinId(std::string inst_name, std::string pin_name)
{
  auto name_pair = std::make_pair(inst_name, pin_name);
  auto it = _pin_name_map.find(name_pair);
  if (it != _pin_name_map.end()) {
    return it->second;
  }

  return -1;
}
/// @brief
/// @param id
/// @return first : instance name, second : pin name
std::pair<std::string, std::string> VecLayout::findPinName(int id)
{
  auto it = _pin_id_map.find(id);
  if (it != _pin_id_map.end()) {
    return it->second;
  }

  return std::make_pair("", "");
}

int VecLayout::findInstId(std::string name)
{
  auto it = _inst_name_map.find(name);
  if (it != _inst_name_map.end()) {
    return it->second;
  }

  return -1;
}

std::string VecLayout::findInstName(int id)
{
  auto it = _inst_id_map.find(id);
  if (it != _inst_id_map.end()) {
    return it->second;
  }

  return "";
}

}  // namespace ivec
