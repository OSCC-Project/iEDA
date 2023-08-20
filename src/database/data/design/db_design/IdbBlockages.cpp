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
 * @project		iDB
 * @file		IdbBlockages.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe macros information,.
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbBlockages.h"

#include <algorithm>

#include "IdbInstance.h"

using namespace std;
namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbBlockage::IdbBlockage()
{
  _type = IdbBlockageType::kNone;
  _instance_name = "";
  _instance = nullptr;
  _is_pushdown = false;
}

IdbBlockage::~IdbBlockage()
{
  for (IdbRect* rect : _rect_list) {
    if (rect) {
      delete rect;
      rect = nullptr;
    }
  }
  _rect_list.clear();
  std::vector<IdbRect*>().swap(_rect_list);
}

IdbRect* IdbBlockage::get_rect(size_t index)
{
  if ((index > 0) && (index < _rect_list.size())) {
    return _rect_list.at(index);
  }

  return nullptr;
}

IdbRect* IdbBlockage::add_rect()
{
  IdbRect* rect = new IdbRect();
  _rect_list.emplace_back(rect);

  return rect;
}

void IdbBlockage::add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
  _rect_list.emplace_back(rect);
}

void IdbBlockage::reset_rect()
{
  for (auto& rect : _rect_list) {
    if (rect != nullptr) {
      delete rect;
      rect = nullptr;
    }
  }

  _rect_list.clear();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbRoutingBlockage::IdbRoutingBlockage()
{
  _layer_name = "";
  _layer = nullptr;

  _is_slots = false;
  _is_fills = false;
  _is_except_pgnet = false;
  _min_spacing = -1;
  _effective_width = -1;
}

IdbRoutingBlockage::~IdbRoutingBlockage()
{
  _layer = nullptr;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbPlacementBlockage::IdbPlacementBlockage()
{
  _is_soft = false;
  _is_partial = false;
  _max_density = 0;
}

IdbPlacementBlockage::~IdbPlacementBlockage()
{
}

IdbLayer* IdbPlacementBlockage::get_layer()
{
  IdbInstance* instance = get_instance();
  return instance == nullptr ? nullptr : instance->get_cell_master()->get_top_layer();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbBlockageList::IdbBlockageList()
{
  _num = 0;
}

IdbBlockageList::~IdbBlockageList()
{
  for (IdbBlockage* blockage : _blockage_list) {
    if (blockage != nullptr) {
      delete blockage;
      blockage = nullptr;
    }
  }
}

// vector<IdbBlockage*> IdbBlockageList::find_routing_blockage(string layer_name)
// {
//     vector<IdbBlockage*> blockage_find;
//     for(IdbBlockage* blockage: _routing_blockage)
//     {
//         IdbRoutingBlockage* routing_blockage = dynamic_cast<IdbRoutingBlockage*>(blockage);
//         if(routing_blockage->get_layer_name() == layer_name)
//         {
//             return blockage_find.push_back(blockage);
//         }
//     }

//     return blockage_find;
// }

// IdbBlockage* IdbBlockageList::get_blockage(size_t index)
// {
//     if(_num > index)
//     {
//         return _blockage_list.at(index);
//     }

//     return nullptr;
// }

IdbRoutingBlockage* IdbBlockageList::add_blockage_routing(string layer_name)
{
  IdbRoutingBlockage* routing_Blockage = new IdbRoutingBlockage();
  routing_Blockage->set_layer_name(layer_name);

  IdbBlockage* pBlockage = dynamic_cast<IdbBlockage*>(routing_Blockage);
  pBlockage->set_type_routing();

  _blockage_list.emplace_back(pBlockage);
  _num++;

  return routing_Blockage;
}

IdbPlacementBlockage* IdbBlockageList::add_blockage_placement()
{
  IdbPlacementBlockage* placement_Blockage = new IdbPlacementBlockage();
  IdbBlockage* pBlockage = dynamic_cast<IdbBlockage*>(placement_Blockage);
  pBlockage->set_type_placement();

  _blockage_list.emplace_back(pBlockage);
  _num++;

  return placement_Blockage;
}

void IdbBlockageList::reset()
{
  for (auto& blockage : _blockage_list) {
    if (blockage != nullptr) {
      delete blockage;
      blockage = nullptr;
    }
  }

  _blockage_list.clear();
}

void IdbBlockageList::clearPlacementBlockage()
{
  for (auto it = _blockage_list.begin(); it != _blockage_list.end();) {
    IdbBlockage* blockage = *it;
    if (blockage != nullptr && blockage->is_palcement_blockage()) {
      it = _blockage_list.erase(it);
      delete blockage;
      blockage = nullptr;
      continue;
    }

    it++;
  }
}

void IdbBlockageList::clearRoutingBlockage()
{
  for (auto it = _blockage_list.begin(); it != _blockage_list.end();) {
    IdbBlockage* blockage = *it;
    if (blockage != nullptr && blockage->is_routing_blockage()) {
      it = _blockage_list.erase(it);
      delete blockage;
      blockage = nullptr;
      continue;
    }

    it++;
  }
}

void IdbBlockageList::removeExceptPgNetBlockageList()
{
  for (auto it = _blockage_list.begin(); it != _blockage_list.end();) {
    IdbBlockage* blockage = *it;
    if (blockage != nullptr && blockage->is_routing_blockage()) {
      IdbRoutingBlockage* routing_blockage = dynamic_cast<IdbRoutingBlockage*>(blockage);
      if (routing_blockage->is_except_pgnet()) {
        it = _blockage_list.erase(it);
        delete blockage;
        blockage = nullptr;
        continue;
      }

      it++;
    }
  }
}

}  // namespace idb
