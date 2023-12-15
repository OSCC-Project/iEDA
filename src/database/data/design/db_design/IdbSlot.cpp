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
 * @file		IdbSlot.h
 * @date		25/05/2021
 * @version		0.1
* @description


    Defines the rectangular shapes that form the slotting of the wires in the design. Each slot is
    defined as an individual rectangle.
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbSlot.h"

#include <algorithm>

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbSlot::IdbSlot()
{
  _layer_name = "";
  _layer = nullptr;
}

IdbSlot::~IdbSlot()
{
  _layer = nullptr;

  for (IdbRect* rect : _rect_list) {
    if (rect) {
      delete rect;
      rect = nullptr;
    }
  }
  _rect_list.clear();
  std::vector<IdbRect*>().swap(_rect_list);
}

IdbRect* IdbSlot::get_rect(size_t index)
{
  if (index < _rect_list.size() && _rect_list.size() > 0) {
    return _rect_list.at(index);
  }

  return nullptr;
}

IdbRect* IdbSlot::add_rect()
{
  IdbRect* rect = new IdbRect();
  _rect_list.emplace_back(rect);

  return rect;
}

void IdbSlot::add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
  _rect_list.emplace_back(rect);
}

void IdbSlot::reset_rect()
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
IdbSlotList::IdbSlotList()
{
  _num = 0;
}

IdbSlotList::~IdbSlotList()
{
  reset();
}

IdbSlot* IdbSlotList::add_slot()
{
  IdbSlot* slot = new IdbSlot();

  _slot_list.emplace_back(slot);
  _num++;

  return slot;
}

void IdbSlotList::reset()
{
  for (auto* slot : _slot_list) {
    if (slot != nullptr) {
      delete slot;
      slot = nullptr;
    }
  }

  _slot_list.clear();
  std::vector<IdbSlot*>().swap(_slot_list);
}

}  // namespace idb
