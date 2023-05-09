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

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"

using std::string;
using std::vector;

namespace idb {

// using std::vector;

class IdbLayer;
class IdbSlot
{
 public:
  IdbSlot();
  ~IdbSlot();

  // getter
  const string& get_layer_name() const { return _layer_name; }
  IdbLayer* get_layer() { return _layer; }

  vector<IdbRect*>& get_rect_list() { return _rect_list; }
  int32_t get_rect_num() const { return _rect_list.size(); }
  IdbRect* get_rect(size_t index);

  // setter
  void set_layer_name(string name) { _layer_name = name; }
  void set_layer(IdbLayer* layer) { _layer = layer; }

  void set_rect_list(vector<IdbRect*> rect_list) { _rect_list = rect_list; }
  IdbRect* add_rect();
  void add_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void reset_rect();

 private:
  string _layer_name;
  IdbLayer* _layer;
  vector<IdbRect*> _rect_list;
};

class IdbSlotList
{
 public:
  IdbSlotList();
  ~IdbSlotList();

  // getter
  const int32_t get_num() const { return _slot_list.size(); };
  vector<IdbSlot*>& get_slot_list() { return _slot_list; }

  // setter
  IdbSlot* add_slot();

  void reset();

  // operator

 private:
  int32_t _num;
  vector<IdbSlot*> _slot_list;
};

}  // namespace idb
