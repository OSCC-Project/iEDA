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
 * @file		IdbObject.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Basic class for all Idb structure.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "../../../basic/geometry/IdbGeometry.h"

namespace idb {

// static uint64_t GLOBAL_ID = 0;
class IdbObject
{
 public:
  IdbObject();
  virtual ~IdbObject();

  // getter
  IdbRect* get_bounding_box() { return _bounding_box; }
  uint64_t& get_id() { return _id; }

  // setter
  bool set_bounding_box(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y);
  void set_bounding_box(IdbRect* bounding_box) { _bounding_box = bounding_box; }
  void set_bounding_box(IdbRect bounding_box);
  void set_id(uint64_t id) { _id = id; }

  // operator

 private:
  IdbRect* _bounding_box;

  uint64_t _id;
};

}  // namespace idb
