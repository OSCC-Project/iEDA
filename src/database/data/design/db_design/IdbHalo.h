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
 * @file		IdbHalo.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe halo & routed halo information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

// #include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbObject.h"

namespace idb {

using std::vector;

class IdbLayer;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @description


        Specifies a placement blockage around the component. The halo extends from the LEF macro’s left edge(s) by left,
 from the bottom edge(s) by bottom, from the right edge(s) by right, and from the top edge(s) by top.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class IdbHalo : public IdbObject
{
 public:
  IdbHalo();
  ~IdbHalo() = default;

  // getter
  const int32_t get_extend_lef() const { return _extend_left; }
  const int32_t get_extend_right() const { return _extend_right; }
  const int32_t get_extend_top() const { return _extend_top; }
  const int32_t get_extend_bottom() const { return _extend_bottom; }
  const bool is_soft() { return _is_soft; }
  //  IdbRect* get_bounding_box(){return _bounding_box;}

  // setter
  void set_extend_lef(int32_t value) { _extend_left = value; }
  void set_extend_right(int32_t value) { _extend_right = value; }
  void set_extend_top(int32_t value) { _extend_top = value; }
  void set_extend_bottom(int32_t value) { _extend_bottom = value; }
  void set_soft(bool value) { _is_soft = value; }
  bool set_bounding_box(IdbRect* instance_bounding_box);

  // operator

 private:
  int32_t _extend_left;
  int32_t _extend_right;
  int32_t _extend_top;
  int32_t _extend_bottom;
  bool _is_soft;
  // IdbRect* _bounding_box;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @description


        Specifies that signal routing in the “halo area” around the block boundary should be
    perpendicular to the block edge in order to reach the block pins. The halo area is the area
    within haloDist of the block boundary
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class IdbRouteHalo
{
 public:
  IdbRouteHalo();
  ~IdbRouteHalo() = default;

  // getter
  const int32_t get_route_distance() { return _route_distance; }
  IdbLayer* get_layer_top() { return _layer_top; }
  IdbLayer* get_layer_bottom() { return _layer_bottom; }
  // setter
  void set_route_distance(int32_t value) { _route_distance = value; }
  void set_layer_top(IdbLayer* layer) { _layer_top = layer; }
  void set_layer_bottom(IdbLayer* layer) { _layer_bottom = layer; }

  // operator

 private:
  int32_t _route_distance;
  IdbLayer* _layer_bottom;
  IdbLayer* _layer_top;
};

}  // namespace idb
