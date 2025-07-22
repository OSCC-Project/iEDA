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
 * @file		IdbMaxViaStack.h
 * @date		13/05/2025
 * @version		0.1
 * @description


        Describe Core information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "../IdbObject.h"
#include "IdbLayer.h"

namespace idb {

using std::vector;

class IdbMaxViaStack : public IdbObject
{
 public:
  IdbMaxViaStack() = default;
  ~IdbMaxViaStack() = default;

  // getter
  bool is_no_single() { return _no_single; }
  bool is_range() { return _layer_top_name != "" && _layer_bottom_name != ""; }
  uint32_t get_stacked_via_num() { return _stacked_via_num; }
  std::string get_layer_top() { return _layer_top_name; }
  std::string get_layer_bottom() { return _layer_bottom_name; }

  // setter
  void set_no_single(bool no_single) { _no_single = no_single; }
  void set_stacked_via_num(uint32_t stacked_via_num) { _stacked_via_num = stacked_via_num; }
  void set_layer_top(std::string layer_top) { _layer_top_name = layer_top; }
  void set_layer_bottom(std::string layer_bottom) { _layer_bottom_name = layer_bottom; }

  // operator

 private:
  bool _no_single = false;
  uint32_t _stacked_via_num = 0;
  std::string _layer_bottom_name = "";
  std::string _layer_top_name = "";
};

}  // namespace idb
