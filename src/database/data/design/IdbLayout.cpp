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
 * @file		IdbDesign.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe lef.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbLayout.h"

#include <limits.h>

#include <algorithm>

namespace idb {

IdbLayout::IdbLayout()
{
  _manufacture_grid = -1;
  _die = new IdbDie();
  _core = new IdbCore();
  // _units            = new IdbUnits();
  _units = nullptr;
  _layers = new IdbLayers();
  _sites = new IdbSites();
  _rows = new IdbRows();
  _gcell_grid_list = new IdbGCellGridList();
  _track_grid_list = new IdbTrackGridList();
  // _macros = new IdbMacros();
  _cell_master_list = new IdbCellMasterList();
  _via_list = new IdbVias();
  _via_rule_list = new IdbViaRuleList();
  _max_via_stack = nullptr;
}

IdbLayout::~IdbLayout()
{
  if (_die != nullptr) {
    delete _die;
    _die = nullptr;
  }
  if (_core != nullptr) {
    delete _core;
    _core = nullptr;
  }
  if (_units != nullptr) {
    delete _units;
    _units = nullptr;
  }
  if (_layers != nullptr) {
    delete _layers;
    _layers = nullptr;
  }
  if (_sites != nullptr) {
    delete _sites;
    _sites = nullptr;
  }
  if (_rows != nullptr) {
    delete _rows;
    _rows = nullptr;
  }
  if (_gcell_grid_list != nullptr) {
    delete _gcell_grid_list;
    _gcell_grid_list = nullptr;
  }
  if (_track_grid_list != nullptr) {
    delete _track_grid_list;
    _track_grid_list = nullptr;
  }
  if (_cell_master_list != nullptr) {
    delete _cell_master_list;
    _cell_master_list = nullptr;
  }
  if (_via_list != nullptr) {
    delete _via_list;
    _via_list = nullptr;
  }
  if (_via_rule_list != nullptr) {
    delete _via_rule_list;
    _via_rule_list = nullptr;
  }
}

IdbCore* IdbLayout::get_core()
{
  if (_rows->get_row_num() > 0) {
    int32_t min_x = INT_MAX;
    int32_t min_y = INT_MAX;
    int32_t max_x = INT_MIN;
    int32_t max_y = INT_MIN;
    for (IdbRow* row : _rows->get_row_list()) {
      IdbRect* row_rect = row->get_bounding_box();
      min_x = std::min(min_x, row_rect->get_low_x());
      min_y = std::min(min_y, row_rect->get_low_y());
      max_x = std::max(max_x, row_rect->get_high_x());
      max_y = std::max(max_y, row_rect->get_high_y());
    }
    _core->set_bounding_box(min_x, min_y, max_x, max_y);
  } else {
    _core->set_bounding_box(_die->get_bounding_box());
  }

  return _core;
}

}  // namespace idb
