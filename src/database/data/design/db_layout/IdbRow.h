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
 * @file		IdbRow.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Row information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbObject.h"
#include "IdbSite.h"

using std::string;
using std::vector;

namespace idb {

class IdbSite;

class IdbRow : public IdbObject
{
 public:
  IdbRow();
  ~IdbRow();

  // getter
  const string get_name() const { return _name; }
  IdbSite* get_site() { return _site; }
  const int32_t get_site_count() { return _row_num_x * _row_num_y; }
  IdbCoordinate<int32_t>* get_original_coordinate() { return _original_coordinate; }
  const int32_t get_row_num_x() const { return _row_num_x; }
  const int32_t get_row_num_y() const { return _row_num_y; }
  const int32_t get_step_x() const { return _step_x; }
  const int32_t get_step_y() const { return _step_y; }
  bool is_horizontal() { return _row_num_y == 1 ? true : false; }
  const IdbOrient get_orient() { return _orient; }

  // setter
  void set_name(string name) { _name = name; }
  void set_site(IdbSite* site)
  {
    if (_site != nullptr) {
      delete _site;
      _site = nullptr;
    }
    _site = site;
  }
  void set_original_coordinate(IdbCoordinate<int32_t>* original_coordinate)
  {
    if (_original_coordinate != nullptr) {
      delete _original_coordinate;
      _original_coordinate = nullptr;
    }
    _original_coordinate = original_coordinate;
  }
  void set_original_coordinate(int32_t x, int32_t y) { _original_coordinate->set_xy(x, y); }
  void set_row_num_x(int32_t row_num_x);
  void set_row_num_y(int32_t row_num_y);
  void set_step_x(int32_t step_x) { _step_x = step_x; }
  void set_step_y(int32_t step_y) { _step_y = step_y; }
  void set_orient(IdbOrient orient) { _orient = orient; }
  bool set_bounding_box();

 private:
  IdbSite* _site;
  string _name;
  IdbCoordinate<int32_t>* _original_coordinate;
  int32_t _row_num_x;
  int32_t _row_num_y;
  int32_t _step_x;
  int32_t _step_y;
  IdbOrient _orient;
};

class IdbRows
{
 public:
  IdbRows();
  ~IdbRows();

  // getter
  vector<IdbRow*>& get_row_list() { return _row_list; }
  const uint32_t get_row_num() const { return _row_num; }
  IdbRow* find_row(string row_name);
  // setter
  void set_row_number(uint32_t number) { _row_num = number; }
  IdbRow* add_row_list(IdbRow* row = nullptr);
  IdbRow* add_row_list(string row_name);
  IdbRow* createRow(string row_name, IdbSite* site, int32_t orig_x, int32_t orig_y, IdbOrient site_orient, int32_t num_x, int32_t num_y,
                    int32_t step_x, int32_t step_y);
  void reset();

  // operator
  int32_t get_row_height()
  {
    if (_row_list.size() > 0) {
      IdbRow* row = _row_list[0];
      IdbSite* site = row->get_site();
      return row->is_horizontal() ? site->get_height() : site->get_width();
    }

    return -1;
  }

 private:
  uint32_t _row_num;
  vector<IdbRow*> _row_list;
};

}  // namespace idb
