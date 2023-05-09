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
 * @file		IdbGCellGrid.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Gcell Grid information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbEnum.h"

namespace idb {

using std::vector;

class IdbGCellGrid
{
 public:
  IdbGCellGrid();
  ~IdbGCellGrid();

  // getter
  const IdbTrackDirection get_direction() const { return _direction; }
  const int32_t get_start() const { return _start; }
  const int32_t get_num() const { return _num; }
  const int32_t get_space() const { return _space; }

  // setter
  void set_direction(IdbTrackDirection direction) { _direction = direction; }
  void set_direction(std::string direction);
  void set_start(int32_t start) { _start = start; }
  void set_num(int32_t num) { _num = num; }
  void set_space(int32_t space) { _space = space; }

  // operator

 private:
  IdbTrackDirection _direction;
  int32_t _start;
  int32_t _num;
  int32_t _space;
};

/*
###The x coordinate of the last vertical track must be less than, and not equal to, the x
   coordinate of the last vertical gcell line.
###The y coordinate of the last horizontal track must be less than, and not equal to, the y
   coordinate of the last horizontal gcell line.
###Each GCELLGRID statement must define two lines
###Every gcell need not contain the vertex of a track grid. But, those that do must be at least
   as large in both directions as the default wire widths on all layers
*/
class IdbGCellGridList
{
 public:
  IdbGCellGridList();
  ~IdbGCellGridList();

  // getter
  vector<IdbGCellGrid*>& get_gcell_grid_list() { return _gcelll_grid_list; }
  int32_t get_gcell_grid_num() { return _gcelll_grid_list.size(); }

  // setter
  IdbGCellGrid* add_gcell_grid(IdbGCellGrid* gcell_grid = nullptr);
  void clear()
  {
    for (auto gcell : _gcelll_grid_list) {
      if (gcell != nullptr) {
        delete gcell;
        gcell = nullptr;
      }
    }

    _gcelll_grid_list.clear();
    vector<IdbGCellGrid*>().swap(_gcelll_grid_list);
  }

 private:
  vector<IdbGCellGrid*> _gcelll_grid_list;
};

}  // namespace idb
