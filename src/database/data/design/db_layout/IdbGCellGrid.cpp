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
 * @file		IdbGCellGrid.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe GCell Grid information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbGCellGrid.h"

namespace idb {

IdbGCellGrid::IdbGCellGrid()
{
  _direction = IdbTrackDirection::kNone;
  _start = -1;
  _num = -1;
  _space = -1;
}

IdbGCellGrid::~IdbGCellGrid()
{
}

void IdbGCellGrid::set_direction(std::string direction)
{
  _direction = IdbEnum::GetInstance()->get_layer_property()->get_track_direction(direction);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbGCellGridList::IdbGCellGridList()
{
  //   _gcell_vertical->set_direction(IdbTrackDirection::kDirectionX);
  //   _gcell_horizontal = new IdbGCellGrid();
  //   _gcell_horizontal->set_direction(IdbTrackDirection::kDirectionY);
}

IdbGCellGridList::~IdbGCellGridList()
{
}

IdbGCellGrid* IdbGCellGridList::add_gcell_grid(IdbGCellGrid* gcell_grid)
{
  if (gcell_grid == nullptr) {
    gcell_grid = new IdbGCellGrid();
  }

  _gcelll_grid_list.emplace_back(gcell_grid);

  return gcell_grid;
}

}  // namespace idb
