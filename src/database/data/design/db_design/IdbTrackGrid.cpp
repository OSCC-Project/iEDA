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
 * @file		IdbTracks.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Tracks information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbTrackGrid.h"

namespace idb {

IdbTrack::IdbTrack()
{
  _direction = IdbTrackDirection::kNone;
  _pitch = 0;
  _width = 0;

  //_layer = nullptr;
  // _wires = nullptr;
  // _vias = nullptr;
}

IdbTrack::IdbTrack(IdbTrackDirection dir, uint32_t pitch, uint32_t width)
{
  _direction = dir;
  _pitch = pitch;
  _width = width;

  //_layer = nullptr;
  // _wires = nullptr;
  // _vias = nullptr;
}

IdbTrack::~IdbTrack()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbTrackGrid::IdbTrackGrid()
{
  _track = new IdbTrack();
  _track_num = 0;
}

IdbTrackGrid::~IdbTrackGrid()
{
  if (_track != nullptr) {
    delete _track;
    _track = nullptr;
  }
}

void IdbTrackGrid::add_layer_list(IdbLayer* layer)
{
  _layer_list.emplace_back(layer);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbTrackGridList::IdbTrackGridList()
{
  _track_grid_num = 0;
}

IdbTrackGridList::~IdbTrackGridList()
{
  reset();
}

IdbTrackGrid* IdbTrackGridList::add_track_grid(IdbTrackGrid* track_grid)
{
  IdbTrackGrid* pGrid = track_grid;
  if (pGrid == nullptr) {
    pGrid = new IdbTrackGrid();
  }
  _track_grid_list.emplace_back(pGrid);
  _track_grid_num++;

  return pGrid;
}

void IdbTrackGridList::reset()
{
  for (auto* tg : _track_grid_list) {
    if (nullptr != tg) {
      delete tg;
      tg = nullptr;
    }
  }
  _track_grid_list.clear();
  std::vector<IdbTrackGrid*>().swap(_track_grid_list);

  _track_grid_num = 0;
}

}  // namespace idb
