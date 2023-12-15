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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../IdbEnum.h"

using std::vector;

namespace idb {

class IdbWires;
class IdbVias;
class IdbLayer;

class IdbTrack
{
 public:
  IdbTrack();
  IdbTrack(IdbTrackDirection dir, uint32_t pitch, uint32_t width);
  ~IdbTrack();

  // getter
  const uint32_t get_start() const { return _start; }
  const IdbTrackDirection get_direction() const { return _direction; }
  const uint32_t get_pitch() const { return _pitch; }
  const uint32_t get_width() const { return _width; }
  bool is_track_direction_x() { return _direction == IdbTrackDirection::kDirectionX ? true : false; }
  bool is_track_direction_y() { return _direction == IdbTrackDirection::kDirectionY ? true : false; }
  bool is_track_vertical() { return _direction == IdbTrackDirection::kDirectionX ? true : false; }
  bool is_track_horizontal() { return _direction == IdbTrackDirection::kDirectionX ? true : false; }

  // IdbLayer* get_layer(){return _layer;}

  // setter
  void set_start(uint32_t start) { _start = start; }
  void set_direction(IdbTrackDirection direction) { _direction = direction; }
  void set_pitch(uint32_t pitch) { _pitch = pitch; }
  void set_width(uint32_t width) { _width = width; }

  // void set_layer(IdbLayer* layer){_layer = layer;}

 private:
  uint32_t _start;
  IdbTrackDirection _direction;  //!< PreferDirection
  uint32_t _pitch;               //!< spacing: The track spacing is the PITCH value for the layer defined in LEF.
  uint32_t _width;

  // <<---tbd--->>
  // IdbLayer* _layer;
  // IdbWires *_wires;
  // IdbVias *_vias;
};

class IdbTrackGrid
{
 public:
  IdbTrackGrid();
  ~IdbTrackGrid();

  // getter
  IdbTrack* get_track() { return _track; }
  const uint32_t get_track_num() const { return _track_num; }
  vector<IdbLayer*> get_layer_list() { return _layer_list; }
  IdbLayer* get_first_layer() { return _layer_list.size() > 0 ? _layer_list[0] : nullptr; }
  // setter
  void set_track(IdbTrack* track) { _track = track; }
  void set_track_number(uint32_t number) { _track_num = number; }
  void add_layer_list(IdbLayer* layer);

 private:
  IdbTrack* _track;
  uint32_t _track_num;

  // <<---tbd--->>
  vector<IdbLayer*> _layer_list;
};

class IdbTrackGridList
{
 public:
  IdbTrackGridList();
  ~IdbTrackGridList();
  // getter
  const uint32_t get_track_grid_num() const { return _track_grid_list.size(); }
  vector<IdbTrackGrid*>& get_track_grid_list() { return _track_grid_list; }

  // setter
  // void set_track_grid_number(uint32_t number){_track_grid_num = number;}
  IdbTrackGrid* add_track_grid(IdbTrackGrid* track_grid = nullptr);

  void reset();

 private:
  vector<IdbTrackGrid*> _track_grid_list;
  uint32_t _track_grid_num;
};

}  // namespace idb
