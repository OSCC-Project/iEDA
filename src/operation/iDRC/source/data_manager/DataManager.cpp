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
#include "DataManager.hpp"

namespace idrc {

// public

void DataManager::initInst()
{
  if (_dm_instance == nullptr) {
    _dm_instance = new DataManager();
  }
}

DataManager& DataManager::getInst()
{
  if (_dm_instance == nullptr) {
    DRCLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_dm_instance;
}

void DataManager::destroyInst()
{
  if (_dm_instance != nullptr) {
    delete _dm_instance;
    _dm_instance = nullptr;
  }
}

// function

#if 1  // 获得唯一的pitch

int32_t DataManager::getOnlyPitch()
{
  return 200;
  // std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  // std::vector<int32_t> pitch_list;
  // for (RoutingLayer& routing_layer : routing_layer_list) {
  //   for (ScaleGrid& x_grid : routing_layer.get_track_axis().get_x_grid_list()) {
  //     pitch_list.push_back(x_grid.get_step_length());
  //   }
  //   for (ScaleGrid& y_grid : routing_layer.get_track_axis().get_y_grid_list()) {
  //     pitch_list.push_back(y_grid.get_step_length());
  //   }
  // }
  // for (int32_t pitch : pitch_list) {
  //   if (pitch_list.front() != pitch) {
  //     RTLOG.error(Loc::current(), "The pitch is not equal!");
  //   }
  // }
  // return pitch_list.front();
}

#endif

// private

DataManager* DataManager::_dm_instance = nullptr;

}  // namespace idrc
