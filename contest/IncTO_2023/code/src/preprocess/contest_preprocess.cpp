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
 * @File Name: contest_evaluation.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "contest_preprocess.h"

#include <iostream>

#include "contest_dm.h"
#include "contest_util.h"
#include "flute3/flute.h"

namespace ieda_contest {

ContestPreprocess::ContestPreprocess(ContestDataManager* data_manager)
{
  /**
   * GCell在每个轴上是均匀的
   * instance的GCell坐标：instance的bounding_box中点的Gcell坐标
   */
  _data_manager = data_manager;
  makeLayerInfo();
  makeGCellInfo();

  Flute::readLUT();
}

void ContestPreprocess::makeLayerInfo()
{
  std::map<std::string, int>& layer_name_to_idx_map = _data_manager->get_database()->get_layer_name_to_idx_map();
  std::map<int, std::string>& layer_idx_to_name_map = _data_manager->get_database()->get_layer_idx_to_name_map();
  std::map<int, idb::IdbLayerDirection>& layer_idx_to_direction_map = _data_manager->get_database()->get_layer_idx_to_direction_map();

  int layer_idx = 0;
  for (idb::IdbLayer* idb_layer : _data_manager->get_idb_layout()->get_layers()->get_layers()) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
      layer_name_to_idx_map[idb_routing_layer->get_name()] = layer_idx;
      layer_idx_to_name_map[layer_idx] = idb_routing_layer->get_name();
      layer_idx_to_direction_map[layer_idx] = idb_routing_layer->get_direction();
      layer_idx++;
    }
  }
}

void ContestPreprocess::makeGCellInfo()
{
  std::map<std::string, int>& layer_name_to_idx_map = _data_manager->get_database()->get_layer_name_to_idx_map();

  if (_data_manager->get_database()->get_single_gcell_x_span() == 0 || _data_manager->get_database()->get_single_gcell_y_span() == 0) {
    std::set<int> single_gcell_x_span_set;
    std::set<int> single_gcell_y_span_set;
    for (idb::IdbGCellGrid* idb_gcell_grid : _data_manager->get_idb_layout()->get_gcell_grid_list()->get_gcell_grid_list()) {
      if (idb_gcell_grid->get_direction() == idb::IdbTrackDirection::kDirectionX) {
        single_gcell_x_span_set.insert(idb_gcell_grid->get_space());
      } else if (idb_gcell_grid->get_direction() == idb::IdbTrackDirection::kDirectionY) {
        single_gcell_y_span_set.insert(idb_gcell_grid->get_space());
      } else {
        std::cout << "[error] The direction of gcell grid is invaild!" << std::endl;
        exit(1);
      }
    }
    if (single_gcell_x_span_set.size() == 0) {
      std::cout << "[error] The x_gcell_grid is empty!" << std::endl;
      exit(1);
    }
    if (single_gcell_y_span_set.size() == 0) {
      std::cout << "[error] The y_gcell_grid is empty!" << std::endl;
      exit(1);
    }
    if (single_gcell_x_span_set.size() > 1) {
      std::cout << "[error] The x_gcell_grid is uneven!" << std::endl;
      exit(1);
    }
    if (single_gcell_y_span_set.size() > 1) {
      std::cout << "[error] The y_gcell_grid is uneven!" << std::endl;
      exit(1);
    }
    int single_gcell_x_span = *single_gcell_x_span_set.begin();
    int single_gcell_y_span = *single_gcell_y_span_set.begin();

    _data_manager->get_database()->set_single_gcell_x_span(single_gcell_x_span);
    _data_manager->get_database()->set_single_gcell_y_span(single_gcell_y_span);
  }
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();
  _data_manager->get_database()->set_single_gcell_area(single_gcell_x_span * single_gcell_y_span);
  // layer_gcell_supply_map
  std::map<int, int>& layer_gcell_supply_map = _data_manager->get_database()->get_layer_gcell_supply_map();
  for (idb::IdbLayer* idb_layer : _data_manager->get_idb_layout()->get_layers()->get_layers()) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);

      int prefer_pitch = 0;
      for (idb::IdbTrackGrid* idb_track_grid : idb_routing_layer->get_track_grid_list()) {
        idb::IdbTrack* idb_track = idb_track_grid->get_track();

        if (idb_routing_layer->get_direction() == idb::IdbLayerDirection::kHorizontal
            && idb_track->get_direction() == idb::IdbTrackDirection::kDirectionY) {
          prefer_pitch = idb_track->get_pitch();
        } else if (idb_routing_layer->get_direction() == idb::IdbLayerDirection::kVertical
                   && idb_track->get_direction() == idb::IdbTrackDirection::kDirectionX) {
          prefer_pitch = idb_track->get_pitch();
        }
      }
      if (idb_routing_layer->get_direction() == idb::IdbLayerDirection::kHorizontal) {
        layer_gcell_supply_map[layer_name_to_idx_map[idb_routing_layer->get_name()]] = (single_gcell_y_span / prefer_pitch);
      } else if (idb_routing_layer->get_direction() == idb::IdbLayerDirection::kVertical) {
        layer_gcell_supply_map[layer_name_to_idx_map[idb_routing_layer->get_name()]] = (single_gcell_x_span / prefer_pitch);
      }
    }
  }
}

void ContestPreprocess::doPreprocess()
{
  // 移动单元
  std::cout << "*********** do Preprocess place *********" << std::endl;
  place();

  // 三维布线
  std::cout << "*********** do Preprocess route *********" << std::endl;
  route();
}

}  // namespace ieda_contest
