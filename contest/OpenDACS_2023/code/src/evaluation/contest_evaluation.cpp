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
#include "contest_evaluation.h"

#include <iostream>

#include "contest_dm.h"
#include "contest_util.h"

namespace ieda_contest {

ContestEvaluation::ContestEvaluation(ContestDataManager* data_manager)
{
  /**
   * GCell在每个轴上是均匀的
   * instance的GCell坐标：instance的bounding_box中点的Gcell坐标
   */
  _data_manager = data_manager;
  makeLayerInfo();
  makeGCellInfo();
  makeInstanceList();
  makeNetList();
}

void ContestEvaluation::makeLayerInfo()
{
  std::map<std::string, int>& layer_name_to_idx_map = _data_manager->get_database()->get_layer_name_to_idx_map();

  int layer_idx = 0;
  for (idb::IdbLayer* idb_layer : _data_manager->get_idb_layout()->get_layers()->get_layers()) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
      layer_name_to_idx_map[idb_routing_layer->get_name()] = layer_idx++;
    }
  }
}

void ContestEvaluation::makeGCellInfo()
{
  std::map<std::string, int>& layer_name_to_idx_map = _data_manager->get_database()->get_layer_name_to_idx_map();

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
  ///////////////
  single_gcell_x_span_set.insert(14000);
  single_gcell_y_span_set.insert(7500);
  ///////////////
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
  /**
   * single_gcell_x_span
   * single_gcell_y_span
   * single_gcell_area
   */
  _data_manager->get_database()->set_single_gcell_x_span(single_gcell_x_span);
  _data_manager->get_database()->set_single_gcell_y_span(single_gcell_y_span);
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

void ContestEvaluation::makeInstanceList()
{
  std::vector<ContestInstance>& contest_instance_list = _data_manager->get_database()->get_instance_list();

  for (idb::IdbInstance* idb_instance : _data_manager->get_idb_design()->get_instance_list()->get_instance_list()) {
    ContestCoord coord(idb_instance->get_bounding_box()->get_middle_point_x(), idb_instance->get_bounding_box()->get_middle_point_y(), 0);

    ContestInstance contest_instance;
    contest_instance.set_name(idb_instance->get_name());
    contest_instance.set_coord(getGCellCoord(coord));
    contest_instance.set_area(idb_instance->get_cell_master()->get_width() * idb_instance->get_cell_master()->get_height());
    contest_instance_list.push_back(contest_instance);
  }
}

ContestCoord ContestEvaluation::getGCellCoord(const ContestCoord& coord)
{
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();

  ContestCoord gcell_coord;
  gcell_coord.set_x((coord.get_x() > 0 ? coord.get_x() - 1 : coord.get_x()) / single_gcell_x_span);
  gcell_coord.set_y((coord.get_y() > 0 ? coord.get_y() - 1 : coord.get_y()) / single_gcell_y_span);
  gcell_coord.set_layer_idx(coord.get_layer_idx());
  return gcell_coord;
}

void ContestEvaluation::makeNetList()
{
  std::map<std::string, int>& layer_name_to_idx_map = _data_manager->get_database()->get_layer_name_to_idx_map();
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();

  for (ContestNet& contest_net : _data_manager->get_database()->get_net_list()) {
    idb::IdbNet* idb_net = _data_manager->get_idb_design()->get_net_list()->find_net(contest_net.get_net_name());
    // pin_list
    std::vector<ContestPin>& pin_list = contest_net.get_pin_list();
    for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      ContestCoord coord(idb_pin->get_instance()->get_bounding_box()->get_middle_point_x(),
                         idb_pin->get_instance()->get_bounding_box()->get_middle_point_y());

      ContestPin contest_pin;
      contest_pin.set_coord(getGCellCoord(coord));
      contest_pin.get_contained_instance_list().push_back(idb_pin->get_instance()->get_name());
      pin_list.push_back(contest_pin);
    }
    if (idb_net->get_io_pin() != nullptr) {
      idb::IdbPin* io_pin = idb_net->get_io_pin();
      ContestCoord coord(io_pin->get_bounding_box()->get_middle_point_x(), io_pin->get_bounding_box()->get_middle_point_y());

      ContestPin contest_pin;
      contest_pin.set_coord(getGCellCoord(coord));
      contest_pin.get_contained_instance_list().push_back("io");
      pin_list.push_back(contest_pin);
    }
    // routing_segment_list
    std::set<ContestCoord, CmpContestCoord> proximal_coord_set;
    for (ContestGuide& contest_guide : contest_net.get_guide_list()) {
      int lb_x = contest_guide.get_lb_x() + (single_gcell_x_span / 2);
      int lb_y = contest_guide.get_lb_y() + (single_gcell_y_span / 2);
      int rt_x = contest_guide.get_rt_x() - (single_gcell_x_span / 2);
      int rt_y = contest_guide.get_rt_y() - (single_gcell_y_span / 2);
      int layer_idx = layer_name_to_idx_map[contest_guide.get_layer_name()];
      for (int x = lb_x; x <= rt_x; x++) {
        for (int y = lb_y; y <= rt_y; y++) {
          proximal_coord_set.insert(ContestCoord(x, y, layer_idx));
        }
      }
    }
    std::vector<ContestSegment>& routing_segment_list = contest_net.get_routing_segment_list();
    for (ContestGuide& contest_guide : contest_net.get_guide_list()) {
      ContestCoord first_coord(contest_guide.get_lb_x() + (single_gcell_x_span / 2), contest_guide.get_lb_y() + (single_gcell_y_span / 2));
      ContestCoord second_coord(contest_guide.get_rt_x() - (single_gcell_x_span / 2), contest_guide.get_rt_y() - (single_gcell_y_span / 2));

      if (first_coord == second_coord) {
        // 竖直
        ContestCoord up_coord = first_coord;
        up_coord.set_layer_idx(first_coord.get_layer_idx() + 1);
        if (ContestUtil::exist(proximal_coord_set, up_coord)) {
          routing_segment_list.emplace_back(first_coord, up_coord);
        }
        ContestCoord down_coord = first_coord;
        down_coord.set_layer_idx(first_coord.get_layer_idx() - 1);
        if (ContestUtil::exist(proximal_coord_set, down_coord)) {
          routing_segment_list.emplace_back(first_coord, down_coord);
        }
      } else {
        // 平面
        routing_segment_list.emplace_back(first_coord, second_coord);
      }
    }
  }
}

bool ContestEvaluation::doEvaluation(std::string report_file)
{
  // overlap检查
  if (!overlapCheckPassed()) {
    std::cout << "Overlap check failed!" << std::endl;
    return false;
  }

  // 连通性检查
  if (!connectivityCheckPassed()) {
    std::cout << "Connectivity check failed!" << std::endl;
    return false;
  }

  // overflow检查
  if (!overflowCheckPassed()) {
    std::cout << "Overflow check failed!" << std::endl;
    return false;
  }

  double score = 0;
  // 计算时序分数
  score += calcTimingScore();
  std::cout << "##############################" << std::endl;
  std::cout << "Final score: " << score << std::endl;
  std::cout << "##############################" << std::endl;

  return true;
}

bool ContestEvaluation::overlapCheckPassed()
{
  int single_gcell_area = _data_manager->get_database()->get_single_gcell_area();

  std::map<ContestCoord, int, CmpContestCoord> coord_used_area_map;
  for (ContestInstance& contest_instance : _data_manager->get_database()->get_instance_list()) {
    ContestCoord& contest_coord = contest_instance.get_coord();
    if ((coord_used_area_map[contest_coord] + contest_instance.get_area()) > single_gcell_area) {
      std::cout << "Overlap GCell: " << contest_coord.get_x() << contest_coord.get_y() << contest_coord.get_layer_idx() << std::endl;
      return false;
    } else {
      coord_used_area_map[contest_coord] += contest_instance.get_area();
    }
  }
  return true;
}

bool ContestEvaluation::connectivityCheckPassed()
{
  for (ContestNet& contest_net : _data_manager->get_database()->get_net_list()) {
    std::vector<ContestCoord> key_coord_list;
    for (ContestPin& contest_pin : contest_net.get_pin_list()) {
      key_coord_list.push_back(contest_pin.get_coord());
    }
    if (!ContestUtil::passCheckingConnectivity(key_coord_list, contest_net.get_routing_segment_list())) {
      std::cout << "Disconnected Net: " << contest_net.get_net_name() << std::endl;
      return false;
    }
  }

  return true;
}

bool ContestEvaluation::overflowCheckPassed()
{
  std::map<int, int>& layer_gcell_supply_map = _data_manager->get_database()->get_layer_gcell_supply_map();

  std::map<ContestCoord, int, CmpContestCoord> coord_demand_map;
  for (ContestNet& contest_net : _data_manager->get_database()->get_net_list()) {
    std::set<ContestCoord, CmpContestCoord> coord_set;
    for (ContestSegment& routing_segment : contest_net.get_routing_segment_list()) {
      int lb_x = routing_segment.get_first().get_x();
      int lb_y = routing_segment.get_first().get_y();
      int lb_layer_idx = routing_segment.get_first().get_layer_idx();

      int rt_x = routing_segment.get_second().get_x();
      int rt_y = routing_segment.get_second().get_y();
      int rt_layer_idx = routing_segment.get_second().get_layer_idx();

      for (int x = lb_x; x <= rt_x; x++) {
        for (int y = lb_y; y <= rt_y; y++) {
          for (int layer_idx = lb_layer_idx; layer_idx <= rt_layer_idx; layer_idx++) {
            coord_set.insert(ContestCoord(x, y, layer_idx));
          }
        }
      }
    }
    for (const ContestCoord& coord : coord_set) {
      if ((coord_demand_map[coord] + 1) > layer_gcell_supply_map[coord.get_layer_idx()]) {
        std::cout << "Overflow GCell: " << coord.get_x() << coord.get_y() << coord.get_layer_idx() << std::endl;
        return false;
      } else {
        coord_demand_map[coord] += 1;
      }
    }
  }
  return true;
}

double ContestEvaluation::calcTimingScore()
{
  return 101;
}

}  // namespace ieda_contest
