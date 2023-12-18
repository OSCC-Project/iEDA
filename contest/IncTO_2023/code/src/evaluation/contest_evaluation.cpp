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
#include "idm.h"

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
    contest_instance.set_real_coord(coord);
    contest_instance.set_grid_coord(getGCellCoord(coord));
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
                         idb_pin->get_instance()->get_bounding_box()->get_middle_point_y(), 0);

      ContestPin contest_pin;
      contest_pin.set_real_coord(coord);
      contest_pin.set_grid_coord(getGCellCoord(coord));
      contest_pin.get_contained_instance_list().push_back(idb_pin->get_instance()->get_name());
      contest_pin.get_contained_pin_list().push_back(std::string(idb_pin->get_instance()->get_name() + ":" + idb_pin->get_pin_name()));
      pin_list.push_back(contest_pin);
    }
    if (idb_net->get_io_pin() != nullptr) {
      idb::IdbPin* io_pin = idb_net->get_io_pin();
      ContestCoord coord(io_pin->get_bounding_box()->get_middle_point_x(), io_pin->get_bounding_box()->get_middle_point_y(), 0);

      ContestPin contest_pin;
      contest_pin.set_real_coord(coord);
      contest_pin.set_grid_coord(getGCellCoord(coord));
      contest_pin.get_contained_instance_list().push_back("io_pin");
      contest_pin.get_contained_pin_list().push_back(io_pin->get_pin_name());
      pin_list.push_back(contest_pin);
    }
    // routing_segment_list
    std::map<std::pair<int, int>, std::set<int>> planar_layer_map;
    for (ContestGuide& contest_guide : contest_net.get_guide_list()) {
      int layer_idx = layer_name_to_idx_map[contest_guide.get_layer_name()];
      ContestCoord real_lb_coord(contest_guide.get_lb_x() + (single_gcell_x_span / 2), contest_guide.get_lb_y() + (single_gcell_y_span / 2),
                                 layer_idx);
      ContestCoord real_rt_coord(contest_guide.get_rt_x() - (single_gcell_x_span / 2), contest_guide.get_rt_y() - (single_gcell_y_span / 2),
                                 layer_idx);
      ContestCoord grid_lb_coord = getGCellCoord(real_lb_coord);
      ContestCoord grid_rt_coord = getGCellCoord(real_rt_coord);

      std::pair<int, int> lb_planar_coord(grid_lb_coord.get_x(), grid_lb_coord.get_y());
      std::pair<int, int> rt_planar_coord(grid_rt_coord.get_x(), grid_rt_coord.get_y());

      planar_layer_map[lb_planar_coord].insert(layer_idx);
      planar_layer_map[rt_planar_coord].insert(layer_idx);
    }
    // 删除layer只有一层的
    std::set<std::pair<int, int>> remove_key_set;
    for (auto& [planar_coord, layer_set] : planar_layer_map) {
      if (layer_set.size() == 1) {
        remove_key_set.insert(planar_coord);
      }
    }
    for (const auto& key : remove_key_set) {
      planar_layer_map.erase(key);
    }
    /////////////////////////////////////////////
    std::vector<ContestSegment>& routing_segment_list = contest_net.get_routing_segment_list();
    // 对这些添加proximal线段
    for (auto& [planar_coord, layer_set] : planar_layer_map) {
      int low_layer_idx = *layer_set.begin();
      auto layer_set_end = layer_set.end();
      layer_set_end--;
      int high_layer_idx = *layer_set_end;

      routing_segment_list.emplace_back(ContestCoord(planar_coord.first, planar_coord.second, low_layer_idx),
                                        ContestCoord(planar_coord.first, planar_coord.second, high_layer_idx));
    }
    // 对这些添加平面线段
    for (ContestGuide& contest_guide : contest_net.get_guide_list()) {
      int layer_idx = layer_name_to_idx_map[contest_guide.get_layer_name()];
      ContestCoord real_lb_coord(contest_guide.get_lb_x() + (single_gcell_x_span / 2), contest_guide.get_lb_y() + (single_gcell_y_span / 2),
                                 layer_idx);
      ContestCoord real_rt_coord(contest_guide.get_rt_x() - (single_gcell_x_span / 2), contest_guide.get_rt_y() - (single_gcell_y_span / 2),
                                 layer_idx);
      ContestCoord grid_lb_coord = getGCellCoord(real_lb_coord);
      ContestCoord grid_rt_coord = getGCellCoord(real_rt_coord);
      if (grid_lb_coord == grid_rt_coord) {
        // 删除点线段
        continue;
      }
      routing_segment_list.emplace_back(grid_lb_coord, grid_rt_coord);
    }
  }
}

bool ContestEvaluation::doEvaluation()
{
  // check overlap
  if (overlapCheckPassed()) {
    std::cout << "Overlap check successful!" << std::endl;
  } else {
    std::cout << "Overlap check failed!" << std::endl;
  }

  // check connectivity
  if (connectivityCheckPassed()) {
    std::cout << "Connectivity check successful!" << std::endl;
  } else {
    std::cout << "Connectivity check failed!" << std::endl;
  }

  // check overflow
  if (overflowCheckPassed()) {
    std::cout << "Overflow check successful!" << std::endl;
  } else {
    std::cout << "Overflow check failed!" << std::endl;
  }
  printInstanceArea();
  printWnsAndTNS();
  return true;
}

bool ContestEvaluation::overlapCheckPassed()
{
  int single_gcell_area = _data_manager->get_database()->get_single_gcell_area();

  std::map<ContestCoord, int, CmpContestCoord> coord_used_area_map;
  for (ContestInstance& contest_instance : _data_manager->get_database()->get_instance_list()) {
    ContestCoord& grid_coord = contest_instance.get_grid_coord();
    if ((coord_used_area_map[grid_coord] + contest_instance.get_area()) > single_gcell_area) {
      std::cout << "Overlap GCell: " << grid_coord.get_x() << grid_coord.get_y() << grid_coord.get_layer_idx() << std::endl;
      return false;
    } else {
      coord_used_area_map[grid_coord] += contest_instance.get_area();
    }
  }
  return true;
}

bool ContestEvaluation::connectivityCheckPassed()
{
  int net_count = 0;
  int disconnected_net_count = 0;
  bool check_result = true;
  for (ContestNet& contest_net : _data_manager->get_database()->get_net_list()) {
    net_count++;
    std::vector<ContestCoord> key_coord_list;
    for (ContestPin& contest_pin : contest_net.get_pin_list()) {
      key_coord_list.push_back(contest_pin.get_grid_coord());
    }
    if (!ContestUtil::passCheckingConnectivity(key_coord_list, contest_net.get_routing_segment_list())) {
      std::cout << "Disconnected Net: " << contest_net.get_net_name() << std::endl;
      disconnected_net_count++;
      check_result = false;
    }
  }

  std::cout << "Disconnected Net Count: " << disconnected_net_count << "/" << net_count << std::endl;

  return check_result;
}

bool ContestEvaluation::overflowCheckPassed()
{
  int overflow_gcell_count = 0;
  bool check_result = true;
  int max_overflow = 0;
  std::map<int, int>& layer_gcell_supply_map = _data_manager->get_database()->get_layer_gcell_supply_map();
  std::map<int, int> layer_to_overflow_map;

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

      if (lb_layer_idx != rt_layer_idx) {
        continue;
      }

      for (int x = lb_x; x <= rt_x; x++) {
        for (int y = lb_y; y <= rt_y; y++) {
          for (int layer_idx = lb_layer_idx; layer_idx <= rt_layer_idx; layer_idx++) {
            if ((lb_x != rt_x && (x == lb_x || x == rt_x)) || (lb_y != rt_y && (y == lb_y || y == rt_y))) {
              continue;
            }
            coord_set.insert(ContestCoord(x, y, layer_idx));
          }
        }
      }
    }
    for (const ContestCoord& coord : coord_set) {
      if (coord_demand_map[coord] == layer_gcell_supply_map[coord.get_layer_idx()]) {
        std::cout << "Overflow GCell: (" << coord.get_x() << "," << coord.get_y() << "," << coord.get_layer_idx() << ")" << std::endl;
        check_result = false;
        overflow_gcell_count++;
        layer_to_overflow_map[coord.get_layer_idx()]++;
      }
      coord_demand_map[coord] += 1;
      auto overflow = coord_demand_map[coord] - layer_gcell_supply_map[coord.get_layer_idx()];
      max_overflow = std::max(max_overflow, overflow);
    }
  }

  std::cout << "Max Overflow: " << max_overflow << std::endl;
  // output layer to overflow map
  std::cout << "Layer to Overflow Map: " << std::endl;
  for (auto& item : layer_to_overflow_map) {
    std::cout << item.first << ": " << item.second << std::endl;
  }

  return check_result;
}

void ContestEvaluation::printInstanceArea()
{
  uint64_t instance_area = 0;
  for (ContestInstance& instance : _data_manager->get_database()->get_instance_list()) {
    instance_area += instance.get_area();
  }

  std::cout << "total instance area: " << instance_area << std::endl;
}

void ContestEvaluation::printWnsAndTNS()
{
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();
  /////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////
  struct RCPin
  {
    RCPin() = default;
    RCPin(ContestCoord coord, bool is_real_pin, std::string pin_name)
    {
      _coord = coord;
      _is_real_pin = is_real_pin;
      _pin_name = pin_name;
    }
    RCPin(ContestCoord coord, bool is_real_pin, int fake_pin_id)
    {
      _coord = coord;
      _is_real_pin = is_real_pin;
      _fake_pin_id = fake_pin_id;
    }
    ~RCPin() = default;

    ContestCoord _coord;
    bool _is_real_pin = false;
    std::string _pin_name;
    int _fake_pin_id;
  };
  using RCPinSegment = std::pair<RCPin, RCPin>;
  auto getRCSegmentList = [](std::map<ContestCoord, std::vector<std::string>, CmpContestCoord>& coord_real_pin_map,
                             std::vector<ContestSegment>& routing_segment_list) {
    std::vector<RCPinSegment> rc_segment_list;
    // 生成线长为0的线段
    for (auto& [coord, real_pin_list] : coord_real_pin_map) {
      for (size_t i = 1; i < real_pin_list.size(); i++) {
        RCPin first_rc_pin(coord, true, real_pin_list[i - 1]);
        RCPin second_rc_pin(coord, true, real_pin_list[i]);
        rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
      }
    }
    // 构建coord_fake_pin_map
    std::map<ContestCoord, int, CmpContestCoord> coord_fake_pin_map;
    int fake_id = 0;
    for (ContestSegment& routing_segment : routing_segment_list) {
      ContestCoord& first_coord = routing_segment.get_first();
      ContestCoord& second_coord = routing_segment.get_second();

      if (!ContestUtil::exist(coord_real_pin_map, first_coord) && !ContestUtil::exist(coord_fake_pin_map, first_coord)) {
        coord_fake_pin_map[first_coord] = fake_id++;
      }
      if (!ContestUtil::exist(coord_real_pin_map, second_coord) && !ContestUtil::exist(coord_fake_pin_map, second_coord)) {
        coord_fake_pin_map[second_coord] = fake_id++;
      }
    }
    // 将routing_segment_list生成rc_segment_list
    for (ContestSegment& routing_segment : routing_segment_list) {
      ContestCoord& first_coord = routing_segment.get_first();
      ContestCoord& second_coord = routing_segment.get_second();

      RCPin first_rc_pin;
      if (ContestUtil::exist(coord_real_pin_map, first_coord)) {
        first_rc_pin = RCPin(first_coord, true, coord_real_pin_map[first_coord].front());
      } else if (ContestUtil::exist(coord_fake_pin_map, first_coord)) {
        first_rc_pin = RCPin(first_coord, false, coord_fake_pin_map[first_coord]);
      } else {
        std::cout << "The coord is not exist!" << std::endl;
      }
      RCPin second_rc_pin;
      if (ContestUtil::exist(coord_real_pin_map, second_coord)) {
        second_rc_pin = RCPin(second_coord, true, coord_real_pin_map[second_coord].front());
      } else if (ContestUtil::exist(coord_fake_pin_map, second_coord)) {
        second_rc_pin = RCPin(second_coord, false, coord_fake_pin_map[second_coord]);
      } else {
        std::cout << "The coord is not exist!" << std::endl;
      }
      rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
    }
    return rc_segment_list;
  };
  /////////////////////////////////////////////////////////////
  ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  timing_engine->set_num_threads(40);
  timing_engine->buildGraph();
  timing_engine->initRcTree();

  ista::Netlist* sta_netlist = timing_engine->get_netlist();

  for (ContestNet& net : _data_manager->get_database()->get_net_list()) {
    // coord_real_pin_map
    std::map<ContestCoord, std::vector<std::string>, CmpContestCoord> coord_real_pin_map;
    for (ContestPin& pin : net.get_pin_list()) {
      for (std::string& pin_name : pin.get_contained_pin_list()) {
        ContestCoord gcell_real_coord(pin.get_grid_coord().get_x() * single_gcell_x_span,
                                      pin.get_grid_coord().get_y() * single_gcell_y_span);
        coord_real_pin_map[gcell_real_coord].push_back(pin_name);
      }
    }
    // routing_segment_list
    std::vector<ContestSegment> routing_segment_list;
    if (net.get_routing_segment_list().empty()) {
      // std::set<ContestCoord, CmpContestCoord> coord_set;
      // for (ContestPin& pin : net.get_pin_list()) {
      //   coord_set.insert(pin.get_real_coord());
      // }
      // if (coord_set.size() != 1) {
      //   std::cout << "coord_set.size() != 1" << std::endl;
      // }
      // net.get_routing_segment_list().emplace_back(*coord_set.begin(), *coord_set.begin());
    } else {
      for (ContestSegment& grid_segment : net.get_routing_segment_list()) {
        ContestCoord first_coord(grid_segment.get_first().get_x() * single_gcell_x_span,
                                 grid_segment.get_first().get_y() * single_gcell_y_span);
        ContestCoord second_coord(grid_segment.get_second().get_x() * single_gcell_x_span,
                                  grid_segment.get_second().get_y() * single_gcell_y_span);
        if (first_coord == second_coord) {
          continue;
        }
        routing_segment_list.emplace_back(first_coord, second_coord);
      }
    }
    // 构建RC-tree
    ista::Net* ista_net = sta_netlist->findNet(net.get_net_name().c_str());
    for (RCPinSegment& segment : getRCSegmentList(coord_real_pin_map, routing_segment_list)) {
      auto getRctNode = [timing_engine, sta_netlist, ista_net](RCPin& rc_pin) {
        ista::RctNode* rct_node = nullptr;
        if (rc_pin._is_real_pin) {
          ista::DesignObject* pin_port = nullptr;
          auto pin_port_list = sta_netlist->findPin(rc_pin._pin_name.c_str(), false, false);
          if (!pin_port_list.empty()) {
            pin_port = pin_port_list.front();
          } else {
            pin_port = sta_netlist->findPort(rc_pin._pin_name.c_str());
          }
          rct_node = timing_engine->makeOrFindRCTreeNode(pin_port);
        } else {
          rct_node = timing_engine->makeOrFindRCTreeNode(ista_net, rc_pin._fake_pin_id);
        }
        return rct_node;
      };
      RCPin& first_rc_pin = segment.first;
      RCPin& second_rc_pin = segment.second;

      int distance = ContestUtil::getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
      int unit = dmInst->get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu();
      std::optional<double> width = std::nullopt;
      double cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getCapacitance(3, distance / 1.0 / unit, width);
      double res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getResistance(3, distance / 1.0 / unit, width);

      ista::RctNode* first_node = getRctNode(first_rc_pin);
      ista::RctNode* second_node = getRctNode(second_rc_pin);
      timing_engine->makeResistor(ista_net, first_node, second_node, res);
      timing_engine->incrCap(first_node, cap / 2);
      timing_engine->incrCap(second_node, cap / 2);
    }
    timing_engine->updateRCTreeInfo(ista_net);
  }
  timing_engine->updateTiming();
  timing_engine->reportTiming();
  /////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////

  const char* master_clock_name = "CLK_osc_in";
  double wns = timing_engine->get_ista()->getWNS(master_clock_name, AnalysisMode::kMax);
  double tns = timing_engine->get_ista()->getTNS(master_clock_name, AnalysisMode::kMax);
  // TODO calc score
  std::cout << "wns: " << wns << std::endl;
  std::cout << "tns: " << tns << std::endl;
}

}  // namespace ieda_contest
