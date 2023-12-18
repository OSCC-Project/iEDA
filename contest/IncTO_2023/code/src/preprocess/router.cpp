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
#include <iostream>

#include "contest_dm.h"
#include "contest_preprocess.h"
#include "contest_util.h"

namespace ieda_contest {

typedef gridmap::Map<astar::Node<3>> Map;

void ContestPreprocess::route()
{
  makeNetList();
  routeGuide();
  updateGuideList();
  outputGCellGrid();
}

void ContestPreprocess::makeNetList()
{
  std::vector<ContestNet>& net_list = _data_manager->get_database()->get_net_list();

  for (idb::IdbNet* idb_net : _data_manager->get_idb_design()->get_net_list()->get_net_list()) {
    ContestNet contest_net;
    contest_net.set_net_name(idb_net->get_net_name());
    std::vector<ContestPin>& pin_list = contest_net.get_pin_list();
    for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      ContestCoord coord(idb_pin->get_instance()->get_bounding_box()->get_middle_point_x(),
                         idb_pin->get_instance()->get_bounding_box()->get_middle_point_y(), 0);

      ContestPin contest_pin;
      contest_pin.set_real_coord(coord);
      contest_pin.set_grid_coord(getGCellCoord(coord));
      contest_pin.get_contained_instance_list().push_back(idb_pin->get_instance()->get_name());
      pin_list.push_back(contest_pin);
    }
    if (idb_net->get_io_pin() != nullptr) {
      idb::IdbPin* io_pin = idb_net->get_io_pin();
      ContestCoord coord(io_pin->get_bounding_box()->get_middle_point_x(), io_pin->get_bounding_box()->get_middle_point_y(), 0);

      ContestPin contest_pin;
      contest_pin.set_real_coord(coord);
      contest_pin.set_grid_coord(getGCellCoord(coord));
      contest_pin.get_contained_instance_list().push_back("io");
      pin_list.push_back(contest_pin);
    }
    net_list.push_back(contest_net);
  }
}

ContestCoord ContestPreprocess::getGCellCoord(const ContestCoord& coord)
{
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();

  ContestCoord gcell_coord;
  gcell_coord.set_x((coord.get_x() > 0 ? coord.get_x() - 1 : coord.get_x()) / single_gcell_x_span);
  gcell_coord.set_y((coord.get_y() > 0 ? coord.get_y() - 1 : coord.get_y()) / single_gcell_y_span);
  gcell_coord.set_layer_idx(coord.get_layer_idx());
  return gcell_coord;
}

struct CmpContestNet
{
  bool operator()(ContestNet& a, ContestNet& b)
  {
    auto aabb_a = getAABB(a);
    auto aabb_b = getAABB(b);
    auto area_a = (aabb_a.get_second().get_x() - aabb_a.get_first().get_x()) * (aabb_a.get_second().get_y() - aabb_a.get_first().get_y());
    auto area_b = (aabb_b.get_second().get_x() - aabb_b.get_first().get_x()) * (aabb_b.get_second().get_y() - aabb_b.get_first().get_y());
    if (area_a == area_b) {
      auto length_a
          = std::max(aabb_a.get_second().get_x() - aabb_a.get_first().get_x(), aabb_a.get_second().get_y() - aabb_a.get_first().get_y());
      auto length_b
          = std::max(aabb_b.get_second().get_x() - aabb_b.get_first().get_x(), aabb_b.get_second().get_y() - aabb_b.get_first().get_y());
      return length_a < length_b;
    }
    return area_a > area_b;
  }

  ContestSegment getAABB(ContestNet& n)
  {
    auto& pins = n.get_pin_list();

    ContestSegment aabb(pins[0].get_grid_coord(), pins[0].get_grid_coord());
    for (ContestPin& pin : pins) {
      if (pin.get_grid_coord().get_x() < aabb.get_first().get_x()) {
        aabb.get_first().set_x(pin.get_grid_coord().get_x());
      }
      if (pin.get_grid_coord().get_y() < aabb.get_first().get_y()) {
        aabb.get_first().set_y(pin.get_grid_coord().get_y());
      }
      if (pin.get_grid_coord().get_x() > aabb.get_second().get_x()) {
        aabb.get_second().set_x(pin.get_grid_coord().get_x());
      }
      if (pin.get_grid_coord().get_y() > aabb.get_second().get_y()) {
        aabb.get_second().set_y(pin.get_grid_coord().get_y());
      }
    }
    return aabb;
  }
};

void ContestPreprocess::routeGuide()
{
  ieda::Stats route_state;
  createMap();

  std::vector<ContestNet>& contest_net_list = _data_manager->get_database()->get_net_list();

  std::sort(contest_net_list.begin(), contest_net_list.end(), CmpContestNet());

  // int batch_size = ContestUtil::getBatchSize(contest_net_list.size());
  int net_num = contest_net_list.size();
  int percent = 0;
  double time_delta = 0;
  double time_last = 0;

  for (int i = 0; i < net_num; i++) {
    routeNet(contest_net_list[i]);

    time_delta = route_state.elapsedRunTime();
    if (time_delta - time_last > 300) {  // ((i + 1) % batch_size == 0) {
      percent = static_cast<int>((i + 1.0) / net_num * 10000);
      std::cout << " " << percent / 100 << "." << std::setw(2) << std::setfill('0') << percent % 100 << "% ";

      std::cout << "Routed " << (i + 1) << " nets, ";

      std::cout << "Time: " << time_delta << "s " << std::endl;
      time_last = time_delta;
    }
  }
  std::cout << "Routed " << contest_net_list.size() << " nets, Time: " << route_state.elapsedRunTime() << "s " << std::endl;

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
}

void ContestPreprocess::createMap()
{
  std::map<int, int>& layer_gcell_supply_map = _data_manager->get_database()->get_layer_gcell_supply_map();

  double gcellWidth = _data_manager->get_database()->get_single_gcell_x_span();
  double gcellHeight = _data_manager->get_database()->get_single_gcell_y_span();
  int gcell_map_size_x = static_cast<int>(_data_manager->get_idb_layout()->get_die()->get_width() / gcellWidth + 1);
  int gcell_map_size_y = static_cast<int>(_data_manager->get_idb_layout()->get_die()->get_height() / gcellHeight + 1);
  int gcell_map_size_z = _data_manager->get_database()->get_layer_name_to_idx_map().size();

  _grid_map = Map(_data_manager, gcell_map_size_x, gcell_map_size_y, gcell_map_size_z, static_cast<int>(gcellWidth),
                  static_cast<int>(gcellHeight));
  _pathfinder.set_map(_grid_map);

  for (int i = 0; i < gcell_map_size_z; ++i) {
    _grid_map.fillLayerSupplyResourceCnt(layer_gcell_supply_map[i] / 2, i);
  }
}

void ContestPreprocess::routeNet(ContestNet& contest_net)
{
  // key_coord and routing_segment_list
  std::vector<ContestCoord> key_coord_list = makeKeyCoordList(contest_net);
  std::vector<ContestSegment>& routing_segment_list = contest_net.get_routing_segment_list();

  // route by astar
  for (ContestSegment& topo : getTopoListByFlute(key_coord_list)) {
    for (ContestSegment& routing_segment : getRoutingSegmentList(topo, contest_net)) {
      routing_segment_list.push_back(routing_segment);
    }
  }
}

std::vector<ContestCoord> ContestPreprocess::makeKeyCoordList(ContestNet& contest_net)
{
  std::vector<ContestCoord> key_coord_list;
  for (ContestPin& pin : contest_net.get_pin_list()) {
    key_coord_list.push_back(pin.get_grid_coord());
  }
  std::sort(key_coord_list.begin(), key_coord_list.end(), CmpContestCoord());
  key_coord_list.erase(std::unique(key_coord_list.begin(), key_coord_list.end()), key_coord_list.end());
  return key_coord_list;
}

std::vector<ContestSegment> ContestPreprocess::getTopoListByGreedy(std::vector<ContestCoord>& coord_list)
{
  std::vector<ContestSegment> topo_list;
  for (int i = 0; i < (int) coord_list.size() - 1; ++i) {
    ContestCoord first_coord = coord_list[i];
    ContestCoord second_coord = coord_list[i + 1];
    topo_list.emplace_back(first_coord, second_coord);
  }
  return topo_list;
}

std::vector<ContestSegment> ContestPreprocess::getTopoListByFlute(std::vector<ContestCoord>& coord_list)
{
  size_t coord_num = coord_list.size();
  if (coord_num == 1) {
    return {};
  }
  int max_layer_idx = INT_MIN;
  int min_layer_idx = INT_MAX;
  for (ContestCoord& coord : coord_list) {
    std::max(max_layer_idx, coord.get_layer_idx());
    std::min(min_layer_idx, coord.get_layer_idx());
  }
  int layer_idx = (max_layer_idx + min_layer_idx) / 2;

  Flute::DTYPE* x_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (coord_num));
  Flute::DTYPE* y_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (coord_num));
  for (size_t i = 0; i < coord_num; i++) {
    x_list[i] = coord_list[i].get_x();
    y_list[i] = coord_list[i].get_y();
  }
  Flute::Tree flute_tree = Flute::flute(coord_num, x_list, y_list, FLUTE_ACCURACY);
  // Flute::printtree(flute_tree);
  free(x_list);
  free(y_list);

  std::vector<ContestSegment> topo_list;
  for (int i = 0; i < 2 * flute_tree.deg - 2; i++) {
    int n_id = flute_tree.branch[i].n;
    ContestCoord first_coord(flute_tree.branch[i].x, flute_tree.branch[i].y, layer_idx);
    ContestCoord second_coord(flute_tree.branch[n_id].x, flute_tree.branch[n_id].y, layer_idx);
    topo_list.emplace_back(first_coord, second_coord);
  }
  Flute::free_tree(flute_tree);
  return topo_list;
}

std::vector<ContestSegment> ContestPreprocess::getRoutingSegmentList(ContestSegment& topo, ContestNet& contest_net)
{
  std::vector<ContestSegment> routing_segment_list;

  _pathfinder.set_starting_position({topo.get_first().get_x(), topo.get_first().get_y(), topo.get_first().get_layer_idx()});
  _pathfinder.set_ending_position({topo.get_second().get_x(), topo.get_second().get_y(), topo.get_second().get_layer_idx()});

  auto path = _pathfinder.findPath();

  if (!path.empty()) {
    ContestCoord segment_starting_coord(path[0]->get_position().get<0>(), path[0]->get_position().get<1>(),
                                        path[0]->get_position().get<2>());

    for (int i = 1; i < (int) path.size(); ++i) {
      ContestCoord starting_coord(path[i - 1]->get_position().get<0>(), path[i - 1]->get_position().get<1>(),
                                  path[i - 1]->get_position().get<2>());
      ContestCoord ending_coord(path[i]->get_position().get<0>(), path[i]->get_position().get<1>(), path[i]->get_position().get<2>());
      if (i + 1 != (int) path.size()) {
        ContestCoord next_coord(path[i + 1]->get_position().get<0>(), path[i + 1]->get_position().get<1>(),
                                path[i + 1]->get_position().get<2>());
        if (ending_coord - starting_coord != next_coord - ending_coord) {
          ContestSegment new_segment(segment_starting_coord, ending_coord);
          routing_segment_list.emplace_back(new_segment);

          segment_starting_coord = ending_coord;
        }
      } else {
        ContestSegment new_segment(segment_starting_coord, ending_coord);
        routing_segment_list.emplace_back(new_segment);
      }

      // decrease remaining resourse when through a gcell
      if (starting_coord.get_layer_idx() == ending_coord.get_layer_idx()) {
        Map::point starting_gcell = Map::point(starting_coord.get_x(), starting_coord.get_y(), starting_coord.get_layer_idx());
        Map::point ending_gcell = Map::point(ending_coord.get_x(), ending_coord.get_y(), ending_coord.get_layer_idx());
        _grid_map[ending_gcell].addNetId(reinterpret_cast<long>(&contest_net));
        _grid_map[starting_gcell].addNetId(reinterpret_cast<long>(&contest_net));
      }
    }
  }

  return routing_segment_list;
}

bool ContestPreprocess::connectivityCheckPassed()
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

bool ContestPreprocess::overflowCheckPassed()
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
        // std::cout << "Overflow GCell: (" << coord.get_x() << "," << coord.get_y() << "," << coord.get_layer_idx() << ")" << std::endl;
        check_result = false;
        overflow_gcell_count++;
        layer_to_overflow_map[coord.get_layer_idx()]++;
      }
      coord_demand_map[coord] += 1;
      auto overflow = coord_demand_map[coord] - layer_gcell_supply_map[coord.get_layer_idx()];
      max_overflow = std::max(max_overflow, overflow);
    }
  }

  std::cout << "Overflow GCell Count: " << overflow_gcell_count << "/"
            << _grid_map.get_map_size().get<0>() * _grid_map.get_map_size().get<1>() * _grid_map.get_map_size().get<2>() << std::endl;

  std::cout << "Max Overflow: " << max_overflow << std::endl;
  // output layer to overflow map
  std::cout << "Layer to Overflow Map: " << std::endl;
  for (auto& item : layer_to_overflow_map) {
    std::cout << item.first << ": " << item.second << std::endl;
  }

  return check_result;
}

void ContestPreprocess::updateGuideList()
{
  for (ContestNet& net : _data_manager->get_database()->get_net_list()) {
    for (ContestSegment& routing_segment : net.get_routing_segment_list()) {
      for (ContestGuide& guide : getGuide(routing_segment)) {
        net.get_guide_list().push_back(guide);
      }
    }
  }
}

std::vector<ContestGuide> ContestPreprocess::getGuide(ContestSegment& routing_segment)
{
  std::map<int, std::string>& layer_idx_to_name_map = _data_manager->get_database()->get_layer_idx_to_name_map();
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();

  std::vector<ContestGuide> guide_list;

  ContestCoord first_coord = routing_segment.get_first();
  ContestCoord second_coord = routing_segment.get_second();

  int first_layer_idx = first_coord.get_layer_idx();
  int second_layer_idx = second_coord.get_layer_idx();
  if (first_layer_idx != second_layer_idx) {
    ContestUtil::swapByASC(first_layer_idx, second_layer_idx);
    for (int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
      ContestGuide guide;
      guide.set_lb_x(first_coord.get_x() * single_gcell_x_span);
      guide.set_lb_y(first_coord.get_y() * single_gcell_y_span);
      guide.set_rt_x((second_coord.get_x() + 1) * single_gcell_x_span);
      guide.set_rt_y((second_coord.get_y() + 1) * single_gcell_y_span);
      guide.set_layer_name(layer_idx_to_name_map[layer_idx]);
      guide_list.emplace_back(guide);
    }
  } else {
    ContestUtil::swapByCMP(first_coord, second_coord, CmpContestCoord());

    ContestGuide guide;
    guide.set_lb_x(first_coord.get_x() * single_gcell_x_span);
    guide.set_lb_y(first_coord.get_y() * single_gcell_y_span);
    guide.set_rt_x((second_coord.get_x() + 1) * single_gcell_x_span);
    guide.set_rt_y((second_coord.get_y() + 1) * single_gcell_y_span);
    guide.set_layer_name(layer_idx_to_name_map[first_layer_idx]);
    guide_list.emplace_back(guide);
  }
  return guide_list;
}

void ContestPreprocess::outputGCellGrid()
{
  int single_gcell_x_span = _data_manager->get_database()->get_single_gcell_x_span();
  int single_gcell_y_span = _data_manager->get_database()->get_single_gcell_y_span();

  idb::IdbDie* die = _data_manager->get_idb_layout()->get_die();
  int32_t start_x = die->get_llx();
  int32_t start_y = die->get_lly();
  int32_t width = die->get_width();
  int32_t heigth = die->get_height();

  idb::IdbGCellGridList* idb_gcell_grid_list = _data_manager->get_idb_layout()->get_gcell_grid_list();
  idb_gcell_grid_list->clear();

  for (idb::IdbTrackDirection idb_track_direction : {idb::IdbTrackDirection::kDirectionX, idb::IdbTrackDirection::kDirectionY}) {
    if (idb_track_direction == idb::IdbTrackDirection::kDirectionX) {
      idb::IdbGCellGrid* idb_gcell_grid = new idb::IdbGCellGrid();
      idb_gcell_grid->set_start(start_x);
      idb_gcell_grid->set_space(single_gcell_x_span);
      idb_gcell_grid->set_num(std::ceil(width / 1.0 / single_gcell_x_span) + 1);
      idb_gcell_grid->set_direction(idb_track_direction);
      idb_gcell_grid_list->add_gcell_grid(idb_gcell_grid);
    } else {
      idb::IdbGCellGrid* idb_gcell_grid = new idb::IdbGCellGrid();
      idb_gcell_grid->set_start(start_y);
      idb_gcell_grid->set_space(single_gcell_y_span);
      idb_gcell_grid->set_num(std::ceil(heigth / 1.0 / single_gcell_y_span) + 1);
      idb_gcell_grid->set_direction(idb_track_direction);
      idb_gcell_grid_list->add_gcell_grid(idb_gcell_grid);
    }
  }
}

}  // namespace ieda_contest
