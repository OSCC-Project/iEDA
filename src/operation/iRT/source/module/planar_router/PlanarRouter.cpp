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
#include "PlanarRouter.hpp"

#include "GDSPlotter.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void PlanarRouter::initInst()
{
  if (_pr_instance == nullptr) {
    _pr_instance = new PlanarRouter();
  }
}

PlanarRouter& PlanarRouter::getInst()
{
  if (_pr_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_pr_instance;
}

void PlanarRouter::destroyInst()
{
  if (_pr_instance != nullptr) {
    delete _pr_instance;
    _pr_instance = nullptr;
  }
}

// function

void PlanarRouter::route()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  PRModel pr_model = initPRModel();
  setPRParameter(pr_model);
  buildNodeMap(pr_model);
  buildPRNodeNeighbor(pr_model);
  buildOrientSupply(pr_model);
  sortPRModel(pr_model);
  routePRModel(pr_model);
  updatePRModel(pr_model);
  // updateSummary(pr_model);
  // printSummary(pr_model);
  // writeDemandCSV(pr_model);
  // writeOverflowCSV(pr_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

PlanarRouter* PlanarRouter::_pr_instance = nullptr;

PRModel PlanarRouter::initPRModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  PRModel pr_model;
  pr_model.set_pr_net_list(convertToPRNetList(net_list));
  return pr_model;
}

std::vector<PRNet> PlanarRouter::convertToPRNetList(std::vector<Net>& net_list)
{
  std::vector<PRNet> pr_net_list;
  pr_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    pr_net_list.emplace_back(convertToPRNet(net_list[i]));
  }
  return pr_net_list;
}

PRNet PlanarRouter::convertToPRNet(Net& net)
{
  PRNet pr_net;
  pr_net.set_origin_net(&net);
  pr_net.set_net_idx(net.get_net_idx());
  pr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    pr_net.get_pr_pin_list().push_back(PRPin(pin));
  }
  pr_net.set_bounding_box(net.get_bounding_box());
  return pr_net;
}

void PlanarRouter::setPRParameter(PRModel& pr_model)
{
  PRParameter pr_parameter;
  RTLOG.info(Loc::current(), "congestion_unit: ", pr_parameter.get_congestion_unit());
  pr_model.set_pr_parameter(pr_parameter);
}

void PlanarRouter::buildNodeMap(PRModel& pr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  GridMap<PRNode>& pr_node_map = pr_model.get_pr_node_map();
  pr_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      PRNode& pr_node = pr_node_map[x][y];
      pr_node.set_coord(x, y);
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PlanarRouter::buildPRNodeNeighbor(PRModel& pr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  GridMap<PRNode>& pr_node_map = pr_model.get_pr_node_map();
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<Orientation, PRNode*>& neighbor_node_map = pr_node_map[x][y].get_neighbor_node_map();
      if (x != 0) {
        neighbor_node_map[Orientation::kWest] = &pr_node_map[x - 1][y];
      }
      if (x != (pr_node_map.get_x_size() - 1)) {
        neighbor_node_map[Orientation::kEast] = &pr_node_map[x + 1][y];
      }
      if (y != 0) {
        neighbor_node_map[Orientation::kSouth] = &pr_node_map[x][y - 1];
      }
      if (y != (pr_node_map.get_y_size() - 1)) {
        neighbor_node_map[Orientation::kNorth] = &pr_node_map[x][y + 1];
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PlanarRouter::buildOrientSupply(PRModel& pr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  GridMap<PRNode>& pr_node_map = pr_model.get_pr_node_map();

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (int32_t layer_idx = bottom_routing_layer_idx; layer_idx <= top_routing_layer_idx; layer_idx++) {
        for (auto& [orient, supply] : gcell_map[x][y].get_routing_orient_supply_map()[layer_idx]) {
          pr_node_map[x][y].get_orient_supply_map()[orient] += supply;
        }
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PlanarRouter::sortPRModel(PRModel& pr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  std::vector<int32_t>& pr_net_idx_list = pr_model.get_pr_net_idx_list();
  for (PRNet& pr_net : pr_model.get_pr_net_list()) {
    pr_net_idx_list.push_back(pr_net.get_net_idx());
  }
  std::sort(pr_net_idx_list.begin(), pr_net_idx_list.end(),
            [&](int32_t net_idx1, int32_t net_idx2) { return sortByMultiLevel(pr_model, net_idx1, net_idx2); });
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool PlanarRouter::sortByMultiLevel(PRModel& pr_model, int32_t net_idx1, int32_t net_idx2)
{
  PRNet& net1 = pr_model.get_pr_net_list()[net_idx1];
  PRNet& net2 = pr_model.get_pr_net_list()[net_idx2];

  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByClockPriority(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByRoutingAreaASC(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByLengthWidthRatioDESC(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByPinNumDESC(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  return false;
}

// 时钟线网优先
SortStatus PlanarRouter::sortByClockPriority(PRNet& net1, PRNet& net2)
{
  ConnectType net1_connect_type = net1.get_connect_type();
  ConnectType net2_connect_type = net2.get_connect_type();

  if (net1_connect_type == ConnectType::kClock && net2_connect_type != ConnectType::kClock) {
    return SortStatus::kTrue;
  } else if (net1_connect_type != ConnectType::kClock && net2_connect_type == ConnectType::kClock) {
    return SortStatus::kFalse;
  } else {
    return SortStatus::kEqual;
  }
}

// RoutingArea 升序
SortStatus PlanarRouter::sortByRoutingAreaASC(PRNet& net1, PRNet& net2)
{
  double net1_routing_area = net1.get_bounding_box().getTotalSize();
  double net2_routing_area = net2.get_bounding_box().getTotalSize();

  if (net1_routing_area < net2_routing_area) {
    return SortStatus::kTrue;
  } else if (net1_routing_area == net2_routing_area) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

// 长宽比 降序
SortStatus PlanarRouter::sortByLengthWidthRatioDESC(PRNet& net1, PRNet& net2)
{
  BoundingBox& net1_bounding_box = net1.get_bounding_box();
  BoundingBox& net2_bounding_box = net2.get_bounding_box();

  double net1_length_width_ratio = net1_bounding_box.getXSize() / 1.0 / net1_bounding_box.getYSize();
  if (net1_length_width_ratio < 1) {
    net1_length_width_ratio = 1 / net1_length_width_ratio;
  }
  double net2_length_width_ratio = net2_bounding_box.getXSize() / 1.0 / net2_bounding_box.getYSize();
  if (net2_length_width_ratio < 1) {
    net2_length_width_ratio = 1 / net2_length_width_ratio;
  }
  if (net1_length_width_ratio > net2_length_width_ratio) {
    return SortStatus::kTrue;
  } else if (net1_length_width_ratio == net2_length_width_ratio) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

// PinNum 降序
SortStatus PlanarRouter::sortByPinNumDESC(PRNet& net1, PRNet& net2)
{
  int32_t net1_pin_num = static_cast<int32_t>(net1.get_pr_pin_list().size());
  int32_t net2_pin_num = static_cast<int32_t>(net2.get_pr_pin_list().size());

  if (net1_pin_num > net2_pin_num) {
    return SortStatus::kTrue;
  } else if (net1_pin_num == net2_pin_num) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

void PlanarRouter::routePRModel(PRModel& pr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<PRNet>& pr_net_list = pr_model.get_pr_net_list();
  std::vector<int32_t>& pr_net_idx_list = pr_model.get_pr_net_idx_list();

  int32_t batch_size = RTUtil::getBatchSize(pr_net_idx_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < pr_net_idx_list.size(); i++) {
    routePRNet(pr_model, pr_net_list[pr_net_idx_list[i]]);
    if ((i + 1) % batch_size == 0 || (i + 1) == pr_net_idx_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", pr_net_idx_list.size(), "(",
                    RTUtil::getPercentage(i + 1, pr_net_idx_list.size()), ") nets", stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PlanarRouter::routePRNet(PRModel& pr_model, PRNet& pr_net)
{
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (Segment<PlanarCoord>& planar_topo : getPlanarTopoListByFlute(pr_net)) {
    for (Segment<PlanarCoord>& routing_segment : getRoutingSegmentList(pr_model, planar_topo)) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<PlanarCoord> coord_tree = getCoordTree(pr_net, routing_segment_list);
  updateDemand(pr_model, pr_net, coord_tree);
  std::function<Guide(PlanarCoord&)> convertToGuide = [](PlanarCoord& coord) {
    ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
    return Guide(LayerRect(RTUtil::getRealRectByGCell(coord, gcell_axis)), coord);
  };
  pr_net.set_pr_result_tree(RTUtil::convertTree(coord_tree, convertToGuide));
}

std::vector<Segment<PlanarCoord>> PlanarRouter::getPlanarTopoListByFlute(PRNet& pr_net)
{
  std::vector<PlanarCoord> planar_coord_list;
  {
    for (PRPin& pr_pin : pr_net.get_pr_pin_list()) {
      planar_coord_list.push_back(pr_pin.get_key_access_point().get_grid_coord());
    }
    std::sort(planar_coord_list.begin(), planar_coord_list.end(), CmpPlanarCoordByXASC());
    planar_coord_list.erase(std::unique(planar_coord_list.begin(), planar_coord_list.end()), planar_coord_list.end());
  }
  std::vector<Segment<PlanarCoord>> planar_topo_list;
  if (planar_coord_list.size() > 1) {
    size_t point_num = planar_coord_list.size();
    Flute::DTYPE* x_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (point_num));
    Flute::DTYPE* y_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (point_num));
    for (size_t i = 0; i < point_num; i++) {
      x_list[i] = planar_coord_list[i].get_x();
      y_list[i] = planar_coord_list[i].get_y();
    }
    Flute::Tree flute_tree = Flute::flute(point_num, x_list, y_list, FLUTE_ACCURACY);
    free(x_list);
    free(y_list);

    for (int i = 0; i < 2 * flute_tree.deg - 2; i++) {
      int n_id = flute_tree.branch[i].n;
      PlanarCoord first_coord(flute_tree.branch[i].x, flute_tree.branch[i].y);
      PlanarCoord second_coord(flute_tree.branch[n_id].x, flute_tree.branch[n_id].y);
      if (first_coord != second_coord) {
        planar_topo_list.emplace_back(first_coord, second_coord);
      }
    }
    Flute::free_tree(flute_tree);
  }
  return planar_topo_list;
}

std::vector<Segment<PlanarCoord>> PlanarRouter::getRoutingSegmentList(PRModel& pr_model, Segment<PlanarCoord>& planar_topo)
{
  int32_t candidate_num = 0;
  {
    int32_t x_span = std::max(0, std::abs(planar_topo.get_first().get_x() - planar_topo.get_second().get_x()) - 1);
    int32_t y_span = std::max(0, std::abs(planar_topo.get_first().get_y() - planar_topo.get_second().get_y()) - 1);
    // Straight + LPattern + ZPattern + Inner3Bends
    candidate_num = (1) + (2) + (x_span + y_span) + (2 * x_span * y_span);
  }
  std::vector<std::pair<double, std::vector<Segment<PlanarCoord>>>> candidate_pair_list;
  candidate_pair_list.reserve(candidate_num);
  for (auto getInflectionListList : {std::bind(&PlanarRouter::getInflectionListListByStraight, this, std::placeholders::_1),
                                     std::bind(&PlanarRouter::getInflectionListListByLPattern, this, std::placeholders::_1),
                                     std::bind(&PlanarRouter::getInflectionListListByZPattern, this, std::placeholders::_1),
                                     std::bind(&PlanarRouter::getInflectionListListByInner3Bends, this, std::placeholders::_1)}) {
    for (std::vector<PlanarCoord> inflection_list : getInflectionListList(planar_topo)) {
      std::vector<Segment<PlanarCoord>> routing_segment_list;
      if (inflection_list.empty()) {
        routing_segment_list.emplace_back(planar_topo.get_first(), planar_topo.get_second());
      } else {
        routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
        for (size_t i = 1; i < inflection_list.size(); i++) {
          routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
        }
        routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
      }
      candidate_pair_list.emplace_back(0, routing_segment_list);
    }
  }
#pragma omp parallel for
  for (auto& [cost, routing_segment_list] : candidate_pair_list) {
    cost = getNodeCost(pr_model, routing_segment_list);
  }
  double min_cost = DBL_MAX;
  std::vector<Segment<PlanarCoord>> best_routing_segment_list;
  for (auto& [cost, routing_segment_list] : candidate_pair_list) {
    if (cost < min_cost) {
      min_cost = cost;
      best_routing_segment_list = routing_segment_list;
    }
  }
  return best_routing_segment_list;
}

std::vector<std::vector<PlanarCoord>> PlanarRouter::getInflectionListListByStraight(Segment<PlanarCoord>& planar_topo)
{
  std::vector<std::vector<PlanarCoord>> inflection_list_list;

  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUtil::isOblique(first_coord, second_coord)) {
    return inflection_list_list;
  }
  inflection_list_list.emplace_back();
  return inflection_list_list;
}

std::vector<std::vector<PlanarCoord>> PlanarRouter::getInflectionListListByLPattern(Segment<PlanarCoord>& planar_topo)
{
  std::vector<std::vector<PlanarCoord>> inflection_list_list;

  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUtil::isRightAngled(first_coord, second_coord)) {
    return inflection_list_list;
  }
  PlanarCoord Inflection_coord1(first_coord.get_x(), second_coord.get_y());
  inflection_list_list.push_back({Inflection_coord1});
  PlanarCoord Inflection_coord2(second_coord.get_x(), first_coord.get_y());
  inflection_list_list.push_back({Inflection_coord2});
  return inflection_list_list;
}

std::vector<std::vector<PlanarCoord>> PlanarRouter::getInflectionListListByZPattern(Segment<PlanarCoord>& planar_topo)
{
  std::vector<std::vector<PlanarCoord>> inflection_list_list;

  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUtil::isRightAngled(first_coord, second_coord)) {
    return inflection_list_list;
  }
  std::vector<int32_t> x_mid_index_list = getMidIndexList(first_coord.get_x(), second_coord.get_x());
  std::vector<int32_t> y_mid_index_list = getMidIndexList(first_coord.get_y(), second_coord.get_y());
  if (x_mid_index_list.empty() && y_mid_index_list.empty()) {
    return inflection_list_list;
  }
  for (size_t i = 0; i < x_mid_index_list.size(); i++) {
    PlanarCoord Inflection_coord1(x_mid_index_list[i], first_coord.get_y());
    PlanarCoord Inflection_coord2(x_mid_index_list[i], second_coord.get_y());
    inflection_list_list.push_back({Inflection_coord1, Inflection_coord2});
  }
  for (size_t i = 0; i < y_mid_index_list.size(); i++) {
    PlanarCoord Inflection_coord1(first_coord.get_x(), y_mid_index_list[i]);
    PlanarCoord Inflection_coord2(second_coord.get_x(), y_mid_index_list[i]);
    inflection_list_list.push_back({Inflection_coord1, Inflection_coord2});
  }
  return inflection_list_list;
}

std::vector<int32_t> PlanarRouter::getMidIndexList(int32_t start_idx, int32_t end_idx)
{
  std::vector<int32_t> index_list;
  RTUtil::swapByASC(start_idx, end_idx);
  index_list.reserve(end_idx - start_idx - 1);
  for (int32_t i = (start_idx + 1); i <= (end_idx - 1); i++) {
    index_list.push_back(i);
  }
  return index_list;
}

std::vector<std::vector<PlanarCoord>> PlanarRouter::getInflectionListListByInner3Bends(Segment<PlanarCoord>& planar_topo)
{
  std::vector<std::vector<PlanarCoord>> inflection_list_list;

  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUtil::isRightAngled(first_coord, second_coord)) {
    return inflection_list_list;
  }
  std::vector<int32_t> x_mid_index_list = getMidIndexList(first_coord.get_x(), second_coord.get_x());
  std::vector<int32_t> y_mid_index_list = getMidIndexList(first_coord.get_y(), second_coord.get_y());
  if (x_mid_index_list.empty() || y_mid_index_list.empty()) {
    return inflection_list_list;
  }
  for (size_t i = 0; i < x_mid_index_list.size(); i++) {
    for (size_t j = 0; j < y_mid_index_list.size(); j++) {
      PlanarCoord Inflection_coord1(x_mid_index_list[i], first_coord.get_y());
      PlanarCoord Inflection_coord2(x_mid_index_list[i], y_mid_index_list[j]);
      PlanarCoord Inflection_coord3(second_coord.get_x(), y_mid_index_list[j]);
      inflection_list_list.push_back({Inflection_coord1, Inflection_coord2, Inflection_coord3});
    }
  }
  for (size_t i = 0; i < x_mid_index_list.size(); i++) {
    for (size_t j = 0; j < y_mid_index_list.size(); j++) {
      PlanarCoord Inflection_coord1(first_coord.get_x(), y_mid_index_list[j]);
      PlanarCoord Inflection_coord2(x_mid_index_list[i], y_mid_index_list[j]);
      PlanarCoord Inflection_coord3(x_mid_index_list[i], second_coord.get_y());
      inflection_list_list.push_back({Inflection_coord1, Inflection_coord2, Inflection_coord3});
    }
  }
  return inflection_list_list;
}

double PlanarRouter::getNodeCost(PRModel& pr_model, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  double congestion_unit = pr_model.get_pr_parameter().get_congestion_unit();

  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> usage_map;
  for (Segment<PlanarCoord>& coord_segment : routing_segment_list) {
    PlanarCoord& first_coord = coord_segment.get_first();
    PlanarCoord& second_coord = coord_segment.get_second();

    Orientation orientation = RTUtil::getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUtil::getOppositeOrientation(orientation);

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();
    RTUtil::swapByASC(first_x, second_x);
    RTUtil::swapByASC(first_y, second_y);

    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        PlanarCoord coord(x, y);
        if (coord != first_coord) {
          usage_map[coord].insert(opposite_orientation);
        }
        if (coord != second_coord) {
          usage_map[coord].insert(orientation);
        }
      }
    }
  }

  double node_cost = 0;
  GridMap<PRNode>& pr_node_map = pr_model.get_pr_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    PRNode& pr_node = pr_node_map[usage_coord.get_x()][usage_coord.get_y()];
    for (const Orientation& orientation : orientation_list) {
      node_cost += (pr_node.getCongestionCost(orientation) * congestion_unit);
    }
  }
  return node_cost;
}

MTree<PlanarCoord> PlanarRouter::getCoordTree(PRNet& pr_net, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  std::vector<PlanarCoord> candidate_root_coord_list;
  std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC> key_coord_pin_map;
  std::vector<PRPin>& pr_pin_list = pr_net.get_pr_pin_list();
  for (size_t i = 0; i < pr_pin_list.size(); i++) {
    PlanarCoord coord = pr_pin_list[i].get_key_access_point().get_grid_coord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUtil::getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void PlanarRouter::updateDemand(PRModel& pr_model, PRNet& pr_net, MTree<PlanarCoord>& coord_tree)
{
  std::set<PlanarCoord, CmpPlanarCoordByXASC> key_coord_set;
  for (PRPin& pr_pin : pr_net.get_pr_pin_list()) {
    key_coord_set.insert(pr_pin.get_key_access_point().get_grid_coord());
  }
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (Segment<TNode<PlanarCoord>*>& coord_segment : RTUtil::getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> usage_map;
  for (Segment<PlanarCoord>& coord_segment : routing_segment_list) {
    PlanarCoord& first_coord = coord_segment.get_first();
    PlanarCoord& second_coord = coord_segment.get_second();

    Orientation orientation = RTUtil::getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUtil::getOppositeOrientation(orientation);

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();
    RTUtil::swapByASC(first_x, second_x);
    RTUtil::swapByASC(first_y, second_y);

    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        PlanarCoord coord(x, y);
        if (coord != first_coord) {
          usage_map[coord].insert(opposite_orientation);
        }
        if (coord != second_coord) {
          usage_map[coord].insert(orientation);
        }
      }
    }
  }
  GridMap<PRNode>& pr_node_map = pr_model.get_pr_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    PRNode& pr_node = pr_node_map[usage_coord.get_x()][usage_coord.get_y()];
    pr_node.updateDemand(orientation_list, ChangeType::kAdd);
  }
}

void PlanarRouter::updatePRModel(PRModel& pr_model)
{
  for (PRNet& pr_net : pr_model.get_pr_net_list()) {
    Net* origin_net = pr_net.get_origin_net();
    origin_net->set_pr_result_tree(pr_net.get_pr_result_tree());
  }
}

}  // namespace irt
