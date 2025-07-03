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
#include "SpaceRouter.hpp"

#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void SpaceRouter::initInst()
{
  if (_sr_instance == nullptr) {
    _sr_instance = new SpaceRouter();
  }
}

SpaceRouter& SpaceRouter::getInst()
{
  if (_sr_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_sr_instance;
}

void SpaceRouter::destroyInst()
{
  if (_sr_instance != nullptr) {
    delete _sr_instance;
    _sr_instance = nullptr;
  }
}

// function

void SpaceRouter::route()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  SRModel sr_model = initSRModel();
  buildLayerNodeMap(sr_model);
  buildOrientSupply(sr_model);
  reviseNodeDemand(sr_model);
  routeSRModel(sr_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

SpaceRouter* SpaceRouter::_sr_instance = nullptr;

SRModel SpaceRouter::initSRModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  SRModel sr_model;
  sr_model.set_sr_net_list(convertToSRNetList(net_list));
  return sr_model;
}

std::vector<SRNet> SpaceRouter::convertToSRNetList(std::vector<Net>& net_list)
{
  std::vector<SRNet> sr_net_list;
  sr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    sr_net_list.emplace_back(convertToSRNet(net));
  }
  return sr_net_list;
}

SRNet SpaceRouter::convertToSRNet(Net& net)
{
  SRNet sr_net;
  sr_net.set_origin_net(&net);
  sr_net.set_net_idx(net.get_net_idx());
  sr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    sr_net.get_sr_pin_list().push_back(SRPin(pin));
  }
  return sr_net;
}

void SpaceRouter::buildLayerNodeMap(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
#pragma omp parallel for
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    sr_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
        SRNode& sr_node = sr_node_map[x][y];
        sr_node.set_coord(x, y);
        sr_node.set_layer_idx(layer_idx);
        sr_node.set_boundary_wire_unit(gcell_map[x][y].get_boundary_wire_unit());
        sr_node.set_internal_wire_unit(gcell_map[x][y].get_internal_wire_unit());
        sr_node.set_internal_via_unit(gcell_map[x][y].get_internal_via_unit());
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void SpaceRouter::buildOrientSupply(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
        layer_node_map[layer_idx][x][y].set_orient_supply_map(gcell_map[x][y].get_routing_orient_supply_map()[layer_idx]);
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void SpaceRouter::reviseNodeDemand(SRModel& sr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
        layer_node_map[layer_idx][x][y].get_orient_net_map().clear();
        layer_node_map[layer_idx][x][y].get_net_orient_map().clear();
      }
    }
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    updateDemandToGraph(sr_model, ChangeType::kAdd, net_idx, segment_set);
  }
}

void SpaceRouter::routeSRModel(SRModel& sr_model)
{
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = 2 * non_prefer_wire_unit;
  double overflow_unit = 4 * non_prefer_wire_unit;
  /**
   * prefer_wire_unit, via_unit, size, offset, schedule_interval, overflow_unit, max_routed_times
   */
  std::vector<SRIterParam> sr_iter_param_list;
  // clang-format off
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 0, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 20, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 40, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 60, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 80, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 0, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 20, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 40, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 60, 3, overflow_unit, 3);
  sr_iter_param_list.emplace_back(prefer_wire_unit, via_unit, 100, 80, 3, overflow_unit, 3);
  // clang-format on
  initRoutingState(sr_model);
  for (int32_t i = 0, iter = 1; i < static_cast<int32_t>(sr_iter_param_list.size()); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", sr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, sr_iter_param_list.size()),
               ") *****");
    setSRIterParam(sr_model, iter, sr_iter_param_list[i]);
    initSRBoxMap(sr_model);
    resetRoutingState(sr_model);
    buildBoxSchedule(sr_model);
    splitNetResult(sr_model);
    routeSRBoxMap(sr_model);
    uploadNetResult(sr_model);
    reviseNodeDemand(sr_model);
    updateBestResult(sr_model);
    updateSummary(sr_model);
    printSummary(sr_model);
    outputGuide(sr_model);
    outputNetCSV(sr_model);
    outputOverflowCSV(sr_model);
    outputNetJson(sr_model);
    outputOverflowJson(sr_model);
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", sr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, sr_iter_param_list.size()), ")",
               iter_monitor.getStatsInfo(), "*****");
    if (stopIteration(sr_model)) {
      break;
    }
  }
  selectBestResult(sr_model);
}

void SpaceRouter::initRoutingState(SRModel& sr_model)
{
  sr_model.set_initial_routing(true);
}

void SpaceRouter::setSRIterParam(SRModel& sr_model, int32_t iter, SRIterParam& sr_iter_param)
{
  sr_model.set_iter(iter);
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", sr_iter_param.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", sr_iter_param.get_via_unit());
  RTLOG.info(Loc::current(), "size: ", sr_iter_param.get_size());
  RTLOG.info(Loc::current(), "offset: ", sr_iter_param.get_offset());
  RTLOG.info(Loc::current(), "schedule_interval: ", sr_iter_param.get_schedule_interval());
  RTLOG.info(Loc::current(), "overflow_unit: ", sr_iter_param.get_overflow_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", sr_iter_param.get_max_routed_times());
  sr_model.set_sr_iter_param(sr_iter_param);
}

void SpaceRouter::initSRBoxMap(SRModel& sr_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  int32_t x_gcell_num = 0;
  for (ScaleGrid& x_grid : gcell_axis.get_x_grid_list()) {
    x_gcell_num += x_grid.get_step_num();
  }
  int32_t y_gcell_num = 0;
  for (ScaleGrid& y_grid : gcell_axis.get_y_grid_list()) {
    y_gcell_num += y_grid.get_step_num();
  }

  SRIterParam& sr_iter_param = sr_model.get_sr_iter_param();
  int32_t size = sr_iter_param.get_size();
  int32_t offset = sr_iter_param.get_offset();
  int32_t x_box_num = static_cast<int32_t>(std::ceil((x_gcell_num - offset) / 1.0 / size));
  int32_t y_box_num = static_cast<int32_t>(std::ceil((y_gcell_num - offset) / 1.0 / size));

  GridMap<SRBox>& sr_box_map = sr_model.get_sr_box_map();
  sr_box_map.init(x_box_num, y_box_num);

  for (int32_t x = 0; x < sr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < sr_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = std::max(offset + x * size, 0);
      int32_t grid_ll_y = std::max(offset + y * size, 0);
      int32_t grid_ur_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_ur_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect ll_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ll_x, grid_ll_y), gcell_axis);
      PlanarRect ur_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ur_x, grid_ur_y), gcell_axis);
      PlanarRect box_real_rect(ll_gcell_rect.get_ll(), ur_gcell_rect.get_ur());

      SRBox& sr_box = sr_box_map[x][y];

      EXTPlanarRect sr_box_rect;
      sr_box_rect.set_real_rect(box_real_rect);
      sr_box_rect.set_grid_rect(RTUTIL.getOpenGCellGridRect(box_real_rect, gcell_axis));
      sr_box.set_box_rect(sr_box_rect);
      SRBoxId sr_box_id;
      sr_box_id.set_x(x);
      sr_box_id.set_y(y);
      sr_box.set_sr_box_id(sr_box_id);
      sr_box.set_sr_iter_param(&sr_iter_param);
      sr_box.set_initial_routing(sr_model.get_initial_routing());
    }
  }
}

void SpaceRouter::resetRoutingState(SRModel& sr_model)
{
  sr_model.set_initial_routing(false);
}

void SpaceRouter::buildBoxSchedule(SRModel& sr_model)
{
  GridMap<SRBox>& sr_box_map = sr_model.get_sr_box_map();
  int32_t schedule_interval = sr_model.get_sr_iter_param().get_schedule_interval();

  std::vector<std::vector<SRBoxId>> sr_box_id_list_list;
  for (int32_t start_x = 0; start_x < schedule_interval; start_x++) {
    for (int32_t start_y = 0; start_y < schedule_interval; start_y++) {
      std::vector<SRBoxId> sr_box_id_list;
      for (int32_t x = start_x; x < sr_box_map.get_x_size(); x += schedule_interval) {
        for (int32_t y = start_y; y < sr_box_map.get_y_size(); y += schedule_interval) {
          sr_box_id_list.emplace_back(x, y);
        }
      }
      sr_box_id_list_list.push_back(sr_box_id_list);
    }
  }
  sr_model.set_sr_box_id_list_list(sr_box_id_list_list);
}

void SpaceRouter::splitNetResult(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::vector<ScaleGrid> box_x_grid_list;
  std::vector<ScaleGrid> box_y_grid_list;
  {
    GridMap<SRBox>& sr_box_map = sr_model.get_sr_box_map();
    std::vector<int32_t> box_x_scale_list;
    std::vector<int32_t> box_y_scale_list;
    for (int32_t x = 0; x < sr_box_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_box_map.get_y_size(); y++) {
        SRBox& sr_box = sr_box_map[x][y];
        box_x_scale_list.push_back(sr_box.get_box_rect().get_grid_ll_x());
        box_x_scale_list.push_back(sr_box.get_box_rect().get_grid_ur_x());
        box_y_scale_list.push_back(sr_box.get_box_rect().get_grid_ll_y());
        box_y_scale_list.push_back(sr_box.get_box_rect().get_grid_ur_y());
      }
    }
    std::sort(box_x_scale_list.begin(), box_x_scale_list.end());
    box_x_scale_list.erase(std::unique(box_x_scale_list.begin(), box_x_scale_list.end()), box_x_scale_list.end());
    box_x_grid_list = RTUTIL.makeScaleGridList(box_x_scale_list);
    std::sort(box_y_scale_list.begin(), box_y_scale_list.end());
    box_y_scale_list.erase(std::unique(box_y_scale_list.begin(), box_y_scale_list.end()), box_y_scale_list.end());
    box_y_grid_list = RTUTIL.makeScaleGridList(box_y_scale_list);
  }

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    std::vector<Segment<LayerCoord>*> del_segment_list;
    std::vector<Segment<LayerCoord>> new_segment_list;
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord& first_coord = segment->get_first();
      LayerCoord& second_coord = segment->get_second();
      if (first_coord.get_layer_idx() != second_coord.get_layer_idx()) {
        continue;
      }
      if (RTUTIL.isHorizontal(first_coord, second_coord)) {
        int32_t first_x = first_coord.get_x();
        int32_t second_x = second_coord.get_x();
        RTUTIL.swapByASC(first_x, second_x);
        std::set<int32_t> x_pre_set;
        std::set<int32_t> x_mid_set;
        std::set<int32_t> x_post_set;
        RTUTIL.getTrackScaleSet(box_x_grid_list, first_x, second_x, x_pre_set, x_mid_set, x_post_set);
        x_mid_set.erase(first_x);
        x_mid_set.erase(second_x);
        if (x_mid_set.empty()) {
          continue;
        }
        std::vector<int32_t> x_scale_list;
        x_scale_list.push_back(first_x);
        for (int32_t x_scale : x_mid_set) {
          x_scale_list.push_back(x_scale);
        }
        x_scale_list.push_back(second_x);
        del_segment_list.push_back(segment);
        for (size_t i = 1; i < x_scale_list.size(); i++) {
          new_segment_list.emplace_back(LayerCoord(x_scale_list[i - 1], first_coord.get_y(), first_coord.get_layer_idx()),
                                        LayerCoord(x_scale_list[i], first_coord.get_y(), first_coord.get_layer_idx()));
        }
      } else if (RTUTIL.isVertical(first_coord, second_coord)) {
        int32_t first_y = first_coord.get_y();
        int32_t second_y = second_coord.get_y();
        RTUTIL.swapByASC(first_y, second_y);
        std::set<int32_t> y_pre_set;
        std::set<int32_t> y_mid_set;
        std::set<int32_t> y_post_set;
        RTUTIL.getTrackScaleSet(box_y_grid_list, first_y, second_y, y_pre_set, y_mid_set, y_post_set);
        y_mid_set.erase(first_y);
        y_mid_set.erase(second_y);
        if (y_mid_set.empty()) {
          continue;
        }
        std::vector<int32_t> y_scale_list;
        y_scale_list.push_back(first_y);
        for (int32_t y_scale : y_mid_set) {
          y_scale_list.push_back(y_scale);
        }
        y_scale_list.push_back(second_y);
        del_segment_list.push_back(segment);
        for (size_t i = 1; i < y_scale_list.size(); i++) {
          new_segment_list.emplace_back(LayerCoord(first_coord.get_x(), y_scale_list[i - 1], first_coord.get_layer_idx()),
                                        LayerCoord(first_coord.get_x(), y_scale_list[i], first_coord.get_layer_idx()));
        }
      }
    }
    for (Segment<LayerCoord>* del_segment : del_segment_list) {
      RTDM.updateNetGlobalResultToGCellMap(ChangeType::kDel, net_idx, del_segment);
    }
    for (Segment<LayerCoord>& new_segment : new_segment_list) {
      RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(new_segment));
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void SpaceRouter::routeSRBoxMap(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<SRBox>& sr_box_map = sr_model.get_sr_box_map();

  size_t total_box_num = 0;
  for (std::vector<SRBoxId>& sr_box_id_list : sr_model.get_sr_box_id_list_list()) {
    total_box_num += sr_box_id_list.size();
  }

  size_t routed_box_num = 0;
  for (std::vector<SRBoxId>& sr_box_id_list : sr_model.get_sr_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (SRBoxId& sr_box_id : sr_box_id_list) {
      SRBox& sr_box = sr_box_map[sr_box_id.get_x()][sr_box_id.get_y()];
      buildNetResult(sr_box);
      initSRTaskList(sr_model, sr_box);
      buildOverflow(sr_model, sr_box);
      if (needRouting(sr_model, sr_box)) {
        buildBoxTrackAxis(sr_box);
        buildLayerNodeMap(sr_model, sr_box);
        buildSRNodeNeighbor(sr_box);
        buildOrientSupply(sr_model, sr_box);
        buildOrientDemand(sr_model, sr_box);
        // debugCheckSRBox(sr_box);
        routeSRBox(sr_box);
      }
      selectBestResult(sr_box);
      freeSRBox(sr_box);
    }
    routed_box_num += sr_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num), ") boxes",
               stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void SpaceRouter::buildNetResult(SRBox& sr_box)
{
  PlanarRect& box_grid_rect = sr_box.get_box_rect().get_grid_rect();

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(sr_box.get_box_rect())) {
    for (Segment<LayerCoord>* segment : segment_set) {
      bool least_one_coord_in_box = false;
      if (RTUTIL.isInside(box_grid_rect, segment->get_first()) && RTUTIL.isInside(box_grid_rect, segment->get_second())) {
        if (RTUTIL.isInside(box_grid_rect, segment->get_first(), false) || RTUTIL.isInside(box_grid_rect, segment->get_second(), false)) {
          // 线段在box_grid_rect内,但不贴边的
          least_one_coord_in_box = true;
        }
      }
      if (least_one_coord_in_box) {
        sr_box.get_net_task_global_result_map()[net_idx].push_back(*segment);
        RTDM.updateNetGlobalResultToGCellMap(ChangeType::kDel, net_idx, segment);
      }
    }
  }
}

void SpaceRouter::initSRTaskList(SRModel& sr_model, SRBox& sr_box)
{
  std::vector<SRNet>& sr_net_list = sr_model.get_sr_net_list();
  std::vector<SRTask*>& sr_task_list = sr_box.get_sr_task_list();

  EXTPlanarRect& box_rect = sr_box.get_box_rect();
  PlanarRect& box_grid_rect = box_rect.get_grid_rect();
  std::map<int32_t, std::set<AccessPoint*>> net_access_point_map = RTDM.getNetAccessPointMap(box_rect);
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_task_global_result_map = sr_box.get_net_task_global_result_map();

  std::map<int32_t, std::vector<SRGroup>> net_group_list_map;
  {
    for (auto& [net_idx, access_point_set] : net_access_point_map) {
      std::map<int32_t, SRGroup> pin_group_map;
      for (AccessPoint* access_point : access_point_set) {
        if (!RTUTIL.isInside(box_grid_rect, access_point->get_grid_coord())) {
          continue;
        }
        pin_group_map[access_point->get_pin_idx()].get_coord_list().push_back(access_point->getGridLayerCoord());
      }
      for (auto& [pin_idx, group] : pin_group_map) {
        net_group_list_map[net_idx].push_back(group);
      }
    }
    for (auto& [net_idx, segment_list] : net_task_global_result_map) {
      std::vector<LayerCoord> coord_list;
      for (const Segment<LayerCoord>& segment : segment_list) {
        const LayerCoord& first = segment.get_first();
        const LayerCoord& second = segment.get_second();
        if (first.get_layer_idx() != second.get_layer_idx()) {
          continue;
        }
        if (RTUTIL.isHorizontal(first, second)) {
          int32_t first_x = first.get_x();
          int32_t second_x = second.get_x();
          if (first.get_y() < box_grid_rect.get_ll_y() || box_grid_rect.get_ur_y() < first.get_y()) {
            continue;
          }
          RTUTIL.swapByASC(first_x, second_x);
          if (first_x <= box_grid_rect.get_ll_x() && box_grid_rect.get_ll_x() <= second_x) {
            coord_list.emplace_back(box_grid_rect.get_ll_x(), first.get_y(), first.get_layer_idx());
          }
          if (first_x <= box_grid_rect.get_ur_x() && box_grid_rect.get_ur_x() <= second_x) {
            coord_list.emplace_back(box_grid_rect.get_ur_x(), first.get_y(), first.get_layer_idx());
          }
        } else if (RTUTIL.isVertical(first, second)) {
          int32_t first_y = first.get_y();
          int32_t second_y = second.get_y();
          if (first.get_x() < box_grid_rect.get_ll_x() || box_grid_rect.get_ur_x() < first.get_x()) {
            continue;
          }
          RTUTIL.swapByASC(first_y, second_y);
          if (first_y <= box_grid_rect.get_ll_y() && box_grid_rect.get_ll_y() <= second_y) {
            coord_list.emplace_back(first.get_x(), box_grid_rect.get_ll_y(), first.get_layer_idx());
          }
          if (first_y <= box_grid_rect.get_ur_y() && box_grid_rect.get_ur_y() <= second_y) {
            coord_list.emplace_back(first.get_x(), box_grid_rect.get_ur_y(), first.get_layer_idx());
          }
        } else {
          RTLOG.error(Loc::current(), "The segment is oblique!");
        }
      }
      for (LayerCoord& coord : coord_list) {
        SRGroup sr_group;
        sr_group.get_coord_list().push_back(coord);
        net_group_list_map[net_idx].push_back(sr_group);
      }
    }
  }
  for (auto& [net_idx, sr_group_list] : net_group_list_map) {
    if (sr_group_list.size() < 2) {
      continue;
    }
    SRTask* sr_task = new SRTask();
    sr_task->set_net_idx(net_idx);
    sr_task->set_connect_type(sr_net_list[net_idx].get_connect_type());
    sr_task->set_sr_group_list(sr_group_list);
    {
      std::vector<PlanarCoord> coord_list;
      for (SRGroup& sr_group : sr_task->get_sr_group_list()) {
        for (LayerCoord& coord : sr_group.get_coord_list()) {
          coord_list.push_back(coord);
        }
      }
      sr_task->set_bounding_box(RTUTIL.getBoundingBox(coord_list));
    }
    sr_task->set_routed_times(0);
    sr_task_list.push_back(sr_task);
  }
  std::sort(sr_task_list.begin(), sr_task_list.end(), CmpSRTask());
}

void SpaceRouter::buildOverflow(SRModel& sr_model, SRBox& sr_box)
{
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();

  EXTPlanarRect& box_rect = sr_box.get_box_rect();

  double total_overflow = 0;
  std::vector<std::set<int32_t>> overflow_net_set_list;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = box_rect.get_grid_ll_x(); x <= box_rect.get_grid_ur_x(); x++) {
      for (int32_t y = box_rect.get_grid_ll_y(); y <= box_rect.get_grid_ur_y(); y++) {
        double node_overflow = sr_node_map[x][y].getOverflow();
        total_overflow += node_overflow;
        if (node_overflow > 0) {
          std::set<int32_t> overflow_net_set;
          for (auto& [orient, net_set] : sr_node_map[x][y].get_orient_net_map()) {
            overflow_net_set.insert(net_set.begin(), net_set.end());
          }
          for (auto& [net_idx, orient_set] : sr_node_map[x][y].get_net_orient_map()) {
            overflow_net_set.insert(net_idx);
          }
          overflow_net_set_list.push_back(overflow_net_set);
        }
      }
    }
  }
  sr_box.set_total_overflow(total_overflow);
  sr_box.set_overflow_net_set_list(overflow_net_set_list);
}

bool SpaceRouter::needRouting(SRModel& sr_model, SRBox& sr_box)
{
  if (sr_box.get_sr_task_list().empty()) {
    return false;
  }
  if (sr_box.get_initial_routing() == false && sr_box.get_total_overflow() <= 0) {
    return false;
  }
  return true;
}

void SpaceRouter::buildBoxTrackAxis(SRBox& sr_box)
{
  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_grid_rect = sr_box.get_box_rect().get_grid_rect();
  for (int32_t x_scale = box_grid_rect.get_ll_x(); x_scale <= box_grid_rect.get_ur_x(); x_scale++) {
    x_scale_list.push_back(x_scale);
  }
  for (int32_t y_scale = box_grid_rect.get_ll_y(); y_scale <= box_grid_rect.get_ur_y(); y_scale++) {
    y_scale_list.push_back(y_scale);
  }

  for (SRTask* sr_task : sr_box.get_sr_task_list()) {
    for (SRGroup& sr_group : sr_task->get_sr_group_list()) {
      for (LayerCoord& coord : sr_group.get_coord_list()) {
        x_scale_list.push_back(coord.get_x());
        y_scale_list.push_back(coord.get_y());
      }
    }
  }

  ScaleAxis& box_track_axis = sr_box.get_box_track_axis();
  std::sort(x_scale_list.begin(), x_scale_list.end());
  x_scale_list.erase(std::unique(x_scale_list.begin(), x_scale_list.end()), x_scale_list.end());
  box_track_axis.set_x_grid_list(RTUTIL.makeScaleGridList(x_scale_list));
  std::sort(y_scale_list.begin(), y_scale_list.end());
  y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  box_track_axis.set_y_grid_list(RTUTIL.makeScaleGridList(y_scale_list));
}

void SpaceRouter::buildLayerNodeMap(SRModel& sr_model, SRBox& sr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  PlanarCoord& grid_ll = sr_box.get_box_rect().get_grid_ll();
  PlanarCoord& grid_ur = sr_box.get_box_rect().get_grid_ur();
  ScaleAxis& box_track_axis = sr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUTIL.getScaleList(grid_ll.get_x(), grid_ur.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUTIL.getScaleList(grid_ll.get_y(), grid_ur.get_y(), box_track_axis.get_y_grid_list());

  std::vector<GridMap<SRNode>>& top_layer_node_map = sr_model.get_layer_node_map();
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& top_sr_node_map = top_layer_node_map[layer_idx];
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    sr_node_map.init(x_list.size(), y_list.size());
    for (size_t x = 0; x < x_list.size(); x++) {
      for (size_t y = 0; y < y_list.size(); y++) {
        SRNode& sr_node = sr_node_map[x][y];
        sr_node.set_x(x_list[x]);
        sr_node.set_y(y_list[y]);
        sr_node.set_layer_idx(layer_idx);
        sr_node.set_boundary_wire_unit(top_sr_node_map[sr_node.get_x()][sr_node.get_y()].get_boundary_wire_unit());
        sr_node.set_internal_wire_unit(top_sr_node_map[sr_node.get_x()][sr_node.get_y()].get_internal_wire_unit());
        sr_node.set_internal_via_unit(top_sr_node_map[sr_node.get_x()][sr_node.get_y()].get_internal_via_unit());
      }
    }
  }
}

void SpaceRouter::buildSRNodeNeighbor(SRBox& sr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        std::map<Orientation, SRNode*>& neighbor_node_map = sr_node_map[x][y].get_neighbor_node_map();
        if (routing_h) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &sr_node_map[x - 1][y];
          }
          if (x != (sr_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &sr_node_map[x + 1][y];
          }
        }
        if (routing_v) {
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &sr_node_map[x][y - 1];
          }
          if (y != (sr_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &sr_node_map[x][y + 1];
          }
        }
        if (layer_idx != 0) {
          neighbor_node_map[Orientation::kBelow] = &layer_node_map[layer_idx - 1][x][y];
        }
        if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
          neighbor_node_map[Orientation::kAbove] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
    }
  }
}

void SpaceRouter::buildOrientSupply(SRModel& sr_model, SRBox& sr_box)
{
  std::vector<GridMap<SRNode>>& top_layer_node_map = sr_model.get_layer_node_map();
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();

  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& top_sr_node_map = top_layer_node_map[layer_idx];
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        SRNode& sr_node = sr_node_map[x][y];
        sr_node.set_orient_supply_map(top_sr_node_map[sr_node.get_x()][sr_node.get_y()].get_orient_supply_map());
      }
    }
  }
}

void SpaceRouter::buildOrientDemand(SRModel& sr_model, SRBox& sr_box)
{
  std::vector<GridMap<SRNode>>& top_layer_node_map = sr_model.get_layer_node_map();
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();

  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& top_sr_node_map = top_layer_node_map[layer_idx];
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        SRNode& sr_node = sr_node_map[x][y];
        sr_node.set_orient_net_map(top_sr_node_map[sr_node.get_x()][sr_node.get_y()].get_orient_net_map());
        sr_node.set_net_orient_map(top_sr_node_map[sr_node.get_x()][sr_node.get_y()].get_net_orient_map());
      }
    }
  }
}

void SpaceRouter::routeSRBox(SRBox& sr_box)
{
  std::vector<SRTask*> routing_task_list = initTaskSchedule(sr_box);
  while (!routing_task_list.empty()) {
    for (SRTask* routing_task : routing_task_list) {
      routeSRTask(sr_box, routing_task);
      routing_task->addRoutedTimes();
    }
    updateOverflow(sr_box);
    updateBestResult(sr_box);
    updateTaskSchedule(sr_box, routing_task_list);
  }
}

std::vector<SRTask*> SpaceRouter::initTaskSchedule(SRBox& sr_box)
{
  bool initial_routing = sr_box.get_initial_routing();

  std::vector<SRTask*> routing_task_list;
  if (initial_routing) {
    for (SRTask* sr_task : sr_box.get_sr_task_list()) {
      routing_task_list.push_back(sr_task);
    }
  } else {
    updateTaskSchedule(sr_box, routing_task_list);
  }
  return routing_task_list;
}

void SpaceRouter::routeSRTask(SRBox& sr_box, SRTask* sr_task)
{
  initSingleTask(sr_box, sr_task);
  while (!isConnectedAllEnd(sr_box)) {
    routeSinglePath(sr_box);
    updatePathResult(sr_box);
    resetStartAndEnd(sr_box);
    resetSinglePath(sr_box);
  }
  updateTaskResult(sr_box);
  resetSingleTask(sr_box);
}

void SpaceRouter::initSingleTask(SRBox& sr_box, SRTask* sr_task)
{
  ScaleAxis& box_track_axis = sr_box.get_box_track_axis();
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();

  // single task
  sr_box.set_curr_sr_task(sr_task);
  {
    std::vector<std::vector<SRNode*>> node_list_list;
    std::vector<SRGroup>& sr_group_list = sr_task->get_sr_group_list();
    for (SRGroup& sr_group : sr_group_list) {
      std::vector<SRNode*> node_list;
      for (LayerCoord& coord : sr_group.get_coord_list()) {
        if (!RTUTIL.existTrackGrid(coord, box_track_axis)) {
          RTLOG.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, box_track_axis);
        SRNode& sr_node = layer_node_map[coord.get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        node_list.push_back(&sr_node);
      }
      node_list_list.push_back(node_list);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        sr_box.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        sr_box.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  sr_box.get_path_node_list().clear();
  sr_box.get_single_task_visited_node_list().clear();
  sr_box.get_routing_segment_list().clear();
}

bool SpaceRouter::isConnectedAllEnd(SRBox& sr_box)
{
  return sr_box.get_end_node_list_list().empty();
}

void SpaceRouter::routeSinglePath(SRBox& sr_box)
{
  initPathHead(sr_box);
  while (!searchEnded(sr_box)) {
    expandSearching(sr_box);
    resetPathHead(sr_box);
  }
}

void SpaceRouter::initPathHead(SRBox& sr_box)
{
  std::vector<std::vector<SRNode*>>& start_node_list_list = sr_box.get_start_node_list_list();
  std::vector<SRNode*>& path_node_list = sr_box.get_path_node_list();

  for (std::vector<SRNode*>& start_node_list : start_node_list_list) {
    for (SRNode* start_node : start_node_list) {
      start_node->set_estimated_cost(getEstimateCostToEnd(sr_box, start_node));
      pushToOpenList(sr_box, start_node);
    }
  }
  for (SRNode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(sr_box, path_node));
    pushToOpenList(sr_box, path_node);
  }
  resetPathHead(sr_box);
}

bool SpaceRouter::searchEnded(SRBox& sr_box)
{
  std::vector<std::vector<SRNode*>>& end_node_list_list = sr_box.get_end_node_list_list();
  SRNode* path_head_node = sr_box.get_path_head_node();

  if (path_head_node == nullptr) {
    sr_box.set_end_node_list_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (SRNode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        sr_box.set_end_node_list_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void SpaceRouter::expandSearching(SRBox& sr_box)
{
  PriorityQueue<SRNode*, std::vector<SRNode*>, CmpSRNodeCost>& open_queue = sr_box.get_open_queue();
  SRNode* path_head_node = sr_box.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double known_cost = getKnownCost(sr_box, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && known_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了,需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpSRNodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(sr_box, neighbor_node));
      pushToOpenList(sr_box, neighbor_node);
    }
  }
}

void SpaceRouter::resetPathHead(SRBox& sr_box)
{
  sr_box.set_path_head_node(popFromOpenList(sr_box));
}

void SpaceRouter::updatePathResult(SRBox& sr_box)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(sr_box.get_path_head_node())) {
    sr_box.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> SpaceRouter::getRoutingSegmentListByNode(SRNode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  SRNode* curr_node = node;
  SRNode* pre_node = curr_node->get_parent_node();

  if (pre_node == nullptr) {
    // 起点和终点重合
    return routing_segment_list;
  }
  Orientation curr_orientation = RTUTIL.getOrientation(*curr_node, *pre_node);
  while (pre_node->get_parent_node() != nullptr) {
    Orientation pre_orientation = RTUTIL.getOrientation(*pre_node, *pre_node->get_parent_node());
    if (curr_orientation != pre_orientation) {
      routing_segment_list.emplace_back(*curr_node, *pre_node);
      curr_orientation = pre_orientation;
      curr_node = pre_node;
    }
    pre_node = pre_node->get_parent_node();
  }
  routing_segment_list.emplace_back(*curr_node, *pre_node);

  return routing_segment_list;
}

void SpaceRouter::resetStartAndEnd(SRBox& sr_box)
{
  std::vector<std::vector<SRNode*>>& start_node_list_list = sr_box.get_start_node_list_list();
  std::vector<std::vector<SRNode*>>& end_node_list_list = sr_box.get_end_node_list_list();
  std::vector<SRNode*>& path_node_list = sr_box.get_path_node_list();
  SRNode* path_head_node = sr_box.get_path_head_node();
  int32_t end_node_list_idx = sr_box.get_end_node_list_idx();

  // 对于抵达的终点pin,只保留到达的node
  end_node_list_list[end_node_list_idx].clear();
  end_node_list_list[end_node_list_idx].push_back(path_head_node);

  SRNode* path_node = path_head_node->get_parent_node();
  if (path_node == nullptr) {
    // 起点和终点重合
    path_node = path_head_node;
  } else {
    // 起点和终点不重合
    while (path_node->get_parent_node() != nullptr) {
      path_node_list.push_back(path_node);
      path_node = path_node->get_parent_node();
    }
  }
  if (start_node_list_list.size() == 1) {
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void SpaceRouter::resetSinglePath(SRBox& sr_box)
{
  PriorityQueue<SRNode*, std::vector<SRNode*>, CmpSRNodeCost> empty_queue;
  sr_box.set_open_queue(empty_queue);

  std::vector<SRNode*>& single_path_visited_node_list = sr_box.get_single_path_visited_node_list();
  for (SRNode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(SRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  sr_box.set_path_head_node(nullptr);
  sr_box.set_end_node_list_idx(-1);
}

void SpaceRouter::updateTaskResult(SRBox& sr_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = getRoutingSegmentList(sr_box);

  int32_t curr_net_idx = sr_box.get_curr_sr_task()->get_net_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = sr_box.get_net_task_global_result_map()[curr_net_idx];

  // 原结果从graph删除,由于task有对应net_idx,所以不需要在布线前进行删除也不会影响结果
  updateDemandToGraph(sr_box, ChangeType::kDel, curr_net_idx, routing_segment_list);
  routing_segment_list = new_routing_segment_list;
  // 新结果添加到graph
  updateDemandToGraph(sr_box, ChangeType::kAdd, curr_net_idx, routing_segment_list);
}

std::vector<Segment<LayerCoord>> SpaceRouter::getRoutingSegmentList(SRBox& sr_box)
{
  SRTask* curr_sr_task = sr_box.get_curr_sr_task();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<SRGroup>& sr_group_list = curr_sr_task->get_sr_group_list();
  for (size_t i = 0; i < sr_group_list.size(); i++) {
    for (LayerCoord& coord : sr_group_list[i].get_coord_list()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, sr_box.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  return routing_segment_list;
}

void SpaceRouter::resetSingleTask(SRBox& sr_box)
{
  sr_box.set_curr_sr_task(nullptr);
  sr_box.get_start_node_list_list().clear();
  sr_box.get_end_node_list_list().clear();
  sr_box.get_path_node_list().clear();
  sr_box.get_single_task_visited_node_list().clear();
  sr_box.get_routing_segment_list().clear();
}

// manager open list

void SpaceRouter::pushToOpenList(SRBox& sr_box, SRNode* curr_node)
{
  PriorityQueue<SRNode*, std::vector<SRNode*>, CmpSRNodeCost>& open_queue = sr_box.get_open_queue();
  std::vector<SRNode*>& single_task_visited_node_list = sr_box.get_single_task_visited_node_list();
  std::vector<SRNode*>& single_path_visited_node_list = sr_box.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(SRNodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

SRNode* SpaceRouter::popFromOpenList(SRBox& sr_box)
{
  PriorityQueue<SRNode*, std::vector<SRNode*>, CmpSRNodeCost>& open_queue = sr_box.get_open_queue();

  SRNode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(SRNodeState::kClose);
  }
  return node;
}

// calculate known

double SpaceRouter::getKnownCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node)
{
  bool exist_neighbor = false;
  for (auto& [orientation, neighbor_ptr] : start_node->get_neighbor_node_map()) {
    if (neighbor_ptr == end_node) {
      exist_neighbor = true;
      break;
    }
  }
  if (!exist_neighbor) {
    RTLOG.error(Loc::current(), "The neighbor not exist!");
  }

  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getNodeCost(sr_box, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(sr_box, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnownWireCost(sr_box, start_node, end_node);
  cost += getKnownViaCost(sr_box, start_node, end_node);
  return cost;
}

double SpaceRouter::getNodeCost(SRBox& sr_box, SRNode* curr_node, Orientation orientation)
{
  double overflow_unit = sr_box.get_sr_iter_param()->get_overflow_unit();

  double node_cost = 0;
  node_cost += curr_node->getOverflowCost(sr_box.get_curr_sr_task()->get_net_idx(), orientation, overflow_unit);
  return node_cost;
}

double SpaceRouter::getKnownWireCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = sr_box.get_sr_iter_param()->get_prefer_wire_unit();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUTIL.getDirection(*start_node, *end_node)) {
      wire_cost *= prefer_wire_unit;
    }
  }
  return wire_cost;
}

double SpaceRouter::getKnownViaCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node)
{
  double via_unit = sr_box.get_sr_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate

double SpaceRouter::getEstimateCostToEnd(SRBox& sr_box, SRNode* curr_node)
{
  std::vector<std::vector<SRNode*>>& end_node_list_list = sr_box.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<SRNode*>& end_node_list : end_node_list_list) {
    for (SRNode* end_node : end_node_list) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(sr_box, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double SpaceRouter::getEstimateCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(sr_box, start_node, end_node);
  estimate_cost += getEstimateViaCost(sr_box, start_node, end_node);
  return estimate_cost;
}

double SpaceRouter::getEstimateWireCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node)
{
  double prefer_wire_unit = sr_box.get_sr_iter_param()->get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double SpaceRouter::getEstimateViaCost(SRBox& sr_box, SRNode* start_node, SRNode* end_node)
{
  double via_unit = sr_box.get_sr_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void SpaceRouter::updateOverflow(SRBox& sr_box)
{
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();

  double total_overflow = 0;
  std::vector<std::set<int32_t>> overflow_net_set_list;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        double node_overflow = sr_node_map[x][y].getOverflow();
        total_overflow += node_overflow;
        if (node_overflow > 0) {
          std::set<int32_t> overflow_net_set;
          for (auto& [orient, net_set] : sr_node_map[x][y].get_orient_net_map()) {
            overflow_net_set.insert(net_set.begin(), net_set.end());
          }
          for (auto& [net_idx, orient_set] : sr_node_map[x][y].get_net_orient_map()) {
            overflow_net_set.insert(net_idx);
          }
          overflow_net_set_list.push_back(overflow_net_set);
        }
      }
    }
  }
  sr_box.set_total_overflow(total_overflow);
  sr_box.set_overflow_net_set_list(overflow_net_set_list);
}

void SpaceRouter::updateBestResult(SRBox& sr_box)
{
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_global_result_map = sr_box.get_best_net_task_global_result_map();
  double best_total_overflow = sr_box.get_best_total_overflow();

  double curr_total_overflow = sr_box.get_total_overflow();
  if (!best_net_task_global_result_map.empty()) {
    if (best_total_overflow < curr_total_overflow) {
      return;
    }
  }
  best_net_task_global_result_map = sr_box.get_net_task_global_result_map();
  sr_box.set_best_total_overflow(curr_total_overflow);
}

void SpaceRouter::updateTaskSchedule(SRBox& sr_box, std::vector<SRTask*>& routing_task_list)
{
  int32_t max_routed_times = sr_box.get_sr_iter_param()->get_max_routed_times();

  std::set<SRTask*> visited_routing_task_set;
  std::vector<SRTask*> new_routing_task_list;
  for (std::set<int32_t>& overflow_net_set : sr_box.get_overflow_net_set_list()) {
    for (SRTask* sr_task : sr_box.get_sr_task_list()) {
      if (!RTUTIL.exist(overflow_net_set, sr_task->get_net_idx())) {
        continue;
      }
      if (sr_task->get_routed_times() < max_routed_times && !RTUTIL.exist(visited_routing_task_set, sr_task)) {
        visited_routing_task_set.insert(sr_task);
        new_routing_task_list.push_back(sr_task);
      }
      break;
    }
  }
  routing_task_list = new_routing_task_list;

  std::vector<SRTask*> new_sr_task_list;
  for (SRTask* sr_task : sr_box.get_sr_task_list()) {
    if (!RTUTIL.exist(visited_routing_task_set, sr_task)) {
      new_sr_task_list.push_back(sr_task);
    }
  }
  for (SRTask* routing_task : routing_task_list) {
    new_sr_task_list.push_back(routing_task);
  }
  sr_box.set_sr_task_list(new_sr_task_list);
}

void SpaceRouter::selectBestResult(SRBox& sr_box)
{
  updateBestResult(sr_box);
  uploadBestResult(sr_box);
}

void SpaceRouter::uploadBestResult(SRBox& sr_box)
{
  for (auto& [net_idx, segment_list] : sr_box.get_best_net_task_global_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
}

void SpaceRouter::freeSRBox(SRBox& sr_box)
{
  for (SRTask* sr_task : sr_box.get_sr_task_list()) {
    delete sr_task;
    sr_task = nullptr;
  }
  sr_box.get_sr_task_list().clear();
  sr_box.get_layer_node_map().clear();
}

double SpaceRouter::getOverflow(SRModel& sr_model)
{
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();

  double total_overflow = 0;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        total_overflow += sr_node_map[x][y].getOverflow();
      }
    }
  }
  return total_overflow;
}

void SpaceRouter::uploadNetResult(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  std::vector<SRNet>& sr_net_list = sr_model.get_sr_net_list();

  // global result
  {
    std::map<int32_t, std::set<Segment<LayerCoord>*>> net_global_result_map = RTDM.getNetGlobalResultMap(die);
    for (auto& [net_idx, segment_set] : net_global_result_map) {
      std::vector<Segment<LayerCoord>> routing_segment_list;
      for (Segment<LayerCoord>* segment : segment_set) {
        routing_segment_list.emplace_back(segment->get_first(), segment->get_second());
      }
      std::vector<LayerCoord> candidate_root_coord_list;
      std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
      std::vector<SRPin>& sr_pin_list = sr_net_list[net_idx].get_sr_pin_list();
      for (size_t i = 0; i < sr_pin_list.size(); i++) {
        LayerCoord coord = sr_pin_list[i].get_access_point().getGridLayerCoord();
        candidate_root_coord_list.push_back(coord);
        key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
      }
      MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
      for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
        RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, net_idx,
                                             new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value()));
      }
    }
    for (auto& [net_idx, segment_set] : net_global_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        RTDM.updateNetGlobalResultToGCellMap(ChangeType::kDel, net_idx, segment);
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void SpaceRouter::updateBestResult(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_global_result_map = sr_model.get_best_net_task_global_result_map();
  double best_overflow = sr_model.get_best_overflow();

  double curr_overflow = getOverflow(sr_model);
  if (!best_net_task_global_result_map.empty()) {
    if (best_overflow < curr_overflow) {
      return;
    }
  }
  best_net_task_global_result_map.clear();
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      best_net_task_global_result_map[net_idx].push_back(*segment);
    }
  }
  sr_model.set_best_overflow(curr_overflow);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool SpaceRouter::stopIteration(SRModel& sr_model)
{
  if (getOverflow(sr_model) == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

void SpaceRouter::selectBestResult(SRModel& sr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  sr_model.set_iter(sr_model.get_iter() + 1);
  uploadBestResult(sr_model);
  reviseNodeDemand(sr_model);
  updateSummary(sr_model);
  printSummary(sr_model);
  outputGuide(sr_model);
  outputNetCSV(sr_model);
  outputOverflowCSV(sr_model);
  outputNetJson(sr_model);
  outputOverflowJson(sr_model);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void SpaceRouter::uploadBestResult(SRModel& sr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateNetGlobalResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }
  for (auto& [net_idx, segment_list] : sr_model.get_best_net_task_global_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
}

#if 1  // update env

void SpaceRouter::updateDemandToGraph(SRModel& sr_model, ChangeType change_type, int32_t net_idx, std::set<Segment<LayerCoord>*>& segment_set)
{
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> usage_map;
  for (Segment<LayerCoord>* segment : segment_set) {
    LayerCoord& first_coord = segment->get_first();
    LayerCoord& second_coord = segment->get_second();

    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUTIL.getOppositeOrientation(orientation);

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t first_layer_idx = first_coord.get_layer_idx();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();
    int32_t second_layer_idx = second_coord.get_layer_idx();
    RTUTIL.swapByASC(first_x, second_x);
    RTUTIL.swapByASC(first_y, second_y);
    RTUTIL.swapByASC(first_layer_idx, second_layer_idx);

    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          LayerCoord coord(x, y, layer_idx);
          if (coord != first_coord) {
            usage_map[coord].insert(opposite_orientation);
          }
          if (coord != second_coord) {
            usage_map[coord].insert(orientation);
          }
        }
      }
    }
  }
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    SRNode& sr_node = layer_node_map[usage_coord.get_layer_idx()][usage_coord.get_x()][usage_coord.get_y()];
    sr_node.updateDemand(net_idx, orientation_list, change_type);
  }
}

void SpaceRouter::updateDemandToGraph(SRBox& sr_box, ChangeType change_type, int32_t net_idx, std::vector<Segment<LayerCoord>>& segment_list)
{
  int32_t grid_ll_x = sr_box.get_box_rect().get_grid_ll_x();
  int32_t grid_ll_y = sr_box.get_box_rect().get_grid_ll_y();

  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> usage_map;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUTIL.getOppositeOrientation(orientation);

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t first_layer_idx = first_coord.get_layer_idx();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();
    int32_t second_layer_idx = second_coord.get_layer_idx();
    RTUTIL.swapByASC(first_x, second_x);
    RTUTIL.swapByASC(first_y, second_y);
    RTUTIL.swapByASC(first_layer_idx, second_layer_idx);

    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          LayerCoord coord(x, y, layer_idx);
          if (coord != first_coord) {
            usage_map[coord].insert(opposite_orientation);
          }
          if (coord != second_coord) {
            usage_map[coord].insert(orientation);
          }
        }
      }
    }
  }
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    SRNode& sr_node = layer_node_map[usage_coord.get_layer_idx()][usage_coord.get_x() - grid_ll_x][usage_coord.get_y() - grid_ll_y];
    sr_node.updateDemand(net_idx, orientation_list, change_type);
  }
}

#endif

#if 1  // exhibit

void SpaceRouter::updateSummary(SRModel& sr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_demand_map = summary.iter_sr_summary_map[sr_model.get_iter()].routing_demand_map;
  double& total_demand = summary.iter_sr_summary_map[sr_model.get_iter()].total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.iter_sr_summary_map[sr_model.get_iter()].routing_overflow_map;
  double& total_overflow = summary.iter_sr_summary_map[sr_model.get_iter()].total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.iter_sr_summary_map[sr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_sr_summary_map[sr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_sr_summary_map[sr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_sr_summary_map[sr_model.get_iter()].total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.iter_sr_summary_map[sr_model.get_iter()].clock_timing;
  std::map<std::string, double>& power_map = summary.iter_sr_summary_map[sr_model.get_iter()].power_map;

  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();
  std::vector<SRNet>& sr_net_list = sr_model.get_sr_net_list();

  routing_demand_map.clear();
  total_demand = 0;
  routing_overflow_map.clear();
  total_overflow = 0;
  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  clock_timing.clear();
  power_map.clear();

  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        double node_demand = sr_node_map[x][y].getDemand();
        double node_overflow = sr_node_map[x][y].getOverflow();
        routing_demand_map[layer_idx] += node_demand;
        total_demand += node_demand;
        routing_overflow_map[layer_idx] += node_overflow;
        total_overflow += node_overflow;
      }
    }
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord& first_coord = segment->get_first();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      LayerCoord& second_coord = segment->get_second();
      int32_t second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        GCell& first_gcell = gcell_map[first_coord.get_x()][first_coord.get_y()];
        GCell& second_gcell = gcell_map[second_coord.get_x()][second_coord.get_y()];
        double wire_length = RTUTIL.getManhattanDistance(first_gcell.getMidPoint(), second_gcell.getMidPoint()) / 1.0 / micron_dbu;
        routing_wire_length_map[first_layer_idx] += wire_length;
        total_wire_length += wire_length;
      } else {
        RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          cut_via_num_map[layer_via_master_list[layer_idx].front().get_cut_layer_idx()]++;
          total_via_num++;
        }
      }
    }
  }
  if (enable_timing) {
    std::vector<std::map<std::string, std::vector<LayerCoord>>> real_pin_coord_map_list;
    real_pin_coord_map_list.resize(sr_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(sr_net_list.size());
    for (SRNet& sr_net : sr_net_list) {
      for (SRPin& sr_pin : sr_net.get_sr_pin_list()) {
        LayerCoord layer_coord = sr_pin.get_access_point().getGridLayerCoord();
        real_pin_coord_map_list[sr_net.get_net_idx()][sr_pin.get_pin_name()].emplace_back(RTUTIL.getRealRectByGCell(layer_coord, gcell_axis).getMidPoint(),
                                                                                          layer_coord.get_layer_idx());
      }
    }
    for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        LayerCoord first_layer_coord = segment->get_first();
        LayerCoord first_real_coord(RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint(), first_layer_coord.get_layer_idx());
        LayerCoord second_layer_coord = segment->get_second();
        LayerCoord second_real_coord(RTUTIL.getRealRectByGCell(second_layer_coord, gcell_axis).getMidPoint(), second_layer_coord.get_layer_idx());

        routing_segment_list_list[net_idx].emplace_back(first_real_coord, second_real_coord);
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing, power_map);
  }
}

void SpaceRouter::printSummary(SRModel& sr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_demand_map = summary.iter_sr_summary_map[sr_model.get_iter()].routing_demand_map;
  double& total_demand = summary.iter_sr_summary_map[sr_model.get_iter()].total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.iter_sr_summary_map[sr_model.get_iter()].routing_overflow_map;
  double& total_overflow = summary.iter_sr_summary_map[sr_model.get_iter()].total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.iter_sr_summary_map[sr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_sr_summary_map[sr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_sr_summary_map[sr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_sr_summary_map[sr_model.get_iter()].total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.iter_sr_summary_map[sr_model.get_iter()].clock_timing;
  std::map<std::string, double>& power_map = summary.iter_sr_summary_map[sr_model.get_iter()].power_map;

  fort::char_table routing_demand_map_table;
  {
    routing_demand_map_table.set_cell_text_align(fort::text_align::right);
    routing_demand_map_table << fort::header << "routing"
                             << "demand"
                             << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_demand_map_table << routing_layer.get_layer_name() << routing_demand_map[routing_layer.get_layer_idx()]
                               << RTUTIL.getPercentage(routing_demand_map[routing_layer.get_layer_idx()], total_demand) << fort::endr;
    }
    routing_demand_map_table << fort::header << "Total" << total_demand << RTUTIL.getPercentage(total_demand, total_demand) << fort::endr;
  }
  fort::char_table routing_overflow_map_table;
  {
    routing_overflow_map_table.set_cell_text_align(fort::text_align::right);
    routing_overflow_map_table << fort::header << "routing"
                               << "overflow"
                               << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_overflow_map_table << routing_layer.get_layer_name() << routing_overflow_map[routing_layer.get_layer_idx()]
                                 << RTUTIL.getPercentage(routing_overflow_map[routing_layer.get_layer_idx()], total_overflow) << fort::endr;
    }
    routing_overflow_map_table << fort::header << "Total" << total_overflow << RTUTIL.getPercentage(total_overflow, total_overflow) << fort::endr;
  }
  fort::char_table routing_wire_length_map_table;
  {
    routing_wire_length_map_table.set_cell_text_align(fort::text_align::right);
    routing_wire_length_map_table << fort::header << "routing"
                                  << "wire_length"
                                  << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_wire_length_map_table << routing_layer.get_layer_name() << routing_wire_length_map[routing_layer.get_layer_idx()]
                                    << RTUTIL.getPercentage(routing_wire_length_map[routing_layer.get_layer_idx()], total_wire_length) << fort::endr;
    }
    routing_wire_length_map_table << fort::header << "Total" << total_wire_length << RTUTIL.getPercentage(total_wire_length, total_wire_length) << fort::endr;
  }
  fort::char_table cut_via_num_map_table;
  {
    cut_via_num_map_table.set_cell_text_align(fort::text_align::right);
    cut_via_num_map_table << fort::header << "cut"
                          << "#via"
                          << "prop" << fort::endr;
    for (CutLayer& cut_layer : cut_layer_list) {
      cut_via_num_map_table << cut_layer.get_layer_name() << cut_via_num_map[cut_layer.get_layer_idx()]
                            << RTUTIL.getPercentage(cut_via_num_map[cut_layer.get_layer_idx()], total_via_num) << fort::endr;
    }
    cut_via_num_map_table << fort::header << "Total" << total_via_num << RTUTIL.getPercentage(total_via_num, total_via_num) << fort::endr;
  }
  fort::char_table timing_table;
  timing_table.set_cell_text_align(fort::text_align::right);
  fort::char_table power_table;
  power_table.set_cell_text_align(fort::text_align::right);
  if (enable_timing) {
    timing_table << fort::header << "clock_name"
                 << "tns"
                 << "wns"
                 << "freq" << fort::endr;
    for (auto& [clock_name, timing_map] : clock_timing) {
      timing_table << clock_name << timing_map["TNS"] << timing_map["WNS"] << timing_map["Freq(MHz)"] << fort::endr;
    }
    power_table << fort::header << "power_type";
    for (auto& [type, power] : power_map) {
      power_table << fort::header << type;
    }
    power_table << fort::endr;
    power_table << "power_value";
    for (auto& [type, power] : power_map) {
      power_table << power;
    }
    power_table << fort::endr;
  }
  RTUTIL.printTableList({routing_demand_map_table, routing_overflow_map_table, routing_wire_length_map_table, cut_via_num_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void SpaceRouter::outputGuide(SRModel& sr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& sr_temp_directory_path = RTDM.getConfig().sr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<SRNet>& sr_net_list = sr_model.get_sr_net_list();

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(RTUTIL.getString(sr_temp_directory_path, "route_", sr_model.get_iter(), ".guide"));
  if (guide_file_stream == nullptr) {
    return;
  }
  RTUTIL.pushStream(guide_file_stream, "guide net_name\n");
  RTUTIL.pushStream(guide_file_stream, "pin grid_x grid_y real_x real_y layer energy name\n");
  RTUTIL.pushStream(guide_file_stream, "wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer\n");
  RTUTIL.pushStream(guide_file_stream, "via grid_x grid_y real_x real_y layer1 layer2\n");

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    SRNet& sr_net = sr_net_list[net_idx];
    RTUTIL.pushStream(guide_file_stream, "guide ", sr_net.get_origin_net()->get_net_name(), "\n");

    for (SRPin& sr_pin : sr_net.get_sr_pin_list()) {
      AccessPoint& access_point = sr_pin.get_access_point();
      double grid_x = access_point.get_grid_x();
      double grid_y = access_point.get_grid_y();
      double real_x = access_point.get_real_x() / 1.0 / micron_dbu;
      double real_y = access_point.get_real_y() / 1.0 / micron_dbu;
      std::string layer = routing_layer_list[access_point.get_layer_idx()].get_layer_name();
      std::string connnect;
      if (sr_pin.get_is_driven()) {
        connnect = "driven";
      } else {
        connnect = "load";
      }
      RTUTIL.pushStream(guide_file_stream, "pin ", grid_x, " ", grid_y, " ", real_x, " ", real_y, " ", layer, " ", connnect, " ", sr_pin.get_pin_name(), "\n");
    }
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord first_layer_coord = segment->get_first();
      double grid1_x = first_layer_coord.get_x();
      double grid1_y = first_layer_coord.get_y();
      int32_t first_layer_idx = first_layer_coord.get_layer_idx();

      PlanarCoord first_mid_coord = RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint();
      double real1_x = first_mid_coord.get_x() / 1.0 / micron_dbu;
      double real1_y = first_mid_coord.get_y() / 1.0 / micron_dbu;

      LayerCoord second_layer_coord = segment->get_second();
      double grid2_x = second_layer_coord.get_x();
      double grid2_y = second_layer_coord.get_y();
      int32_t second_layer_idx = second_layer_coord.get_layer_idx();

      PlanarCoord second_mid_coord = RTUTIL.getRealRectByGCell(second_layer_coord, gcell_axis).getMidPoint();
      double real2_x = second_mid_coord.get_x() / 1.0 / micron_dbu;
      double real2_y = second_mid_coord.get_y() / 1.0 / micron_dbu;

      if (first_layer_idx != second_layer_idx) {
        RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
        std::string layer1 = routing_layer_list[first_layer_idx].get_layer_name();
        std::string layer2 = routing_layer_list[second_layer_idx].get_layer_name();
        RTUTIL.pushStream(guide_file_stream, "via ", grid1_x, " ", grid1_y, " ", real1_x, " ", real1_y, " ", layer1, " ", layer2, "\n");
      } else {
        std::string layer = routing_layer_list[first_layer_idx].get_layer_name();
        RTUTIL.pushStream(guide_file_stream, "wire ", grid1_x, " ", grid1_y, " ", grid2_x, " ", grid2_y, " ", real1_x, " ", real1_y, " ", real2_x, " ", real2_y,
                          " ", layer, "\n");
      }
    }
  }
  RTUTIL.closeFileStream(guide_file_stream);
}

void SpaceRouter::outputNetCSV(SRModel& sr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& sr_temp_directory_path = RTDM.getConfig().sr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* net_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(sr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", sr_model.get_iter(), ".csv"));
    GridMap<SRNode>& sr_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = sr_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
        RTUTIL.pushStream(net_csv_file, sr_node_map[x][y].getDemand(), ",");
      }
      RTUTIL.pushStream(net_csv_file, "\n");
    }
    RTUTIL.closeFileStream(net_csv_file);
  }
}

void SpaceRouter::outputOverflowCSV(SRModel& sr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& sr_temp_directory_path = RTDM.getConfig().sr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* overflow_csv_file = RTUTIL.getOutputFileStream(
        RTUTIL.getString(sr_temp_directory_path, "overflow_map_", routing_layer.get_layer_name(), "_", sr_model.get_iter(), ".csv"));

    GridMap<SRNode>& sr_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = sr_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
        RTUTIL.pushStream(overflow_csv_file, sr_node_map[x][y].getOverflow(), ",");
      }
      RTUTIL.pushStream(overflow_csv_file, "\n");
    }
    RTUTIL.closeFileStream(overflow_csv_file);
  }
}

void SpaceRouter::outputNetJson(SRModel& sr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& sr_temp_directory_path = RTDM.getConfig().sr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<nlohmann::json> net_json_list;
  net_json_list.resize(net_list.size());
  for (Net& net : net_list) {
    net_json_list[net.get_net_idx()]["net_name"] = net.get_net_name();
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      PlanarRect first_gcell = RTUTIL.getRealRectByGCell(segment->get_first(), gcell_axis);
      PlanarRect second_gcell = RTUTIL.getRealRectByGCell(segment->get_second(), gcell_axis);
      if (segment->get_first().get_layer_idx() != segment->get_second().get_layer_idx()) {
        net_json_list[net_idx]["result"].push_back({first_gcell.get_ll_x(), first_gcell.get_ll_y(), first_gcell.get_ur_x(), first_gcell.get_ur_y(),
                                                    routing_layer_list[segment->get_first().get_layer_idx()].get_layer_name()});
        net_json_list[net_idx]["result"].push_back({second_gcell.get_ll_x(), second_gcell.get_ll_y(), second_gcell.get_ur_x(), second_gcell.get_ur_y(),
                                                    routing_layer_list[segment->get_second().get_layer_idx()].get_layer_name()});
      } else {
        PlanarRect gcell = RTUTIL.getBoundingBox({first_gcell, second_gcell});
        net_json_list[net_idx]["result"].push_back({gcell.get_ll_x(), gcell.get_ll_y(), gcell.get_ur_x(), gcell.get_ur_y(),
                                                    routing_layer_list[segment->get_first().get_layer_idx()].get_layer_name()});
      }
    }
  }
  std::string net_json_file_path = RTUTIL.getString(sr_temp_directory_path, "net_map_", sr_model.get_iter(), ".json");
  std::ofstream* net_json_file = RTUTIL.getOutputFileStream(net_json_file_path);
  (*net_json_file) << net_json_list;
  RTUTIL.closeFileStream(net_json_file);
  RTI.sendNotification(RTUTIL.getString("SR_", sr_model.get_iter(), "_net_map"), net_json_file_path);
}

void SpaceRouter::outputOverflowJson(SRModel& sr_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& sr_temp_directory_path = RTDM.getConfig().sr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<SRNode>>& layer_node_map = sr_model.get_layer_node_map();
  std::vector<nlohmann::json> overflow_json_list;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<SRNode>& sr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        PlanarRect gcell = RTUTIL.getRealRectByGCell(PlanarCoord(x, y), gcell_axis);
        overflow_json_list.push_back({gcell.get_ll_x(), gcell.get_ll_y(), gcell.get_ur_x(), gcell.get_ur_y(), routing_layer_list[layer_idx].get_layer_name(),
                                      sr_node_map[x][y].getOverflow()});
      }
    }
  }
  std::string overflow_json_file_path = RTUTIL.getString(sr_temp_directory_path, "overflow_map_", sr_model.get_iter(), ".json");
  std::ofstream* overflow_json_file = RTUTIL.getOutputFileStream(overflow_json_file_path);
  (*overflow_json_file) << overflow_json_list;
  RTUTIL.closeFileStream(overflow_json_file);
  RTI.sendNotification(RTUTIL.getString("SR_", sr_model.get_iter(), "_net_map"), overflow_json_file_path);
}

#endif

#if 1  // debug

void SpaceRouter::debugCheckSRBox(SRBox& sr_box)
{
  SRBoxId& sr_box_id = sr_box.get_sr_box_id();
  if (sr_box_id.get_x() < 0 || sr_box_id.get_y() < 0) {
    RTLOG.error(Loc::current(), "The grid coord is illegal!");
  }
  std::vector<GridMap<SRNode>>& layer_node_map = sr_box.get_layer_node_map();
  for (GridMap<SRNode>& sr_node_map : layer_node_map) {
    for (int32_t x = 0; x < sr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sr_node_map.get_y_size(); y++) {
        SRNode& sr_node = sr_node_map[x][y];
        if (!RTUTIL.isInside(sr_box.get_box_rect().get_grid_rect(), sr_node.get_planar_coord())) {
          RTLOG.error(Loc::current(), "The sr_node is out of box!");
        }
        for (auto& [orient, neighbor] : sr_node.get_neighbor_node_map()) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
            RTLOG.error(Loc::current(), "The sr_node neighbor is not bidirectional!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orient] != &sr_node) {
            RTLOG.error(Loc::current(), "The sr_node neighbor is not bidirectional!");
          }
          if (RTUTIL.getOrientation(LayerCoord(sr_node), LayerCoord(*neighbor)) == orient) {
            continue;
          }
          RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
        }
      }
    }
  }
}

#endif

}  // namespace irt
