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
#include "PinAccessor.hpp"

#include "DRCEngine.hpp"
#include "GDSPlotter.hpp"
#include "Monitor.hpp"
#include "PABox.hpp"
#include "PABoxId.hpp"
#include "PANet.hpp"
#include "PANode.hpp"
#include "PAParameter.hpp"
#include "PinAccessor.hpp"
#include "RTInterface.hpp"

namespace irt {

// public

void PinAccessor::initInst()
{
  if (_pa_instance == nullptr) {
    _pa_instance = new PinAccessor();
  }
}

PinAccessor& PinAccessor::getInst()
{
  if (_pa_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_pa_instance;
}

void PinAccessor::destroyInst()
{
  if (_pa_instance != nullptr) {
    delete _pa_instance;
    _pa_instance = nullptr;
  }
}

// function

void PinAccessor::access()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  PAModel pa_model = initPAModel();
  initAccessPointList(pa_model);
  uploadAccessPointList(pa_model);
  // debugPlotPAModel(pa_model, "before_eliminate");
  setPAParameter(pa_model);
  initPABoxMap(pa_model);
  buildBoxSchedule(pa_model);
  routePABoxMap(pa_model);
  updatePAModel(pa_model);
  // debugPlotPAModel(pa_model, "after_eliminate");
  updateSummary(pa_model);
  printSummary(pa_model);
  writePlanarPinCSV(pa_model);
  writeLayerPinCSV(pa_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

PinAccessor* PinAccessor::_pa_instance = nullptr;

PAModel PinAccessor::initPAModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  PAModel pa_model;
  pa_model.set_pa_net_list(convertToPANetList(net_list));
  return pa_model;
}

std::vector<PANet> PinAccessor::convertToPANetList(std::vector<Net>& net_list)
{
  std::vector<PANet> pa_net_list;
  pa_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    pa_net_list.emplace_back(convertToPANet(net));
  }
  return pa_net_list;
}

PANet PinAccessor::convertToPANet(Net& net)
{
  PANet pa_net;
  pa_net.set_origin_net(&net);
  pa_net.set_net_idx(net.get_net_idx());
  pa_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    pa_net.get_pa_pin_list().push_back(PAPin(pin));
  }
  pa_net.set_bounding_box(net.get_bounding_box());
  return pa_net;
}

void PinAccessor::initAccessPointList(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();

  std::vector<std::pair<int32_t, PAPin*>> net_pin_pair_list;
  for (PANet& pa_net : pa_net_list) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      net_pin_pair_list.emplace_back(pa_net.get_net_idx(), &pa_pin);
    }
  }
#pragma omp parallel for
  for (std::pair<int32_t, PAPin*>& net_pin_pair : net_pin_pair_list) {
    PAPin* pin = net_pin_pair.second;
    std::vector<AccessPoint>& access_point_list = net_pin_pair.second->get_access_point_list();
    std::vector<LayerRect> legal_shape_list = getLegalShapeList(pa_model, net_pin_pair.first, pin);
    for (auto getAccessPointList :
         {std::bind(&PinAccessor::getAccessPointListByTrackGrid, this, std::placeholders::_1, std::placeholders::_2),
          std::bind(&PinAccessor::getAccessPointListByOnTrack, this, std::placeholders::_1, std::placeholders::_2),
          std::bind(&PinAccessor::getAccessPointListByShapeCenter, this, std::placeholders::_1, std::placeholders::_2)}) {
      for (AccessPoint& access_point : getAccessPointList(pin->get_pin_idx(), legal_shape_list)) {
        access_point_list.push_back(access_point);
      }
      if (!access_point_list.empty()) {
        std::sort(access_point_list.begin(), access_point_list.end(),
                  [](AccessPoint& a, AccessPoint& b) { return CmpLayerCoordByXASC()(a.getRealLayerCoord(), b.getRealLayerCoord()); });
        break;
      }
    }
    if (access_point_list.empty()) {
      RTLOG.error(Loc::current(), "No access point was generated!");
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<LayerRect> PinAccessor::getLegalShapeList(PAModel& pa_model, int32_t net_idx, Pin* pin)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, std::vector<EXTLayerRect>> layer_pin_shape_list;
  for (EXTLayerRect& routing_shape : pin->get_routing_shape_list()) {
    layer_pin_shape_list[routing_shape.get_layer_idx()].emplace_back(routing_shape);
  }
  std::vector<LayerRect> legal_rect_list;
  for (auto& [layer_idx, pin_shape_list] : layer_pin_shape_list) {
    std::vector<PlanarRect> planar_legal_rect_list = getPlanarLegalRectList(pa_model, net_idx, pin_shape_list);
    // 对legal rect进行融合，prefer横就竖着切，prefer竖就横着切
    if (routing_layer_list[layer_idx].isPreferH()) {
      planar_legal_rect_list = RTUTIL.mergeRectListByBoost(planar_legal_rect_list, Direction::kVertical);
    } else {
      planar_legal_rect_list = RTUTIL.mergeRectListByBoost(planar_legal_rect_list, Direction::kHorizontal);
    }
    for (PlanarRect planar_legal_rect : planar_legal_rect_list) {
      legal_rect_list.emplace_back(planar_legal_rect, layer_idx);
    }
  }
  if (!legal_rect_list.empty()) {
    return legal_rect_list;
  }
  for (EXTLayerRect& routing_shape : pin->get_routing_shape_list()) {
    legal_rect_list.emplace_back(routing_shape.getRealLayerRect());
  }
  return legal_rect_list;
}

std::vector<PlanarRect> PinAccessor::getPlanarLegalRectList(PAModel& pa_model, int32_t curr_net_idx,
                                                            std::vector<EXTLayerRect>& pin_shape_list)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();

  int32_t curr_layer_idx;
  {
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      if (pin_shape_list.front().get_layer_idx() != pin_shape.get_layer_idx()) {
        RTLOG.error(Loc::current(), "The pin_shape_list is not on the same layer!");
      }
    }
    curr_layer_idx = pin_shape_list.front().get_layer_idx();
  }
  // 当前层缩小后的结果
  std::vector<EXTLayerRect> reduced_rect_list;
  {
    std::vector<PlanarRect> origin_pin_shape_list;
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      origin_pin_shape_list.push_back(pin_shape.get_real_rect());
    }
    PlanarRect& enclosure = layer_enclosure_map[curr_layer_idx];
    int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
    int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;
    int32_t half_min_width = routing_layer_list[curr_layer_idx].get_min_width() / 2;
    int32_t reduced_x_size = std::max(half_min_width, enclosure_half_x_span);
    int32_t reduced_y_size = std::max(half_min_width, enclosure_half_y_span);
    for (PlanarRect& real_rect :
         RTUTIL.getClosedReducedRectListByBoost(origin_pin_shape_list, reduced_x_size, reduced_y_size, reduced_x_size, reduced_y_size)) {
      EXTLayerRect reduced_rect;
      reduced_rect.set_real_rect(real_rect);
      reduced_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(reduced_rect.get_real_rect(), gcell_axis));
      reduced_rect.set_layer_idx(curr_layer_idx);
      reduced_rect_list.push_back(reduced_rect);
    }
  }
  // 要被剪裁的obstacle的集合 排序按照 本层 上层
  /**
   * 要被剪裁的obstacle的集合
   * 如果不是最顶层就往上取一层
   * 是最顶层就往下取一层
   */
  std::vector<int32_t> obs_layer_idx_list;
  if (curr_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
    obs_layer_idx_list = {curr_layer_idx, curr_layer_idx + 1};
  } else {
    obs_layer_idx_list = {curr_layer_idx, curr_layer_idx - 1};
  }
  std::vector<std::vector<PlanarRect>> routing_obs_shape_list_list;
  for (int32_t obs_layer_idx : obs_layer_idx_list) {
    RoutingLayer& routing_layer = routing_layer_list[obs_layer_idx];
    PlanarRect& enclosure = layer_enclosure_map[obs_layer_idx];
    int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
    int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;

    std::vector<PlanarRect> routing_obs_shape_list;
    for (EXTLayerRect& reduced_rect : reduced_rect_list) {
      for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(reduced_rect)) {
        if (!is_routing) {
          continue;
        }
        for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
          if (obs_layer_idx != layer_idx) {
            continue;
          }
          for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
            if (net_idx == curr_net_idx) {
              continue;
            }
            for (EXTLayerRect* fixed_rect : fixed_rect_set) {
              int32_t min_spacing = routing_layer.getMinSpacing(fixed_rect->get_real_rect());
              int32_t enlarged_x_size = min_spacing + enclosure_half_x_span;
              int32_t enlarged_y_size = min_spacing + enclosure_half_y_span;
              PlanarRect enlarged_rect
                  = RTUTIL.getEnlargedRect(fixed_rect->get_real_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
              if (RTUTIL.isOpenOverlap(reduced_rect.get_real_rect(), enlarged_rect)) {
                routing_obs_shape_list.push_back(enlarged_rect);
              }
            }
          }
        }
      }
    }
    if (!routing_obs_shape_list.empty()) {
      routing_obs_shape_list_list.push_back(routing_obs_shape_list);
    }
  }
  std::vector<PlanarRect> legal_rect_list;
  for (EXTLayerRect& reduced_rect : reduced_rect_list) {
    legal_rect_list.push_back(reduced_rect.get_real_rect());
  }
  for (std::vector<PlanarRect>& routing_obs_shape_list : routing_obs_shape_list_list) {
    std::vector<PlanarRect> legal_rect_list_temp = RTUTIL.getClosedCuttingRectListByBoost(legal_rect_list, routing_obs_shape_list);
    if (!legal_rect_list_temp.empty()) {
      legal_rect_list = legal_rect_list_temp;
    } else {
      break;
    }
  }
  return legal_rect_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByTrackGrid(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    // curr layer track grid
    for (int32_t x : RTUTIL.getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
      for (int32_t y : RTUTIL.getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kTrackGrid);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByOnTrack(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    // on track
    int32_t mid_x = (ll_x + ur_x) / 2;
    int32_t mid_y = (ll_y + ur_y) / 2;
    for (int32_t y : RTUTIL.getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
      layer_coord_list.emplace_back(mid_x, y, curr_layer_idx);
    }
    for (int32_t x : RTUTIL.getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
      layer_coord_list.emplace_back(x, mid_y, curr_layer_idx);
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kOnTrack);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByShapeCenter(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    // on shape
    int32_t mid_x = (ll_x + ur_x) / 2;
    int32_t mid_y = (ll_y + ur_y) / 2;
    layer_coord_list.emplace_back(mid_x, mid_y, curr_layer_idx);
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kShapeCenter);
  }
  return access_point_list;
}

void PinAccessor::uploadAccessPointList(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    std::vector<PlanarCoord> coord_list;
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        coord_list.push_back(access_point.get_real_coord());
      }
    }
    BoundingBox& bounding_box = pa_net.get_bounding_box();
    bounding_box.set_real_rect(RTUTIL.getBoundingBox(coord_list));
    bounding_box.set_grid_rect(RTUTIL.getOpenGCellGridRect(bounding_box.get_real_rect(), gcell_axis));
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        access_point.set_grid_coord(RTUTIL.getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
        RTDM.updateAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_point);
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::setPAParameter(PAModel& pa_model)
{
  int32_t cost_unit = 8;
  PAParameter pa_parameter(1, 0, 128 * cost_unit, 32 * cost_unit, 32 * cost_unit, 4);
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", pa_parameter.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "non_prefer_wire_unit: ", pa_parameter.get_non_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", pa_parameter.get_via_unit());
  RTLOG.info(Loc::current(), "corner_unit: ", pa_parameter.get_corner_unit());
  RTLOG.info(Loc::current(), "size: ", pa_parameter.get_size());
  RTLOG.info(Loc::current(), "offset: ", pa_parameter.get_offset());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", pa_parameter.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", pa_parameter.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", pa_parameter.get_violation_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", pa_parameter.get_max_routed_times());
  pa_model.set_pa_parameter(pa_parameter);
}

void PinAccessor::initPABoxMap(PAModel& pa_model)
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

  PAParameter& pa_parameter = pa_model.get_pa_parameter();
  int32_t size = pa_parameter.get_size();
  int32_t offset = pa_parameter.get_offset();
  int32_t x_box_num = std::ceil((x_gcell_num - offset) / 1.0 / size);
  int32_t y_box_num = std::ceil((y_gcell_num - offset) / 1.0 / size);

  GridMap<PABox>& pa_box_map = pa_model.get_pa_box_map();
  pa_box_map.init(x_box_num, y_box_num);

  for (int32_t x = 0; x < pa_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < pa_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = std::max(offset + x * size, 0);
      int32_t grid_ll_y = std::max(offset + y * size, 0);
      int32_t grid_ur_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_ur_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect ll_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ll_x, grid_ll_y), gcell_axis);
      PlanarRect ur_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ur_x, grid_ur_y), gcell_axis);
      PlanarRect box_real_rect(ll_gcell_rect.get_ll(), ur_gcell_rect.get_ur());

      PABox& pa_box = pa_box_map[x][y];

      EXTPlanarRect pa_box_rect;
      pa_box_rect.set_real_rect(box_real_rect);
      pa_box_rect.set_grid_rect(RTUTIL.getOpenGCellGridRect(box_real_rect, gcell_axis));
      pa_box.set_box_rect(pa_box_rect);
      PABoxId pa_box_id;
      pa_box_id.set_x(x);
      pa_box_id.set_y(y);
      pa_box.set_pa_box_id(pa_box_id);
      pa_box.set_pa_parameter(&pa_parameter);
    }
  }
}

void PinAccessor::buildBoxSchedule(PAModel& pa_model)
{
  GridMap<PABox>& pa_box_map = pa_model.get_pa_box_map();

  int32_t range = 2;

  std::vector<std::vector<PABoxId>> pa_box_id_list_list;
  for (int32_t start_x = 0; start_x < range; start_x++) {
    for (int32_t start_y = 0; start_y < range; start_y++) {
      std::vector<PABoxId> pa_box_id_list;
      for (int32_t x = start_x; x < pa_box_map.get_x_size(); x += range) {
        for (int32_t y = start_y; y < pa_box_map.get_y_size(); y += range) {
          pa_box_id_list.emplace_back(x, y);
        }
      }
      pa_box_id_list_list.push_back(pa_box_id_list);
    }
  }
  pa_model.set_pa_box_id_list_list(pa_box_id_list_list);
}

void PinAccessor::routePABoxMap(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<PABox>& pa_box_map = pa_model.get_pa_box_map();

  size_t total_box_num = 0;
  for (std::vector<PABoxId>& pa_box_id_list : pa_model.get_pa_box_id_list_list()) {
    total_box_num += pa_box_id_list.size();
  }

  size_t routed_box_num = 0;
  for (std::vector<PABoxId>& pa_box_id_list : pa_model.get_pa_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (PABoxId& pa_box_id : pa_box_id_list) {
      PABox& pa_box = pa_box_map[pa_box_id.get_x()][pa_box_id.get_y()];
      buildFixedRectList(pa_box);
      initPATaskList(pa_model, pa_box);
      if (needRouting(pa_box)) {
        buildBoxTrackAxis(pa_box);
        buildLayerNodeMap(pa_box);
        buildPANodeNeighbor(pa_box);
        buildOrientNetMap(pa_box);
        // debugCheckPABox(pa_box);
        // debugPlotPABox(pa_box, -1, "before_routing");
        routePABox(pa_box);
        // debugPlotPABox(pa_box, -1, "after_routing");
        buildAccessInfo(pa_box);
        uploadViolation(pa_box);
      }
      freePABox(pa_box);
    }
    routed_box_num += pa_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num),
               ") boxes with ", getViolationNum(), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::buildFixedRectList(PABox& pa_box)
{
  pa_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(pa_box.get_box_rect()));
}

void PinAccessor::initPATaskList(PAModel& pa_model, PABox& pa_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();
  std::vector<PATask*>& pa_task_list = pa_box.get_pa_task_list();

  EXTPlanarRect& box_rect = pa_box.get_box_rect();
  PlanarRect& box_real_rect = box_rect.get_real_rect();
  std::map<int32_t, std::set<AccessPoint*>> net_access_point_map = RTDM.getNetAccessPointMap(box_rect);

  for (auto& [net_idx, access_point_set] : net_access_point_map) {
    PANet& pa_net = pa_net_list[net_idx];

    std::map<int32_t, std::set<AccessPoint*>> pin_access_point_map;
    for (AccessPoint* access_point : access_point_set) {
      pin_access_point_map[access_point->get_pin_idx()].insert(access_point);
    }
    for (auto& [pin_idx, pin_access_point_set] : pin_access_point_map) {
      PAPin& pa_pin = pa_net.get_pa_pin_list()[pin_idx];
      std::vector<PAGroup> pa_group_list(2);
      {
        pa_group_list.front().set_is_target(false);
        for (AccessPoint* pin_access_point : pin_access_point_set) {
          pa_group_list.front().get_coord_list().push_back(pin_access_point->getRealLayerCoord());
        }
        std::set<LayerCoord, CmpLayerCoordByXASC> coord_set;
        for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
          int32_t curr_layer_idx = routing_shape.get_layer_idx();
          // 构建目标层
          std::vector<int32_t> point_layer_idx_list;
          if (curr_layer_idx < bottom_routing_layer_idx) {
            point_layer_idx_list.push_back(bottom_routing_layer_idx);
          } else if (top_routing_layer_idx < curr_layer_idx) {
            point_layer_idx_list.push_back(top_routing_layer_idx);
          } else if (curr_layer_idx < top_routing_layer_idx) {
            point_layer_idx_list.push_back(curr_layer_idx);
            point_layer_idx_list.push_back(curr_layer_idx + 1);
          } else {
            point_layer_idx_list.push_back(curr_layer_idx);
            point_layer_idx_list.push_back(curr_layer_idx - 1);
          }
          // 构建有效形状
          std::vector<ScaleGrid>& x_track_grid_list = routing_layer_list[curr_layer_idx].getXTrackGridList();
          std::vector<ScaleGrid>& y_track_grid_list = routing_layer_list[curr_layer_idx].getYTrackGridList();
          int32_t enlarged_x_size = x_track_grid_list.front().get_step_num();
          int32_t enlarged_y_size = y_track_grid_list.front().get_step_num();
          PlanarRect real_rect
              = RTUTIL.getEnlargedRect(routing_shape.get_real_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
          if (!RTUTIL.hasRegularRect(real_rect, box_real_rect)) {
            continue;
          }
          real_rect = RTUTIL.getRegularRect(real_rect, box_real_rect);
          // 构建点
          for (int32_t x : RTUTIL.getScaleList(real_rect.get_ll_x(), real_rect.get_ur_x(), x_track_grid_list)) {
            for (int32_t y : RTUTIL.getScaleList(real_rect.get_ll_y(), real_rect.get_ur_y(), y_track_grid_list)) {
              for (int32_t point_layer_idx : point_layer_idx_list) {
                coord_set.insert(LayerCoord(x, y, point_layer_idx));
              }
            }
          }
        }
        pa_group_list.back().set_is_target(true);
        for (const LayerCoord& coord : coord_set) {
          pa_group_list.back().get_coord_list().push_back(coord);
        }
      }
      PATask* pa_task = new PATask();
      pa_task->set_net_idx(net_idx);
      pa_task->set_task_idx(static_cast<int32_t>(pa_task_list.size()));
      pa_task->set_pa_pin(&pa_pin);
      pa_task->set_connect_type(pa_net.get_connect_type());
      pa_task->set_pa_group_list(pa_group_list);
      {
        std::vector<PlanarCoord> coord_list;
        for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
          for (LayerCoord& coord : pa_group.get_coord_list()) {
            coord_list.push_back(coord);
          }
        }
        pa_task->set_bounding_box(RTUTIL.getBoundingBox(coord_list));
      }
      pa_task->set_routed_times(0);
      pa_task_list.push_back(pa_task);
    }
  }
  std::sort(pa_task_list.begin(), pa_task_list.end(), CmpPATask());
}

bool PinAccessor::needRouting(PABox& pa_box)
{
  if (pa_box.get_pa_task_list().empty()) {
    return false;
  }
  return true;
}

void PinAccessor::buildBoxTrackAxis(PABox& pa_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_region = pa_box.get_box_rect().get_real_rect();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (int32_t x_scale : RTUTIL.getScaleList(box_region.get_ll_x(), box_region.get_ur_x(), routing_layer.getXTrackGridList())) {
      x_scale_list.push_back(x_scale);
    }
    for (int32_t y_scale : RTUTIL.getScaleList(box_region.get_ll_y(), box_region.get_ur_y(), routing_layer.getYTrackGridList())) {
      y_scale_list.push_back(y_scale);
    }
  }
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
      for (LayerCoord& coord : pa_group.get_coord_list()) {
        x_scale_list.push_back(coord.get_x());
        y_scale_list.push_back(coord.get_y());
      }
    }
  }

  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::sort(x_scale_list.begin(), x_scale_list.end());
  x_scale_list.erase(std::unique(x_scale_list.begin(), x_scale_list.end()), x_scale_list.end());
  box_track_axis.set_x_grid_list(RTUTIL.makeScaleGridList(x_scale_list));
  std::sort(y_scale_list.begin(), y_scale_list.end());
  y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  box_track_axis.set_y_grid_list(RTUTIL.makeScaleGridList(y_scale_list));
}

void PinAccessor::buildLayerNodeMap(PABox& pa_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  PlanarCoord& real_ll = pa_box.get_box_rect().get_real_ll();
  PlanarCoord& real_ur = pa_box.get_box_rect().get_real_ur();
  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());

  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<PANode>& pa_node_map = layer_node_map[layer_idx];
    pa_node_map.init(x_list.size(), y_list.size());
    for (size_t x = 0; x < x_list.size(); x++) {
      for (size_t y = 0; y < y_list.size(); y++) {
        PANode& pa_node = pa_node_map[x][y];
        pa_node.set_x(x_list[x]);
        pa_node.set_y(y_list[y]);
        pa_node.set_layer_idx(layer_idx);
      }
    }
  }
}

void PinAccessor::buildPANodeNeighbor(PABox& pa_box)
{
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_hv = true;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_hv = false;
    }
    GridMap<PANode>& dr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < dr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < dr_node_map.get_y_size(); y++) {
        std::map<Orientation, PANode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
        if (routing_hv) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &dr_node_map[x - 1][y];
          }
          if (x != (dr_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &dr_node_map[x + 1][y];
          }
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &dr_node_map[x][y - 1];
          }
          if (y != (dr_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &dr_node_map[x][y + 1];
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

void PinAccessor::buildOrientNetMap(PABox& pa_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToGraph(pa_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
}

void PinAccessor::routePABox(PABox& pa_box)
{
  std::vector<PATask*> pa_task_list = initTaskSchedule(pa_box);
  while (!pa_task_list.empty()) {
    for (PATask* pa_task : pa_task_list) {
      routePATask(pa_box, pa_task);
      pa_task->addRoutedTimes();
    }
    updateViolationList(pa_box);
    pa_task_list = getTaskScheduleByViolation(pa_box);
  }
}

std::vector<PATask*> PinAccessor::initTaskSchedule(PABox& pa_box)
{
  std::vector<PATask*> pa_task_list;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    pa_task_list.push_back(pa_task);
  }
  return pa_task_list;
}

std::vector<PATask*> PinAccessor::getTaskScheduleByViolation(PABox& pa_box)
{
  int32_t max_routed_times = pa_box.get_pa_parameter()->get_max_routed_times();

  std::set<int32_t> violation_net_set;
  for (Violation& violation : pa_box.get_violation_list()) {
    for (int32_t violation_net : violation.get_violation_net_set()) {
      violation_net_set.insert(violation_net);
    }
  }
  std::vector<PATask*> pa_task_list;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    if (!RTUTIL.exist(violation_net_set, pa_task->get_net_idx())) {
      continue;
    }
    if (pa_task->get_routed_times() >= max_routed_times) {
      continue;
    }
    pa_task_list.push_back(pa_task);
  }
  return pa_task_list;
}

void PinAccessor::routePATask(PABox& pa_box, PATask* pa_task)
{
  initSingleTask(pa_box, pa_task);
  while (!isConnectedAllEnd(pa_box)) {
    routeSinglePath(pa_box);
    updatePathResult(pa_box);
    updateDirectionSet(pa_box);
    resetStartAndEnd(pa_box);
    resetSinglePath(pa_box);
  }
  updateTaskResult(pa_box);
  resetSingleTask(pa_box);
}

void PinAccessor::initSingleTask(PABox& pa_box, PATask* pa_task)
{
  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();

  // single task
  pa_box.set_curr_pa_task(pa_task);
  {
    std::vector<std::vector<PANode*>> node_list_list;
    std::vector<PAGroup>& pa_group_list = pa_task->get_pa_group_list();
    for (PAGroup& pa_group : pa_group_list) {
      std::vector<PANode*> node_list;
      for (LayerCoord& coord : pa_group.get_coord_list()) {
        if (!RTUTIL.existTrackGrid(coord, box_track_axis)) {
          RTLOG.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, box_track_axis);
        PANode& pa_node = layer_node_map[coord.get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        node_list.push_back(&pa_node);
      }
      node_list_list.push_back(node_list);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        pa_box.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        pa_box.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  pa_box.get_path_node_list().clear();
  pa_box.get_single_task_visited_node_list().clear();
  pa_box.get_routing_segment_list().clear();
}

bool PinAccessor::isConnectedAllEnd(PABox& pa_box)
{
  return pa_box.get_end_node_list_list().empty();
}

void PinAccessor::routeSinglePath(PABox& pa_box)
{
  initPathHead(pa_box);
  while (!searchEnded(pa_box)) {
    expandSearching(pa_box);
    resetPathHead(pa_box);
  }
}

void PinAccessor::initPathHead(PABox& pa_box)
{
  std::vector<std::vector<PANode*>>& start_node_list_list = pa_box.get_start_node_list_list();
  std::vector<PANode*>& path_node_list = pa_box.get_path_node_list();

  for (std::vector<PANode*>& start_node_list : start_node_list_list) {
    for (PANode* start_node : start_node_list) {
      start_node->set_estimated_cost(getEstimateCostToEnd(pa_box, start_node));
      pushToOpenList(pa_box, start_node);
    }
  }
  for (PANode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(pa_box, path_node));
    pushToOpenList(pa_box, path_node);
  }
  resetPathHead(pa_box);
}

bool PinAccessor::searchEnded(PABox& pa_box)
{
  std::vector<std::vector<PANode*>>& end_node_list_list = pa_box.get_end_node_list_list();
  PANode* path_head_node = pa_box.get_path_head_node();

  if (path_head_node == nullptr) {
    pa_box.set_end_node_list_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (PANode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        pa_box.set_end_node_list_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void PinAccessor::expandSearching(PABox& pa_box)
{
  PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost>& open_queue = pa_box.get_open_queue();
  PANode* path_head_node = pa_box.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(pa_box, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了，需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpPANodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(pa_box, neighbor_node));
      pushToOpenList(pa_box, neighbor_node);
    }
  }
}

void PinAccessor::resetPathHead(PABox& pa_box)
{
  pa_box.set_path_head_node(popFromOpenList(pa_box));
}

bool PinAccessor::isRoutingFailed(PABox& pa_box)
{
  return pa_box.get_end_node_list_idx() == -1;
}

void PinAccessor::resetSinglePath(PABox& pa_box)
{
  PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost> empty_queue;
  pa_box.set_open_queue(empty_queue);

  std::vector<PANode*>& single_path_visited_node_list = pa_box.get_single_path_visited_node_list();
  for (PANode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(PANodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  pa_box.set_path_head_node(nullptr);
  pa_box.set_end_node_list_idx(-1);
}

void PinAccessor::updatePathResult(PABox& pa_box)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(pa_box.get_path_head_node())) {
    pa_box.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> PinAccessor::getRoutingSegmentListByNode(PANode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  PANode* curr_node = node;
  PANode* pre_node = curr_node->get_parent_node();

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

void PinAccessor::updateDirectionSet(PABox& pa_box)
{
  PANode* path_head_node = pa_box.get_path_head_node();

  PANode* curr_node = path_head_node;
  PANode* pre_node = curr_node->get_parent_node();
  while (pre_node != nullptr) {
    curr_node->get_direction_set().insert(RTUTIL.getDirection(*curr_node, *pre_node));
    pre_node->get_direction_set().insert(RTUTIL.getDirection(*pre_node, *curr_node));
    curr_node = pre_node;
    pre_node = curr_node->get_parent_node();
  }
}

void PinAccessor::resetStartAndEnd(PABox& pa_box)
{
  std::vector<std::vector<PANode*>>& start_node_list_list = pa_box.get_start_node_list_list();
  std::vector<std::vector<PANode*>>& end_node_list_list = pa_box.get_end_node_list_list();
  std::vector<PANode*>& path_node_list = pa_box.get_path_node_list();
  PANode* path_head_node = pa_box.get_path_head_node();
  int32_t end_node_list_idx = pa_box.get_end_node_list_idx();

  // 对于抵达的终点pin，只保留到达的node
  end_node_list_list[end_node_list_idx].clear();
  end_node_list_list[end_node_list_idx].push_back(path_head_node);

  PANode* path_node = path_head_node->get_parent_node();
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
    // 初始化时，要把start_node_list_list的pin只留一个ap点
    // 后续只要将end_node_list_list的pin保留一个ap点
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void PinAccessor::updateTaskResult(PABox& pa_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = getRoutingSegmentList(pa_box);

  int32_t curr_net_idx = pa_box.get_curr_pa_task()->get_net_idx();
  int32_t curr_task_idx = pa_box.get_curr_pa_task()->get_task_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = pa_box.get_net_task_result_map()[curr_net_idx][curr_task_idx];
  // 原结果从graph删除
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateNetResultToGraph(pa_box, ChangeType::kDel, curr_net_idx, routing_segment);
  }
  routing_segment_list = new_routing_segment_list;
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateNetResultToGraph(pa_box, ChangeType::kAdd, curr_net_idx, routing_segment);
  }
}

std::vector<Segment<LayerCoord>> PinAccessor::getRoutingSegmentList(PABox& pa_box)
{
  PATask* curr_pa_task = pa_box.get_curr_pa_task();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<PAGroup>& pa_group_list = curr_pa_task->get_pa_group_list();
  for (size_t i = 0; i < pa_group_list.size(); i++) {
    for (LayerCoord& coord : pa_group_list[i].get_coord_list()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, pa_box.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  return routing_segment_list;
}

void PinAccessor::resetSingleTask(PABox& pa_box)
{
  pa_box.set_curr_pa_task(nullptr);
  pa_box.get_start_node_list_list().clear();
  pa_box.get_end_node_list_list().clear();
  pa_box.get_path_node_list().clear();

  std::vector<PANode*>& single_task_visited_node_list = pa_box.get_single_task_visited_node_list();
  for (PANode* single_task_visited_node : single_task_visited_node_list) {
    single_task_visited_node->get_direction_set().clear();
  }
  single_task_visited_node_list.clear();

  pa_box.get_routing_segment_list().clear();
}

// manager open list

void PinAccessor::pushToOpenList(PABox& pa_box, PANode* curr_node)
{
  PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost>& open_queue = pa_box.get_open_queue();
  std::vector<PANode*>& single_task_visited_node_list = pa_box.get_single_task_visited_node_list();
  std::vector<PANode*>& single_path_visited_node_list = pa_box.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(PANodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

PANode* PinAccessor::popFromOpenList(PABox& pa_box)
{
  PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost>& open_queue = pa_box.get_open_queue();

  PANode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(PANodeState::kClose);
  }
  return node;
}

// calculate known cost

double PinAccessor::getKnowCost(PABox& pa_box, PANode* start_node, PANode* end_node)
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
  cost += getNodeCost(pa_box, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(pa_box, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(pa_box, start_node, end_node);
  cost += getKnowCornerCost(pa_box, start_node, end_node);
  cost += getKnowViaCost(pa_box, start_node, end_node);
  return cost;
}

double PinAccessor::getNodeCost(PABox& pa_box, PANode* curr_node, Orientation orientation)
{
  double fixed_rect_unit = pa_box.get_pa_parameter()->get_fixed_rect_unit();
  double routed_rect_unit = pa_box.get_pa_parameter()->get_routed_rect_unit();
  double violation_unit = pa_box.get_pa_parameter()->get_violation_unit();

  int32_t net_idx = pa_box.get_curr_pa_task()->get_net_idx();

  double cost = 0;
  cost += curr_node->getFixedRectCost(net_idx, orientation, fixed_rect_unit);
  cost += curr_node->getRoutedRectCost(net_idx, orientation, routed_rect_unit);
  cost += curr_node->getViolationCost(orientation, violation_unit);
  return cost;
}

double PinAccessor::getKnowWireCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = pa_box.get_pa_parameter()->get_prefer_wire_unit();
  double non_prefer_wire_unit = pa_box.get_pa_parameter()->get_non_prefer_wire_unit();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUTIL.getDirection(*start_node, *end_node)) {
      wire_cost *= prefer_wire_unit;
    } else {
      wire_cost *= non_prefer_wire_unit;
    }
  }
  return wire_cost;
}

double PinAccessor::getKnowCornerCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double corner_unit = pa_box.get_pa_parameter()->get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    std::set<Direction> direction_set;
    // 添加start direction
    std::set<Direction>& start_direction_set = start_node->get_direction_set();
    direction_set.insert(start_direction_set.begin(), start_direction_set.end());
    // 添加start到parent的direction
    if (start_node->get_parent_node() != nullptr) {
      direction_set.insert(RTUTIL.getDirection(*start_node->get_parent_node(), *start_node));
    }
    // 添加end direction
    std::set<Direction>& end_direction_set = end_node->get_direction_set();
    direction_set.insert(end_direction_set.begin(), end_direction_set.end());
    // 添加start到end的direction
    direction_set.insert(RTUTIL.getDirection(*start_node, *end_node));

    if (direction_set.size() == 2) {
      corner_cost += corner_unit;
    } else if (direction_set.size() == 2) {
      RTLOG.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

double PinAccessor::getKnowViaCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double via_unit = pa_box.get_pa_parameter()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double PinAccessor::getEstimateCostToEnd(PABox& pa_box, PANode* curr_node)
{
  std::vector<std::vector<PANode*>>& end_node_list_list = pa_box.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<PANode*>& end_node_list : end_node_list_list) {
    for (PANode* end_node : end_node_list) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(pa_box, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double PinAccessor::getEstimateCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(pa_box, start_node, end_node);
  estimate_cost += getEstimateCornerCost(pa_box, start_node, end_node);
  estimate_cost += getEstimateViaCost(pa_box, start_node, end_node);
  return estimate_cost;
}

double PinAccessor::getEstimateWireCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double prefer_wire_unit = pa_box.get_pa_parameter()->get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double PinAccessor::getEstimateCornerCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double corner_unit = pa_box.get_pa_parameter()->get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUTIL.isOblique(*start_node, *end_node)) {
      corner_cost += corner_unit;
    }
  }
  return corner_cost;
}

double PinAccessor::getEstimateViaCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double via_unit = pa_box.get_pa_parameter()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void PinAccessor::updateViolationList(PABox& pa_box)
{
  std::vector<Violation> new_violation_list = getViolationList(pa_box);

  std::vector<Violation>& violation_list = pa_box.get_violation_list();
  // 原结果从graph删除
  for (Violation& violation : violation_list) {
    updateViolationToGraph(pa_box, ChangeType::kDel, violation);
  }
  violation_list = new_violation_list;
  // 新结果添加到graph
  for (Violation& violation : violation_list) {
    updateViolationToGraph(pa_box, ChangeType::kAdd, violation);
  }
}

std::vector<Violation> PinAccessor::getViolationList(PABox& pa_box)
{
  std::string top_name = RTUTIL.getString("pa_box_", pa_box.get_pa_box_id().get_x(), "_", pa_box.get_pa_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          for (auto& fixed_rect : fixed_rect_set) {
            env_shape_list.emplace_back(fixed_rect, is_routing);
          }
        } else {
          for (auto& fixed_rect : fixed_rect_set) {
            net_pin_shape_map[net_idx].emplace_back(fixed_rect, is_routing);
          }
        }
      }
    }
  }
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_result_map;
  for (auto& [net_idx, task_result_map] : pa_box.get_net_task_result_map()) {
    for (auto& [task_idx, segment_list] : task_result_map) {
      for (Segment<LayerCoord>& segment : segment_list) {
        net_result_map[net_idx].emplace_back(segment);
      }
    }
  }
  std::string stage = "PA";
  return RTDE.getViolationList(top_name, env_shape_list, net_pin_shape_map, net_result_map, stage);
}

void PinAccessor::buildAccessInfo(PABox& pa_box)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& net_task_result_map = pa_box.get_net_task_result_map();

  std::vector<PATask*>& pa_task_list = pa_box.get_pa_task_list();
  for (PATask* pa_task : pa_task_list) {
    std::vector<Segment<LayerCoord>>& segment_list = net_task_result_map[pa_task->get_net_idx()][pa_task->get_task_idx()];
    PAPin* pa_pin = pa_task->get_pa_pin();
    pa_pin->set_access_segment_list(segment_list);
    for (NetShape& net_shape : RTDM.getNetShapeList(pa_task->get_net_idx(), segment_list)) {
      EXTLayerRect ext_layer_rect;
      ext_layer_rect.set_real_rect(net_shape.get_rect());
      ext_layer_rect.set_layer_idx(net_shape.get_layer_idx());
      ext_layer_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(ext_layer_rect.get_real_rect(), gcell_axis));
      if (net_shape.get_is_routing()) {
        pa_pin->get_access_routing_shape_list().push_back(ext_layer_rect);
      } else {
        pa_pin->get_access_cut_shape_list().push_back(ext_layer_rect);
      }
    }
    std::vector<LayerCoord> origin_coord_list;
    std::vector<LayerCoord> target_coord_list;
    if (segment_list.empty()) {
      for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
        for (LayerCoord& coord : pa_group.get_coord_list()) {
          if (!pa_group.get_is_target()) {
            origin_coord_list.push_back(coord);
          } else {
            target_coord_list.push_back(coord);
          }
        }
      }
    } else {
      for (Segment<LayerCoord>& segment : segment_list) {
        origin_coord_list.push_back(segment.get_first());
        origin_coord_list.push_back(segment.get_second());
      }
      for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
        if (pa_group.get_is_target()) {
          for (LayerCoord& coord : pa_group.get_coord_list()) {
            target_coord_list.push_back(coord);
          }
        }
      }
    }
    std::sort(origin_coord_list.begin(), origin_coord_list.end(), CmpLayerCoordByXASC());
    origin_coord_list.erase(std::unique(origin_coord_list.begin(), origin_coord_list.end()), origin_coord_list.end());
    std::sort(target_coord_list.begin(), target_coord_list.end(), CmpLayerCoordByXASC());
    target_coord_list.erase(std::unique(target_coord_list.begin(), target_coord_list.end()), target_coord_list.end());

    for (size_t i = 0, j = 0; i < origin_coord_list.size() && j < target_coord_list.size();) {
      if (origin_coord_list[i] == target_coord_list[j]) {
        AccessPoint access_point(pa_pin->get_pin_idx(), origin_coord_list[i], AccessPointType::kTrackGrid);
        pa_task->get_pa_pin()->set_access_point(access_point);
        break;
      } else if (CmpLayerCoordByXASC()(origin_coord_list[i], target_coord_list[j])) {
        i++;
      } else {
        j++;
      }
    }
    if (pa_task->get_pa_pin()->get_access_point().get_real_coord() == PlanarCoord(-1, -1)) {
      RTLOG.error(Loc::current(), "The access_point creation failed!");
    }
  }
}

void PinAccessor::uploadViolation(PABox& pa_box)
{
  for (Violation& violation : pa_box.get_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
}

void PinAccessor::freePABox(PABox& pa_box)
{
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    delete pa_task;
    pa_task = nullptr;
  }
  pa_box.get_pa_task_list().clear();
  pa_box.get_layer_node_map().clear();
}

int32_t PinAccessor::getViolationNum()
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void PinAccessor::updatePAModel(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    Net* origin_net = pa_net.get_origin_net();
    if (origin_net->get_net_idx() != pa_net.get_net_idx()) {
      RTLOG.error(Loc::current(), "The net idx is not equal!");
    }
    std::vector<PlanarCoord> coord_list;
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      coord_list.push_back(pa_pin.get_access_point().get_real_coord());
    }
    BoundingBox& bounding_box = pa_net.get_bounding_box();
    bounding_box.set_real_rect(RTUTIL.getBoundingBox(coord_list));
    bounding_box.set_grid_rect(RTUTIL.getOpenGCellGridRect(bounding_box.get_real_rect(), gcell_axis));
    origin_net->set_bounding_box(bounding_box);
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      Pin& origin_pin = origin_net->get_pin_list()[pa_pin.get_pin_idx()];
      if (origin_pin.get_pin_idx() != pa_pin.get_pin_idx()) {
        RTLOG.error(Loc::current(), "The pin idx is not equal!");
      }
      origin_pin.set_access_segment_list(pa_pin.get_access_segment_list());
      origin_pin.set_access_routing_shape_list(pa_pin.get_access_routing_shape_list());
      origin_pin.set_access_cut_shape_list(pa_pin.get_access_cut_shape_list());
      for (EXTLayerRect& access_routing_shape : origin_pin.get_access_routing_shape_list()) {
        RTDM.updateFixedRectToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_routing_shape, true);
      }
      for (EXTLayerRect& access_cut_shape : origin_pin.get_access_cut_shape_list()) {
        RTDM.updateFixedRectToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_cut_shape, false);
      }
      AccessPoint& access_point = pa_pin.get_access_point();
      access_point.set_grid_coord(RTUTIL.getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
      origin_pin.set_access_point(access_point);
      RTDM.updateAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &origin_pin.get_access_point());
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

#if 1  // update env

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        pa_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        pa_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void PinAccessor::updateNetResultToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape)) {
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          pa_node->get_orient_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          pa_node->get_orient_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void PinAccessor::updateViolationToGraph(PABox& pa_box, ChangeType change_type, Violation& violation)
{
  NetShape net_shape(-1, violation.get_violation_shape().getRealLayerRect(), violation.get_is_routing());
  for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        pa_node->get_orient_violation_number_map()[orientation]++;
      } else if (change_type == ChangeType::kDel) {
        pa_node->get_orient_violation_number_map()[orientation]--;
      }
    }
  }
}

std::map<PANode*, std::set<Orientation>> PinAccessor::getNodeOrientationMap(PABox& pa_box, NetShape& net_shape)
{
  std::map<PANode*, std::set<Orientation>> node_orientation_map;
  if (net_shape.get_is_routing()) {
    node_orientation_map = getRoutingNodeOrientationMap(pa_box, net_shape);
  } else {
    node_orientation_map = getCutNodeOrientationMap(pa_box, net_shape);
  }
  return node_orientation_map;
}

std::map<PANode*, std::set<Orientation>> PinAccessor::getRoutingNodeOrientationMap(PABox& pa_box, NetShape& net_shape)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();
  if (!net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  int32_t layer_idx = net_shape.get_layer_idx();
  RoutingLayer& routing_layer = routing_layer_list[layer_idx];
  int32_t min_spacing = routing_layer.getMinSpacing(net_shape.get_rect());
  int32_t half_wire_width = routing_layer.get_min_width() / 2;
  PlanarRect& enclosure = layer_enclosure_map[layer_idx];
  int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
  int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;

  GridMap<PANode>& pa_node_map = pa_box.get_layer_node_map()[layer_idx];
  std::map<PANode*, std::set<Orientation>> node_orientation_map;
  // wire
  {
    // 膨胀size为 min_spacing + half_wire_width
    int32_t enlarged_size = min_spacing + half_wire_width;
    // 贴合的也不算违例
    enlarged_size -= 1;
    PlanarRect planar_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_size);
    for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(planar_enlarged_rect, pa_box.get_box_track_axis())) {
      PANode& node = pa_node_map[grid_coord.get_x()][grid_coord.get_y()];
      for (const Orientation& orientation : orientation_set) {
        if (orientation == Orientation::kAbove || orientation == Orientation::kBelow) {
          continue;
        }
        if (!RTUTIL.exist(node.get_neighbor_node_map(), orientation)) {
          continue;
        }
        node_orientation_map[&node].insert(orientation);
        node_orientation_map[node.get_neighbor_node_map()[orientation]].insert(RTUTIL.getOppositeOrientation(orientation));
      }
    }
  }
  // via
  {
    // 膨胀size为 min_spacing + enclosure_half_span
    int32_t enlarged_x_size = min_spacing + enclosure_half_x_span;
    int32_t enlarged_y_size = min_spacing + enclosure_half_y_span;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect space_enlarged_rect
        = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, pa_box.get_box_track_axis())) {
      PANode& node = pa_node_map[grid_coord.get_x()][grid_coord.get_y()];
      for (const Orientation& orientation : orientation_set) {
        if (orientation == Orientation::kEast || orientation == Orientation::kWest || orientation == Orientation::kSouth
            || orientation == Orientation::kNorth) {
          continue;
        }
        if (!RTUTIL.exist(node.get_neighbor_node_map(), orientation)) {
          continue;
        }
        node_orientation_map[&node].insert(orientation);
        node_orientation_map[node.get_neighbor_node_map()[orientation]].insert(RTUTIL.getOppositeOrientation(orientation));
      }
    }
  }
  return node_orientation_map;
}

std::map<PANode*, std::set<Orientation>> PinAccessor::getCutNodeOrientationMap(PABox& pa_box, NetShape& net_shape)
{
  return {};
}

#endif

#if 1  // exhibit

void PinAccessor::updateSummary(PAModel& pa_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, int32_t>& routing_access_point_num_map = RTDM.getSummary().pa_summary.routing_access_point_num_map;
  std::map<AccessPointType, int32_t>& type_access_point_num_map = RTDM.getSummary().pa_summary.type_access_point_num_map;
  int32_t& total_access_point_num = RTDM.getSummary().pa_summary.total_access_point_num;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_access_point_num_map[routing_layer.get_layer_idx()] = 0;
  }
  type_access_point_num_map
      = {{AccessPointType::kNone, 0}, {AccessPointType::kTrackGrid, 0}, {AccessPointType::kOnTrack, 0}, {AccessPointType::kShapeCenter, 0}};
  total_access_point_num = 0;

  for (auto& [net_idx, access_point_list] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_list) {
      routing_access_point_num_map[access_point->get_layer_idx()]++;
      type_access_point_num_map[access_point->get_type()]++;
      total_access_point_num++;
    }
  }
}

void PinAccessor::printSummary(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, int32_t>& routing_access_point_num_map = RTDM.getSummary().pa_summary.routing_access_point_num_map;
  std::map<AccessPointType, int32_t>& type_access_point_num_map = RTDM.getSummary().pa_summary.type_access_point_num_map;
  int32_t& total_access_point_num = RTDM.getSummary().pa_summary.total_access_point_num;

  fort::char_table routing_access_point_num_map_table;
  {
    routing_access_point_num_map_table << fort::header << "routing_layer" << "access_point_num" << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_access_point_num_map_table << routing_layer.get_layer_name() << routing_access_point_num_map[routing_layer.get_layer_idx()]
                                         << RTUTIL.getPercentage(routing_access_point_num_map[routing_layer.get_layer_idx()],
                                                                 total_access_point_num)
                                         << fort::endr;
    }
    routing_access_point_num_map_table << fort::header << "Total" << total_access_point_num
                                       << RTUTIL.getPercentage(total_access_point_num, total_access_point_num) << fort::endr;
  }
  fort::char_table type_access_point_num_map_table;
  {
    type_access_point_num_map_table << fort::header << "type" << "access_point_num" << "proportion" << fort::endr;
    for (auto& [type, access_point_num] : type_access_point_num_map) {
      type_access_point_num_map_table << GetAccessPointTypeName()(type) << access_point_num
                                      << RTUTIL.getPercentage(access_point_num, total_access_point_num) << fort::endr;
    }
    type_access_point_num_map_table << fort::header << "Total" << total_access_point_num
                                    << RTUTIL.getPercentage(total_access_point_num, total_access_point_num) << fort::endr;
  }
  std::vector<fort::char_table> table_list;
  table_list.push_back(routing_access_point_num_map_table);
  table_list.push_back(type_access_point_num_map_table);
  RTUTIL.printTableList(table_list);
}

void PinAccessor::writePlanarPinCSV(PAModel& pa_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;
  int32_t output_csv = RTDM.getConfig().output_csv;
  if (!output_csv) {
    return;
  }
  GridMap<int32_t> planar_pin_map;
  planar_pin_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [net_idx, access_point_list] : gcell_map[x][y].get_net_access_point_map()) {
        planar_pin_map[x][y] += static_cast<int32_t>(access_point_list.size());
      }
    }
  }
  std::ofstream* pin_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(pa_temp_directory_path, "pin_map_planar.csv"));
  for (int32_t y = planar_pin_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < planar_pin_map.get_x_size(); x++) {
      RTUTIL.pushStream(pin_csv_file, planar_pin_map[x][y], ",");
    }
    RTUTIL.pushStream(pin_csv_file, "\n");
  }
  RTUTIL.closeFileStream(pin_csv_file);
}

void PinAccessor::writeLayerPinCSV(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;
  int32_t output_csv = RTDM.getConfig().output_csv;
  if (!output_csv) {
    return;
  }
  std::vector<GridMap<int32_t>> layer_pin_map;
  layer_pin_map.resize(routing_layer_list.size());
  for (GridMap<int32_t>& pin_map : layer_pin_map) {
    pin_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [net_idx, access_point_list] : gcell_map[x][y].get_net_access_point_map()) {
        for (AccessPoint* access_point : access_point_list) {
          layer_pin_map[access_point->get_layer_idx()][x][y]++;
        }
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* pin_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(pa_temp_directory_path, "pin_map_", routing_layer.get_layer_name(), ".csv"));
    GridMap<int32_t>& pin_map = layer_pin_map[routing_layer.get_layer_idx()];
    for (int32_t y = pin_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < pin_map.get_x_size(); x++) {
        RTUTIL.pushStream(pin_csv_file, pin_map[x][y], ",");
      }
      RTUTIL.pushStream(pin_csv_file, "\n");
    }
    RTUTIL.closeFileStream(pin_csv_file);
  }
}

#endif

#if 1  // debug

void PinAccessor::debugPlotPAModel(PAModel& pa_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;

  GPGDS gp_gds;

  // track_axis_struct
  GPStruct track_axis_struct("track_axis_struct");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<int32_t> x_list = RTUTIL.getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), routing_layer.getXTrackGridList());
    std::vector<int32_t> y_list = RTUTIL.getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), routing_layer.getYTrackGridList());
    for (int32_t x : x_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
      gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
      track_axis_struct.push(gp_path);
    }
    for (int32_t y : y_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
      gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
      track_axis_struct.push(gp_path);
    }
  }
  gp_gds.addStruct(track_axis_struct);

  // 整张版图的fixed_rect
  for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(die)) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        GPStruct fixed_rect_struct(RTUTIL.getString("fixed_rect(net_", net_idx, ")"));
        for (auto& fixed_rect : fixed_rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(fixed_rect->get_real_rect());
          if (is_routing) {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          } else {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(layer_idx));
          }
          fixed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(fixed_rect_struct);
      }
    }
  }

  // gcell_axis
  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<int32_t> gcell_x_list = RTUTIL.getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), gcell_axis.get_x_grid_list());
  std::vector<int32_t> gcell_y_list = RTUTIL.getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), gcell_axis.get_y_grid_list());
  for (int32_t x : gcell_x_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
    gcell_axis_struct.push(gp_path);
  }
  for (int32_t y : gcell_y_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
    gcell_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(gcell_axis_struct);

  // access_point
  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    GPStruct access_point_struct(RTUTIL.getString("access_point(net_", net_idx, ")"));
    for (AccessPoint* access_point : access_point_set) {
      int32_t x = access_point->get_real_x();
      int32_t y = access_point->get_real_y();

      GPBoundary access_point_boundary;
      access_point_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(access_point->get_layer_idx()));
      access_point_boundary.set_data_type(static_cast<int32_t>(GPDataType::kAccessPoint));
      access_point_boundary.set_rect(x - 10, y - 10, x + 10, y + 10);
      access_point_struct.push(access_point_boundary);
    }
    gp_gds.addStruct(access_point_struct);
  }

  std::string gds_file_path = RTUTIL.getString(pa_temp_directory_path, flag, "_access_point.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
