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
  // debugPlotPAModel(pa_model, "before");
  setPAParameter(pa_model);
  initAccessPointList(pa_model);
  uploadAccessPointList(pa_model);
  // debugPlotPAModel(pa_model, "middle");
  initPABoxMap(pa_model);
  buildBoxSchedule(pa_model);
  routePABoxMap(pa_model);
  uploadAccessPoint(pa_model);
  ignoreViolation(pa_model);
  // debugPlotPAModel(pa_model, "after");
  updateSummary(pa_model);
  printSummary(pa_model);
  outputPlanarPinCSV(pa_model);
  outputLayerPinCSV(pa_model);
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

void PinAccessor::setPAParameter(PAModel& pa_model)
{
  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5;
  double via_unit = cost_unit;
  double fixed_rect_unit = 4 * non_prefer_wire_unit * cost_unit;
  double routed_rect_unit = 2 * via_unit;
  double violation_unit = 4 * non_prefer_wire_unit * cost_unit;
  /**
   * prefer_wire_unit, non_prefer_wire_unit, via_unit, size, offset, fixed_rect_unit, routed_rect_unit, violation_unit, max_routed_times,
   * max_candidate_point_num
   */
  // clang-format off
  PAParameter pa_parameter(prefer_wire_unit, non_prefer_wire_unit, via_unit, 1, 0, fixed_rect_unit, routed_rect_unit, violation_unit, 10, 10);
  // clang-format on
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", pa_parameter.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "non_prefer_wire_unit: ", pa_parameter.get_non_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", pa_parameter.get_via_unit());
  RTLOG.info(Loc::current(), "size: ", pa_parameter.get_size());
  RTLOG.info(Loc::current(), "offset: ", pa_parameter.get_offset());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", pa_parameter.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", pa_parameter.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", pa_parameter.get_violation_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", pa_parameter.get_max_routed_times());
  RTLOG.info(Loc::current(), "max_candidate_point_num: ", pa_parameter.get_max_candidate_point_num());
  pa_model.set_pa_parameter(pa_parameter);
}

void PinAccessor::initAccessPointList(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();

  PlanarRect die_valid_rect = die.get_real_rect();
  int32_t shrinked_size = RTDM.getOnlyPitch();
  if (RTUTIL.hasShrinkedRect(die_valid_rect, shrinked_size)) {
    die_valid_rect = RTUTIL.getShrinkedRect(die_valid_rect, shrinked_size);
  }
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
    for (AccessPoint& access_point : getAccessPointList(pa_model, pin->get_pin_idx(), legal_shape_list)) {
      if (!RTUTIL.isInside(die_valid_rect, access_point.get_real_coord())) {
        continue;
      }
      access_point_list.push_back(access_point);
    }
    // 对于分散在多个gcell内的ap,取最多的留下
    {
      std::map<PlanarCoord, std::set<size_t>, CmpPlanarCoordByXASC> grid_access_point_idx_map;
      for (size_t i = 0; i < access_point_list.size(); i++) {
        PlanarCoord grid_coord = RTUTIL.getGCellGridCoordByBBox(access_point_list[i].get_real_coord(), gcell_axis, die);
        grid_access_point_idx_map[grid_coord].insert(i);
      }
      size_t max_point_num = 0;
      std::set<size_t> max_access_point_idx_set;
      for (auto& [coord, access_point_idx_set] : grid_access_point_idx_map) {
        size_t point_num = access_point_idx_set.size();
        if (max_point_num < point_num) {
          max_point_num = point_num;
          max_access_point_idx_set = access_point_idx_set;
        }
      }
      std::vector<AccessPoint> max_access_point_list;
      for (size_t access_point_idx : max_access_point_idx_set) {
        max_access_point_list.push_back(access_point_list[access_point_idx]);
      }
      access_point_list = max_access_point_list;
    }
    std::sort(access_point_list.begin(), access_point_list.end(),
              [](AccessPoint& a, AccessPoint& b) { return CmpLayerCoordByXASC()(a.getRealLayerCoord(), b.getRealLayerCoord()); });
    if (access_point_list.empty()) {
      RTLOG.error(Loc::current(), "No access point was generated!");
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<LayerRect> PinAccessor::getLegalShapeList(PAModel& pa_model, int32_t net_idx, Pin* pin)
{
  std::map<int32_t, std::vector<EXTLayerRect>> layer_pin_shape_list;
  for (EXTLayerRect& routing_shape : pin->get_routing_shape_list()) {
    layer_pin_shape_list[routing_shape.get_layer_idx()].emplace_back(routing_shape);
  }
  std::vector<LayerRect> legal_rect_list;
  for (auto& [layer_idx, pin_shape_list] : layer_pin_shape_list) {
    std::vector<PlanarRect> planar_legal_rect_list = getPlanarLegalRectList(pa_model, net_idx, pin_shape_list);
    for (PlanarRect planar_legal_rect : RTUTIL.mergeRectListByBoost(planar_legal_rect_list, Direction::kVertical)) {
      legal_rect_list.emplace_back(planar_legal_rect, layer_idx);
    }
    for (PlanarRect planar_legal_rect : RTUTIL.mergeRectListByBoost(planar_legal_rect_list, Direction::kHorizontal)) {
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
  std::vector<EXTLayerRect> shrinked_rect_list;
  {
    std::vector<PlanarRect> origin_pin_shape_list;
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      origin_pin_shape_list.push_back(pin_shape.get_real_rect());
    }
    PlanarRect& enclosure = layer_enclosure_map[curr_layer_idx];
    int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
    int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;
    int32_t half_min_width = routing_layer_list[curr_layer_idx].get_min_width() / 2;
    int32_t shrinked_x_size = std::max(half_min_width, enclosure_half_x_span);
    int32_t shrinked_y_size = std::max(half_min_width, enclosure_half_y_span);
    for (PlanarRect& real_rect : RTUTIL.getClosedShrinkedRectListByBoost(origin_pin_shape_list, shrinked_x_size, shrinked_y_size,
                                                                         shrinked_x_size, shrinked_y_size)) {
      EXTLayerRect shrinked_rect;
      shrinked_rect.set_real_rect(real_rect);
      shrinked_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(shrinked_rect.get_real_rect(), gcell_axis));
      shrinked_rect.set_layer_idx(curr_layer_idx);
      shrinked_rect_list.push_back(shrinked_rect);
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
    for (EXTLayerRect& shrinked_rect : shrinked_rect_list) {
      for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(shrinked_rect)) {
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
              // x_spacing y_spacing
              std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
              {
                // prl
                int32_t prl_spacing = routing_layer.getPRLSpacing(fixed_rect->get_real_rect());
                spacing_pair_list.emplace_back(prl_spacing, prl_spacing);
                if (layer_idx != curr_layer_idx) {
                  // eol
                  if (routing_layer.isPreferH()) {
                    spacing_pair_list.emplace_back(routing_layer.get_eol_spacing(), routing_layer.get_eol_within());
                  } else {
                    spacing_pair_list.emplace_back(routing_layer.get_eol_within(), routing_layer.get_eol_spacing());
                  }
                }
              }
              for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
                int32_t enlarged_x_size = x_spacing + enclosure_half_x_span;
                int32_t enlarged_y_size = y_spacing + enclosure_half_y_span;
                PlanarRect enlarged_rect = RTUTIL.getEnlargedRect(fixed_rect->get_real_rect(), enlarged_x_size, enlarged_y_size,
                                                                  enlarged_x_size, enlarged_y_size);
                if (RTUTIL.isOpenOverlap(shrinked_rect.get_real_rect(), enlarged_rect)) {
                  routing_obs_shape_list.push_back(enlarged_rect);
                }
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
  for (EXTLayerRect& shrinked_rect : shrinked_rect_list) {
    legal_rect_list.push_back(shrinked_rect.get_real_rect());
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

std::vector<AccessPoint> PinAccessor::getAccessPointList(PAModel& pa_model, int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    int32_t mid_x = (ll_x + ur_x) / 2;
    int32_t mid_y = (ll_y + ur_y) / 2;
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    // track grid
    for (int32_t x : RTUTIL.getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
      for (int32_t y : RTUTIL.getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
    // on track
    {
      for (int32_t x : {ll_x, mid_x, ur_x}) {
        for (int32_t y : RTUTIL.getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
      for (int32_t x : RTUTIL.getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
        for (int32_t y : {ll_y, mid_y, ur_y}) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
    }
    // on shape
    for (int32_t x : {ll_x, mid_x, ur_x}) {
      for (int32_t y : {ll_y, mid_y, ur_y}) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());
  // 均匀采样,删除一些靠的很近的点
  uniformSampleCoordList(pa_model, layer_coord_list);
  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kNoAccess);
  }
  return access_point_list;
}

void PinAccessor::uniformSampleCoordList(PAModel& pa_model, std::vector<LayerCoord>& layer_coord_list)
{
  int32_t max_candidate_point_num = pa_model.get_pa_parameter().get_max_candidate_point_num();

  PlanarRect bounding_box = RTUTIL.getBoundingBox(layer_coord_list);
  int32_t grid_num = static_cast<int32_t>(std::sqrt(max_candidate_point_num));
  double grid_x_span = bounding_box.getXSpan() / grid_num;
  double grid_y_span = bounding_box.getYSpan() / grid_num;

  std::set<PlanarCoord, CmpPlanarCoordByXASC> visited_set;
  std::vector<LayerCoord> new_layer_coord_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    PlanarCoord grid_coord(static_cast<int32_t>((layer_coord.get_x() - bounding_box.get_ll_x()) / grid_x_span),
                           static_cast<int32_t>((layer_coord.get_y() - bounding_box.get_ll_y()) / grid_y_span));
    if (!RTUTIL.exist(visited_set, grid_coord)) {
      new_layer_coord_list.push_back(layer_coord);
      visited_set.insert(grid_coord);
      if (static_cast<int32_t>(new_layer_coord_list.size()) >= max_candidate_point_num) {
        break;
      }
    }
  }
  layer_coord_list = new_layer_coord_list;
}

void PinAccessor::uploadAccessPointList(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kDel, net_idx, access_point);
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
        RTDM.updateAccessNetPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_point);
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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
  int32_t x_box_num = static_cast<int32_t>(std::ceil((x_gcell_num - offset) / 1.0 / size));
  int32_t y_box_num = static_cast<int32_t>(std::ceil((y_gcell_num - offset) / 1.0 / size));

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

  int32_t range = 3;

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
      buildFixedRect(pa_box);
      buildAccessResult(pa_box);
      buildViolation(pa_box);
      initPATaskList(pa_model, pa_box);
      if (needRouting(pa_box)) {
        buildBoxTrackAxis(pa_box);
        buildLayerNodeMap(pa_box);
        buildPANodeNeighbor(pa_box);
        buildOrientNetMap(pa_box);
        exemptPinShape(pa_box);
        // debugCheckPABox(pa_box);
        // debugPlotPABox(pa_box, -1, "before");
        routePABox(pa_box);
        // debugPlotPABox(pa_box, -1, "after");
        uploadAccessResult(pa_box);
      }
      uploadViolation(pa_box);
      freePABox(pa_box);
    }
    routed_box_num += pa_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num),
               ") boxes with ", getViolationNum(), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::buildFixedRect(PABox& pa_box)
{
  pa_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(pa_box.get_box_rect()));
}

void PinAccessor::buildAccessResult(PABox& pa_box)
{
  pa_box.set_net_access_result_map(RTDM.getNetAccessResultMap(pa_box.get_box_rect()));
  pa_box.set_net_access_patch_map(RTDM.getNetAccessPatchMap(pa_box.get_box_rect()));
}

void PinAccessor::buildViolation(PABox& pa_box)
{
  for (Violation* violation : RTDM.getViolationSet(pa_box.get_box_rect())) {
    if (!RTUTIL.isInside(pa_box.get_box_rect().get_real_rect(), violation->get_violation_shape().get_real_rect())) {
      continue;
    }
    pa_box.get_violation_list().push_back(*violation);
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
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
      if (pa_pin.get_is_accessed()) {
        continue;
      }
      pa_pin.set_is_accessed(true);
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
            point_layer_idx_list.push_back(bottom_routing_layer_idx + 1);
          } else if (top_routing_layer_idx < curr_layer_idx) {
            point_layer_idx_list.push_back(top_routing_layer_idx);
            point_layer_idx_list.push_back(top_routing_layer_idx - 1);
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
          int32_t enlarged_x_size = x_track_grid_list.front().get_step_length();
          int32_t enlarged_y_size = y_track_grid_list.front().get_step_length();
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
          bool in_shape = false;
          for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
            if (routing_shape.get_layer_idx() == coord.get_layer_idx() && RTUTIL.isInside(routing_shape.get_real_rect(), coord)) {
              in_shape = true;
              break;
            }
          }
          if (!in_shape) {
            pa_group_list.back().get_coord_list().push_back(coord);
          }
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
    GridMap<PANode>& pa_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < pa_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < pa_node_map.get_y_size(); y++) {
        std::map<Orientation, PANode*>& neighbor_node_map = pa_node_map[x][y].get_neighbor_node_map();
        if (routing_hv) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &pa_node_map[x - 1][y];
          }
          if (x != (pa_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &pa_node_map[x + 1][y];
          }
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &pa_node_map[x][y - 1];
          }
          if (y != (pa_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &pa_node_map[x][y + 1];
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
  for (auto& [net_idx, segment_set] : pa_box.get_net_access_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      updateFixedRectToGraph(pa_box, ChangeType::kAdd, net_idx, *segment);
    }
  }
  for (auto& [net_idx, patch_set] : pa_box.get_net_access_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      updateFixedRectToGraph(pa_box, ChangeType::kAdd, net_idx, *patch);
    }
  }
  for (Violation& violation : pa_box.get_violation_list()) {
    addViolationToGraph(pa_box, violation);
  }
}

void PinAccessor::exemptPinShape(PABox& pa_box)
{
  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  for (GridMap<PANode>& pa_node_map : layer_node_map) {
    for (int32_t x = 0; x < pa_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < pa_node_map.get_y_size(); y++) {
        PANode& pa_node = pa_node_map[x][y];
        for (auto& [orient, net_set] : pa_node.get_orient_fixed_rect_map()) {
          if (RTUTIL.exist(net_set, -1) && net_set.size() >= 2) {
            net_set.erase(-1);
          }
        }
      }
    }
  }
}

void PinAccessor::routePABox(PABox& pa_box)
{
  std::vector<PATask*> routing_task_list = initTaskSchedule(pa_box);
  while (!routing_task_list.empty()) {
    for (PATask* routing_task : routing_task_list) {
      routePATask(pa_box, routing_task);
      routing_task->addRoutedTimes();
    }
    updateViolationList(pa_box);
    updateTaskSchedule(pa_box, routing_task_list);
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

void PinAccessor::routePATask(PABox& pa_box, PATask* pa_task)
{
  initSingleTask(pa_box, pa_task);
  while (!isConnectedAllEnd(pa_box)) {
    routeSinglePath(pa_box);
    updatePathResult(pa_box);
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
      // 对优先队列中的值修改了,需要重新建堆
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

void PinAccessor::resetStartAndEnd(PABox& pa_box)
{
  std::vector<std::vector<PANode*>>& start_node_list_list = pa_box.get_start_node_list_list();
  std::vector<std::vector<PANode*>>& end_node_list_list = pa_box.get_end_node_list_list();
  std::vector<PANode*>& path_node_list = pa_box.get_path_node_list();
  PANode* path_head_node = pa_box.get_path_head_node();
  int32_t end_node_list_idx = pa_box.get_end_node_list_idx();

  // 对于抵达的终点pin,只保留到达的node
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
    // 初始化时,要把start_node_list_list的pin只留一个ap点
    // 后续只要将end_node_list_list的pin保留一个ap点
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
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

void PinAccessor::updateTaskResult(PABox& pa_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = getRoutingSegmentList(pa_box);

  int32_t curr_net_idx = pa_box.get_curr_pa_task()->get_net_idx();
  int32_t curr_task_idx = pa_box.get_curr_pa_task()->get_task_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = pa_box.get_net_task_result_map()[curr_net_idx][curr_task_idx];

  // 原结果从graph删除,由于task有对应net_idx,所以不需要在布线前进行删除也不会影响结果
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
  pa_box.get_single_task_visited_node_list().clear();
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
  estimate_cost += getEstimateViaCost(pa_box, start_node, end_node);
  return estimate_cost;
}

double PinAccessor::getEstimateWireCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double prefer_wire_unit = pa_box.get_pa_parameter()->get_prefer_wire_unit();
  double non_prefer_wire_unit = pa_box.get_pa_parameter()->get_non_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= std::min(prefer_wire_unit, non_prefer_wire_unit);
  return wire_cost;
}

double PinAccessor::getEstimateViaCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double via_unit = pa_box.get_pa_parameter()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void PinAccessor::updateViolationList(PABox& pa_box)
{
  pa_box.get_violation_list().clear();
  for (Violation new_violation : getCostViolationList(pa_box)) {
    pa_box.get_violation_list().push_back(new_violation);
  }
  // 新结果添加到graph
  for (Violation& violation : pa_box.get_violation_list()) {
    addViolationToGraph(pa_box, violation);
  }
}

std::vector<Violation> PinAccessor::getCostViolationList(PABox& pa_box)
{
  std::string top_name = RTUTIL.getString("pa_box_", pa_box.get_pa_box_id().get_x(), "_", pa_box.get_pa_box_id().get_y());
  PlanarRect check_region = pa_box.get_box_rect().get_real_rect();
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
  std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
  for (auto& [net_idx, segment_set] : pa_box.get_net_access_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, task_result_map] : pa_box.get_net_task_result_map()) {
    for (auto& [task_idx, segment_list] : task_result_map) {
      for (Segment<LayerCoord>& segment : segment_list) {
        net_result_map[net_idx].emplace_back(&segment);
      }
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : pa_box.get_net_access_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, task_patch_map] : pa_box.get_net_task_patch_map()) {
    for (auto& [task_idx, patch_list] : task_patch_map) {
      for (EXTLayerRect& patch : patch_list) {
        net_patch_map[net_idx].emplace_back(&patch);
      }
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    need_checked_net_set.insert(pa_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_process_type(DEProcessType::kMultiNet);
  de_task.set_top_name(top_name);
  de_task.set_check_region(check_region);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  return RTDE.getViolationList(de_task);
}

void PinAccessor::updateTaskSchedule(PABox& pa_box, std::vector<PATask*>& routing_task_list)
{
  int32_t max_routed_times = pa_box.get_pa_parameter()->get_max_routed_times();

  std::set<PATask*> visited_routing_task_set;
  std::vector<PATask*> new_routing_task_list;
  for (Violation& violation : pa_box.get_violation_list()) {
    for (PATask* pa_task : pa_box.get_pa_task_list()) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), pa_task->get_net_idx())) {
        continue;
      }
      if (pa_task->get_routed_times() < max_routed_times && !RTUTIL.exist(visited_routing_task_set, pa_task)) {
        visited_routing_task_set.insert(pa_task);
        new_routing_task_list.push_back(pa_task);
      }
      break;
    }
  }
  routing_task_list = new_routing_task_list;

  std::vector<PATask*> new_pa_task_list;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    if (!RTUTIL.exist(visited_routing_task_set, pa_task)) {
      new_pa_task_list.push_back(pa_task);
    }
  }
  for (PATask* routing_task : routing_task_list) {
    new_pa_task_list.push_back(routing_task);
  }
  pa_box.set_pa_task_list(new_pa_task_list);
}

void PinAccessor::uploadAccessResult(PABox& pa_box)
{
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& net_task_result_map = pa_box.get_net_task_result_map();
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& net_task_patch_map = pa_box.get_net_task_patch_map();

  std::vector<PATask*>& pa_task_list = pa_box.get_pa_task_list();
  for (PATask* pa_task : pa_task_list) {
    PAPin* pa_pin = pa_task->get_pa_pin();
    std::vector<Segment<LayerCoord>>& segment_list = net_task_result_map[pa_task->get_net_idx()][pa_task->get_task_idx()];
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetAccessResultToGCellMap(ChangeType::kAdd, pa_task->get_net_idx(), new Segment<LayerCoord>(segment));
    }
    std::vector<EXTLayerRect>& patch_list = net_task_patch_map[pa_task->get_net_idx()][pa_task->get_task_idx()];
    for (EXTLayerRect& patch : patch_list) {
      RTDM.updateNetAccessPatchToGCellMap(ChangeType::kAdd, pa_task->get_net_idx(), new EXTLayerRect(patch));
    }
    // access_point
    {
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
          pa_pin->set_access_point(access_point);
          break;
        } else if (CmpLayerCoordByXASC()(origin_coord_list[i], target_coord_list[j])) {
          i++;
        } else {
          j++;
        }
      }
      if (pa_pin->get_access_point().get_real_coord() == PlanarCoord(-1, -1)) {
        RTLOG.error(Loc::current(), "The access_point creation failed!");
      }
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

void PinAccessor::uploadAccessPoint(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kDel, net_idx, access_point);
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
      AccessPoint& access_point = pa_pin.get_access_point();
      access_point.set_grid_coord(RTUTIL.getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
      origin_pin.set_access_point(access_point);
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &origin_pin.get_access_point());
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::ignoreViolation(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  RTDE.updateIgnoreViolationSet();
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

#if 1  // update env

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape, true)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        pa_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        pa_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape, true)) {
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          pa_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          pa_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch)
{
  NetShape net_shape(net_idx, patch.getRealLayerRect(), true);
  for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape, true)) {
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
    for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape, true)) {
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

void PinAccessor::updateNetResultToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch)
{
  NetShape net_shape(net_idx, patch.getRealLayerRect(), true);
  for (auto& [pa_node, orientation_set] : getNodeOrientationMap(pa_box, net_shape, true)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        pa_node->get_orient_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        pa_node->get_orient_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void PinAccessor::addViolationToGraph(PABox& pa_box, Violation& violation)
{
  LayerRect searched_rect;
  {
    EXTLayerRect& violation_shape = violation.get_violation_shape();
    searched_rect.set_rect(RTUTIL.getEnlargedRect(violation_shape.get_real_rect(), RTDM.getOnlyPitch()));
    if (violation.get_is_routing()) {
      searched_rect.set_layer_idx(violation_shape.get_layer_idx());
    } else {
      RTLOG.error(Loc::current(), "The violation layer is cut!");
    }
  }
  std::vector<Segment<LayerCoord>> overlap_segment_list;
  for (auto& [net_idx, task_result_map] : pa_box.get_net_task_result_map()) {
    for (auto& [task_idx, segment_list] : task_result_map) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), net_idx)) {
        continue;
      }
      for (Segment<LayerCoord>& segment : segment_list) {
        if (!RTUTIL.isOverlap(searched_rect, segment)) {
          continue;
        }
        overlap_segment_list.push_back(segment);
      }
    }
  }
  addViolationToGraph(pa_box, searched_rect, overlap_segment_list);
}

void PinAccessor::addViolationToGraph(PABox& pa_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list)
{
  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();

  for (Segment<LayerCoord>& overlap_segment : overlap_segment_list) {
    LayerCoord& first_coord = overlap_segment.get_first();
    LayerCoord& second_coord = overlap_segment.get_second();
    if (first_coord == second_coord) {
      continue;
    }
    PlanarRect real_rect = RTUTIL.getEnlargedRect(first_coord, second_coord, 0);
    if (!RTUTIL.existTrackGrid(real_rect, box_track_axis)) {
      continue;
    }
    PlanarRect grid_rect = RTUTIL.getTrackGrid(real_rect, box_track_axis);
    std::map<int32_t, std::set<PANode*>> distance_node_map;
    {
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
      for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
        for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
          for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
            PANode* pa_node = &layer_node_map[layer_idx][x][y];
            if (searched_rect.get_layer_idx() != pa_node->get_layer_idx()) {
              continue;
            }
            int32_t distance = 0;
            if (!RTUTIL.isInside(searched_rect.get_rect(), pa_node->get_planar_coord())) {
              distance = RTUTIL.getManhattanDistance(searched_rect.getMidPoint(), pa_node->get_planar_coord());
            }
            distance_node_map[distance].insert(pa_node);
          }
        }
      }
    }
    std::set<PANode*> valid_node_set;
    if (!distance_node_map[0].empty()) {
      valid_node_set = distance_node_map[0];
    } else {
      for (auto& [distance, node_set] : distance_node_map) {
        valid_node_set.insert(node_set.begin(), node_set.end());
        if (valid_node_set.size() >= 2) {
          break;
        }
      }
    }
    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    Orientation oppo_orientation = RTUTIL.getOppositeOrientation(orientation);
    for (PANode* valid_node : valid_node_set) {
      if (LayerCoord(*valid_node) != first_coord) {
        valid_node->get_orient_violation_number_map()[oppo_orientation]++;
        if (RTUTIL.exist(valid_node->get_neighbor_node_map(), oppo_orientation)) {
          valid_node->get_neighbor_node_map()[oppo_orientation]->get_orient_violation_number_map()[orientation]++;
        }
      }
      if (LayerCoord(*valid_node) != second_coord) {
        valid_node->get_orient_violation_number_map()[orientation]++;
        if (RTUTIL.exist(valid_node->get_neighbor_node_map(), orientation)) {
          valid_node->get_neighbor_node_map()[orientation]->get_orient_violation_number_map()[oppo_orientation]++;
        }
      }
    }
  }
}

std::map<PANode*, std::set<Orientation>> PinAccessor::getNodeOrientationMap(PABox& pa_box, NetShape& net_shape, bool need_enlarged)
{
  std::map<PANode*, std::set<Orientation>> node_orientation_map;
  if (net_shape.get_is_routing()) {
    node_orientation_map = getRoutingNodeOrientationMap(pa_box, net_shape, need_enlarged);
  } else {
    node_orientation_map = getCutNodeOrientationMap(pa_box, net_shape, need_enlarged);
  }
  return node_orientation_map;
}

std::map<PANode*, std::set<Orientation>> PinAccessor::getRoutingNodeOrientationMap(PABox& pa_box, NetShape& net_shape, bool need_enlarged)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();
  if (!net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  int32_t layer_idx = net_shape.get_layer_idx();
  RoutingLayer& routing_layer = routing_layer_list[layer_idx];
  // x_spacing y_spacing
  std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
  {
    // prl
    int32_t prl_spacing = routing_layer.getPRLSpacing(net_shape.get_rect());
    spacing_pair_list.emplace_back(prl_spacing, prl_spacing);
    // eol
    if (routing_layer.isPreferH()) {
      spacing_pair_list.emplace_back(routing_layer.get_eol_spacing(), routing_layer.get_eol_within());
    } else {
      spacing_pair_list.emplace_back(routing_layer.get_eol_within(), routing_layer.get_eol_spacing());
    }
  }
  int32_t half_wire_width = routing_layer.get_min_width() / 2;
  PlanarRect& enclosure = layer_enclosure_map[layer_idx];
  int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
  int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;

  GridMap<PANode>& pa_node_map = pa_box.get_layer_node_map()[layer_idx];
  std::map<PANode*, std::set<Orientation>> node_orientation_map;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    int32_t enlarged_x_size = half_wire_width;
    int32_t enlarged_y_size = half_wire_width;
    if (need_enlarged) {
      // 膨胀size为 half_wire_width + spacing
      enlarged_x_size += x_spacing;
      enlarged_y_size += y_spacing;
    }
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect planar_enlarged_rect
        = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
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
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    int32_t enlarged_x_size = enclosure_half_x_span;
    int32_t enlarged_y_size = enclosure_half_y_span;
    if (need_enlarged) {
      // 膨胀size为 enclosure_half_span + spacing
      enlarged_x_size += x_spacing;
      enlarged_y_size += y_spacing;
    }
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

std::map<PANode*, std::set<Orientation>> PinAccessor::getCutNodeOrientationMap(PABox& pa_box, NetShape& net_shape, bool need_enlarged)
{
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  if (net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is routing!");
  }
  CutLayer& cut_layer = cut_layer_list[net_shape.get_layer_idx()];
  std::map<int32_t, std::vector<std::pair<int32_t, int32_t>>> cut_spacing_map;
  {
    int32_t curr_cut_layer_idx = net_shape.get_layer_idx();
    if (0 <= curr_cut_layer_idx && curr_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[curr_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        spacing_pair_list.emplace_back(cut_layer.get_curr_x_spacing(), cut_layer.get_curr_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_prl_spacing(), cut_layer.get_curr_y_spacing());
        cut_spacing_map[curr_cut_layer_idx] = spacing_pair_list;
      }
    }
    int32_t below_cut_layer_idx = net_shape.get_layer_idx() - 1;
    if (0 <= below_cut_layer_idx && below_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[below_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        spacing_pair_list.emplace_back(cut_layer.get_below_x_spacing(), cut_layer.get_below_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_below_prl_spacing(), cut_layer.get_below_y_spacing());
        cut_spacing_map[below_cut_layer_idx] = spacing_pair_list;
      }
    }
    int32_t above_cut_layer_idx = net_shape.get_layer_idx() + 1;
    if (0 <= above_cut_layer_idx && above_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[above_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        spacing_pair_list.emplace_back(cut_layer.get_above_x_spacing(), cut_layer.get_above_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_above_prl_spacing(), cut_layer.get_above_y_spacing());
        cut_spacing_map[above_cut_layer_idx] = spacing_pair_list;
      }
    }
  }
  std::map<PANode*, std::set<Orientation>> node_orientation_map;
  for (auto& [cut_layer_idx, spacing_pair_list] : cut_spacing_map) {
    std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    int32_t below_routing_layer_idx = adjacent_routing_layer_idx_list.front();
    int32_t above_routing_layer_idx = adjacent_routing_layer_idx_list.back();
    RTUTIL.swapByASC(below_routing_layer_idx, above_routing_layer_idx);
    PlanarRect& cut_shape = layer_via_master_list[below_routing_layer_idx].front().get_cut_shape_list().front();
    int32_t cut_shape_half_x_span = cut_shape.getXSpan() / 2;
    int32_t cut_shape_half_y_span = cut_shape.getYSpan() / 2;
    std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
    for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
      int32_t enlarged_x_size = cut_shape_half_x_span;
      int32_t enlarged_y_size = cut_shape_half_y_span;
      if (need_enlarged) {
        // 膨胀size为 cut_shape_half_span + spacing
        enlarged_x_size += x_spacing;
        enlarged_y_size += y_spacing;
      }
      // 贴合的也不算违例
      enlarged_x_size -= 1;
      enlarged_y_size -= 1;
      PlanarRect space_enlarged_rect
          = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
      for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, pa_box.get_box_track_axis())) {
        if (!RTUTIL.exist(orientation_set, Orientation::kAbove) && !RTUTIL.exist(orientation_set, Orientation::kBelow)) {
          continue;
        }
        PANode& below_node = layer_node_map[below_routing_layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (RTUTIL.exist(below_node.get_neighbor_node_map(), Orientation::kAbove)) {
          node_orientation_map[&below_node].insert(Orientation::kAbove);
        }
        PANode& above_node = layer_node_map[above_routing_layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (RTUTIL.exist(above_node.get_neighbor_node_map(), Orientation::kBelow)) {
          node_orientation_map[&above_node].insert(Orientation::kBelow);
        }
      }
    }
  }
  return node_orientation_map;
}

#endif

#if 1  // exhibit

void PinAccessor::updateSummary(PAModel& pa_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, int32_t>& routing_access_point_num_map = summary.pa_summary.routing_access_point_num_map;
  std::map<AccessPointType, int32_t>& type_access_point_num_map = summary.pa_summary.type_access_point_num_map;
  int32_t& total_access_point_num = summary.pa_summary.total_access_point_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.pa_summary.routing_violation_num_map;
  int32_t& total_violation_num = summary.pa_summary.total_violation_num;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_access_point_num_map[routing_layer.get_layer_idx()] = 0;
    routing_violation_num_map[routing_layer.get_layer_idx()] = 0;
  }
  type_access_point_num_map = {{AccessPointType::kNone, 0}, {AccessPointType::kNoAccess, 0}, {AccessPointType::kTrackGrid, 0}};
  total_access_point_num = 0;
  total_violation_num = 0;

  for (auto& [net_idx, access_point_list] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_list) {
      routing_access_point_num_map[access_point->get_layer_idx()]++;
      type_access_point_num_map[access_point->get_type()]++;
      total_access_point_num++;
    }
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    total_violation_num++;
  }
}

void PinAccessor::printSummary(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, int32_t>& routing_access_point_num_map = summary.pa_summary.routing_access_point_num_map;
  std::map<AccessPointType, int32_t>& type_access_point_num_map = summary.pa_summary.type_access_point_num_map;
  int32_t& total_access_point_num = summary.pa_summary.total_access_point_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.pa_summary.routing_violation_num_map;
  int32_t& total_violation_num = summary.pa_summary.total_violation_num;

  fort::char_table routing_access_point_num_map_table;
  {
    routing_access_point_num_map_table << fort::header << "routing_layer"
                                       << "access_point_num"
                                       << "proportion" << fort::endr;
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
    type_access_point_num_map_table << fort::header << "type"
                                    << "access_point_num"
                                    << "proportion" << fort::endr;
    for (auto& [type, access_point_num] : type_access_point_num_map) {
      type_access_point_num_map_table << GetAccessPointTypeName()(type) << access_point_num
                                      << RTUTIL.getPercentage(access_point_num, total_access_point_num) << fort::endr;
    }
    type_access_point_num_map_table << fort::header << "Total" << total_access_point_num
                                    << RTUTIL.getPercentage(total_access_point_num, total_access_point_num) << fort::endr;
  }
  fort::char_table routing_violation_num_map_table;
  {
    routing_violation_num_map_table << fort::header << "routing_layer"
                                    << "violation_num"
                                    << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_violation_num_map_table << routing_layer.get_layer_name() << routing_violation_num_map[routing_layer.get_layer_idx()]
                                      << RTUTIL.getPercentage(routing_violation_num_map[routing_layer.get_layer_idx()], total_violation_num)
                                      << fort::endr;
    }
    routing_violation_num_map_table << fort::header << "Total" << total_violation_num
                                    << RTUTIL.getPercentage(total_violation_num, total_violation_num) << fort::endr;
  }
  RTUTIL.printTableList({routing_access_point_num_map_table, type_access_point_num_map_table, routing_violation_num_map_table});
}

void PinAccessor::outputPlanarPinCSV(PAModel& pa_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
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

void PinAccessor::outputLayerPinCSV(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
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

  int32_t point_size = 5;

  GPGDS gp_gds;

  // gcell_axis
  {
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
  }

  // track_axis_struct
  {
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
  }

  // fixed_rect
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

  // net_access_result
  for (auto& [net_idx, segment_set] : RTDM.getNetAccessResultMap(die)) {
    GPStruct access_result_struct(RTUTIL.getString("access_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        access_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(access_result_struct);
  }

  // net_access_patch
  for (auto& [net_idx, patch_set] : RTDM.getNetAccessPatchMap(die)) {
    GPStruct access_patch_struct(RTUTIL.getString("access_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      access_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(access_patch_struct);
  }

  // access_point
  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    GPStruct access_point_struct(RTUTIL.getString("access_point(net_", net_idx, ")"));
    for (AccessPoint* access_point : access_point_set) {
      int32_t x = access_point->get_real_x();
      int32_t y = access_point->get_real_y();

      GPBoundary access_point_boundary;
      access_point_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(access_point->get_layer_idx()));
      access_point_boundary.set_data_type(static_cast<int32_t>(GPDataType::kAccessPoint));
      access_point_boundary.set_rect(x - point_size, y - point_size, x + point_size, y + point_size);
      access_point_struct.push(access_point_boundary);
    }
    gp_gds.addStruct(access_point_struct);
  }

  // violation
  {
    for (Violation* violation : RTDM.getViolationSet(die)) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation->get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      violation_struct.push(gp_boundary);
      gp_gds.addStruct(violation_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(pa_temp_directory_path, flag, "_pa_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

void PinAccessor::debugCheckPABox(PABox& pa_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  PABoxId& pa_box_id = pa_box.get_pa_box_id();
  if (pa_box_id.get_x() < 0 || pa_box_id.get_y() < 0) {
    RTLOG.error(Loc::current(), "The grid coord is illegal!");
  }

  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  for (GridMap<PANode>& pa_node_map : layer_node_map) {
    for (int32_t x = 0; x < pa_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < pa_node_map.get_y_size(); y++) {
        PANode& pa_node = pa_node_map[x][y];
        if (!RTUTIL.isInside(pa_box.get_box_rect().get_real_rect(), pa_node.get_planar_coord())) {
          RTLOG.error(Loc::current(), "The pa_node is out of box!");
        }
        for (auto& [orient, neighbor] : pa_node.get_neighbor_node_map()) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
            RTLOG.error(Loc::current(), "The pa_node neighbor is not bidirectional!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orient] != &pa_node) {
            RTLOG.error(Loc::current(), "The pa_node neighbor is not bidirectional!");
          }
          if (RTUTIL.getOrientation(LayerCoord(pa_node), LayerCoord(*neighbor)) == orient) {
            continue;
          }
          RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
        }
      }
    }
  }

  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    if (pa_task->get_net_idx() < 0) {
      RTLOG.error(Loc::current(), "The idx of origin net is illegal!");
    }
    for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
      if (pa_group.get_coord_list().empty()) {
        RTLOG.error(Loc::current(), "The coord_list is empty!");
      }
      for (LayerCoord& coord : pa_group.get_coord_list()) {
        int32_t layer_idx = coord.get_layer_idx();
        if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
          RTLOG.error(Loc::current(), "The layer idx of group coord is illegal!");
        }
        if (!RTUTIL.existTrackGrid(coord, pa_box.get_box_track_axis())) {
          RTLOG.error(Loc::current(), "There is no grid coord for real coord(", coord.get_x(), ",", coord.get_y(), ")!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, pa_box.get_box_track_axis());
        PANode& pa_node = layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (pa_node.get_neighbor_node_map().empty()) {
          RTLOG.error(Loc::current(), "The neighbor of group coord (", coord.get_x(), ",", coord.get_y(), ",", layer_idx,
                      ") is empty in box(", pa_box_id.get_x(), ",", pa_box_id.get_y(), ")");
        }
        if (RTUTIL.isInside(pa_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        RTLOG.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
  }
}

void PinAccessor::debugPlotPABox(PABox& pa_box, int32_t curr_task_idx, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;

  PlanarRect box_rect = pa_box.get_box_rect().get_real_rect();

  int32_t point_size = 5;

  GPGDS gp_gds;

  // base_region
  {
    GPStruct base_region_struct("base_region");
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(box_rect);
    base_region_struct.push(gp_boundary);
    gp_gds.addStruct(base_region_struct);
  }

  // gcell_axis
  {
    GPStruct gcell_axis_struct("gcell_axis");
    for (int32_t x : RTUTIL.getScaleList(box_rect.get_ll_x(), box_rect.get_ur_x(), gcell_axis.get_x_grid_list())) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(x, box_rect.get_ll_y(), x, box_rect.get_ur_y());
      gcell_axis_struct.push(gp_path);
    }
    for (int32_t y : RTUTIL.getScaleList(box_rect.get_ll_y(), box_rect.get_ur_y(), gcell_axis.get_y_grid_list())) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(box_rect.get_ll_x(), y, box_rect.get_ur_x(), y);
      gcell_axis_struct.push(gp_path);
    }
    gp_gds.addStruct(gcell_axis_struct);
  }

  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  // pa_node_map
  {
    GPStruct pa_node_map_struct("pa_node_map");
    for (GridMap<PANode>& pa_node_map : layer_node_map) {
      for (int32_t grid_x = 0; grid_x < pa_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < pa_node_map.get_y_size(); grid_y++) {
          PANode& pa_node = pa_node_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getEnlargedRect(pa_node.get_planar_coord(), point_size);
          int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
          int32_t y = real_rect.get_ur_y();

          GPBoundary gp_boundary;
          switch (pa_node.get_state()) {
            case PANodeState::kNone:
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kNone));
              break;
            case PANodeState::kOpen:
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kOpen));
              break;
            case PANodeState::kClose:
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kClose));
              break;
            default:
              RTLOG.error(Loc::current(), "The type is error!");
              break;
          }
          gp_boundary.set_rect(real_rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
          pa_node_map_struct.push(gp_boundary);

          y -= y_reduced_span;
          GPText gp_text_node_real_coord;
          gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_real_coord.set_message(
              RTUTIL.getString("(", pa_node.get_x(), " , ", pa_node.get_y(), " , ", pa_node.get_layer_idx(), ")"));
          gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
          gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          pa_node_map_struct.push(gp_text_node_real_coord);

          y -= y_reduced_span;
          GPText gp_text_node_grid_coord;
          gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", pa_node.get_layer_idx(), ")"));
          gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
          gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          pa_node_map_struct.push(gp_text_node_grid_coord);

          y -= y_reduced_span;
          GPText gp_text_orient_fixed_rect_map;
          gp_text_orient_fixed_rect_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_fixed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_fixed_rect_map.set_message("orient_fixed_rect_map: ");
          gp_text_orient_fixed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
          gp_text_orient_fixed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
          pa_node_map_struct.push(gp_text_orient_fixed_rect_map);

          if (!pa_node.get_orient_fixed_rect_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_fixed_rect_map_info;
            gp_text_orient_fixed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_fixed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_fixed_rect_map_info_message = "--";
            for (auto& [orient, net_set] : pa_node.get_orient_fixed_rect_map()) {
              orient_fixed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
              for (int32_t net_idx : net_set) {
                orient_fixed_rect_map_info_message += RTUTIL.getString(",", net_idx);
              }
              orient_fixed_rect_map_info_message += RTUTIL.getString(")");
            }
            gp_text_orient_fixed_rect_map_info.set_message(orient_fixed_rect_map_info_message);
            gp_text_orient_fixed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
            gp_text_orient_fixed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            pa_node_map_struct.push(gp_text_orient_fixed_rect_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_orient_routed_rect_map;
          gp_text_orient_routed_rect_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_routed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_routed_rect_map.set_message("orient_routed_rect_map: ");
          gp_text_orient_routed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
          gp_text_orient_routed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
          pa_node_map_struct.push(gp_text_orient_routed_rect_map);

          if (!pa_node.get_orient_routed_rect_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_routed_rect_map_info;
            gp_text_orient_routed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_routed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_routed_rect_map_info_message = "--";
            for (auto& [orient, net_set] : pa_node.get_orient_routed_rect_map()) {
              orient_routed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
              for (int32_t net_idx : net_set) {
                orient_routed_rect_map_info_message += RTUTIL.getString(",", net_idx);
              }
              orient_routed_rect_map_info_message += RTUTIL.getString(")");
            }
            gp_text_orient_routed_rect_map_info.set_message(orient_routed_rect_map_info_message);
            gp_text_orient_routed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
            gp_text_orient_routed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            pa_node_map_struct.push(gp_text_orient_routed_rect_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_orient_violation_number_map;
          gp_text_orient_violation_number_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_violation_number_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_violation_number_map.set_message("orient_violation_number_map: ");
          gp_text_orient_violation_number_map.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
          gp_text_orient_violation_number_map.set_presentation(GPTextPresentation::kLeftMiddle);
          pa_node_map_struct.push(gp_text_orient_violation_number_map);

          if (!pa_node.get_orient_violation_number_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_violation_number_map_info;
            gp_text_orient_violation_number_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_violation_number_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_violation_number_map_info_message = "--";
            for (auto& [orient, violation_number] : pa_node.get_orient_violation_number_map()) {
              orient_violation_number_map_info_message
                  += RTUTIL.getString("(", GetOrientationName()(orient), ",", violation_number != 0, ")");
            }
            gp_text_orient_violation_number_map_info.set_message(orient_violation_number_map_info_message);
            gp_text_orient_violation_number_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
            gp_text_orient_violation_number_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            pa_node_map_struct.push(gp_text_orient_violation_number_map_info);
          }
        }
      }
    }
    gp_gds.addStruct(pa_node_map_struct);
  }

  // neighbor_map
  {
    GPStruct neighbor_map_struct("neighbor_map");
    for (GridMap<PANode>& pa_node_map : layer_node_map) {
      for (int32_t grid_x = 0; grid_x < pa_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < pa_node_map.get_y_size(); grid_y++) {
          PANode& pa_node = pa_node_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getEnlargedRect(pa_node.get_planar_coord(), point_size);

          int32_t ll_x = real_rect.get_ll_x();
          int32_t ll_y = real_rect.get_ll_y();
          int32_t ur_x = real_rect.get_ur_x();
          int32_t ur_y = real_rect.get_ur_y();
          int32_t mid_x = (ll_x + ur_x) / 2;
          int32_t mid_y = (ll_y + ur_y) / 2;
          int32_t x_reduced_span = (ur_x - ll_x) / 4;
          int32_t y_reduced_span = (ur_y - ll_y) / 4;

          for (auto& [orientation, neighbor_node] : pa_node.get_neighbor_node_map()) {
            GPPath gp_path;
            switch (orientation) {
              case Orientation::kEast:
                gp_path.set_segment(ur_x - x_reduced_span, mid_y, ur_x, mid_y);
                break;
              case Orientation::kSouth:
                gp_path.set_segment(mid_x, ll_y, mid_x, ll_y + y_reduced_span);
                break;
              case Orientation::kWest:
                gp_path.set_segment(ll_x, mid_y, ll_x + x_reduced_span, mid_y);
                break;
              case Orientation::kNorth:
                gp_path.set_segment(mid_x, ur_y - y_reduced_span, mid_x, ur_y);
                break;
              case Orientation::kAbove:
                gp_path.set_segment(ur_x - x_reduced_span, ur_y - y_reduced_span, ur_x, ur_y);
                break;
              case Orientation::kBelow:
                gp_path.set_segment(ll_x, ll_y, ll_x + x_reduced_span, ll_y + y_reduced_span);
                break;
              default:
                RTLOG.error(Loc::current(), "The orientation is oblique!");
                break;
            }
            gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(pa_node.get_layer_idx()));
            gp_path.set_width(std::min(x_reduced_span, y_reduced_span) / 2);
            gp_path.set_data_type(static_cast<int32_t>(GPDataType::kNeighbor));
            neighbor_map_struct.push(gp_path);
          }
        }
      }
    }
    gp_gds.addStruct(neighbor_map_struct);
  }

  // box_track_axis
  {
    GPStruct box_track_axis_struct("box_track_axis");
    PlanarCoord& real_ll = box_rect.get_ll();
    PlanarCoord& real_ur = box_rect.get_ur();
    ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
    std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
    std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());
    for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
      for (int32_t x : x_list) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
        gp_path.set_segment(x, real_ll.get_y(), x, real_ur.get_y());
        gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
        box_track_axis_struct.push(gp_path);
      }
      for (int32_t y : y_list) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
        gp_path.set_segment(real_ll.get_x(), y, real_ur.get_x(), y);
        gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
        box_track_axis_struct.push(gp_path);
      }
    }
    gp_gds.addStruct(box_track_axis_struct);
  }

  // fixed_rect
  for (auto& [is_routing, layer_net_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_rect_map] : layer_net_rect_map) {
      for (auto& [net_idx, rect_set] : net_rect_map) {
        GPStruct fixed_rect_struct(RTUTIL.getString("fixed_rect(net_", net_idx, ")"));
        for (EXTLayerRect* rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(rect->get_real_rect());
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

  // net_access_result
  for (auto& [net_idx, segment_set] : pa_box.get_net_access_result_map()) {
    GPStruct access_result_struct(RTUTIL.getString("access_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        access_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(access_result_struct);
  }

  // net_access_patch
  for (auto& [net_idx, patch_set] : pa_box.get_net_access_patch_map()) {
    GPStruct access_patch_struct(RTUTIL.getString("access_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      access_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(access_patch_struct);
  }

  // task
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", pa_task->get_net_idx(), ")"));

    if (curr_task_idx == -1 || pa_task->get_net_idx() == curr_task_idx) {
      for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
        for (LayerCoord& coord : pa_group.get_coord_list()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kKey));
          gp_boundary.set_rect(RTUTIL.getEnlargedRect(coord, point_size));
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(coord.get_layer_idx()));
          task_struct.push(gp_boundary);
        }
      }
    }
    {
      // bounding_box
      GPBoundary gp_boundary;
      gp_boundary.set_layer_idx(0);
      gp_boundary.set_data_type(2);
      gp_boundary.set_rect(pa_task->get_bounding_box());
      task_struct.push(gp_boundary);
    }
    for (Segment<LayerCoord>& segment : pa_box.get_net_task_result_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
      LayerCoord first_coord = segment.get_first();
      LayerCoord second_coord = segment.get_second();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      int32_t half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
        gp_boundary.set_rect(RTUTIL.getEnlargedRect(first_coord, second_coord, half_width));
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(first_layer_idx));
        task_struct.push(gp_boundary);
      } else {
        RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          ViaMaster& via_master = layer_via_master_list[layer_idx].front();

          LayerRect& above_enclosure = via_master.get_above_enclosure();
          LayerRect offset_above_enclosure(RTUTIL.getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
          GPBoundary above_boundary;
          above_boundary.set_rect(offset_above_enclosure);
          above_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
          above_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          task_struct.push(above_boundary);

          LayerRect& below_enclosure = via_master.get_below_enclosure();
          LayerRect offset_below_enclosure(RTUTIL.getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
          GPBoundary below_boundary;
          below_boundary.set_rect(offset_below_enclosure);
          below_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
          below_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          task_struct.push(below_boundary);

          for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
            LayerRect offset_cut_shape(RTUTIL.getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
            GPBoundary cut_boundary;
            cut_boundary.set_rect(offset_cut_shape);
            cut_boundary.set_layer_idx(RTGP.getGDSIdxByCut(via_master.get_cut_layer_idx()));
            cut_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
            task_struct.push(cut_boundary);
          }
        }
      }
    }
    for (EXTLayerRect& patch : pa_box.get_net_task_patch_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
      GPBoundary gp_boundary;
      gp_boundary.set_rect(patch.get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch.get_layer_idx()));
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
      task_struct.push(gp_boundary);
    }
    gp_gds.addStruct(task_struct);
  }

  // violation
  {
    for (Violation& violation : pa_box.get_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      violation_struct.push(gp_boundary);
      gp_gds.addStruct(violation_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(pa_temp_directory_path, flag, "_pa_box_", pa_box.get_pa_box_id().get_x(), "_",
                                               pa_box.get_pa_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
