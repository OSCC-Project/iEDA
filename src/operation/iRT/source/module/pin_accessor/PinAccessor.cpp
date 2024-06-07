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

#include "GDSPlotter.hpp"
#include "GPGDS.hpp"

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
  buildAccessPointList(pa_model);
  uploadAccessPoint(pa_model);
  // debugPlotPAModel(pa_model, "before_eliminate");
  buildConflictGroupList(pa_model);
  eliminateConflict(pa_model);
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
  RTLOG.warn(Loc::current(), "The pin ", pin->get_pin_name(), " without legal shape!");
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
    int32_t reduced_size = routing_layer_list[curr_layer_idx].get_min_width() / 2;
    for (PlanarRect& real_rect : RTUTIL.getClosedReducedRectListByBoost(origin_pin_shape_list, reduced_size)) {
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

void PinAccessor::buildAccessPointList(PAModel& pa_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();

#pragma omp parallel for
  for (PANet& pa_net : pa_net_list) {
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
      }
    }
  }
}

void PinAccessor::uploadAccessPoint(PAModel& pa_model)
{
  // 更新到gcell_map
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        RTDM.updateAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_point);
      }
    }
  }
}

void PinAccessor::buildConflictGroupList(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<ConflictGroup>& conflict_group_list = pa_model.get_conflict_group_list();

  std::map<PAPin*, std::set<PAPin*>> pin_conflict_map;
  for (auto& [curr_pin, conflict_pin_set] : getPinConlictMap(pa_model)) {
    pin_conflict_map[curr_pin] = conflict_pin_set;
  }
  for (auto& [curr_pin, conflict_pin_set] : pin_conflict_map) {
    if (conflict_pin_set.empty()) {
      continue;
    }
    std::vector<std::pair<PAPin*, PAPin*>> conflict_list;
    std::map<PAPin*, int32_t> pin_idx_map;
    std::queue<PAPin*> pin_queue = RTUTIL.initQueue(curr_pin);
    while (!pin_queue.empty()) {
      PAPin* pa_pin = RTUTIL.getFrontAndPop(pin_queue);
      if (!RTUTIL.exist(pin_idx_map, pa_pin)) {
        pin_idx_map[pa_pin] = pin_idx_map.size();
      }
      if (!RTUTIL.exist(pin_conflict_map, pa_pin)) {
        continue;
      }
      std::set<PAPin*>& conflict_pin_set = pin_conflict_map[pa_pin];
      for (PAPin* conflict_pin : conflict_pin_set) {
        conflict_list.emplace_back(pa_pin, conflict_pin);
        pin_queue.push(conflict_pin);
      }
      conflict_pin_set.clear();
    }
    ConflictGroup conflict_group;
    std::vector<std::vector<ConflictAccessPoint>>& conflict_ap_list_list = conflict_group.get_conflict_ap_list_list();
    conflict_ap_list_list.resize(pin_idx_map.size());
    for (auto& [pa_pin, conflict_ap_list_idx] : pin_idx_map) {
      std::vector<ConflictAccessPoint> conflict_ap_list;
      for (AccessPoint& access_point : pa_pin->get_access_point_list()) {
        ConflictAccessPoint conflict_ap;
        conflict_ap.set_pa_pin(pa_pin);
        conflict_ap.set_access_point(&access_point);
        conflict_ap.set_coord(access_point.get_real_coord());
        conflict_ap_list.push_back(conflict_ap);
      }
      conflict_ap_list_list[conflict_ap_list_idx] = conflict_ap_list;
    }
    std::map<int32_t, std::vector<int32_t>>& conflict_map = conflict_group.get_conflict_map();
    for (std::pair<PAPin*, PAPin*>& conflict_pair : conflict_list) {
      conflict_map[pin_idx_map[conflict_pair.first]].push_back(pin_idx_map[conflict_pair.second]);
    }
    conflict_group_list.push_back(conflict_group);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<std::pair<PAPin*, std::set<PAPin*>>> PinAccessor::getPinConlictMap(PAModel& pa_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  Die& die = RTDM.getDatabase().get_die();

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();

  std::vector<std::pair<PAPin*, std::set<PAPin*>>> pin_conflict_list;
  for (PANet& pa_net : pa_net_list) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      pin_conflict_list.emplace_back(&pa_pin, std::set<PAPin*>{});
    }
  }
#pragma omp parallel for
  for (std::pair<PAPin*, std::set<PAPin*>>& pin_conflict_pair : pin_conflict_list) {
    PAPin* pa_pin = pin_conflict_pair.first;
    std::set<PAPin*>& conflict_pin_set = pin_conflict_pair.second;

    for (AccessPoint& access_point : pa_pin->get_access_point_list()) {
      PlanarCoord& grid_coord = access_point.get_grid_coord();
      for (int32_t x : {grid_coord.get_x() - 1, grid_coord.get_x(), grid_coord.get_x() + 1}) {
        for (int32_t y : {grid_coord.get_y() - 1, grid_coord.get_y(), grid_coord.get_y() + 1}) {
          if (!RTUTIL.isInside(die.get_grid_rect(), PlanarCoord(x, y))) {
            continue;
          }
          for (auto& [net_idx, access_point_set] : gcell_map[x][y].get_net_access_point_map()) {
            for (AccessPoint* gcell_access_point : access_point_set) {
              PAPin* gcell_pin = &pa_net_list[net_idx].get_pa_pin_list()[gcell_access_point->get_pin_idx()];
              if (gcell_pin == pa_pin) {
                continue;
              }
              if (hasConflict(pa_model, access_point, *gcell_access_point)) {
                conflict_pin_set.insert(gcell_pin);
              }
            }
          }
        }
      }
    }
  }
  return pin_conflict_list;
}

bool PinAccessor::hasConflict(PAModel& pa_model, AccessPoint& curr_access_point, AccessPoint& gcell_access_point)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();

  std::set<int32_t> conflict_layer_idx_set;
  {
    int32_t start_layer_idx = curr_access_point.get_layer_idx();
    int32_t end_layer_idx = gcell_access_point.get_layer_idx();
    RTUTIL.swapByASC(start_layer_idx, end_layer_idx);
    for (int32_t layer_idx = start_layer_idx; layer_idx <= end_layer_idx; layer_idx++) {
      if (layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
        conflict_layer_idx_set.insert(layer_idx);
        conflict_layer_idx_set.insert(layer_idx + 1);
      } else {
        conflict_layer_idx_set.insert(layer_idx);
        conflict_layer_idx_set.insert(layer_idx - 1);
      }
    }
  }
  int32_t x_searched_distance = 0;
  int32_t y_searched_distance = 0;
  for (int32_t conflict_layer_idx : conflict_layer_idx_set) {
    PlanarRect& enclosure = layer_enclosure_map[conflict_layer_idx];
    RoutingLayer& routing_layer = routing_layer_list[conflict_layer_idx];
    int32_t min_spacing = routing_layer.getMinSpacing(enclosure);
    int32_t x_distance = enclosure.getXSpan() + min_spacing;
    int32_t y_distance = enclosure.getYSpan() + min_spacing;

    x_searched_distance = std::max(x_searched_distance, x_distance);
    y_searched_distance = std::max(y_searched_distance, y_distance);
  }
  PlanarCoord& curr_real_coord = curr_access_point.get_real_coord();
  int32_t left = curr_real_coord.get_x() - x_searched_distance;
  int32_t right = curr_real_coord.get_x() + x_searched_distance;
  int32_t bottom = curr_real_coord.get_y() - y_searched_distance;
  int32_t top = curr_real_coord.get_y() + y_searched_distance;
  PlanarCoord& gcell_real_coord = gcell_access_point.get_real_coord();
  if (left <= gcell_real_coord.get_x() && gcell_real_coord.get_x() <= right) {
    if (bottom <= gcell_real_coord.get_y() && gcell_real_coord.get_y() <= top) {
      return true;
    }
  }
  return false;
}

void PinAccessor::eliminateConflict(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  for (ConflictGroup& conflict_group : pa_model.get_conflict_group_list()) {
    for (ConflictAccessPoint& best_point : getBestPointList(conflict_group)) {
      best_point.get_pa_pin()->set_key_access_point(*best_point.get_access_point());
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      // 将没有conflict的pin设置key
      if (pa_pin.get_key_access_point().get_layer_idx() < 0) {
        pa_pin.set_key_access_point(pa_pin.get_access_point_list().front());
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<ConflictAccessPoint> PinAccessor::getBestPointList(ConflictGroup& conflict_group)
{
  std::vector<std::vector<ConflictAccessPoint>>& conflict_ap_list_list = conflict_group.get_conflict_ap_list_list();
  std::map<int32_t, std::vector<int32_t>>& conflict_map = conflict_group.get_conflict_map();

  std::vector<ConflictAccessPoint> curr_ap_list;
  for (std::vector<ConflictAccessPoint>& conflict_ap_list : conflict_ap_list_list) {
    curr_ap_list.push_back(conflict_ap_list.front());
  }
  bool improved = true;
  while (improved) {
    improved = false;
    for (int32_t i = 0; i < static_cast<int32_t>(conflict_ap_list_list.size()); ++i) {
      std::vector<int32_t> conflict_j_list;
      if (RTUTIL.exist(conflict_map, i)) {
        conflict_j_list = conflict_map[i];
      } else {
        RTLOG.error(Loc::current(), "The conflict_map is not exist i!");
      }
      int32_t max_min_distance = INT32_MIN;
      ConflictAccessPoint best_ap = curr_ap_list[i];
      for (ConflictAccessPoint& conflict_ap : conflict_ap_list_list[i]) {
        int32_t min_distance = INT32_MAX;
        for (int32_t j : conflict_j_list) {
          min_distance = std::min(min_distance, RTUTIL.getManhattanDistance(conflict_ap, curr_ap_list[j]));
        }
        if (max_min_distance < min_distance) {
          max_min_distance = min_distance;
          best_ap = conflict_ap;
        }
      }
      if (best_ap.get_access_point() != curr_ap_list[i].get_access_point()) {
        curr_ap_list[i] = best_ap;
        improved = true;
      }
    }
  }
  return curr_ap_list;
}

void PinAccessor::updatePAModel(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      Pin& origin_pin = pa_net.get_origin_net()->get_pin_list()[pa_pin.get_pin_idx()];
      if (origin_pin.get_pin_idx() != pa_pin.get_pin_idx()) {
        RTLOG.error(Loc::current(), "The pin idx is not equal!");
      }
      origin_pin.set_key_access_point(pa_pin.get_key_access_point());
      RTDM.updateAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &origin_pin.get_key_access_point());
    }
    pa_net.get_origin_net()->set_bounding_box(pa_net.get_bounding_box());
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

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
