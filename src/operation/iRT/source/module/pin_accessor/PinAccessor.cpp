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
#include "PAComParam.hpp"
#include "PAIterParam.hpp"
#include "PANet.hpp"
#include "PANode.hpp"
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
  setPAComParam(pa_model);
  initAccessPointList(pa_model);
  uploadAccessPointList(pa_model);
  // debugPlotPAModel(pa_model, "init");
  routePAModel(pa_model);
  uploadAccessPoint(pa_model);
  uploadAccessResult(pa_model);
  uploadAccessPatch(pa_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

PinAccessor* PinAccessor::_pa_instance = nullptr;

PAModel PinAccessor::initPAModel()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  PAModel pa_model;
  pa_model.set_pa_net_list(convertToPANetList(net_list));
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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
  pa_net.get_pa_pin_list().reserve(net.get_pin_list().size());
  for (Pin& pin : net.get_pin_list()) {
    pa_net.get_pa_pin_list().push_back(PAPin(pin));
  }
  pa_net.set_bounding_box(net.get_bounding_box());
  return pa_net;
}

void PinAccessor::setPAComParam(PAModel& pa_model)
{
  /**
   * max_candidate_point_num, extra_via_master_num, ap_per_via_master
   */
  PAComParam pa_com_param(15, 2, 8);
  RTLOG.info(Loc::current(), "max_candidate_point_num: ", pa_com_param.get_max_candidate_point_num());
  RTLOG.info(Loc::current(), "extra_via_master_num: ", pa_com_param.get_extra_via_master_num());
  RTLOG.info(Loc::current(), "ap_per_via_master: ", pa_com_param.get_ap_per_via_master());
  pa_model.set_pa_com_param(pa_com_param);
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
  // try default via master only
  updateAccessPointList(pa_model, net_pin_pair_list, false);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::updateAccessPointList(PAModel& pa_model, std::vector<std::pair<int32_t, PAPin*>>& net_pin_pair_list, bool enable_via_candidate)
{

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t detection_distance = RTDM.getDatabase().get_detection_distance();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  for (auto& [net_idx, pa_pin] : net_pin_pair_list) {
    for (AccessPoint& access_point : pa_pin->get_access_point_list()) {
      delCurrAccessPoint(pa_model, net_idx, &access_point);
      RTDM.updateNetAccessPointToGCellMap(ChangeType::kDel, net_idx, &access_point);
    }
  }

  std::map<int32_t, std::vector<ViaMaster*>> selected_via_master_list_map;
  if (enable_via_candidate) {
    for (RoutingLayer& routing_layer : routing_layer_list) {
      selected_via_master_list_map[routing_layer.get_layer_idx()] = getSelectedViaMasterList(pa_model, routing_layer.get_layer_idx());
    }
  }
  std::map<int32_t, std::vector<ViaMaster*>> empty_via_master_list_map;
#pragma omp parallel for
  for (std::pair<int32_t, PAPin*>& net_pin_pair : net_pin_pair_list) {
    PAPin* pa_pin = net_pin_pair.second;
    pa_pin->get_access_point_list().clear();
    pa_pin->get_grid_coord_set().clear();
    pa_pin->get_pin_shape_coord_list().clear();
    pa_pin->get_target_coord_list().clear();

    std::vector<AccessPoint>& access_point_list = pa_pin->get_access_point_list();
    bool via_candidate_generated = false;
    if (enable_via_candidate && pa_pin->get_is_core()) {
      std::vector<PALegalShape> legal_shape_list = getLegalShapeList(pa_model, net_pin_pair.first, pa_pin, selected_via_master_list_map);
      for (AccessPoint& access_point : getAccessPointList(pa_model, pa_pin->get_pin_idx(), legal_shape_list)) {
        access_point_list.push_back(access_point);
      }
      via_candidate_generated = !access_point_list.empty();
    }
    if (!via_candidate_generated) {
      std::vector<PALegalShape> legal_shape_list = getLegalShapeList(pa_model, net_pin_pair.first, pa_pin, empty_via_master_list_map);
      for (AccessPoint& access_point : getAccessPointList(pa_model, pa_pin->get_pin_idx(), legal_shape_list)) {
        access_point_list.push_back(access_point);
      }
    }
    std::sort(access_point_list.begin(), access_point_list.end(),
              [](AccessPoint& a, AccessPoint& b) { return CmpLayerCoordByXASC()(a.getRealLayerCoord(), b.getRealLayerCoord()); });
    if (access_point_list.empty()) {
      RTLOG.error(Loc::current(), RTUTIL.getString("No access point was generated! Pin ", pa_pin->get_pin_name()));
    }
    for (AccessPoint& access_point : pa_pin->get_access_point_list()) {
      pa_pin->get_pin_shape_coord_list().push_back(access_point.getRealLayerCoord());
    }
    std::vector<LayerCoord> coord_list;
    for (AccessPoint& access_point : pa_pin->get_access_point_list()) {
      int32_t curr_layer_idx = access_point.get_layer_idx();
      // 构建目标层
      std::vector<int32_t> point_layer_idx_list;
      if (pa_pin->get_is_core()) {
        if (curr_layer_idx < bottom_routing_layer_idx) {
          point_layer_idx_list.push_back(bottom_routing_layer_idx + 1);
        } else if (top_routing_layer_idx < curr_layer_idx) {
          point_layer_idx_list.push_back(top_routing_layer_idx - 1);
        } else if (curr_layer_idx < top_routing_layer_idx) {
          point_layer_idx_list.push_back(curr_layer_idx + 1);
        } else {
          point_layer_idx_list.push_back(curr_layer_idx - 1);
        }
      } else {
        if (curr_layer_idx < bottom_routing_layer_idx) {
          point_layer_idx_list.push_back(bottom_routing_layer_idx);
        } else if (top_routing_layer_idx < curr_layer_idx) {
          point_layer_idx_list.push_back(top_routing_layer_idx);
        } else if (curr_layer_idx < top_routing_layer_idx) {
          point_layer_idx_list.push_back(curr_layer_idx);
        } else {
          point_layer_idx_list.push_back(curr_layer_idx);
        }
      }
      // 构建搜索形状
      PlanarRect real_rect = RTUTIL.getEnlargedRect(access_point.get_real_coord(), detection_distance);
      // 构建点
      std::vector<ScaleGrid>& x_track_grid_list = routing_layer_list[curr_layer_idx].getXTrackGridList();
      std::vector<ScaleGrid>& y_track_grid_list = routing_layer_list[curr_layer_idx].getYTrackGridList();
      std::vector<int32_t> x_scale_list = RTUTIL.getScaleList(real_rect.get_ll_x(), real_rect.get_ur_x(), x_track_grid_list);
      std::vector<int32_t> y_scale_list = RTUTIL.getScaleList(real_rect.get_ll_y(), real_rect.get_ur_y(), y_track_grid_list);
      for (int32_t x : x_scale_list) {
        for (int32_t y : y_scale_list) {
          for (int32_t point_layer_idx : point_layer_idx_list) {
            coord_list.emplace_back(x, y, point_layer_idx);
          }
        }
      }
    }
    std::sort(coord_list.begin(), coord_list.end(), CmpLayerCoordByXASC());
    coord_list.erase(std::unique(coord_list.begin(), coord_list.end()), coord_list.end());
    for (LayerCoord& coord : coord_list) {
      pa_pin->get_target_coord_list().push_back(coord);
    }
  }
}

std::vector<PALegalShape> PinAccessor::getLegalShapeList(PAModel& pa_model, int32_t net_idx, PAPin* pa_pin,
                                                         const std::map<int32_t, std::vector<ViaMaster*>>& selected_via_master_list_map)
{
  std::map<int32_t, std::vector<EXTLayerRect>> routing_pin_shape_map;
  for (EXTLayerRect& routing_shape : pa_pin->get_routing_shape_list()) {
    routing_pin_shape_map[routing_shape.get_layer_idx()].emplace_back(routing_shape);
  }
  std::vector<int32_t> routing_layer_idx_list;
  routing_layer_idx_list.reserve(routing_pin_shape_map.size());
  for (auto& routing_pin_shape_pair : routing_pin_shape_map) {
    routing_layer_idx_list.push_back(routing_pin_shape_pair.first);
  }
  if (pa_pin->get_is_core() || !selected_via_master_list_map.empty()) {
    std::sort(routing_layer_idx_list.begin(), routing_layer_idx_list.end(), [](int32_t a, int32_t b) { return a > b; });
  } else {
    std::sort(routing_layer_idx_list.begin(), routing_layer_idx_list.end(),
              [](int32_t a, int32_t b) { return (a % 2 != 0 && b % 2 == 0) || (a % 2 == b % 2 && a > b); });
  }
  std::vector<PALegalShape> legal_shape_list;
  for (int32_t routing_layer_idx : routing_layer_idx_list) {
    std::vector<EXTLayerRect>& pin_shape_list = routing_pin_shape_map.at(routing_layer_idx);
    if (!selected_via_master_list_map.empty()) {
      auto iter = selected_via_master_list_map.find(routing_layer_idx);
      if (iter == selected_via_master_list_map.end()) {
        continue;
      }
      for (ViaMaster* via_master : iter->second) {
        for (PALegalShape& legal_shape : getPlanarLegalShapeList(pa_model, net_idx, pa_pin, pin_shape_list, via_master)) {
          legal_shape_list.push_back(legal_shape);
        }
      }
    } else {
      for (PALegalShape& legal_shape : getPlanarLegalShapeList(pa_model, net_idx, pa_pin, pin_shape_list, nullptr)) {
        legal_shape_list.push_back(legal_shape);
      }
    }
    if (!legal_shape_list.empty()) {
      break;
    }
  }
  if (!legal_shape_list.empty() || !selected_via_master_list_map.empty()) {
    return legal_shape_list;
  }
  legal_shape_list.reserve(pa_pin->get_routing_shape_list().size());
  for (EXTLayerRect& routing_shape : pa_pin->get_routing_shape_list()) {
    legal_shape_list.push_back({routing_shape.getRealLayerRect(), ViaMasterIdx()});
  }
  return legal_shape_list;
}

std::vector<PALegalShape> PinAccessor::getPlanarLegalShapeList(PAModel& pa_model, int32_t curr_net_idx, PAPin* pa_pin,
                                                               std::vector<EXTLayerRect>& pin_shape_list, ViaMaster* via_master)
{
  (void) pa_model;
  (void) pa_pin;
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
  std::vector<PlanarRect> origin_pin_shape_list;
  origin_pin_shape_list.reserve(pin_shape_list.size());
  {
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      origin_pin_shape_list.push_back(pin_shape.get_real_rect());
    }
  }
  // 当前层缩小后的结果
  std::vector<EXTLayerRect> shrinked_rect_list;
  shrinked_rect_list.reserve(pin_shape_list.size());
  {
    PlanarRect enclosure = (via_master == nullptr ? layer_enclosure_map[curr_layer_idx] : getViaEnclosure(*via_master, curr_layer_idx));
    int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
    int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;
    int32_t half_min_width = routing_layer_list[curr_layer_idx].get_min_width() / 2;
    int32_t shrinked_x_size = std::max(half_min_width, enclosure_half_x_span);
    int32_t shrinked_y_size = std::max(half_min_width, enclosure_half_y_span);
    for (PlanarRect& real_rect :
         RTUTIL.getClosedShrinkedRectListByBoost(origin_pin_shape_list, shrinked_x_size, shrinked_y_size, shrinked_x_size, shrinked_y_size)) {
      EXTLayerRect shrinked_rect;
      shrinked_rect.set_real_rect(real_rect);
      shrinked_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(shrinked_rect.get_real_rect(), gcell_axis));
      shrinked_rect.set_layer_idx(curr_layer_idx);
      shrinked_rect_list.push_back(shrinked_rect);
    }
  }
  std::vector<int32_t> obs_layer_idx_list;
  if (via_master != nullptr) {
    int32_t below_layer_idx = via_master->get_below_enclosure().get_layer_idx();
    int32_t above_layer_idx = via_master->get_above_enclosure().get_layer_idx();
    if (curr_layer_idx == below_layer_idx) {
      obs_layer_idx_list = {curr_layer_idx, above_layer_idx};
    } else if (curr_layer_idx == above_layer_idx) {
      obs_layer_idx_list = {below_layer_idx, curr_layer_idx};
    } else {
      RTLOG.error(Loc::current(), "The via_master is not adjacent to curr_layer_idx!");
    }
  } else if (curr_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
    obs_layer_idx_list = {curr_layer_idx, curr_layer_idx + 1};
  } else {
    obs_layer_idx_list = {curr_layer_idx, curr_layer_idx - 1};
  }
  std::vector<PlanarRect> legal_rect_list;
  legal_rect_list.reserve(shrinked_rect_list.size());
  for (EXTLayerRect& shrinked_rect : shrinked_rect_list) {
    legal_rect_list.push_back(shrinked_rect.get_real_rect());
  }
  for (int32_t obs_layer_idx : obs_layer_idx_list) {
    RoutingLayer& routing_layer = routing_layer_list[obs_layer_idx];
    PlanarRect enclosure = (via_master == nullptr ? layer_enclosure_map[obs_layer_idx] : getViaEnclosure(*via_master, obs_layer_idx));
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
              int32_t prl_spacing = routing_layer.getPRLSpacing(fixed_rect->get_real_rect());
              int32_t enlarged_x_size = prl_spacing + enclosure_half_x_span;
              int32_t enlarged_y_size = prl_spacing + enclosure_half_y_span;
              PlanarRect enlarged_rect
                  = RTUTIL.getEnlargedRect(fixed_rect->get_real_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
              if (RTUTIL.isOpenOverlap(shrinked_rect.get_real_rect(), enlarged_rect)) {
                routing_obs_shape_list.push_back(enlarged_rect);
              }
              if (layer_idx != curr_layer_idx) {
                int32_t eol_x_spacing = routing_layer.get_eol_spacing();
                int32_t eol_y_spacing = routing_layer.get_eol_within();
                if (!routing_layer.isPreferH()) {
                  std::swap(eol_x_spacing, eol_y_spacing);
                }
                enlarged_x_size = eol_x_spacing + enclosure_half_x_span;
                enlarged_y_size = eol_y_spacing + enclosure_half_y_span;
                enlarged_rect = RTUTIL.getEnlargedRect(fixed_rect->get_real_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
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
      std::vector<PlanarRect> legal_rect_list_temp = RTUTIL.getClosedCuttingRectListByBoost(legal_rect_list, routing_obs_shape_list);
      if (!legal_rect_list_temp.empty()) {
        legal_rect_list = legal_rect_list_temp;
      } else {
        break;
      }
    }
  }
  std::vector<PALegalShape> legal_shape_list;
  ViaMasterIdx via_master_idx;
  if (via_master != nullptr) {
    via_master_idx = via_master->get_via_master_idx();
  }
  for (PlanarRect planar_legal_rect : RTUTIL.mergeRectListByBoost(legal_rect_list, Direction::kVertical)) {
    legal_shape_list.push_back({LayerRect(planar_legal_rect, curr_layer_idx), via_master_idx});
  }
  for (PlanarRect planar_legal_rect : RTUTIL.mergeRectListByBoost(legal_rect_list, Direction::kHorizontal)) {
    legal_shape_list.push_back({LayerRect(planar_legal_rect, curr_layer_idx), via_master_idx});
  }
  return legal_shape_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointList(PAModel& pa_model, int32_t pin_idx, std::vector<PALegalShape>& legal_shape_list)
{
  Die& die = RTDM.getDatabase().get_die();
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t ap_per_via_master = std::max(1, pa_model.get_pa_com_param().get_ap_per_via_master());
  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double violation_unit = 4 * non_prefer_wire_unit * cost_unit;

  PlanarRect die_valid_rect = die.get_real_rect();
  int32_t shrinked_size = cost_unit;
  if (RTUTIL.hasShrinkedRect(die_valid_rect, shrinked_size)) {
    die_valid_rect = RTUTIL.getShrinkedRect(die_valid_rect, shrinked_size);
  }

  struct CandidateAccessPoint
  {
    LayerCoord coord;
    ViaMasterIdx via_master_idx;
    int32_t track_num = 0;
    double init_cost = 0;
  };
  bool use_via_candidate = false;
  std::map<LayerCoord, int32_t, CmpLayerCoordByXASC> coord_track_num_map;
  std::map<std::pair<int32_t, int32_t>, std::set<LayerCoord, CmpLayerCoordByXASC>> via_coord_set_map;
  std::map<std::pair<int32_t, int32_t>, std::vector<CandidateAccessPoint>> via_candidate_list_map;
  for (PALegalShape& legal_shape : legal_shape_list) {
    LayerRect& shape = legal_shape.shape;
    int32_t ll_x = shape.get_ll_x();
    int32_t ll_y = shape.get_ll_y();
    int32_t ur_x = shape.get_ur_x();
    int32_t ur_y = shape.get_ur_y();
    int32_t curr_layer_idx = shape.get_layer_idx();
    while (ll_x % manufacture_grid != 0) {
      ll_x++;
    }
    while (ll_y % manufacture_grid != 0) {
      ll_y++;
    }
    while (ur_x % manufacture_grid != 0) {
      ur_x--;
    }
    while (ur_y % manufacture_grid != 0) {
      ur_y--;
    }
    if (ll_x > ur_x || ll_y > ur_y) {
      continue;
    }

    RoutingLayer& curr_routing_layer = routing_layer_list[curr_layer_idx];
    std::vector<int32_t> x_track_list = RTUTIL.getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList());
    std::vector<int32_t> y_track_list = RTUTIL.getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList());
    std::vector<int32_t> x_shape_list;
    x_shape_list.reserve(4);
    x_shape_list.emplace_back(ll_x);
    if ((ur_x - ll_x) / manufacture_grid % 2 == 0) {
      x_shape_list.emplace_back((ll_x + ur_x) / 2);
    } else {
      x_shape_list.emplace_back((ll_x + ur_x - manufacture_grid) / 2);
      x_shape_list.emplace_back((ll_x + ur_x + manufacture_grid) / 2);
    }
    x_shape_list.emplace_back(ur_x);
    std::vector<int32_t> y_shape_list;
    y_shape_list.reserve(4);
    y_shape_list.emplace_back(ll_y);
    if ((ur_y - ll_y) / manufacture_grid % 2 == 0) {
      y_shape_list.emplace_back((ll_y + ur_y) / 2);
    } else {
      y_shape_list.emplace_back((ll_y + ur_y - manufacture_grid) / 2);
      y_shape_list.emplace_back((ll_y + ur_y + manufacture_grid) / 2);
    }
    y_shape_list.emplace_back(ur_y);

    std::vector<LayerCoord> curr_coord_list;
    curr_coord_list.reserve(x_track_list.size() * y_track_list.size() + x_shape_list.size() * y_track_list.size()
                            + x_track_list.size() * y_shape_list.size() + x_shape_list.size() * y_shape_list.size());
    for (int32_t x : x_track_list) {
      for (int32_t y : y_track_list) {
        curr_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
    for (int32_t x : x_shape_list) {
      for (int32_t y : y_track_list) {
        curr_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
    for (int32_t x : x_track_list) {
      for (int32_t y : y_shape_list) {
        curr_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
    for (int32_t x : x_shape_list) {
      for (int32_t y : y_shape_list) {
        curr_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }

    std::pair<int32_t, int32_t> via_key(legal_shape.via_master_idx.get_below_layer_idx(), legal_shape.via_master_idx.get_via_idx());
    bool is_via_valid = legal_shape.via_master_idx.isValid();
    std::set<LayerCoord, CmpLayerCoordByXASC>* via_coord_set = nullptr;
    std::vector<CandidateAccessPoint>* via_candidate_list = nullptr;
    if (is_via_valid) {
      use_via_candidate = true;
      via_coord_set = &via_coord_set_map[via_key];
      via_candidate_list = &via_candidate_list_map[via_key];
    }
    for (LayerCoord& layer_coord : curr_coord_list) {
      if (!RTUTIL.isInside(die_valid_rect, layer_coord)) {
        continue;
      }
      if (layer_coord.get_x() % manufacture_grid != 0 || layer_coord.get_y() % manufacture_grid != 0) {
        RTLOG.error(Loc::current(), "The coord is off_grid!");
      }
      int32_t track_num = (std::binary_search(x_track_list.begin(), x_track_list.end(), layer_coord.get_x()) ? 1 : 0)
                          + (std::binary_search(y_track_list.begin(), y_track_list.end(), layer_coord.get_y()) ? 1 : 0);
      if (is_via_valid) {
        if (RTUTIL.exist(*via_coord_set, layer_coord)) {
          continue;
        }
        via_coord_set->insert(layer_coord);
        double init_cost = (track_num == 2 ? 0 : (track_num == 1 ? violation_unit / 4.0 : violation_unit / 2.0));
        via_candidate_list->push_back({layer_coord, legal_shape.via_master_idx, track_num, init_cost});
      } else {
        auto [iter, inserted] = coord_track_num_map.emplace(layer_coord, track_num);
        if (!inserted) {
          iter->second = std::max(iter->second, track_num);
        }
      }
    }
  }

  auto getInitCost = [&](int32_t track_num) {
    return (track_num == 2 ? 0 : (track_num == 1 ? violation_unit / 4.0 : violation_unit / 2.0));
  };
  if (use_via_candidate) {
    std::vector<CandidateAccessPoint> selected_candidate_list;
    selected_candidate_list.reserve(via_candidate_list_map.size() * ap_per_via_master);
    for (auto& via_candidate_pair : via_candidate_list_map) {
      std::vector<CandidateAccessPoint>& via_candidate_list = via_candidate_pair.second;
      std::sort(via_candidate_list.begin(), via_candidate_list.end(), [](const CandidateAccessPoint& a, const CandidateAccessPoint& b) {
        if (a.track_num != b.track_num) {
          return a.track_num > b.track_num;
        }
        return CmpLayerCoordByXASC()(a.coord, b.coord);
      });
      int32_t selected_num = 0;
      for (CandidateAccessPoint& candidate : via_candidate_list) {
        if (selected_num >= ap_per_via_master) {
          break;
        }
        selected_candidate_list.push_back(candidate);
        selected_num++;
      }
    }
    std::vector<AccessPoint> access_point_list;
    access_point_list.reserve(selected_candidate_list.size());
    std::map<LayerCoord, size_t, CmpLayerCoordByXASC> coord_access_point_map;
    for (CandidateAccessPoint& candidate : selected_candidate_list) {
      LayerCoord& layer_coord = candidate.coord;
      ViaMasterIdx& via_master_idx = candidate.via_master_idx;
      auto [iter, inserted] = coord_access_point_map.emplace(layer_coord, access_point_list.size());
      if (inserted) {
        access_point_list.emplace_back(pin_idx, layer_coord);
        access_point_list.back().set_init_cost(candidate.init_cost);
      }
      AccessPoint& access_point = access_point_list[iter->second];
      if (candidate.init_cost < access_point.get_init_cost()) {
        access_point.set_init_cost(candidate.init_cost);
      }
      if (!RTUTIL.exist(access_point.get_candidate_via_list(), via_master_idx)) {
        access_point.get_candidate_via_list().push_back(via_master_idx);
      }
    }
    return access_point_list;
  }

  std::vector<CandidateAccessPoint> candidate_list;
  candidate_list.reserve(coord_track_num_map.size());
  for (auto& [coord, track_num] : coord_track_num_map) {
    candidate_list.push_back({coord, ViaMasterIdx(), track_num, getInitCost(track_num)});
  }
  std::sort(candidate_list.begin(), candidate_list.end(), [](const CandidateAccessPoint& a, const CandidateAccessPoint& b) {
    if (a.track_num != b.track_num) {
      return a.track_num > b.track_num;
    }
    return CmpLayerCoordByXASC()(a.coord, b.coord);
  });
  std::vector<LayerCoord> layer_coord_list;
  layer_coord_list.reserve(candidate_list.size());
  for (CandidateAccessPoint& candidate : candidate_list) {
    layer_coord_list.push_back(candidate.coord);
  }
  uniformSampleCoordList(pa_model, layer_coord_list);
  std::vector<AccessPoint> access_point_list;
  access_point_list.reserve(layer_coord_list.size());
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord);
    access_point_list.back().set_init_cost(getInitCost(coord_track_num_map[layer_coord]));
  }
  return access_point_list;
}

std::vector<ViaMaster*> PinAccessor::getSelectedViaMasterList(PAModel& pa_model, int32_t routing_layer_idx)
{
  struct CmpViaMasterByPA
  {
    int32_t getSymmetry(const LayerRect& rect) const
    {
      return std::abs(rect.get_ll_x() + rect.get_ur_x()) + std::abs(rect.get_ll_y() + rect.get_ur_y());
    }
    int64_t getArea(const LayerRect& rect) const { return static_cast<int64_t>(rect.getXSpan()) * rect.getYSpan(); }
    bool operator()(ViaMaster* a, ViaMaster* b) const
    {
      int32_t a_symmetry = getSymmetry(a->get_below_enclosure()) + getSymmetry(a->get_above_enclosure());
      int32_t b_symmetry = getSymmetry(b->get_below_enclosure()) + getSymmetry(b->get_above_enclosure());
      if (a_symmetry != b_symmetry) {
        return a_symmetry < b_symmetry;
      }
      int64_t a_below_area = getArea(a->get_below_enclosure());
      int64_t b_below_area = getArea(b->get_below_enclosure());
      if (a_below_area != b_below_area) {
        return a_below_area < b_below_area;
      }
      int64_t a_above_area = getArea(a->get_above_enclosure());
      int64_t b_above_area = getArea(b->get_above_enclosure());
      if (a_above_area != b_above_area) {
        return a_above_area < b_above_area;
      }
      return a->get_via_master_idx().get_via_idx() < b->get_via_master_idx().get_via_idx();
    }
  };

  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  int32_t extra_via_master_num = pa_model.get_pa_com_param().get_extra_via_master_num();

  std::vector<ViaMaster*> selected_via_master_list;
  for (std::vector<ViaMaster>& via_master_list : layer_via_master_list) {
    if (via_master_list.empty()) {
      continue;
    }
    ViaMaster& front_via_master = via_master_list.front();
    if (front_via_master.get_below_enclosure().get_layer_idx() != routing_layer_idx
        && front_via_master.get_above_enclosure().get_layer_idx() != routing_layer_idx) {
      continue;
    }
    selected_via_master_list.push_back(&front_via_master);
    if (extra_via_master_num <= 0) {
      continue;
    }

    std::vector<ViaMaster*> sorted_via_master_list;
    for (ViaMaster& via_master : via_master_list) {
      if (&via_master != &front_via_master) {
        sorted_via_master_list.push_back(&via_master);
      }
    }
    std::sort(sorted_via_master_list.begin(), sorted_via_master_list.end(), CmpViaMasterByPA());
    int32_t selected_num = 0;
    for (ViaMaster* via_master : sorted_via_master_list) {
      if (selected_num >= extra_via_master_num) {
        break;
      }
      selected_via_master_list.push_back(via_master);
      selected_num++;
    }
  }
  if (!selected_via_master_list.empty()) {
    std::string via_master_name_string = RTUTIL.getString("PA selected via masters for layer ", routing_layer_list[routing_layer_idx].get_layer_name(), ":");
    for (ViaMaster* via_master : selected_via_master_list) {
      via_master_name_string += RTUTIL.getString(" ", via_master->get_via_name());
    }
    RTLOG.info(Loc::current(), via_master_name_string);
  }
  return selected_via_master_list;
}

PlanarRect PinAccessor::getViaEnclosure(ViaMaster& via_master, int32_t routing_layer_idx)
{
  if (via_master.get_below_enclosure().get_layer_idx() == routing_layer_idx) {
    return via_master.get_below_enclosure();
  }
  if (via_master.get_above_enclosure().get_layer_idx() == routing_layer_idx) {
    return via_master.get_above_enclosure();
  }
  RTLOG.error(Loc::current(), "The routing_layer_idx is not in via_master!");
  return PlanarRect();
}

void PinAccessor::uniformSampleCoordList(PAModel& pa_model, std::vector<LayerCoord>& layer_coord_list)
{
  int32_t max_candidate_point_num = pa_model.get_pa_com_param().get_max_candidate_point_num();

  PlanarRect bounding_box = RTUTIL.getBoundingBox(layer_coord_list);
  int32_t grid_num = static_cast<int32_t>(std::sqrt(max_candidate_point_num));
  double grid_x_span = bounding_box.getXSpan() * 1.0 / grid_num;
  double grid_y_span = bounding_box.getYSpan() * 1.0 / grid_num;

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

  for (auto& [net_idx, access_point_list] : pa_model.get_curr_net_access_point_map()) {
    for (AccessPoint* access_point : access_point_list) {
      RTDM.updateNetAccessPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  resetCurrAccessPoint(pa_model);
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
        pa_pin.get_grid_coord_set().insert(access_point.get_grid_coord());
      }
    }
  }
  for (PANet& pa_net : pa_net_list) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        RTDM.updateNetAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_point);
        addCurrAccessPoint(pa_model, pa_net.get_net_idx(), &access_point);
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<std::pair<int32_t, PAPin*>> PinAccessor::getReroutePinList(PAModel& pa_model, const std::vector<Violation>& extra_violation_list)
{
  Die& die = RTDM.getDatabase().get_die();

  // reroute current violation pins and via_only violation
  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> net_layer_violation_rect_map;
  for (Violation* violation : RTDM.getViolationSet(die)) {
    EXTLayerRect& violation_shape = violation->get_violation_shape();
    for (int32_t net_idx : violation->get_violation_net_set()) {
      net_layer_violation_rect_map[net_idx][violation_shape.get_layer_idx()].push_back(violation_shape.get_real_rect());
    }
  }
  for (const Violation& violation : extra_violation_list) {
    const EXTLayerRect& violation_shape = violation.get_violation_shape();
    for (int32_t net_idx : violation.get_violation_net_set()) {
      net_layer_violation_rect_map[net_idx][violation_shape.get_layer_idx()].push_back(violation_shape.get_real_rect());
    }
  }

  std::set<std::pair<int32_t, int32_t>> net_pin_set;
  for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
    if (!RTUTIL.exist(net_layer_violation_rect_map, net_idx)) {
      continue;
    }
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      bool overlap = false;
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
          if (!RTUTIL.exist(net_layer_violation_rect_map[net_idx], net_shape.get_layer_idx())) {
            continue;
          }
          PlanarRect shape_rect = net_shape.get_rect();
          for (PlanarRect& violation_rect : net_layer_violation_rect_map[net_idx][net_shape.get_layer_idx()]) {
            if (RTUTIL.isClosedOverlap(shape_rect, violation_rect)) {
              overlap = true;
              break;
            }
          }
          if (overlap) {
            break;
          }
        }
        if (overlap) {
          break;
        }
      }
      if (overlap) {
        net_pin_set.emplace(net_idx, pin_idx);
      }
    }
  }
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
    if (!RTUTIL.exist(net_layer_violation_rect_map, net_idx)) {
      continue;
    }
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      bool overlap = false;
      for (EXTLayerRect* patch : patch_set) {
        if (!RTUTIL.exist(net_layer_violation_rect_map[net_idx], patch->get_layer_idx())) {
          continue;
        }
        PlanarRect patch_rect = patch->get_real_rect();
        for (PlanarRect& violation_rect : net_layer_violation_rect_map[net_idx][patch->get_layer_idx()]) {
          if (RTUTIL.isClosedOverlap(patch_rect, violation_rect)) {
            overlap = true;
            break;
          }
        }
        if (overlap) {
          break;
        }
      }
      if (overlap) {
        net_pin_set.emplace(net_idx, pin_idx);
      }
    }
  }

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();
  std::vector<std::pair<int32_t, PAPin*>> net_pin_pair_list;
  for (auto& [net_idx, pin_idx] : net_pin_set) {
    if (net_idx < 0 || net_idx >= static_cast<int32_t>(pa_net_list.size())) {
      continue;
    }
    std::vector<PAPin>& pa_pin_list = pa_net_list[net_idx].get_pa_pin_list();
    if (pin_idx < 0 || pin_idx >= static_cast<int32_t>(pa_pin_list.size())) {
      continue;
    }
    net_pin_pair_list.emplace_back(net_idx, &pa_pin_list[pin_idx]);
  }
  return net_pin_pair_list;
}

bool PinAccessor::updateRerouteAccessPointList(PAModel& pa_model, const std::vector<Violation>& ap_via_only_violation_list)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<std::pair<int32_t, PAPin*>> net_pin_pair_list = getReroutePinList(pa_model, ap_via_only_violation_list);
  if (net_pin_pair_list.empty()) {
    RTLOG.info(Loc::current(), "No reroute access point needs update", monitor.getStatsInfo());
    return false;
  }

  int32_t del_result_num = 0;
  int32_t del_patch_num = 0;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>*>>>& net_pin_access_result_map = pa_model.get_curr_net_pin_access_result_map();
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect*>>>& net_pin_access_patch_map = pa_model.get_curr_net_pin_access_patch_map();
  for (auto& [net_idx, pa_pin] : net_pin_pair_list) {
    int32_t pin_idx = pa_pin->get_pin_idx();
    pa_model.get_reroute_pin_set().emplace(net_idx, pin_idx);
    std::vector<Segment<LayerCoord>*> result_list;
    std::vector<EXTLayerRect*> patch_list;
    if (RTUTIL.exist(net_pin_access_result_map, net_idx) && RTUTIL.exist(net_pin_access_result_map[net_idx], pin_idx)) {
      for (Segment<LayerCoord>* segment : net_pin_access_result_map[net_idx][pin_idx]) {
        result_list.push_back(segment);
      }
    }
    if (RTUTIL.exist(net_pin_access_patch_map, net_idx) && RTUTIL.exist(net_pin_access_patch_map[net_idx], pin_idx)) {
      for (EXTLayerRect* patch : net_pin_access_patch_map[net_idx][pin_idx]) {
        patch_list.push_back(patch);
      }
    }
    for (Segment<LayerCoord>* segment : result_list) {
      delCurrAccessResult(pa_model, net_idx, pin_idx, segment);
      RTDM.updateNetPinAccessResultToGCellMap(ChangeType::kDel, net_idx, pin_idx, segment);
      del_result_num++;
    }
    for (EXTLayerRect* patch : patch_list) {
      delCurrAccessPatch(pa_model, net_idx, pin_idx, patch);
      RTDM.updateNetPinAccessPatchToGCellMap(ChangeType::kDel, net_idx, pin_idx, patch);
      del_patch_num++;
    }
  }
  updateAccessPointList(pa_model, net_pin_pair_list, true);
  uploadAccessPointList(pa_model);
  RTLOG.info(Loc::current(), "Updated ", net_pin_pair_list.size(), " reroute pins, deleted ", del_result_num, " access results and ", del_patch_num,
             " access patches", monitor.getStatsInfo());
  return true;
}

void PinAccessor::routePAModel(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = 2 * non_prefer_wire_unit * cost_unit;
  double fixed_rect_unit = 4 * non_prefer_wire_unit * cost_unit;
  double routed_rect_unit = 2 * non_prefer_wire_unit * cost_unit;
  double violation_unit = 4 * non_prefer_wire_unit * cost_unit;
  /**
   * prefer_wire_unit, non_prefer_wire_unit, via_unit, size, offset, schedule_interval, fixed_rect_unit, routed_rect_unit, violation_unit, max_routed_times,
   * max_candidate_patch_num
   */
  std::vector<PAIterParam> pa_iter_param_list;
  // clang-format off
  pa_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 3, 0, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 20, 10);
  pa_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 3, 1, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 20, 10);
  pa_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 3, 2, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 20, 10);
  pa_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 3, 0, 3, 2 * fixed_rect_unit, 2 * routed_rect_unit, 2 * violation_unit, 40, 10);
  pa_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 3, 1, 3, 2 * fixed_rect_unit, 2 * routed_rect_unit, 2 * violation_unit, 40, 10);
  pa_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 3, 2, 3, 2 * fixed_rect_unit, 2 * routed_rect_unit, 2 * violation_unit, 40, 10);
  // clang-format on
  initRoutingState(pa_model);
  bool checked_ap_via_only = false;
  for (int32_t i = 0, iter = 1; i < static_cast<int32_t>(pa_iter_param_list.size()); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", pa_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, pa_iter_param_list.size()),
               ") *****");
    // debugPlotPAModel(pa_model, "before");
    setPAIterParam(pa_model, iter, pa_iter_param_list[i]);
    initPABoxMap(pa_model);
    resetRoutingState(pa_model);
    buildBoxSchedule(pa_model);
    // debugPlotPAModel(pa_model, "middle");
    routePABoxMap(pa_model);
    uploadViolation(pa_model, pa_model.get_iter() > 3);
    std::vector<Violation> ap_via_only_violation_list;
    bool need_reroute_access_point = checkAPViaOnlyViolation(pa_model, iter, checked_ap_via_only, ap_via_only_violation_list);
    bool reroute_access_point_updated = false;
    updateBestResult(pa_model, !ap_via_only_violation_list.empty());
    // debugPlotPAModel(pa_model, "after");
    updateSummary(pa_model);
    printSummary(pa_model);
    outputNetCSV(pa_model);
    outputViolationCSV(pa_model);
    outputJson(pa_model);
    if (need_reroute_access_point) {
      reroute_access_point_updated = updateRerouteAccessPointList(pa_model, ap_via_only_violation_list);
    }
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", pa_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, pa_iter_param_list.size()), ")",
               iter_monitor.getStatsInfo(), "*****");
    if (iter > 3 && !pa_model.get_reroute_pin_set().empty()) {
      RTLOG.info(Loc::current(), "Cleared ", pa_model.get_reroute_pin_set().size(), " pending reroute pins");
      pa_model.get_reroute_pin_set().clear();
    }
    if (!reroute_access_point_updated && checked_ap_via_only && stopIteration(pa_model, pa_iter_param_list)) {
      break;
    }
  }
  selectBestResult(pa_model);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool PinAccessor::checkAPViaOnlyViolation(PAModel& pa_model, int32_t iter, bool& checked_ap_via_only,
                                          std::vector<Violation>& ap_via_only_violation_list)
{
  bool need_check_ap_via_only = (iter == 3 || (!checked_ap_via_only && getRouteViolationNum(pa_model) == 0));
  if (!need_check_ap_via_only) {
    return false;
  }
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  checked_ap_via_only = true;
  ap_via_only_violation_list = getRouteViolationList(pa_model, true);
  RTLOG.info(Loc::current(), "AP via-only violation num: ", ap_via_only_violation_list.size());
  std::map<ViolationType, int32_t> ap_via_only_violation_type_num_map;
  for (Violation& violation : ap_via_only_violation_list) {
    ap_via_only_violation_type_num_map[violation.get_violation_type()]++;
  }
  for (auto& [violation_type, violation_num] : ap_via_only_violation_type_num_map) {
    RTLOG.info(Loc::current(), "AP via-only ", GetViolationTypeName()(violation_type), " violation num: ", violation_num);
  }

  bool need_reroute_access_point = (getRouteViolationNum(pa_model) > 0 || !ap_via_only_violation_list.empty());
  if (!ap_via_only_violation_list.empty()) {
    Die& die = RTDM.getDatabase().get_die();
    std::set<Violation, CmpViolation> route_violation_set;
    for (Violation* violation : RTDM.getViolationSet(die)) {
      route_violation_set.insert(*violation);
    }
    int32_t add_violation_num = uploadRouteViolationList(route_violation_set, ap_via_only_violation_list, &pa_model.get_curr_route_violation_list());
    RTLOG.info(Loc::current(), "Uploaded ", add_violation_num, " AP via-only violations for reroute");
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
  return need_reroute_access_point;
}

void PinAccessor::initRoutingState(PAModel& pa_model)
{
  pa_model.set_initial_routing(true);
  pa_model.get_reroute_pin_set().clear();
}

void PinAccessor::setPAIterParam(PAModel& pa_model, int32_t iter, PAIterParam& pa_iter_param)
{
  pa_model.set_iter(iter);
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", pa_iter_param.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "non_prefer_wire_unit: ", pa_iter_param.get_non_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", pa_iter_param.get_via_unit());
  RTLOG.info(Loc::current(), "size: ", pa_iter_param.get_size());
  RTLOG.info(Loc::current(), "offset: ", pa_iter_param.get_offset());
  RTLOG.info(Loc::current(), "schedule_interval: ", pa_iter_param.get_schedule_interval());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", pa_iter_param.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", pa_iter_param.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", pa_iter_param.get_violation_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", pa_iter_param.get_max_routed_times());
  RTLOG.info(Loc::current(), "max_candidate_patch_num: ", pa_iter_param.get_max_candidate_patch_num());
  pa_model.set_pa_iter_param(pa_iter_param);
}

void PinAccessor::initPABoxMap(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  PAIterParam& pa_iter_param = pa_model.get_pa_iter_param();

  int32_t size = pa_iter_param.get_size();
  int32_t offset = pa_iter_param.get_offset();
  while (offset >= size) {
    offset -= size;
  }
  std::vector<int32_t> x_scale_list;
  {
    int32_t x_gcell_num = 0;
    for (ScaleGrid& x_grid : gcell_axis.get_x_grid_list()) {
      x_gcell_num += x_grid.get_step_num();
    }
    x_scale_list.push_back(0);
    for (int32_t x_scale = offset; x_scale <= x_gcell_num; x_scale += size) {
      x_scale_list.push_back(x_scale);
    }
    x_scale_list.push_back(x_gcell_num);
    std::sort(x_scale_list.begin(), x_scale_list.end());
    x_scale_list.erase(std::unique(x_scale_list.begin(), x_scale_list.end()), x_scale_list.end());
  }
  std::vector<int32_t> y_scale_list;
  {
    int32_t y_gcell_num = 0;
    for (ScaleGrid& y_grid : gcell_axis.get_y_grid_list()) {
      y_gcell_num += y_grid.get_step_num();
    }
    y_scale_list.push_back(0);
    for (int32_t y_scale = offset; y_scale <= y_gcell_num; y_scale += size) {
      y_scale_list.push_back(y_scale);
    }
    y_scale_list.push_back(y_gcell_num);
    std::sort(y_scale_list.begin(), y_scale_list.end());
    y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  }
  GridMap<PABox>& pa_box_map = pa_model.get_pa_box_map();
  {
    int32_t x_box_num = static_cast<int32_t>(x_scale_list.size()) - 1;
    int32_t y_box_num = static_cast<int32_t>(y_scale_list.size()) - 1;
    pa_box_map.init(x_box_num, y_box_num);
  }
  int32_t x_box_num = pa_box_map.get_x_size();
  int32_t y_box_num = pa_box_map.get_y_size();
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < x_box_num; x++) {
    for (int32_t y = 0; y < y_box_num; y++) {
      int32_t grid_ll_x = x_scale_list[x];
      int32_t grid_ll_y = y_scale_list[y];
      int32_t grid_ur_x = x_scale_list[x + 1] - 1;
      int32_t grid_ur_y = y_scale_list[y + 1] - 1;

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
      pa_box.set_iter(pa_model.get_iter());
      pa_box.set_pa_iter_param(&pa_iter_param);
      pa_box.set_initial_routing(pa_model.get_initial_routing());
      pa_box.set_hard(pa_model.get_iter() > 3);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::resetRoutingState(PAModel& pa_model)
{
  pa_model.set_initial_routing(false);
  pa_model.get_curr_route_violation_list().clear();
}

void PinAccessor::buildBoxSchedule(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<PABox>& pa_box_map = pa_model.get_pa_box_map();
  int32_t schedule_interval = pa_model.get_pa_iter_param().get_schedule_interval();

  std::vector<std::vector<PABoxId>> pa_box_id_list_list;
  pa_box_id_list_list.reserve(schedule_interval * schedule_interval);
  for (int32_t start_x = 0; start_x < schedule_interval; start_x++) {
    for (int32_t start_y = 0; start_y < schedule_interval; start_y++) {
      std::vector<PABoxId> pa_box_id_list;
      int32_t x_box_num = (pa_box_map.get_x_size() - start_x + schedule_interval - 1) / schedule_interval;
      int32_t y_box_num = (pa_box_map.get_y_size() - start_y + schedule_interval - 1) / schedule_interval;
      pa_box_id_list.reserve(x_box_num * y_box_num);
      for (int32_t x = start_x; x < pa_box_map.get_x_size(); x += schedule_interval) {
        for (int32_t y = start_y; y < pa_box_map.get_y_size(); y += schedule_interval) {
          pa_box_id_list.emplace_back(x, y);
        }
      }
      if (!pa_box_id_list.empty()) {
        pa_box_id_list_list.push_back(pa_box_id_list);
      }
    }
  }
  pa_model.set_pa_box_id_list_list(pa_box_id_list_list);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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
    int32_t thread_num = omp_get_max_threads();
    std::vector<std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>*>>>> del_result_map_list(thread_num);
    std::vector<std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect*>>>> del_patch_map_list(thread_num);
    std::vector<std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>*>>>> add_result_map_list(thread_num);
    std::vector<std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect*>>>> add_patch_map_list(thread_num);
#pragma omp parallel for schedule(dynamic, 1)
    for (PABoxId& pa_box_id : pa_box_id_list) {
      int32_t thread_id = omp_get_thread_num();
      PABox& pa_box = pa_box_map[pa_box_id.get_x()][pa_box_id.get_y()];
      buildFixedRect(pa_box);
      buildAccessPoint(pa_box);
      buildAccessResult(pa_box);
      buildAccessPatch(pa_box);
      initPATaskList(pa_model, pa_box, del_result_map_list[thread_id], del_patch_map_list[thread_id]);
      pa_box.get_reroute_task_idx_set().clear();
      // 前3轮或者early stop之后将reroute相关的segment和patch都删了，违例找不到pin，所以强制reroute这个box
      for (PATask* pa_task : pa_box.get_pa_task_list()) {
        std::pair<int32_t, int32_t> net_pin_pair(pa_task->get_net_idx(), pa_task->get_pa_pin()->get_pin_idx());
        if (RTUTIL.exist(pa_model.get_reroute_pin_set(), net_pin_pair)) {
          pa_box.get_reroute_task_idx_set().insert(pa_task->get_task_idx());
        }
      }
      buildRouteViolation(pa_box);
      if (needRouting(pa_box)) {
        buildBoxTrackAxis(pa_box);
        buildLayerNodeMap(pa_box);
        buildLayerShadowMap(pa_box);
        buildPANodeNeighbor(pa_box);
        buildOrientNetMap(pa_box);
        buildNetShadowMap(pa_box);
        exemptPinShape(pa_model, pa_box);
        // debugCheckPABox(pa_box);
        // debugPlotPABox(pa_box, "before");
        routePABox(pa_box);
        // debugPlotPABox(pa_box, "after");
      }
      updateBestResult(pa_box);
      uploadBestResult(pa_box, add_result_map_list[thread_id], add_patch_map_list[thread_id]);
      freePABox(pa_box);
    }
    for (auto& del_result_map : del_result_map_list) {
      for (auto& [net_idx, pin_result_map] : del_result_map) {
        for (auto& [pin_idx, segment_list] : pin_result_map) {
          for (Segment<LayerCoord>* segment : segment_list) {
            delCurrAccessResult(pa_model, net_idx, pin_idx, segment);
          }
        }
      }
    }
    for (auto& del_patch_map : del_patch_map_list) {
      for (auto& [net_idx, pin_patch_map] : del_patch_map) {
        for (auto& [pin_idx, patch_list] : pin_patch_map) {
          for (EXTLayerRect* patch : patch_list) {
            delCurrAccessPatch(pa_model, net_idx, pin_idx, patch);
          }
        }
      }
    }
    for (auto& add_result_map : add_result_map_list) {
      for (auto& [net_idx, pin_result_map] : add_result_map) {
        for (auto& [pin_idx, segment_list] : pin_result_map) {
          for (Segment<LayerCoord>* segment : segment_list) {
            addCurrAccessResult(pa_model, net_idx, pin_idx, segment);
          }
        }
      }
    }
    for (auto& add_patch_map : add_patch_map_list) {
      for (auto& [net_idx, pin_patch_map] : add_patch_map) {
        for (auto& [pin_idx, patch_list] : pin_patch_map) {
          for (EXTLayerRect* patch : patch_list) {
            addCurrAccessPatch(pa_model, net_idx, pin_idx, patch);
          }
        }
      }
    }
    routed_box_num += pa_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num), ") boxes with ",
               getRouteViolationNum(pa_model), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::buildFixedRect(PABox& pa_box)
{
  pa_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(pa_box.get_box_rect()));
  std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list = pa_box.get_env_shape_list();
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map = pa_box.get_net_pin_shape_map();
  env_shape_list.clear();
  net_pin_shape_map.clear();
  for (auto& [is_routing, layer_net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          env_shape_list.reserve(env_shape_list.size() + fixed_rect_set.size());
          for (EXTLayerRect* fixed_rect : fixed_rect_set) {
            env_shape_list.emplace_back(fixed_rect, is_routing);
          }
        } else {
          std::vector<std::pair<EXTLayerRect*, bool>>& pin_shape_list = net_pin_shape_map[net_idx];
          pin_shape_list.reserve(pin_shape_list.size() + fixed_rect_set.size());
          for (EXTLayerRect* fixed_rect : fixed_rect_set) {
            pin_shape_list.emplace_back(fixed_rect, is_routing);
          }
        }
      }
    }
  }
}

void PinAccessor::buildAccessPoint(PABox& pa_box)
{
  pa_box.set_net_access_point_map(RTDM.getNetAccessPointMap(pa_box.get_box_rect()));
}

void PinAccessor::buildAccessResult(PABox& pa_box)
{
  pa_box.set_net_pin_access_result_map(RTDM.getNetPinAccessResultMap(pa_box.get_box_rect()));
}

void PinAccessor::buildAccessPatch(PABox& pa_box)
{
  pa_box.set_net_pin_access_patch_map(RTDM.getNetPinAccessPatchMap(pa_box.get_box_rect()));
}

void PinAccessor::initPATaskList(PAModel& pa_model, PABox& pa_box,
                                 std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>*>>>& del_result_map,
                                 std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect*>>>& del_patch_map)
{
  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();
  std::vector<PATask*>& pa_task_list = pa_box.get_pa_task_list();

  EXTPlanarRect& box_rect = pa_box.get_box_rect();
  PlanarRect& box_real_rect = box_rect.get_real_rect();
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& net_pin_access_result_map = pa_box.get_net_pin_access_result_map();
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& net_task_access_result_map = pa_box.get_net_task_access_result_map();
  std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>& net_pin_access_patch_map = pa_box.get_net_pin_access_patch_map();
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& net_task_access_patch_map = pa_box.get_net_task_access_patch_map();
  int32_t enlarged_size = RTDM.getOnlyPitch();

  std::map<PANet*, std::map<PAPin*, std::set<AccessPoint*>>> net_pin_access_point_map;
  {
    for (auto& [net_idx, access_point_set] : pa_box.get_net_access_point_map()) {
      PANet& pa_net = pa_net_list[net_idx];
      for (AccessPoint* access_point : access_point_set) {
        if (!RTUTIL.isInside(box_real_rect, access_point->get_real_coord())) {
          continue;
        }
        PAPin& pa_pin = pa_net.get_pa_pin_list()[access_point->get_pin_idx()];
        net_pin_access_point_map[&pa_net][&pa_pin].insert(access_point);
      }
    }
  }
  std::map<int32_t, std::vector<std::pair<int32_t, PlanarRect>>> layer_routing_obs_rect_map;
  {
    for (auto& [routing_layer_idx, net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()[true]) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        layer_routing_obs_rect_map[routing_layer_idx].reserve(layer_routing_obs_rect_map[routing_layer_idx].size() + fixed_rect_set.size());
        for (EXTLayerRect* fixed_rect : fixed_rect_set) {
          layer_routing_obs_rect_map[routing_layer_idx].emplace_back(net_idx, RTUTIL.getEnlargedRect(fixed_rect->get_real_rect(), enlarged_size));
        }
      }
    }
  }
  for (auto& [pa_net, pin_access_point_map] : net_pin_access_point_map) {
    std::map<int32_t, std::vector<PlanarRect>> routing_obs_rect_map;
    for (auto& [routing_layer_idx, obs_rect_list] : layer_routing_obs_rect_map) {
      std::vector<PlanarRect>& curr_obs_rect_list = routing_obs_rect_map[routing_layer_idx];
      curr_obs_rect_list.reserve(obs_rect_list.size());
      for (auto& [net_idx, obs_rect] : obs_rect_list) {
        if (pa_net->get_net_idx() != net_idx) {
          curr_obs_rect_list.push_back(obs_rect);
        }
      }
    }
    for (auto& [pa_pin, access_point_set] : pin_access_point_map) {
      bool inside_box = false;
      for (const PlanarCoord& grid_coord : pa_pin->get_grid_coord_set()) {
        if (RTUTIL.isInside(box_rect.get_grid_rect(), grid_coord)) {
          inside_box = true;
          break;
        }
      }
      if (!inside_box) {
        continue;
      }
      if (pa_pin->get_access_point().get_real_coord() != PlanarCoord(-1, -1)) {
        if (!RTUTIL.isInside(box_rect.get_real_rect(), pa_pin->get_access_point().get_real_coord())) {
          continue;
        }
      }
      std::map<int32_t, std::vector<PlanarRect>> routing_pin_rect_map;
      for (EXTLayerRect& routing_shape : pa_pin->get_routing_shape_list()) {
        routing_pin_rect_map[routing_shape.get_layer_idx()].push_back(RTUTIL.getEnlargedRect(routing_shape.get_real_rect(), RTDM.getOnlyPitch()));
      }
      std::vector<PAGroup> pa_group_list(2);
      {
        pa_group_list.front().set_is_target(false);
        for (AccessPoint* access_point : access_point_set) {
          pa_group_list.front().get_coord_list().push_back(access_point->getRealLayerCoord());
        }
        pa_group_list.back().set_is_target(true);
        for (const LayerCoord& coord : pa_pin->get_target_coord_list()) {
          if (!RTUTIL.isInside(box_rect.get_real_rect(), coord.get_planar_coord())) {
            continue;
          }
          pa_group_list.back().get_coord_list().push_back(coord);
        }
      }
      if (pa_group_list.front().get_coord_list().empty() || pa_group_list.back().get_coord_list().empty()) {
        continue;
      }
      PATask* pa_task = new PATask();
      pa_task->set_net_idx(pa_net->get_net_idx());
      pa_task->set_task_idx(static_cast<int32_t>(pa_task_list.size()));
      pa_task->set_pa_pin(pa_pin);
      pa_task->set_connect_type(pa_net->get_connect_type());
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
  {
    // 重载数据
    std::map<int32_t, std::map<int32_t, int32_t>> net_pin_task_map;
    for (PATask* pa_task : pa_task_list) {
      net_pin_task_map[pa_task->get_net_idx()][pa_task->get_pa_pin()->get_pin_idx()] = pa_task->get_task_idx();
    }
    {
      std::vector<std::pair<int32_t, int32_t>> net_pin_pair_list;
      for (auto& [net_idx, pin_access_result_map] : net_pin_access_result_map) {
        for (auto& [pin_idx, segment_set] : pin_access_result_map) {
          if (!RTUTIL.exist(net_pin_task_map, net_idx)) {
            continue;
          }
          if (!RTUTIL.exist(net_pin_task_map[net_idx], pin_idx)) {
            continue;
          }
          for (Segment<LayerCoord>* segment : segment_set) {
            net_task_access_result_map[net_idx][net_pin_task_map[net_idx][pin_idx]].push_back(*segment);
            del_result_map[net_idx][pin_idx].push_back(segment);
            RTDM.updateNetPinAccessResultToGCellMap(ChangeType::kDel, net_idx, pin_idx, segment);
          }
          net_pin_pair_list.emplace_back(net_idx, pin_idx);
        }
      }
      for (auto& [net_idx, pin_idx] : net_pin_pair_list) {
        net_pin_access_result_map[net_idx].erase(pin_idx);
        if (net_pin_access_result_map[net_idx].empty()) {
          net_pin_access_result_map.erase(net_idx);
        }
      }
    }
    {
      std::vector<std::pair<int32_t, int32_t>> net_pin_pair_list;
      for (auto& [net_idx, pin_access_patch_map] : net_pin_access_patch_map) {
        for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
          if (!RTUTIL.exist(net_pin_task_map, net_idx)) {
            continue;
          }
          if (!RTUTIL.exist(net_pin_task_map[net_idx], pin_idx)) {
            continue;
          }
          for (EXTLayerRect* patch : patch_set) {
            net_task_access_patch_map[net_idx][net_pin_task_map[net_idx][pin_idx]].push_back(*patch);
            del_patch_map[net_idx][pin_idx].push_back(patch);
            RTDM.updateNetPinAccessPatchToGCellMap(ChangeType::kDel, net_idx, pin_idx, patch);
          }
          net_pin_pair_list.emplace_back(net_idx, pin_idx);
        }
      }
      for (auto& [net_idx, pin_idx] : net_pin_pair_list) {
        net_pin_access_patch_map[net_idx].erase(pin_idx);
        if (net_pin_access_patch_map[net_idx].empty()) {
          net_pin_access_patch_map.erase(net_idx);
        }
      }
    }
  }
}

void PinAccessor::buildRouteViolation(PABox& pa_box)
{
  std::set<int32_t> need_checked_net_set;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    need_checked_net_set.insert(pa_task->get_net_idx());
  }
  std::set<Violation, CmpViolation> route_violation_set;
  for (Violation* violation : RTDM.getViolationSet(pa_box.get_box_rect())) {
    bool exist_checked_net = false;
    for (int32_t violation_net_idx : violation->get_violation_net_set()) {
      if (RTUTIL.exist(need_checked_net_set, violation_net_idx)) {
        exist_checked_net = true;
        break;
      }
    }
    if (exist_checked_net) {
      if (route_violation_set.insert(*violation).second) {
        pa_box.get_route_violation_list().push_back(*violation);
      }
      RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
    }
  }
  // 后3轮检查要把via only的违例也算进去
  if (pa_box.get_hard()) {
    for (Violation new_violation : getRouteViolationList(pa_box, true)) {
      if (RTUTIL.isOpenOverlap(pa_box.get_box_rect().get_real_rect(),
                               RTUTIL.getEnlargedRect(new_violation.get_violation_shape().get_real_rect(), RTDM.getOnlyPitch()))) {
        if (route_violation_set.insert(new_violation).second) {
          pa_box.get_route_violation_list().push_back(new_violation);
        }
      }
    }
  }
}

bool PinAccessor::needRouting(PABox& pa_box)
{
  if (pa_box.get_pa_task_list().empty()) {
    return false;
  }
  if (pa_box.get_initial_routing() == false && pa_box.get_route_violation_list().empty() && pa_box.get_reroute_task_idx_set().empty()) {
    return false;
  }
  return true;
}

void PinAccessor::buildBoxTrackAxis(PABox& pa_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_real_rect = pa_box.get_box_rect().get_real_rect();
  int32_t ll_x = box_real_rect.get_ll_x();
  int32_t ll_y = box_real_rect.get_ll_y();
  int32_t ur_x = box_real_rect.get_ur_x();
  int32_t ur_y = box_real_rect.get_ur_y();
  // 避免 off_grid
  while (ll_x % manufacture_grid != 0) {
    ll_x++;
  }
  while (ll_y % manufacture_grid != 0) {
    ll_y++;
  }
  while (ur_x % manufacture_grid != 0) {
    ur_x--;
  }
  while (ur_y % manufacture_grid != 0) {
    ur_y--;
  }
  std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> layer_scale_list_list(routing_layer_list.size());
  for (RoutingLayer& routing_layer : routing_layer_list) {
    layer_scale_list_list[routing_layer.get_layer_idx()].first = RTUTIL.getScaleList(ll_x, ur_x, routing_layer.getXTrackGridList());
    layer_scale_list_list[routing_layer.get_layer_idx()].second = RTUTIL.getScaleList(ll_y, ur_y, routing_layer.getYTrackGridList());
  }
  std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> layer_axis_list(routing_layer_list.size());
  for (RoutingLayer& routing_layer : routing_layer_list) {
    auto& layer_scale_list = layer_scale_list_list[routing_layer.get_layer_idx()];
    if (routing_layer.isPreferH()) {
      layer_axis_list[routing_layer.get_layer_idx()].first.insert(layer_axis_list[routing_layer.get_layer_idx()].first.end(),
                                                                  layer_scale_list.first.begin(), layer_scale_list.first.end());
    } else {
      layer_axis_list[routing_layer.get_layer_idx()].second.insert(layer_axis_list[routing_layer.get_layer_idx()].second.end(),
                                                                   layer_scale_list.second.begin(), layer_scale_list.second.end());
    }
  }
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
      for (LayerCoord& coord : pa_group.get_coord_list()) {
        int32_t layer_idx = coord.get_layer_idx();
        layer_axis_list[layer_idx].first.push_back(coord.get_x());
        layer_axis_list[layer_idx].second.push_back(coord.get_y());
      }
    }
  }
  std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>>& layer_axis_map = pa_box.get_layer_axis_map();
  layer_axis_map.clear();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    int32_t layer_idx = routing_layer.get_layer_idx();
    std::vector<int32_t>& layer_x_axis_list = layer_axis_list[layer_idx].first;
    std::vector<int32_t>& layer_y_axis_list = layer_axis_list[layer_idx].second;
    std::sort(layer_x_axis_list.begin(), layer_x_axis_list.end());
    layer_x_axis_list.erase(std::unique(layer_x_axis_list.begin(), layer_x_axis_list.end()), layer_x_axis_list.end());
    std::sort(layer_y_axis_list.begin(), layer_y_axis_list.end());
    layer_y_axis_list.erase(std::unique(layer_y_axis_list.begin(), layer_y_axis_list.end()), layer_y_axis_list.end());
    layer_axis_map[layer_idx].first.insert(layer_x_axis_list.begin(), layer_x_axis_list.end());
    layer_axis_map[layer_idx].second.insert(layer_y_axis_list.begin(), layer_y_axis_list.end());
  }
  size_t x_scale_num = 0;
  size_t y_scale_num = 0;
  size_t task_coord_num = 0;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    auto& layer_scale_list = layer_scale_list_list[routing_layer.get_layer_idx()];
    x_scale_num += layer_scale_list.first.size();
    y_scale_num += layer_scale_list.second.size();
  }
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
      task_coord_num += pa_group.get_coord_list().size();
    }
  }
  x_scale_list.reserve(x_scale_num + task_coord_num);
  y_scale_list.reserve(y_scale_num + task_coord_num);
  for (RoutingLayer& routing_layer : routing_layer_list) {
    auto& layer_scale_list = layer_scale_list_list[routing_layer.get_layer_idx()];
    x_scale_list.insert(x_scale_list.end(), layer_scale_list.first.begin(), layer_scale_list.first.end());
    y_scale_list.insert(y_scale_list.end(), layer_scale_list.second.begin(), layer_scale_list.second.end());
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

void PinAccessor::buildLayerShadowMap(PABox& pa_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  pa_box.get_layer_shadow_map().resize(routing_layer_list.size());
}

void PinAccessor::buildPANodeNeighbor(PABox& pa_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>>& layer_axis_map = pa_box.get_layer_axis_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_hv = true;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_hv = false;
    }
    GridMap<PANode>& pa_node_map = layer_node_map[layer_idx];
    std::set<int32_t> neighbor_layer_x_axis_set;
    std::set<int32_t> neighbor_layer_y_axis_set;
    if (layer_idx != 0) {
      neighbor_layer_x_axis_set.insert(layer_axis_map[layer_idx - 1].first.begin(), layer_axis_map[layer_idx - 1].first.end());
      neighbor_layer_y_axis_set.insert(layer_axis_map[layer_idx - 1].second.begin(), layer_axis_map[layer_idx - 1].second.end());
    }
    if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
      neighbor_layer_x_axis_set.insert(layer_axis_map[layer_idx + 1].first.begin(), layer_axis_map[layer_idx + 1].first.end());
      neighbor_layer_y_axis_set.insert(layer_axis_map[layer_idx + 1].second.begin(), layer_axis_map[layer_idx + 1].second.end());
    }
    std::set<int32_t>& curr_axis = (routing_layer_list[layer_idx].isPreferH()) ? layer_axis_map[layer_idx].first : layer_axis_map[layer_idx].second;
    for (int32_t x = 0; x < pa_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < pa_node_map.get_y_size(); y++) {
        PANode& pa_node = pa_node_map[x][y];
        if (routing_hv) {
          if (!routing_layer_list[layer_idx].isPreferH()) {
            if (RTUTIL.exist(curr_axis, pa_node_map[x][y].get_y()) || RTUTIL.exist(neighbor_layer_y_axis_set, pa_node_map[x][y].get_y())) {
              if (x != 0) {
                pa_node.setNeighborNode(Orientation::kWest, &pa_node_map[x - 1][y]);
              }
              if (x != (pa_node_map.get_x_size() - 1)) {
                pa_node.setNeighborNode(Orientation::kEast, &pa_node_map[x + 1][y]);
              }
            }
            if (RTUTIL.exist(neighbor_layer_x_axis_set, pa_node_map[x][y].get_x())) {
              if (y != 0) {
                pa_node.setNeighborNode(Orientation::kSouth, &pa_node_map[x][y - 1]);
              }
              if (y != (pa_node_map.get_y_size() - 1)) {
                pa_node.setNeighborNode(Orientation::kNorth, &pa_node_map[x][y + 1]);
              }
            }
          } else if (routing_layer_list[layer_idx].isPreferH()) {
            if (RTUTIL.exist(curr_axis, pa_node_map[x][y].get_x()) || RTUTIL.exist(neighbor_layer_x_axis_set, pa_node_map[x][y].get_x())) {
              if (y != 0) {
                pa_node.setNeighborNode(Orientation::kSouth, &pa_node_map[x][y - 1]);
              }
              if (y != (pa_node_map.get_y_size() - 1)) {
                pa_node.setNeighborNode(Orientation::kNorth, &pa_node_map[x][y + 1]);
              }
            }
            if (RTUTIL.exist(neighbor_layer_y_axis_set, pa_node_map[x][y].get_y())) {
              if (x != 0) {
                pa_node.setNeighborNode(Orientation::kWest, &pa_node_map[x - 1][y]);
              }
              if (x != (pa_node_map.get_x_size() - 1)) {
                pa_node.setNeighborNode(Orientation::kEast, &pa_node_map[x + 1][y]);
              }
            }
          }
        }
        if (layer_idx != 0) {
          pa_node.setNeighborNode(Orientation::kBelow, &layer_node_map[layer_idx - 1][x][y]);
        }
        if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
          pa_node.setNeighborNode(Orientation::kAbove, &layer_node_map[layer_idx + 1][x][y]);
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
  for (auto& [net_idx, pin_access_result_map] : pa_box.get_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        updateFixedRectToGraph(pa_box, ChangeType::kAdd, net_idx, segment);
      }
    }
  }
  for (auto& [net_idx, task_access_result_map] : pa_box.get_net_task_access_result_map()) {
    for (auto& [task_idx, segment_list] : task_access_result_map) {
      for (Segment<LayerCoord>& segment : segment_list) {
        updateRoutedRectToGraph(pa_box, ChangeType::kAdd, net_idx, segment);
      }
    }
  }
  for (auto& [net_idx, pin_access_patch_map] : pa_box.get_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        updateFixedRectToGraph(pa_box, ChangeType::kAdd, net_idx, patch, true);
      }
    }
  }
  for (auto& [net_idx, task_access_patch_map] : pa_box.get_net_task_access_patch_map()) {
    for (auto& [task_idx, patch_list] : task_access_patch_map) {
      for (EXTLayerRect& patch : patch_list) {
        updateRoutedRectToGraph(pa_box, ChangeType::kAdd, net_idx, patch, true);
      }
    }
  }
  for (Violation& violation : pa_box.get_route_violation_list()) {
    addRouteViolationToGraph(pa_box, violation);
  }
}

void PinAccessor::buildNetShadowMap(PABox& pa_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToShadow(pa_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
  for (auto& [net_idx, pin_access_result_map] : pa_box.get_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        updateFixedRectToShadow(pa_box, ChangeType::kAdd, net_idx, segment);
      }
    }
  }
  for (auto& [net_idx, task_access_result_map] : pa_box.get_net_task_access_result_map()) {
    for (auto& [task_idx, segment_list] : task_access_result_map) {
      for (Segment<LayerCoord>& segment : segment_list) {
        updateRoutedRectToShadow(pa_box, ChangeType::kAdd, net_idx, segment);
      }
    }
  }
  for (auto& [net_idx, pin_access_patch_map] : pa_box.get_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        updateFixedRectToShadow(pa_box, ChangeType::kAdd, net_idx, patch, true);
      }
    }
  }
  for (auto& [net_idx, task_access_patch_map] : pa_box.get_net_task_access_patch_map()) {
    for (auto& [task_idx, patch_list] : task_access_patch_map) {
      for (EXTLayerRect& patch : patch_list) {
        updateRoutedRectToShadow(pa_box, ChangeType::kAdd, net_idx, patch, true);
      }
    }
  }
}

void PinAccessor::exemptPinShape(PAModel& pa_model, PABox& pa_box)
{
  int32_t detection_distance = RTDM.getDatabase().get_detection_distance();
  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();
  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();

  for (auto& [pa_net_idx, access_point_set] : pa_box.get_net_access_point_map()) {
    std::map<int32_t, std::vector<EXTLayerRect*>> routing_obs_rect_map;
    for (auto& [routing_layer_idx, net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()[true]) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (pa_net_idx == net_idx) {
          continue;
        }
        for (auto& fixed_rect : fixed_rect_set) {
          routing_obs_rect_map[routing_layer_idx].push_back(fixed_rect);
        }
      }
    }
    std::vector<PAPin>& pa_pin_list = pa_net_list[pa_net_idx].get_pa_pin_list();
    for (AccessPoint* access_point : access_point_set) {
      if (pa_pin_list[access_point->get_pin_idx()].get_is_core()) {
        if (!RTUTIL.existTrackGrid(access_point->get_real_coord(), box_track_axis)) {
          continue;
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(access_point->get_real_coord(), box_track_axis);
        PANode& pa_node = layer_node_map[access_point->get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        for (Orientation orient : {Orientation::kAbove, Orientation::kBelow}) {
          if (pa_node.hasFixedRectOrient(orient)) {
            pa_node.delFixedRectNet(orient, -1);
            PANode* neighbor_node = pa_node.getNeighborNode(orient);
            if (neighbor_node == nullptr) {
              continue;
            }
            Orientation oppo_orientation = RTUTIL.getOppositeOrientation(orient);
            neighbor_node->delFixedRectNet(oppo_orientation, -1);
          }
        }
      } else {
        PlanarRect real_rect = RTUTIL.getEnlargedRect(access_point->get_real_coord(), detection_distance);
        if (!RTUTIL.existTrackGrid(real_rect, box_track_axis)) {
          continue;
        }
        PlanarRect grid_rect = RTUTIL.getTrackGrid(real_rect, box_track_axis);
        for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
          for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
            PANode& pa_node = layer_node_map[access_point->get_layer_idx()][x][y];

            bool within_shape = false;
            for (EXTLayerRect* obs_rect : routing_obs_rect_map[pa_node.get_layer_idx()]) {
              if (RTUTIL.isInside(obs_rect->get_real_rect(), pa_node.get_planar_coord())) {
                within_shape = true;
                break;
              }
            }
            if (within_shape) {
              continue;
            }
            for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
              pa_node.delFixedRectNet(orient, -1);
            }
          }
        }
      }
    }
  }
}

void PinAccessor::routePABox(PABox& pa_box)
{
  std::vector<PATask*> routing_task_list = initTaskSchedule(pa_box);
  int routing_rounds = 0;
  while (!routing_task_list.empty()) {
    for (PATask* routing_task : routing_task_list) {
      updateGraph(pa_box, routing_task);
      routePATask(pa_box, routing_task);
      patchPATask(pa_box, routing_task);
      routing_task->addRoutedTimes();
    }
    updateRouteViolationList(pa_box);
    updateAccessPoint(pa_box);
    updateBestResult(pa_box);
    updateTaskSchedule(pa_box, routing_task_list, routing_rounds);
    routing_rounds++;
  }
}

std::vector<PATask*> PinAccessor::initTaskSchedule(PABox& pa_box)
{
  bool initial_routing = pa_box.get_initial_routing();

  std::vector<PATask*> routing_task_list;
  if (initial_routing) {
    for (PATask* pa_task : pa_box.get_pa_task_list()) {
      routing_task_list.push_back(pa_task);
    }
  } else {
    updateTaskSchedule(pa_box, routing_task_list, 0);
  }
  return routing_task_list;
}

void PinAccessor::updateGraph(PABox& pa_box, PATask* pa_task)
{
  int32_t curr_net_idx = pa_task->get_net_idx();
  int32_t curr_task_idx = pa_task->get_task_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = pa_box.get_net_task_access_result_map()[curr_net_idx][curr_task_idx];
  std::vector<EXTLayerRect>& routing_patch_list = pa_box.get_net_task_access_patch_map()[curr_net_idx][curr_task_idx];

  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(pa_box, ChangeType::kDel, curr_net_idx, routing_segment);
    updateRoutedRectToShadow(pa_box, ChangeType::kDel, curr_net_idx, routing_segment);
  }
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(pa_box, ChangeType::kDel, curr_net_idx, routing_patch, true);
    updateRoutedRectToShadow(pa_box, ChangeType::kDel, curr_net_idx, routing_patch, true);
  }
}

void PinAccessor::routePATask(PABox& pa_box, PATask* pa_task)
{
  initSingleRouteTask(pa_box, pa_task);
  while (!isConnectedAllEnd(pa_box)) {
    routeSinglePath(pa_box);
    updatePathResult(pa_box);
    resetStartAndEnd(pa_box);
    resetSinglePath(pa_box);
  }
  updateTaskResult(pa_box);
  resetSingleRouteTask(pa_box);
}

void PinAccessor::initSingleRouteTask(PABox& pa_box, PATask* pa_task)
{
  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();
  std::map<LayerCoord, AccessPoint*, CmpLayerCoordByXASC> source_access_point_map;
  for (AccessPoint* access_point : pa_box.get_net_access_point_map()[pa_task->get_net_idx()]) {
    if (access_point->get_pin_idx() == pa_task->get_pa_pin()->get_pin_idx()) {
      source_access_point_map[access_point->getRealLayerCoord()] = access_point;
    }
  }

  // single task
  pa_box.set_curr_route_task(pa_task);
  pa_box.get_source_node_access_point_map().clear();
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
        if (!pa_group.get_is_target() && RTUTIL.exist(source_access_point_map, coord)) {
          pa_box.get_source_node_access_point_map()[&pa_node] = source_access_point_map[coord];
        }
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
  std::map<PANode*, AccessPoint*>& source_node_access_point_map = pa_box.get_source_node_access_point_map();

  for (std::vector<PANode*>& start_node_list : start_node_list_list) {
    for (PANode* start_node : start_node_list) {
      if (RTUTIL.exist(source_node_access_point_map, start_node)) {
        start_node->set_known_cost(source_node_access_point_map[start_node]->get_init_cost());
      } else {
        start_node->set_known_cost(0);
      }
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
  OpenQueue<PANode>& open_queue = pa_box.get_open_queue();
  PANode* path_head_node = pa_box.get_path_head_node();
  AccessPoint* source_access_point = nullptr;
  if (RTUTIL.exist(pa_box.get_source_node_access_point_map(), path_head_node)) {
    source_access_point = pa_box.get_source_node_access_point_map()[path_head_node];
  }

  path_head_node->forEachNeighborNode([&](Orientation orientation, PANode* neighbor_node) {
    if (neighbor_node->isClose()) {
      return;
    }
    ViaMasterIdx parent_via_master_idx;
    double transition_cost = 0;
    // 在AP点进行via选择时， 如果有多种viaMaster，按照via above enclosure和周围shape的面积重叠来判断
    if (source_access_point != nullptr && (orientation == Orientation::kAbove || orientation == Orientation::kBelow)) {
      LayerCoord via_coord = (path_head_node->get_layer_idx() < neighbor_node->get_layer_idx() ? *path_head_node : *neighbor_node);
      parent_via_master_idx = getSelectedViaMasterIdx(pa_box, source_access_point, via_coord);
      if (parent_via_master_idx.isValid()) {
        Segment<LayerCoord> via_segment(*path_head_node, *neighbor_node, parent_via_master_idx);
        transition_cost = getViaMasterCost(pa_box, pa_box.get_curr_route_task()->get_net_idx(), via_segment);
      }
    }
    double known_cost = getKnownCost(pa_box, path_head_node, neighbor_node) + transition_cost;
    if (neighbor_node->isOpen() && known_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_parent_via_master_idx(parent_via_master_idx);
      open_queue.push(neighbor_node);
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_parent_via_master_idx(parent_via_master_idx);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(pa_box, neighbor_node));
      pushToOpenList(pa_box, neighbor_node);
    }
  });
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
  PANode* planar_first_node = nullptr;
  PANode* planar_second_node = nullptr;
  Orientation planar_orientation = Orientation::kNone;
  // 回溯segment需要保持viaMaster
  auto pushPlanarSegment = [&]() {
    if (planar_first_node == nullptr) {
      return;
    }
    Segment<LayerCoord> routing_segment(*planar_first_node, *planar_second_node);
    updateSegmentViaMaster(routing_segment);
    routing_segment_list.push_back(routing_segment);
    planar_first_node = nullptr;
    planar_second_node = nullptr;
    planar_orientation = Orientation::kNone;
  };
  while (pre_node != nullptr) {
    Orientation curr_orientation = RTUTIL.getOrientation(*curr_node, *pre_node);
    if (curr_node->get_layer_idx() != pre_node->get_layer_idx()) {
      pushPlanarSegment();
      Segment<LayerCoord> routing_segment(*curr_node, *pre_node, curr_node->get_parent_via_master_idx());
      updateSegmentViaMaster(routing_segment);
      routing_segment_list.push_back(routing_segment);
    } else if (planar_first_node != nullptr && curr_orientation == planar_orientation) {
      planar_second_node = pre_node;
    } else {
      pushPlanarSegment();
      planar_first_node = curr_node;
      planar_second_node = pre_node;
      planar_orientation = curr_orientation;
    }
    PANode* next_node = pre_node->get_parent_node();
    curr_node = pre_node;
    pre_node = next_node;
  }
  pushPlanarSegment();

  return routing_segment_list;
}

void PinAccessor::updateSegmentViaMaster(Segment<LayerCoord>& segment)
{
  if (segment.hasValidViaMaster()) {
    return;
  }
  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  if (first_coord.get_layer_idx() == second_coord.get_layer_idx()) {
    return;
  }
  if (std::abs(first_coord.get_layer_idx() - second_coord.get_layer_idx()) != 1) {
    return;
  }
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  int32_t below_layer_idx = std::min(first_coord.get_layer_idx(), second_coord.get_layer_idx());
  if (below_layer_idx < 0 || below_layer_idx >= static_cast<int32_t>(layer_via_master_list.size())) {
    return;
  }
  if (layer_via_master_list[below_layer_idx].empty()) {
    return;
  }
  segment.set_via_master_idx(layer_via_master_list[below_layer_idx].front().get_via_master_idx());
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
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void PinAccessor::resetSinglePath(PABox& pa_box)
{
  pa_box.get_open_queue().clear();
  std::vector<PANode*>& single_path_visited_node_list = pa_box.get_single_path_visited_node_list();
  for (PANode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(PANodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_parent_via_master_idx(ViaMasterIdx());
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  pa_box.set_path_head_node(nullptr);
  pa_box.set_end_node_list_idx(-1);
}

void PinAccessor::updateTaskResult(PABox& pa_box)
{
  int32_t curr_net_idx = pa_box.get_curr_route_task()->get_net_idx();
  int32_t curr_task_idx = pa_box.get_curr_route_task()->get_task_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = pa_box.get_net_task_access_result_map()[curr_net_idx][curr_task_idx];
  routing_segment_list = getRoutingSegmentList(pa_box);
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(pa_box, ChangeType::kAdd, curr_net_idx, routing_segment);
    updateRoutedRectToShadow(pa_box, ChangeType::kAdd, curr_net_idx, routing_segment);
  }
}

std::vector<Segment<LayerCoord>> PinAccessor::getRoutingSegmentList(PABox& pa_box)
{
  PATask* curr_route_task = pa_box.get_curr_route_task();

  // 记录segment的viaMaster，避免MTree重构布线时viaMaster丢失
  auto isViaSegment = [](Segment<LayerCoord>& segment) {
    return segment.get_first().get_planar_coord() == segment.get_second().get_planar_coord()
           && std::abs(segment.get_first().get_layer_idx() - segment.get_second().get_layer_idx()) == 1;
  };
  auto isSameViaSegment = [](Segment<LayerCoord>& a, Segment<LayerCoord>& b) {
    return (a.get_first() == b.get_first() && a.get_second() == b.get_second()) || (a.get_first() == b.get_second() && a.get_second() == b.get_first());
  };
  std::vector<Segment<LayerCoord>> via_segment_list;
  for (Segment<LayerCoord>& routing_segment : pa_box.get_routing_segment_list()) {
    if (isViaSegment(routing_segment)) {
      via_segment_list.push_back(routing_segment);
    }
  }

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<PAGroup>& pa_group_list = curr_route_task->get_pa_group_list();
  for (size_t i = 0; i < pa_group_list.size(); i++) {
    for (LayerCoord& coord : pa_group_list[i].get_coord_list()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, pa_box.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    Segment<LayerCoord> routing_segment(coord_segment.get_first()->value(), coord_segment.get_second()->value());
    if (isViaSegment(routing_segment)) {
      for (Segment<LayerCoord>& via_segment : via_segment_list) {
        if (isSameViaSegment(routing_segment, via_segment) && via_segment.hasValidViaMaster()) {
          routing_segment.set_via_master_idx(via_segment.get_via_master_idx());
          break;
        }
      }
    }
    updateSegmentViaMaster(routing_segment);
    routing_segment_list.push_back(routing_segment);
  }
  return routing_segment_list;
}

void PinAccessor::resetSingleRouteTask(PABox& pa_box)
{
  pa_box.set_curr_route_task(nullptr);
  pa_box.get_start_node_list_list().clear();
  pa_box.get_end_node_list_list().clear();
  pa_box.get_path_node_list().clear();
  pa_box.get_single_task_visited_node_list().clear();
  pa_box.get_routing_segment_list().clear();
  pa_box.get_source_node_access_point_map().clear();
}

// manager open list

void PinAccessor::pushToOpenList(PABox& pa_box, PANode* curr_node)
{
  OpenQueue<PANode>& open_queue = pa_box.get_open_queue();
  std::vector<PANode*>& single_task_visited_node_list = pa_box.get_single_task_visited_node_list();
  std::vector<PANode*>& single_path_visited_node_list = pa_box.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(PANodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

PANode* PinAccessor::popFromOpenList(PABox& pa_box)
{
  PANode* node = pa_box.get_open_queue().pop();
  if (node != nullptr) {
    node->set_state(PANodeState::kClose);
  }
  return node;
}

// calculate known

double PinAccessor::getKnownCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  if (start_node->getNeighborNode(RTUTIL.getOrientation(*start_node, *end_node)) != end_node) {
    RTLOG.error(Loc::current(), "The neighbor not exist!");
  }

  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getNodeCost(pa_box, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(pa_box, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnownWireCost(pa_box, start_node, end_node);
  cost += getKnownViaCost(pa_box, start_node, end_node);
  cost += getKnownSelfCost(pa_box, start_node, end_node);
  return cost;
}

double PinAccessor::getNodeCost(PABox& pa_box, PANode* curr_node, Orientation orientation)
{
  double fixed_rect_unit = pa_box.get_pa_iter_param()->get_fixed_rect_unit();
  double routed_rect_unit = pa_box.get_pa_iter_param()->get_routed_rect_unit();
  double violation_unit = pa_box.get_pa_iter_param()->get_violation_unit();

  int32_t net_idx = pa_box.get_curr_route_task()->get_net_idx();

  double cost = 0;
  cost += curr_node->getFixedRectCost(net_idx, orientation, fixed_rect_unit);
  cost += curr_node->getRoutedRectCost(net_idx, orientation, routed_rect_unit);
  cost += curr_node->getViolationCost(orientation, violation_unit);
  return cost;
}

double PinAccessor::getKnownWireCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = pa_box.get_pa_iter_param()->get_prefer_wire_unit();
  double non_prefer_wire_unit = pa_box.get_pa_iter_param()->get_non_prefer_wire_unit();

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

double PinAccessor::getKnownViaCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double via_unit = pa_box.get_pa_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

double PinAccessor::getKnownSelfCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double routed_rect_unit = pa_box.get_pa_iter_param()->get_routed_rect_unit();

  bool nonprefer_and_segment_end = false;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() != RTUTIL.getDirection(*start_node, *end_node)) {
      for (std::vector<PANode*>& end_node_list : pa_box.get_end_node_list_list()) {
        if (RTUTIL.exist(end_node_list, end_node)) {
          nonprefer_and_segment_end = true;
          break;
        }
      }
      if (!nonprefer_and_segment_end) {
        return 0;
      }
    }
  }
  RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
  int32_t wire_width = routing_layer.get_min_width();
  int32_t target_wire_length = std::max(routing_layer.getPRLSpacing(wire_width), routing_layer.get_notch_spacing()) + wire_width;

  int32_t non_prefer_wire_length = 0;
  {
    PANode* curr_node = start_node;
    PANode* pre_node = curr_node->get_parent_node();
    while (pre_node != nullptr) {
      if (pre_node->get_layer_idx() != curr_node->get_layer_idx()) {
        break;
      }
      if (routing_layer.get_prefer_direction() == RTUTIL.getDirection(*pre_node, *curr_node)) {
        break;
      }
      non_prefer_wire_length += RTUTIL.getManhattanDistance(pre_node->get_planar_coord(), curr_node->get_planar_coord());
      curr_node = pre_node;
      pre_node = curr_node->get_parent_node();
    }
    if (nonprefer_and_segment_end) {
      non_prefer_wire_length += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
    }
  }
  double self_cost = 0;
  if (0 < non_prefer_wire_length && non_prefer_wire_length < target_wire_length) {
    self_cost += routed_rect_unit;
  }
  return self_cost;
}

ViaMasterIdx PinAccessor::getSelectedViaMasterIdx(PABox& pa_box, AccessPoint* access_point, const LayerCoord& via_coord)
{
  if (access_point == nullptr) {
    return ViaMasterIdx();
  }
  std::vector<ViaMasterIdx> candidate_via_list;
  for (ViaMasterIdx& via_master_idx : access_point->get_candidate_via_list()) {
    if (via_master_idx.get_below_layer_idx() == via_coord.get_layer_idx()) {
      candidate_via_list.push_back(via_master_idx);
    }
  }
  if (candidate_via_list.empty()) {
    return ViaMasterIdx();
  }
  if (candidate_via_list.size() == 1) {
    return candidate_via_list.front();
  }
  int select = pa_box.get_curr_route_task()->get_routed_times() % candidate_via_list.size();
  ViaMasterIdx best_via_master_idx = candidate_via_list[select];
  double best_cost = DBL_MAX;
  LayerCoord above_coord(via_coord.get_planar_coord(), via_coord.get_layer_idx() + 1);
  for (size_t i = 0; i < candidate_via_list.size(); i++) {
    ViaMasterIdx& via_master_idx = candidate_via_list[(select + i) % candidate_via_list.size()];
    Segment<LayerCoord> via_segment(via_coord, above_coord, via_master_idx);
    double cost = getViaMasterCost(pa_box, pa_box.get_curr_route_task()->get_net_idx(), via_segment);
    if (cost < best_cost) {
      best_cost = cost;
      best_via_master_idx = via_master_idx;
    }
  }
  return best_via_master_idx;
}

double PinAccessor::getViaMasterCost(PABox& pa_box, int32_t net_idx, const Segment<LayerCoord>& via_segment)
{
  std::vector<NetShape> query_shape_list;
  Segment<LayerCoord> query_segment = via_segment;
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, query_segment)) {
    if (net_shape.get_is_routing()) {
      query_shape_list.push_back(net_shape);
    }
  }

  double cost = 0;
  auto addOverlapCost = [&](int32_t layer_idx, const PlanarRect& rect) {
    for (NetShape& query_shape : query_shape_list) {
      if (query_shape.get_layer_idx() != layer_idx) {
        continue;
      }
      if (RTUTIL.isOpenOverlap(query_shape.get_rect(), rect)) {
        cost += RTUTIL.getOverlap(query_shape.get_rect(), rect).getArea() + 1;
      }
    }
  };

  for (auto& [layer_idx, net_fixed_rect_map] : pa_box.get_type_layer_net_fixed_rect_map()[true]) {
    for (auto& [fixed_net_idx, fixed_rect_set] : net_fixed_rect_map) {
      if (fixed_net_idx == net_idx) {
        continue;
      }
      for (EXTLayerRect* fixed_rect : fixed_rect_set) {
        addOverlapCost(layer_idx, fixed_rect->get_real_rect());
      }
    }
  }
  for (auto& [other_net_idx, pin_access_result_map] : pa_box.get_net_pin_access_result_map()) {
    if (other_net_idx == net_idx) {
      continue;
    }
    for (auto& [pin_idx, access_result_set] : pin_access_result_map) {
      (void) pin_idx;
      for (Segment<LayerCoord>* access_result : access_result_set) {
        Segment<LayerCoord> result_segment = *access_result;
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(other_net_idx, result_segment)) {
          if (net_shape.get_is_routing()) {
            addOverlapCost(net_shape.get_layer_idx(), net_shape.get_rect());
          }
        }
      }
    }
  }
  for (auto& [other_net_idx, task_access_result_map] : pa_box.get_net_task_access_result_map()) {
    if (other_net_idx == net_idx) {
      continue;
    }
    for (auto& [task_idx, access_result_list] : task_access_result_map) {
      (void) task_idx;
      for (Segment<LayerCoord>& access_result : access_result_list) {
        Segment<LayerCoord> result_segment = access_result;
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(other_net_idx, result_segment)) {
          if (net_shape.get_is_routing()) {
            addOverlapCost(net_shape.get_layer_idx(), net_shape.get_rect());
          }
        }
      }
    }
  }
  for (auto& [other_net_idx, pin_access_patch_map] : pa_box.get_net_pin_access_patch_map()) {
    if (other_net_idx == net_idx) {
      continue;
    }
    for (auto& [pin_idx, access_patch_set] : pin_access_patch_map) {
      (void) pin_idx;
      for (EXTLayerRect* access_patch : access_patch_set) {
        addOverlapCost(access_patch->get_layer_idx(), access_patch->get_real_rect());
      }
    }
  }
  for (auto& [other_net_idx, task_access_patch_map] : pa_box.get_net_task_access_patch_map()) {
    if (other_net_idx == net_idx) {
      continue;
    }
    for (auto& [task_idx, access_patch_list] : task_access_patch_map) {
      (void) task_idx;
      for (EXTLayerRect& access_patch : access_patch_list) {
        addOverlapCost(access_patch.get_layer_idx(), access_patch.get_real_rect());
      }
    }
  }
  return cost;
}

// calculate estimate

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
  double prefer_wire_unit = pa_box.get_pa_iter_param()->get_prefer_wire_unit();
  double non_prefer_wire_unit = pa_box.get_pa_iter_param()->get_non_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= std::min(prefer_wire_unit, non_prefer_wire_unit);
  return wire_cost;
}

double PinAccessor::getEstimateViaCost(PABox& pa_box, PANode* start_node, PANode* end_node)
{
  double via_unit = pa_box.get_pa_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void PinAccessor::patchPATask(PABox& pa_box, PATask* pa_task)
{
  initSinglePatchTask(pa_box, pa_task);
  while (searchViolation(pa_box)) {
    addViolationToShadow(pa_box);
    patchSingleViolation(pa_box);
    resetSingleViolation(pa_box);
    clearViolationShadow(pa_box);
  }
  updateTaskPatch(pa_box);
  resetSinglePatchTask(pa_box);
}

void PinAccessor::initSinglePatchTask(PABox& pa_box, PATask* pa_task)
{
  // single task
  pa_box.set_curr_patch_task(pa_task);
  pa_box.get_routing_patch_list().clear();
  pa_box.set_patch_violation_list(getPatchViolationList(pa_box, {ViolationType::kMinimumArea}, {}));
  pa_box.get_tried_fix_violation_set().clear();
}

std::vector<Violation> PinAccessor::getPatchViolationList(PABox& pa_box, const std::set<ViolationType>& check_type_set,
                                                          const std::vector<LayerRect>& check_region_list)
{
  std::string top_name = RTUTIL.getString("pa_box_", pa_box.get_pa_box_id().get_x(), "_", pa_box.get_pa_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list = pa_box.get_env_shape_list();
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map = pa_box.get_net_pin_shape_map();
  std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
  for (auto& [net_idx, pin_access_result_map] : pa_box.get_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      std::vector<Segment<LayerCoord>*>& result_list = net_result_map[net_idx];
      result_list.reserve(result_list.size() + segment_set.size());
      for (Segment<LayerCoord>* segment : segment_set) {
        result_list.push_back(segment);
      }
    }
  }
  for (auto& [net_idx, task_access_result_map] : pa_box.get_net_task_access_result_map()) {
    for (auto& [task_idx, segment_list] : task_access_result_map) {
      std::vector<Segment<LayerCoord>*>& result_list = net_result_map[net_idx];
      result_list.reserve(result_list.size() + segment_list.size());
      for (Segment<LayerCoord>& segment : segment_list) {
        result_list.emplace_back(&segment);
      }
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, pin_access_patch_map] : pa_box.get_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      std::vector<EXTLayerRect*>& patch_list = net_patch_map[net_idx];
      patch_list.reserve(patch_list.size() + patch_set.size());
      for (EXTLayerRect* patch : patch_set) {
        patch_list.push_back(patch);
      }
    }
  }
  for (auto& [net_idx, task_access_patch_map] : pa_box.get_net_task_access_patch_map()) {
    for (auto& [task_idx, patch_list] : task_access_patch_map) {
      std::vector<EXTLayerRect*>& result_patch_list = net_patch_map[net_idx];
      if (net_idx == pa_box.get_curr_patch_task()->get_net_idx() && task_idx == pa_box.get_curr_patch_task()->get_task_idx()) {
        result_patch_list.reserve(result_patch_list.size() + pa_box.get_routing_patch_list().size());
        for (EXTLayerRect& patch : pa_box.get_routing_patch_list()) {
          result_patch_list.emplace_back(&patch);
        }
      } else {
        result_patch_list.reserve(result_patch_list.size() + patch_list.size());
        for (EXTLayerRect& patch : patch_list) {
          result_patch_list.emplace_back(&patch);
        }
      }
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    need_checked_net_set.insert(pa_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kPatchHybrid);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  de_task.set_check_type_set(check_type_set);
  de_task.set_check_region_list(check_region_list);
  return RTDE.getViolationList(de_task);
}

bool PinAccessor::searchViolation(PABox& pa_box)
{
  for (Violation& violation : pa_box.get_patch_violation_list()) {
    if (!isValidPatchViolation(pa_box, violation)) {
      continue;
    }
    if (RTUTIL.exist(pa_box.get_tried_fix_violation_set(), violation)) {
      continue;
    }
    int32_t net_idx = *violation.get_violation_net_set().begin();
    if (pa_box.get_curr_patch_task()->get_net_idx() != net_idx) {
      continue;
    }
    if (getViolationOverlapRect(pa_box, violation).empty()) {
      continue;
    }
    pa_box.set_curr_patch_violation(violation);
    return true;
  }
  return false;
}

bool PinAccessor::isValidPatchViolation(PABox& pa_box, Violation& violation)
{
  PlanarRect& box_real_rect = pa_box.get_box_rect().get_real_rect();

  bool is_valid = true;
  if (!RTUTIL.isOpenOverlap(box_real_rect, violation.get_violation_shape().get_real_rect())) {
    is_valid = false;
  }
  if (violation.get_violation_type() != ViolationType::kMinimumArea) {
    is_valid = false;
  }
  return is_valid;
}

std::vector<PlanarRect> PinAccessor::getViolationOverlapRect(PABox& pa_box, Violation& violation)
{
  int32_t curr_net_idx = pa_box.get_curr_patch_task()->get_net_idx();
  int32_t curr_pin_idx = pa_box.get_curr_patch_task()->get_pa_pin()->get_pin_idx();
  int32_t curr_task_idx = pa_box.get_curr_patch_task()->get_task_idx();
  EXTLayerRect& violation_shape = violation.get_violation_shape();
  PlanarRect violation_real_rect = violation_shape.get_real_rect();
  int32_t violation_layer_idx = violation_shape.get_layer_idx();

  GTLPolySetInt gtl_poly_set;
  {
    for (EXTLayerRect* fixed_rect : pa_box.get_type_layer_net_fixed_rect_map()[true][violation_layer_idx][curr_net_idx]) {
      if (RTUTIL.isClosedOverlap(violation_real_rect, fixed_rect->get_real_rect())) {
        gtl_poly_set += RTUTIL.convertToGTLRectInt(fixed_rect->get_real_rect());
      }
    }
    for (Segment<LayerCoord>* segment : pa_box.get_net_pin_access_result_map()[curr_net_idx][curr_pin_idx]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(curr_net_idx, *segment)) {
        if (!net_shape.get_is_routing()) {
          continue;
        }
        if (violation_layer_idx == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, net_shape.get_rect())) {
          gtl_poly_set += RTUTIL.convertToGTLRectInt(net_shape.get_rect());
        }
      }
    }
    for (Segment<LayerCoord>& segment : pa_box.get_net_task_access_result_map()[curr_net_idx][curr_task_idx]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(curr_net_idx, segment)) {
        if (!net_shape.get_is_routing()) {
          continue;
        }
        if (violation_layer_idx == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, net_shape.get_rect())) {
          gtl_poly_set += RTUTIL.convertToGTLRectInt(net_shape.get_rect());
        }
      }
    }
    for (EXTLayerRect* patch : pa_box.get_net_pin_access_patch_map()[curr_net_idx][curr_pin_idx]) {
      if (violation_layer_idx == patch->get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, patch->get_real_rect())) {
        gtl_poly_set += RTUTIL.convertToGTLRectInt(patch->get_real_rect());
      }
    }
  }
  std::vector<GTLPolyInt> gtl_poly_list;
  gtl_poly_set.get_polygons(gtl_poly_list);
  if (gtl_poly_list.empty()) {
    return {};
  }
  GTLPolyInt best_gtl_poly = gtl_poly_list.front();
  {
    int32_t max_overlap_area = INT32_MIN;
    for (GTLPolyInt& gtl_poly : gtl_poly_list) {
      int32_t overlap_area = static_cast<int32_t>(gtl::area(gtl_poly & RTUTIL.convertToGTLRectInt(violation_real_rect)));
      if (max_overlap_area < overlap_area) {
        max_overlap_area = overlap_area;
        best_gtl_poly = gtl_poly;
      }
    }
  }
  std::vector<GTLRectInt> gtl_rect_list;
  gtl::get_max_rectangles(gtl_rect_list, best_gtl_poly);
  std::vector<PlanarRect> overlap_rect_list;
  for (GTLRectInt& gtl_rect : gtl_rect_list) {
    overlap_rect_list.push_back(RTUTIL.convertToPlanarRect(gtl_rect));
  }
  return overlap_rect_list;
}

void PinAccessor::addViolationToShadow(PABox& pa_box)
{
  for (Violation& patch_violation : pa_box.get_patch_violation_list()) {
    if (patch_violation.get_violation_type() == ViolationType::kMinimumArea) {
      continue;
    }
    addPatchViolationToShadow(pa_box, patch_violation);
  }
}

void PinAccessor::patchSingleViolation(PABox& pa_box)
{
  std::vector<EXTLayerRect>& routing_patch_list = pa_box.get_routing_patch_list();
  std::set<Violation, CmpViolation>& tried_fix_violation_set = pa_box.get_tried_fix_violation_set();
  LayerRect violation_rect = pa_box.get_curr_patch_violation().get_violation_shape().getRealLayerRect();

  std::vector<PAPatch> pa_patch_list = getCandidatePatchList(pa_box);
  if (pa_patch_list.size() == 1) {
    routing_patch_list.push_back(pa_patch_list.front().get_patch());
  } else if (pa_patch_list.size() >= 2) {
    std::vector<Violation> origin_patch_violation_list = getPatchViolationList(pa_box, {}, {violation_rect});

    bool curr_is_solved = false;
    for (PAPatch& pa_patch : pa_patch_list) {
      std::vector<Violation> curr_patch_violation_list;
      {
        routing_patch_list.push_back(pa_patch.get_patch());
        curr_patch_violation_list = getPatchViolationList(pa_box, {}, {violation_rect});
        routing_patch_list.pop_back();
      }
      curr_is_solved = getSolvedStatus(pa_box, origin_patch_violation_list, curr_patch_violation_list);
      if (curr_is_solved) {
        routing_patch_list.push_back(pa_patch.get_patch());
        break;
      }
    }
    if (!curr_is_solved) {
      routing_patch_list.push_back(pa_patch_list.front().get_patch());
    }
  }
  tried_fix_violation_set.insert(pa_box.get_curr_patch_violation());
}

std::vector<PAPatch> PinAccessor::getCandidatePatchList(PABox& pa_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t max_candidate_patch_num = pa_box.get_pa_iter_param()->get_max_candidate_patch_num();

  int32_t curr_net_idx = pa_box.get_curr_patch_task()->get_net_idx();
  Violation& curr_patch_violation = pa_box.get_curr_patch_violation();
  int32_t violation_layer_idx = curr_patch_violation.get_violation_shape().get_layer_idx();

  RoutingLayer& routing_layer = routing_layer_list[violation_layer_idx];
  Direction layer_direction = routing_layer.get_prefer_direction();
  int32_t min_area = routing_layer.get_min_area();
  int32_t wire_width = routing_layer.get_min_width();

  GTLPolyInt gtl_poly;
  {
    GTLPolySetInt gtl_poly_set;
    for (PlanarRect& overlap_rect : getViolationOverlapRect(pa_box, curr_patch_violation)) {
      gtl_poly_set += RTUTIL.convertToGTLRectInt(overlap_rect);
    }
    std::vector<GTLPolyInt> gtl_poly_list;
    gtl_poly_set.get_polygons(gtl_poly_list);
    gtl_poly = gtl_poly_list.front();
    if (min_area <= static_cast<int32_t>(gtl::area(gtl_poly))) {
      return {};
    }
  }
  PlanarRect h_cutting_rect;
  {
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, gtl_poly, gtl::HORIZONTAL);
    GTLRectInt best_gtl_rect;
    int32_t max_x_span = 0;
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      int32_t curr_x_span = std::abs(gtl::xl(gtl_rect) - gtl::xh(gtl_rect));
      if (max_x_span <= curr_x_span) {
        max_x_span = curr_x_span;
        best_gtl_rect = gtl_rect;
      }
    }
    h_cutting_rect = RTUTIL.convertToPlanarRect(best_gtl_rect);
  }
  PlanarRect v_cutting_rect;
  {
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, gtl_poly, gtl::VERTICAL);
    GTLRectInt best_gtl_rect;
    int32_t max_y_span = 0;
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      int32_t curr_y_span = std::abs(gtl::yl(gtl_rect) - gtl::yh(gtl_rect));
      if (max_y_span <= curr_y_span) {
        max_y_span = curr_y_span;
        best_gtl_rect = gtl_rect;
      }
    }
    v_cutting_rect = RTUTIL.convertToPlanarRect(best_gtl_rect);
  }
  std::vector<PAPatch> pa_patch_list;
  {
    int32_t h_wire_length = static_cast<int32_t>(std::ceil((min_area - v_cutting_rect.getArea()) / wire_width) + v_cutting_rect.getXSpan());
    while (h_wire_length % manufacture_grid != 0) {
      h_wire_length++;
    }
    for (int32_t y : {h_cutting_rect.get_ll_y(), v_cutting_rect.get_ll_y(), v_cutting_rect.get_ur_y() - wire_width}) {
      for (int32_t x = v_cutting_rect.get_ur_x() - h_wire_length; x <= v_cutting_rect.get_ll_x(); x += manufacture_grid) {
        PlanarRect h_real_rect = RTUTIL.getEnlargedRect(PlanarCoord(x, y), 0, 0, h_wire_length, wire_width);
        if (!RTUTIL.isInside(die.get_real_rect(), h_real_rect)) {
          continue;
        }
        pa_patch_list.emplace_back(h_real_rect, violation_layer_idx);
      }
    }
    int32_t v_wire_length = static_cast<int32_t>(std::ceil((min_area - h_cutting_rect.getArea()) / wire_width) + h_cutting_rect.getYSpan());
    while (v_wire_length % manufacture_grid != 0) {
      v_wire_length++;
    }
    for (int32_t x : {v_cutting_rect.get_ll_x(), h_cutting_rect.get_ll_x(), h_cutting_rect.get_ur_x() - wire_width}) {
      for (int32_t y = h_cutting_rect.get_ur_y() - v_wire_length; y <= h_cutting_rect.get_ll_y(); y += manufacture_grid) {
        PlanarRect v_real_rect = RTUTIL.getEnlargedRect(PlanarCoord(x, y), 0, 0, wire_width, v_wire_length);
        if (!RTUTIL.isInside(die.get_real_rect(), v_real_rect)) {
          continue;
        }
        pa_patch_list.emplace_back(v_real_rect, violation_layer_idx);
      }
    }
    for (PAPatch& pa_patch : pa_patch_list) {
      EXTLayerRect& patch = pa_patch.get_patch();
      patch.set_grid_rect(RTUTIL.getClosedGCellGridRect(patch.get_real_rect(), gcell_axis));
      pa_patch.set_fixed_rect_cost(getFixedRectCost(pa_box, curr_net_idx, patch));
      pa_patch.set_routed_rect_cost(getRoutedRectCost(pa_box, curr_net_idx, patch));
      pa_patch.set_violation_cost(getViolationCost(pa_box, curr_net_idx, patch));
      pa_patch.set_direction(patch.get_real_rect().getRectDirection(layer_direction));
      pa_patch.set_overlap_area(static_cast<int32_t>(gtl::area(gtl_poly & RTUTIL.convertToGTLRectInt(patch.get_real_rect()))));
    }
    std::sort(pa_patch_list.begin(), pa_patch_list.end(), [&layer_direction](PAPatch& a, PAPatch& b) { return CmpPAPatch()(a, b, layer_direction); });
    if (pa_patch_list.empty()) {
      RTLOG.error(Loc::current(), "The pa_patch_list is empty!");
    }
  }
  std::vector<PAPatch> candidate_patch_list;
  {
    std::vector<PAPatch> pa_patch_list_temp;
    for (PAPatch& pa_patch : pa_patch_list) {
      if (pa_patch.getTotalCost() > 0) {
        continue;
      }
      pa_patch_list_temp.push_back(pa_patch);
    }
    if (pa_patch_list_temp.empty()) {
      pa_patch_list_temp.push_back(pa_patch_list.front());
    }
    int32_t patch_size = static_cast<int32_t>(pa_patch_list_temp.size());
    if (patch_size <= max_candidate_patch_num) {
      candidate_patch_list = pa_patch_list_temp;
    } else {
      int32_t candidate_step = (patch_size - 2) / (max_candidate_patch_num - 2);
      candidate_patch_list.push_back(pa_patch_list_temp.front());
      for (int32_t i = candidate_step; i < (patch_size - candidate_step); i += candidate_step) {
        candidate_patch_list.push_back(pa_patch_list_temp[i]);
      }
      candidate_patch_list.push_back(pa_patch_list_temp.back());
    }
  }
  return candidate_patch_list;
}

bool PinAccessor::getSolvedStatus(PABox& pa_box, std::vector<Violation>& origin_patch_violation_list, std::vector<Violation>& curr_patch_violation_list)
{
  std::map<ViolationType, std::pair<int32_t, int32_t>> env_type_origin_curr_map;
  std::map<ViolationType, std::pair<int32_t, int32_t>> valid_type_origin_curr_map;
  std::map<ViolationType, std::pair<int32_t, int32_t>> within_net_map;
  for (Violation& origin_violation : origin_patch_violation_list) {
    if (!isValidPatchViolation(pa_box, origin_violation)) {
      env_type_origin_curr_map[origin_violation.get_violation_type()].first++;
    } else {
      valid_type_origin_curr_map[origin_violation.get_violation_type()].first++;
    }
    if (origin_violation.get_violation_net_set().size() > 1) {
      within_net_map[origin_violation.get_violation_type()].first++;
    }
  }
  for (Violation& curr_violation : curr_patch_violation_list) {
    if (!isValidPatchViolation(pa_box, curr_violation)) {
      env_type_origin_curr_map[curr_violation.get_violation_type()].second++;
    } else {
      valid_type_origin_curr_map[curr_violation.get_violation_type()].second++;
    }
    if (curr_violation.get_violation_net_set().size() > 1) {
      within_net_map[curr_violation.get_violation_type()].second++;
    }
  }
  bool curr_is_solved = true;
  for (auto& [violation_type, origin_curr] : env_type_origin_curr_map) {
    if (!curr_is_solved) {
      break;
    }
    curr_is_solved = origin_curr.second <= origin_curr.first;
  }
  for (auto& [violation_type, origin_curr] : valid_type_origin_curr_map) {
    if (!curr_is_solved) {
      break;
    }
    curr_is_solved = origin_curr.second < origin_curr.first;
  }
  for (auto& [violation_type, origin_curr] : within_net_map) {
    if (!curr_is_solved) {
      break;
    }
    curr_is_solved = origin_curr.second <= origin_curr.first;
  }
  return curr_is_solved;
}

void PinAccessor::resetSingleViolation(PABox& pa_box)
{
  pa_box.set_curr_patch_violation(Violation());
}

void PinAccessor::clearViolationShadow(PABox& pa_box)
{
  for (PAShadow& pa_shadow : pa_box.get_layer_shadow_map()) {
    pa_shadow.get_violation_set().clear();
  }
}

void PinAccessor::updateTaskPatch(PABox& pa_box)
{
  int32_t curr_net_idx = pa_box.get_curr_patch_task()->get_net_idx();
  int32_t curr_task_idx = pa_box.get_curr_patch_task()->get_task_idx();
  std::vector<EXTLayerRect>& routing_patch_list = pa_box.get_net_task_access_patch_map()[curr_net_idx][curr_task_idx];
  routing_patch_list = pa_box.get_routing_patch_list();
  // 新结果添加到graph
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(pa_box, ChangeType::kAdd, curr_net_idx, routing_patch, true);
    updateRoutedRectToShadow(pa_box, ChangeType::kAdd, curr_net_idx, routing_patch, true);
  }
}

void PinAccessor::resetSinglePatchTask(PABox& pa_box)
{
  pa_box.set_curr_patch_task(nullptr);
  pa_box.get_routing_patch_list().clear();
  pa_box.get_patch_violation_list().clear();
  pa_box.get_tried_fix_violation_set().clear();
}

void PinAccessor::updateRouteViolationList(PABox& pa_box)
{
  pa_box.get_route_violation_list().clear();
  std::set<Violation, CmpViolation> route_violation_set;
  for (Violation new_violation : getRouteViolationList(pa_box, false)) {
    if (RTUTIL.isClosedOverlap(pa_box.get_box_rect().get_real_rect(),
                             RTUTIL.getEnlargedRect(new_violation.get_violation_shape().get_real_rect(), RTDM.getOnlyPitch()))
        && route_violation_set.insert(new_violation).second) {
      pa_box.get_route_violation_list().push_back(new_violation);
    }
  }
  if (pa_box.get_hard()) {
    for (Violation new_violation : getRouteViolationList(pa_box, true)) {
      if (RTUTIL.isClosedOverlap(pa_box.get_box_rect().get_real_rect(),
                               RTUTIL.getEnlargedRect(new_violation.get_violation_shape().get_real_rect(), RTDM.getOnlyPitch()))
          && route_violation_set.insert(new_violation).second) {
        pa_box.get_route_violation_list().push_back(new_violation);
      }
    }
  }
  // 新结果添加到graph
  for (Violation& violation : pa_box.get_route_violation_list()) {
    addRouteViolationToGraph(pa_box, violation);
  }
}

std::vector<Violation> PinAccessor::getRouteViolationList(PABox& pa_box, bool ap_via_only)
{
  std::string top_name = RTUTIL.getString("pa_box_", pa_box.get_pa_box_id().get_x(), "_", pa_box.get_pa_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list = pa_box.get_env_shape_list();
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map = pa_box.get_net_pin_shape_map();
  std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
  std::map<int32_t, std::map<int32_t, PATask*>> net_task_map;
  // ap via only的检查只使用ap上的via和env shape, 避免result和patch掩盖一些违例
  if (ap_via_only) {
    for (PATask* pa_task : pa_box.get_pa_task_list()) {
      net_task_map[pa_task->get_net_idx()][pa_task->get_task_idx()] = pa_task;
    }
  }
  for (auto& [net_idx, pin_access_result_map] : pa_box.get_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      if (ap_via_only) {
        continue;
      }
      std::vector<Segment<LayerCoord>*>& result_list = net_result_map[net_idx];
      result_list.reserve(result_list.size() + segment_set.size());
      for (Segment<LayerCoord>* segment : segment_set) {
        result_list.push_back(segment);
      }
    }
  }
  for (auto& [net_idx, task_access_result_map] : pa_box.get_net_task_access_result_map()) {
    for (auto& [task_idx, segment_list] : task_access_result_map) {
      if (!ap_via_only) {
        std::vector<Segment<LayerCoord>*>& result_list = net_result_map[net_idx];
        result_list.reserve(result_list.size() + segment_list.size());
        for (Segment<LayerCoord>& segment : segment_list) {
          result_list.emplace_back(&segment);
        }
      } else if (RTUTIL.exist(net_task_map, net_idx) && RTUTIL.exist(net_task_map[net_idx], task_idx)) {
        PATask* pa_task = net_task_map[net_idx][task_idx];
        LayerCoord access_coord = getAccessCoord(pa_task, segment_list);
        std::vector<Segment<LayerCoord>*>& result_list = net_result_map[net_idx];
        result_list.reserve(result_list.size() + 1);
        for (Segment<LayerCoord>& segment : segment_list) {
          if (isAPViaSegment(segment, access_coord)) {
            result_list.emplace_back(&segment);
            break;
          }
        }
      }
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  if (!ap_via_only) {
    for (auto& [net_idx, pin_access_patch_map] : pa_box.get_net_pin_access_patch_map()) {
      for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
        std::vector<EXTLayerRect*>& patch_list = net_patch_map[net_idx];
        patch_list.reserve(patch_list.size() + patch_set.size());
        for (EXTLayerRect* patch : patch_set) {
          patch_list.push_back(patch);
        }
      }
    }
    for (auto& [net_idx, task_access_patch_map] : pa_box.get_net_task_access_patch_map()) {
      for (auto& [task_idx, patch_list] : task_access_patch_map) {
        std::vector<EXTLayerRect*>& result_patch_list = net_patch_map[net_idx];
        result_patch_list.reserve(result_patch_list.size() + patch_list.size());
        for (EXTLayerRect& patch : patch_list) {
          result_patch_list.emplace_back(&patch);
        }
      }
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    need_checked_net_set.insert(pa_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kRouteHybrid);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);

  std::vector<Violation> route_violation_list = RTDE.getViolationList(de_task);
  if (!ap_via_only) {
    return route_violation_list;
  }
  std::vector<Violation> result_violations;
  result_violations.reserve(route_violation_list.size());
  for (Violation& violation : route_violation_list) {
    if (violation.get_violation_net_set().size() <= 1) {
      continue;
    }
    result_violations.push_back(violation);
  }
  return result_violations;
}

int32_t PinAccessor::getViolationWeight(ViolationType violation_type)
{
  if (violation_type == ViolationType::kCutEOLSpacing) {
    return 10;
  }
  return 1;
}

int32_t PinAccessor::getViolationScore(const std::vector<Violation>& violation_list)
{
  int32_t violation_score = 0;
  for (const Violation& violation : violation_list) {
    violation_score += getViolationWeight(violation.get_violation_type());
  }
  return violation_score;
}

LayerCoord PinAccessor::getAccessCoord(PATask* pa_task, std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<LayerCoord>& pin_shape_coord_list = pa_task->get_pa_pin()->get_pin_shape_coord_list();
  std::vector<LayerCoord>& target_coord_list = pa_task->get_pa_pin()->get_target_coord_list();
  std::vector<LayerCoord> segment_coord_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    segment_coord_list.push_back(segment.get_first());
    segment_coord_list.push_back(segment.get_second());
  }
  if (segment_coord_list.empty()) {
    return RTUTIL.getFirstEqualCoord(pin_shape_coord_list, target_coord_list);
  }
  return RTUTIL.getFirstEqualCoord(pin_shape_coord_list, segment_coord_list);
}

bool PinAccessor::isAPViaSegment(const Segment<LayerCoord>& segment, const LayerCoord& access_coord)
{
  bool touch_access_point = (segment.get_first() == access_coord || segment.get_second() == access_coord);
  bool via_segment = (segment.get_first().get_planar_coord() == segment.get_second().get_planar_coord()
                      && std::abs(segment.get_first().get_layer_idx() - segment.get_second().get_layer_idx()) == 1);
  return touch_access_point && via_segment;
}

void PinAccessor::updateAccessPoint(PABox& pa_box)
{
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    int32_t net_idx = pa_task->get_net_idx();
    int32_t task_idx = pa_task->get_task_idx();
    int32_t pin_idx = pa_task->get_pa_pin()->get_pin_idx();
    std::vector<Segment<LayerCoord>>& segment_list = pa_box.get_net_task_access_result_map()[net_idx][task_idx];
    AccessPoint access_point(pin_idx, getAccessCoord(pa_task, segment_list));
    if (access_point.get_real_coord() == PlanarCoord(-1, -1)) {
      RTLOG.error(Loc::current(), "The access_point creation failed!");
    }
    for (AccessPoint* candidate_access_point : pa_box.get_net_access_point_map()[net_idx]) {
      if (candidate_access_point->get_pin_idx() == pin_idx && candidate_access_point->getRealLayerCoord() == access_point.getRealLayerCoord()) {
        access_point.set_init_cost(candidate_access_point->get_init_cost());
        access_point.set_candidate_via_list(candidate_access_point->get_candidate_via_list());
        break;
      }
    }
    ViaMasterIdx selected_via_master_idx;
    LayerCoord access_coord = access_point.getRealLayerCoord();
    for (Segment<LayerCoord>& segment : segment_list) {
      if (!segment.hasValidViaMaster()) {
        continue;
      }
      if (isAPViaSegment(segment, access_coord)) {
        selected_via_master_idx = segment.get_via_master_idx();
        break;
      }
    }
    if (selected_via_master_idx.isValid()) {
      access_point.set_candidate_via_list({selected_via_master_idx});
    } else if (access_point.get_candidate_via_list().size() > 1) {
      access_point.get_candidate_via_list().clear();
    }
    pa_box.get_pin_access_point_map()[pa_task->get_pa_pin()] = access_point;
  }
}

void PinAccessor::updateBestResult(PABox& pa_box)
{
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& best_net_task_access_result_map = pa_box.get_best_net_task_access_result_map();
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& best_net_task_access_patch_map = pa_box.get_best_net_task_access_patch_map();
  std::map<PAPin*, AccessPoint>& best_pin_access_point_map = pa_box.get_best_pin_access_point_map();
  std::vector<Violation>& best_route_violation_list = pa_box.get_best_route_violation_list();

  int32_t curr_violation_score = getViolationScore(pa_box.get_route_violation_list());
  if (!best_net_task_access_result_map.empty()) {
    if (getViolationScore(best_route_violation_list) < curr_violation_score) {
      return;
    }
  }
  best_net_task_access_result_map = pa_box.get_net_task_access_result_map();
  best_net_task_access_patch_map = pa_box.get_net_task_access_patch_map();
  best_pin_access_point_map = pa_box.get_pin_access_point_map();
  best_route_violation_list = pa_box.get_route_violation_list();
}

void PinAccessor::updateTaskSchedule(PABox& pa_box, std::vector<PATask*>& routing_task_list, int routing_rounds)
{
  int32_t max_routed_times = pa_box.get_pa_iter_param()->get_max_routed_times();

  std::set<PATask*, CmpPATask> visited_routing_task_set;
  std::vector<PATask*> new_routing_task_list;
  for (Violation& violation : pa_box.get_route_violation_list()) {
    EXTLayerRect& violation_shape = violation.get_violation_shape();
    if (!RTUTIL.isClosedOverlap(pa_box.get_box_rect().get_real_rect(),
                              RTUTIL.getEnlargedRect(violation_shape.get_real_rect(), RTDM.getOnlyPitch()))) {
      continue;
    }
    PlanarRect enlarged_rect = RTUTIL.getEnlargedRect(violation_shape.get_real_rect(), RTDM.getOnlyPitch());
    for (PATask* pa_task : pa_box.get_pa_task_list()) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), pa_task->get_net_idx())) {
        continue;
      }
      bool result_overlap = false;
      for (Segment<LayerCoord>& segment : pa_box.get_net_task_access_result_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(pa_task->get_net_idx(), segment)) {
          if (violation_shape.get_layer_idx() == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(enlarged_rect, net_shape.get_rect())) {
            result_overlap = true;
            break;
          }
        }
        if (result_overlap) {
          break;
        }
      }
      bool patch_overlap = false;
      for (EXTLayerRect& patch : pa_box.get_net_task_access_patch_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
        if (violation_shape.get_layer_idx() == patch.get_layer_idx() && RTUTIL.isClosedOverlap(enlarged_rect, patch.get_real_rect())) {
          patch_overlap = true;
          break;
        }
      }
      if (!result_overlap && !patch_overlap) {
        continue;
      }
      if (pa_task->get_routed_times() < max_routed_times && !RTUTIL.exist(visited_routing_task_set, pa_task)) {
        visited_routing_task_set.insert(pa_task);
        new_routing_task_list.push_back(pa_task);
      }
    }
  }
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    if (!RTUTIL.exist(pa_box.get_reroute_task_idx_set(), pa_task->get_task_idx())) {
      continue;
    }
    if (pa_task->get_routed_times() > 0 || pa_task->get_routed_times() >= max_routed_times || RTUTIL.exist(visited_routing_task_set, pa_task)) {
      continue;
    }
    visited_routing_task_set.insert(pa_task);
    new_routing_task_list.push_back(pa_task);
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
  if (routing_rounds % 2 == 1) {
    std::reverse(new_pa_task_list.begin(), new_pa_task_list.end());
  }
  pa_box.set_pa_task_list(new_pa_task_list);
}

void PinAccessor::uploadBestResult(PABox& pa_box, std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>*>>>& add_result_map,
                                   std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect*>>>& add_patch_map)
{
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    for (Segment<LayerCoord>& segment : pa_box.get_best_net_task_access_result_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
      Segment<LayerCoord>* new_segment = new Segment<LayerCoord>(segment);
      RTDM.updateNetPinAccessResultToGCellMap(ChangeType::kAdd, pa_task->get_net_idx(), pa_task->get_pa_pin()->get_pin_idx(), new_segment);
      add_result_map[pa_task->get_net_idx()][pa_task->get_pa_pin()->get_pin_idx()].push_back(new_segment);
    }
    for (EXTLayerRect& patch : pa_box.get_best_net_task_access_patch_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
      EXTLayerRect* new_patch = new EXTLayerRect(patch);
      RTDM.updateNetPinAccessPatchToGCellMap(ChangeType::kAdd, pa_task->get_net_idx(), pa_task->get_pa_pin()->get_pin_idx(), new_patch);
      add_patch_map[pa_task->get_net_idx()][pa_task->get_pa_pin()->get_pin_idx()].push_back(new_patch);
    }
  }
  for (auto& [pa_pin, access_point] : pa_box.get_best_pin_access_point_map()) {
    pa_pin->set_access_point(access_point);
  }
  for (Violation& violation : pa_box.get_best_route_violation_list()) {
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
  pa_box.get_reroute_task_idx_set().clear();
  pa_box.get_env_shape_list().clear();
  pa_box.get_net_pin_shape_map().clear();
  pa_box.get_layer_node_map().clear();
  pa_box.get_source_node_access_point_map().clear();
}

int32_t PinAccessor::getRouteViolationNum(PAModel& pa_model)
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void PinAccessor::uploadViolation(PAModel& pa_model, bool include_ap_via_only)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  pa_model.get_curr_route_violation_list().clear();
  std::set<Violation, CmpViolation> route_violation_set;
  std::vector<Violation> route_violation_list = getRouteViolationList(pa_model, false);
  uploadRouteViolationList(route_violation_set, route_violation_list, &pa_model.get_curr_route_violation_list());
  if (include_ap_via_only) {
    std::vector<Violation> ap_via_violation_list = getRouteViolationList(pa_model, true);
    uploadRouteViolationList(route_violation_set, ap_via_violation_list, &pa_model.get_curr_route_violation_list());
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

int32_t PinAccessor::uploadRouteViolationList(std::set<Violation, CmpViolation>& route_violation_set, const std::vector<Violation>& route_violation_list,
                                              std::vector<Violation>* add_violation_list)
{
  int32_t add_violation_num = 0;
  for (const Violation& violation : route_violation_list) {
    if (route_violation_set.insert(violation).second) {
      RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
      if (add_violation_list != nullptr) {
        add_violation_list->push_back(violation);
      }
      add_violation_num++;
    }
  }
  return add_violation_num;
}

std::vector<Violation> PinAccessor::getRouteViolationList(PAModel& pa_model, bool ap_via_only)
{
  Die& die = RTDM.getDatabase().get_die();

  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("pa_model");
    std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list = pa_model.get_fixed_env_shape_list();
    std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map = pa_model.get_fixed_net_pin_shape_map();
    if (env_shape_list.empty() && net_pin_shape_map.empty()) {
      for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(die)) {
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
    }
    std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
    for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
      for (auto& [pin_idx, segment_set] : pin_access_result_map) {
        if (!ap_via_only) {
          for (Segment<LayerCoord>* segment : segment_set) {
            net_result_map[net_idx].push_back(segment);
          }
          continue;
        }
        if (net_idx < 0 || net_idx >= static_cast<int32_t>(pa_model.get_pa_net_list().size())) {
          continue;
        }
        std::vector<PAPin>& pa_pin_list = pa_model.get_pa_net_list()[net_idx].get_pa_pin_list();
        if (pin_idx < 0 || pin_idx >= static_cast<int32_t>(pa_pin_list.size())) {
          continue;
        }
        LayerCoord access_coord = pa_pin_list[pin_idx].get_access_point().getRealLayerCoord();
        for (Segment<LayerCoord>* segment : segment_set) {
          if (isAPViaSegment(*segment, access_coord)) {
            net_result_map[net_idx].push_back(segment);
            break;
          }
        }
      }
    }
    std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
    if (!ap_via_only) {
      for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
        for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
          for (EXTLayerRect* patch : patch_set) {
            net_patch_map[net_idx].push_back(patch);
          }
        }
      }
    }
    std::set<int32_t> need_checked_net_set;
    for (PANet& pa_net : pa_model.get_pa_net_list()) {
      need_checked_net_set.insert(pa_net.get_net_idx());
    }

    de_task.set_proc_type(DEProcType::kGet);
    de_task.set_net_type(DENetType::kRouteHybrid);
    de_task.set_top_name(top_name);
    de_task.set_env_shape_list(env_shape_list);
    de_task.set_net_pin_shape_map(net_pin_shape_map);
    de_task.set_net_result_map(net_result_map);
    de_task.set_net_patch_map(net_patch_map);
    de_task.set_need_checked_net_set(need_checked_net_set);
  }
  std::vector<Violation> route_violation_list = RTDE.getViolationList(de_task);
  if (!ap_via_only) {
    return route_violation_list;
  }
  std::vector<Violation> result_violations;
  result_violations.reserve(route_violation_list.size());
  for (Violation& violation : route_violation_list) {
    if (violation.get_violation_net_set().size() <= 1) {
      continue;
    }
    result_violations.push_back(violation);
  }
  return result_violations;
}

void PinAccessor::updateBestResult(PAModel& pa_model, bool force_update)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& best_net_pin_access_result_map = pa_model.get_best_net_pin_access_result_map();
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& best_net_pin_access_patch_map = pa_model.get_best_net_pin_access_patch_map();
  std::vector<Violation>& best_route_violation_list = pa_model.get_best_route_violation_list();

  Die& die = RTDM.getDatabase().get_die();
  std::vector<Violation> curr_route_violation_list;
  for (Violation* violation : RTDM.getViolationSet(die)) {
    curr_route_violation_list.push_back(*violation);
  }
  int32_t curr_violation_score = getViolationScore(curr_route_violation_list);
  if (!force_update && !best_net_pin_access_result_map.empty()) {
    if (getViolationScore(best_route_violation_list) < curr_violation_score) {
      return;
    }
  }
  best_net_pin_access_result_map.clear();
  for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        best_net_pin_access_result_map[net_idx][pin_idx].push_back(*segment);
      }
    }
  }
  best_net_pin_access_patch_map.clear();
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        best_net_pin_access_patch_map[net_idx][pin_idx].push_back(*patch);
      }
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      pa_pin.set_best_access_point(pa_pin.get_access_point());
    }
  }
  best_route_violation_list.clear();
  for (Violation& violation : pa_model.get_curr_route_violation_list()) {
    best_route_violation_list.push_back(violation);
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool PinAccessor::stopIteration(PAModel& pa_model, std::vector<PAIterParam>& pa_iter_param_list)
{
  if (pa_model.get_iter() != static_cast<int32_t>(pa_iter_param_list.size()) && getRouteViolationNum(pa_model) == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

void PinAccessor::selectBestResult(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  pa_model.set_iter(pa_model.get_iter() + 1);
  uploadBestResult(pa_model);
  updateSummary(pa_model);
  printSummary(pa_model);
  outputNetCSV(pa_model);
  outputViolationCSV(pa_model);
  outputJson(pa_model);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::uploadBestResult(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        RTDM.updateNetPinAccessResultToGCellMap(ChangeType::kDel, net_idx, pin_idx, segment);
      }
    }
  }
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        RTDM.updateNetPinAccessPatchToGCellMap(ChangeType::kDel, net_idx, pin_idx, patch);
      }
    }
  }
  resetCurrResult(pa_model);
  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }

  for (auto& [net_idx, pin_access_result_map] : pa_model.get_best_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_list] : pin_access_result_map) {
      for (Segment<LayerCoord>& segment : segment_list) {
        Segment<LayerCoord>* new_segment = new Segment<LayerCoord>(segment);
        RTDM.updateNetPinAccessResultToGCellMap(ChangeType::kAdd, net_idx, pin_idx, new_segment);
        addCurrAccessResult(pa_model, net_idx, pin_idx, new_segment);
      }
    }
  }
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_best_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_list] : pin_access_patch_map) {
      for (EXTLayerRect& patch : patch_list) {
        EXTLayerRect* new_patch = new EXTLayerRect(patch);
        RTDM.updateNetPinAccessPatchToGCellMap(ChangeType::kAdd, net_idx, pin_idx, new_patch);
        addCurrAccessPatch(pa_model, net_idx, pin_idx, new_patch);
      }
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      pa_pin.set_access_point(pa_pin.get_best_access_point());
    }
  }
  for (Violation violation : pa_model.get_best_route_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
    pa_model.get_curr_route_violation_list().push_back(violation);
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::uploadAccessPoint(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();

  for (auto& [net_idx, access_point_list] : pa_model.get_curr_net_access_point_map()) {
    for (AccessPoint* access_point : access_point_list) {
      RTDM.updateNetAccessPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  resetCurrAccessPoint(pa_model);
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
      // output for debug
      Pin& origin_pin = origin_net->get_pin_list()[pa_pin.get_pin_idx()];
      if (origin_pin.get_pin_idx() != pa_pin.get_pin_idx()) {
        RTLOG.error(Loc::current(), "The pin idx is not equal!");
      }
      AccessPoint& access_point = pa_pin.get_access_point();
      std::vector<ViaMasterIdx>& candidate_via_list = access_point.get_candidate_via_list();
      if (candidate_via_list.size() == 1 && candidate_via_list.front().isValid()) {
        ViaMasterIdx& via_master_idx = candidate_via_list.front();
        int32_t below_layer_idx = via_master_idx.get_below_layer_idx();
        if (0 <= below_layer_idx && below_layer_idx < static_cast<int32_t>(layer_via_master_list.size()) && !layer_via_master_list[below_layer_idx].empty()) {
          ViaMasterIdx& default_via_master_idx = layer_via_master_list[below_layer_idx].front().get_via_master_idx();
          if (via_master_idx != default_via_master_idx) {
            RTLOG.info(Loc::current(), "PA upload non-default via master access point. net_idx=", pa_net.get_net_idx(),
                       " net_name=", origin_net->get_net_name(), " pin_idx=", pa_pin.get_pin_idx(), " pin_name=", origin_pin.get_pin_name(), " coord=(",
                       access_point.get_real_coord().get_x(), ",", access_point.get_real_coord().get_y(), ",", access_point.get_layer_idx(),
                       ") via_master_idx=(", via_master_idx.get_below_layer_idx(), ",", via_master_idx.get_via_idx(), ")");
          }
        }
      }
      access_point.set_grid_coord(RTUTIL.getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
      origin_pin.set_access_point(access_point);
      RTDM.updateNetAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &origin_pin.get_access_point());
      addCurrAccessPoint(pa_model, pa_net.get_net_idx(), &origin_pin.get_access_point());
    }
  }
  {
    std::map<int32_t, std::map<int32_t, std::vector<LayerCoord>>> net_pin_access_coord_map;
    for (auto& [net_idx, access_point_list] : pa_model.get_curr_net_access_point_map()) {
      for (AccessPoint* access_point : access_point_list) {
        net_pin_access_coord_map[net_idx][access_point->get_pin_idx()].push_back(access_point->getRealLayerCoord());
      }
    }
    for (auto& [net_idx, pin_access_coord_map] : net_pin_access_coord_map) {
      for (auto& [pin_idx, access_coord_list] : pin_access_coord_map) {
        if (access_coord_list.size() <= 1) {
          continue;
        }
        std::string net_name = "--";
        std::string pin_name = "--";
        if (0 <= net_idx && net_idx < static_cast<int32_t>(pa_model.get_pa_net_list().size())) {
          PANet& pa_net = pa_model.get_pa_net_list()[net_idx];
          if (pa_net.get_origin_net() != nullptr) {
            net_name = pa_net.get_origin_net()->get_net_name();
            if (0 <= pin_idx && pin_idx < static_cast<int32_t>(pa_net.get_origin_net()->get_pin_list().size())) {
              pin_name = pa_net.get_origin_net()->get_pin_list()[pin_idx].get_pin_name();
            }
          }
        }
        std::string coord_info;
        for (size_t i = 0; i < access_coord_list.size() && i < 4; i++) {
          LayerCoord& coord = access_coord_list[i];
          coord_info += RTUTIL.getString(" (", coord.get_x(), ",", coord.get_y(), ",", coord.get_layer_idx(), ")");
        }
        RTLOG.warn(Loc::current(), "PA upload duplicate access points. net_idx=", net_idx, " net_name=", net_name, " pin_idx=", pin_idx,
                   " pin_name=", pin_name, " access_point_num=", access_coord_list.size(), " coords=", coord_info);
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::uploadAccessResult(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<std::pair<std::pair<int32_t, int32_t>, Segment<LayerCoord>*>> result_list;
  for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        result_list.push_back({{net_idx, pin_idx}, segment});
      }
    }
  }
  for (auto& [net_pin_pair, segment] : result_list) {
    int32_t net_idx = net_pin_pair.first;
    int32_t pin_idx = net_pin_pair.second;
    RTDM.updateNetDetailedResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(*segment));
    delCurrAccessResult(pa_model, net_idx, pin_idx, segment);
    RTDM.updateNetPinAccessResultToGCellMap(ChangeType::kDel, net_idx, pin_idx, segment);
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::uploadAccessPatch(PAModel& pa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<std::pair<std::pair<int32_t, int32_t>, EXTLayerRect*>> patch_list;
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        patch_list.push_back({{net_idx, pin_idx}, patch});
      }
    }
  }
  for (auto& [net_pin_pair, patch] : patch_list) {
    int32_t net_idx = net_pin_pair.first;
    int32_t pin_idx = net_pin_pair.second;
    RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kAdd, net_idx, new EXTLayerRect(*patch));
    delCurrAccessPatch(pa_model, net_idx, pin_idx, patch);
    RTDM.updateNetPinAccessPatchToGCellMap(ChangeType::kDel, net_idx, pin_idx, patch);
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::addCurrAccessPoint(PAModel& pa_model, int32_t net_idx, AccessPoint* access_point)
{
  pa_model.get_curr_net_access_point_map()[net_idx].push_back(access_point);
}

void PinAccessor::delCurrAccessPoint(PAModel& pa_model, int32_t net_idx, AccessPoint* access_point)
{
  std::map<int32_t, std::vector<AccessPoint*>>& curr_access_point_map = pa_model.get_curr_net_access_point_map();
  auto net_iter = curr_access_point_map.find(net_idx);
  if (net_iter == curr_access_point_map.end()) {
    return;
  }
  std::vector<AccessPoint*>& access_point_list = net_iter->second;
  access_point_list.erase(std::remove(access_point_list.begin(), access_point_list.end(), access_point), access_point_list.end());
  if (access_point_list.empty()) {
    curr_access_point_map.erase(net_iter);
  }
}

void PinAccessor::resetCurrAccessPoint(PAModel& pa_model)
{
  pa_model.get_curr_net_access_point_map().clear();
}

void PinAccessor::addCurrAccessResult(PAModel& pa_model, int32_t net_idx, int32_t pin_idx, Segment<LayerCoord>* segment)
{
  pa_model.get_curr_net_pin_access_result_map()[net_idx][pin_idx].push_back(segment);
}

void PinAccessor::delCurrAccessResult(PAModel& pa_model, int32_t net_idx, int32_t pin_idx, Segment<LayerCoord>* segment)
{
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>*>>>& curr_result_map = pa_model.get_curr_net_pin_access_result_map();
  auto net_iter = curr_result_map.find(net_idx);
  if (net_iter != curr_result_map.end()) {
    auto pin_iter = net_iter->second.find(pin_idx);
    if (pin_iter != net_iter->second.end()) {
      std::vector<Segment<LayerCoord>*>& segment_list = pin_iter->second;
      segment_list.erase(std::remove(segment_list.begin(), segment_list.end(), segment), segment_list.end());
      if (segment_list.empty()) {
        net_iter->second.erase(pin_iter);
      }
      if (net_iter->second.empty()) {
        curr_result_map.erase(net_iter);
      }
    }
  }
}

void PinAccessor::addCurrAccessPatch(PAModel& pa_model, int32_t net_idx, int32_t pin_idx, EXTLayerRect* patch)
{
  pa_model.get_curr_net_pin_access_patch_map()[net_idx][pin_idx].push_back(patch);
}

void PinAccessor::delCurrAccessPatch(PAModel& pa_model, int32_t net_idx, int32_t pin_idx, EXTLayerRect* patch)
{
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect*>>>& curr_patch_map = pa_model.get_curr_net_pin_access_patch_map();
  auto net_iter = curr_patch_map.find(net_idx);
  if (net_iter != curr_patch_map.end()) {
    auto pin_iter = net_iter->second.find(pin_idx);
    if (pin_iter != net_iter->second.end()) {
      std::vector<EXTLayerRect*>& patch_list = pin_iter->second;
      patch_list.erase(std::remove(patch_list.begin(), patch_list.end(), patch), patch_list.end());
      if (patch_list.empty()) {
        net_iter->second.erase(pin_iter);
      }
      if (net_iter->second.empty()) {
        curr_patch_map.erase(net_iter);
      }
    }
  }
}

void PinAccessor::resetCurrResult(PAModel& pa_model)
{
  pa_model.get_curr_net_pin_access_result_map().clear();
  pa_model.get_curr_net_pin_access_patch_map().clear();
  pa_model.get_curr_route_violation_list().clear();
}

#if 1  // update env

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  updateNetShapeToGraph(pa_box, change_type, net_shape, true);
}

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  updateNetShapeToGraph(pa_box, change_type, net_shape, true);
}

void PinAccessor::updateFixedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
    updateNetShapeToGraph(pa_box, change_type, net_shape, true);
  }
}

void PinAccessor::updateRoutedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  updateNetShapeToGraph(pa_box, change_type, net_shape, false);
}

void PinAccessor::updateRoutedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, segment)) {
    updateNetShapeToGraph(pa_box, change_type, net_shape, false);
  }
}

void PinAccessor::updateRoutedRectToGraph(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, routed_rect.getRealLayerRect(), is_routing);
  updateNetShapeToGraph(pa_box, change_type, net_shape, false);
}

void PinAccessor::addRouteViolationToGraph(PABox& pa_box, Violation& violation)
{
  LayerRect searched_rect = violation.get_violation_shape().get_real_rect();
  std::vector<Segment<LayerCoord>> overlap_segment_list;
  std::set<int32_t> target_net_set;
  std::set<int32_t> found_net_set;
  for (int32_t net_idx : violation.get_violation_net_set()) {
    if (net_idx != -1) {
      target_net_set.insert(net_idx);
    }
  }
  if (target_net_set.empty()) {
    return;
  }
  int32_t searched_times = 0;
  constexpr int32_t max_searched_times = 3;
  // find segements for all violation nets
  while (true) {
    searched_rect.set_rect(RTUTIL.getEnlargedRect(searched_rect, RTDM.getOnlyPitch()));
    searched_times++;
    if (violation.get_is_routing()) {
      searched_rect.set_layer_idx(violation.get_violation_shape().get_layer_idx());
    } else {
      RTLOG.error(Loc::current(), "The violation layer is cut!");
    }
    for (auto& [net_idx, task_access_result_map] : pa_box.get_net_task_access_result_map()) {
      if (!RTUTIL.exist(target_net_set, net_idx) || RTUTIL.exist(found_net_set, net_idx)) {
        continue;
      }
      for (auto& [task_idx, segment_list] : task_access_result_map) {
        for (Segment<LayerCoord>& segment : segment_list) {
          bool is_overlap = false;
          for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, segment)) {
            if (searched_rect.get_layer_idx() == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(searched_rect, net_shape.get_rect())) {
              is_overlap = true;
              break;
            }
          }
          if (is_overlap) {
            overlap_segment_list.push_back(segment);
            found_net_set.insert(net_idx);
            // break;
          }
        }
      }
    }
    if (found_net_set.size() == target_net_set.size()) {
      break;
    }
    if (searched_times >= max_searched_times) {
      break;
    }
    if (!RTUTIL.isInside(pa_box.get_box_rect().get_real_rect(), searched_rect)) {
      break;
    }
  }
  addRouteViolationToGraph(pa_box, searched_rect, overlap_segment_list);
}

void PinAccessor::addRouteViolationToGraph(PABox& pa_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list)
{
  ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
  std::vector<GridMap<PANode>>& layer_node_map = pa_box.get_layer_node_map();

  for (Segment<LayerCoord>& overlap_segment : overlap_segment_list) {
    LayerCoord& first_coord = overlap_segment.get_first();
    LayerCoord& second_coord = overlap_segment.get_second();
    if (first_coord == second_coord) {
      continue;
    }
    PlanarRect real_rect = RTUTIL.getRect(first_coord, second_coord);
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
        valid_node->addViolationNumber(oppo_orientation);
        PANode* neighbor_node = valid_node->getNeighborNode(oppo_orientation);
        if (neighbor_node != nullptr) {
          neighbor_node->addViolationNumber(orientation);
        }
      }
      if (LayerCoord(*valid_node) != second_coord) {
        valid_node->addViolationNumber(orientation);
        PANode* neighbor_node = valid_node->getNeighborNode(orientation);
        if (neighbor_node != nullptr) {
          neighbor_node->addViolationNumber(oppo_orientation);
        }
      }
    }
  }
}

void PinAccessor::updateNetShapeToGraph(PABox& pa_box, ChangeType change_type, NetShape& net_shape, bool is_fixed)
{
  if (net_shape.get_is_routing()) {
    updateRoutingNetShapeToGraph(pa_box, change_type, net_shape, is_fixed);
  } else {
    updateCutNetShapeToGraph(pa_box, change_type, net_shape, is_fixed);
  }
}

void PinAccessor::updateNodeNetToGraph(PANode& pa_node, ChangeType change_type, int32_t net_idx, Orientation orientation, bool is_fixed)
{
  if (change_type == ChangeType::kAdd) {
    if (is_fixed) {
      pa_node.addFixedRectNet(orientation, net_idx);
    } else {
      pa_node.addRoutedRectNet(orientation, net_idx);
    }
  } else if (change_type == ChangeType::kDel) {
    if (is_fixed) {
      pa_node.delFixedRectNet(orientation, net_idx);
    } else {
      pa_node.delRoutedRectNet(orientation, net_idx);
    }
  }
}

void PinAccessor::updateRoutingNetShapeToGraph(PABox& pa_box, ChangeType change_type, NetShape& net_shape, bool is_fixed)
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
    int32_t max_eol_spacing = std::max(routing_layer.get_eol_spacing(), routing_layer.get_eol_ete());
    if (routing_layer.isPreferH()) {
      spacing_pair_list.emplace_back(max_eol_spacing, routing_layer.get_eol_within());
    } else {
      spacing_pair_list.emplace_back(routing_layer.get_eol_within(), max_eol_spacing);
    }
  }
  int32_t half_wire_width = routing_layer.get_min_width() / 2;
  PlanarRect& enclosure = layer_enclosure_map[layer_idx];
  int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
  int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;

  GridMap<PANode>& pa_node_map = pa_box.get_layer_node_map()[layer_idx];
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 half_wire_width + spacing
    int32_t enlarged_x_size = half_wire_width + x_spacing;
    int32_t enlarged_y_size = half_wire_width + y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect planar_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [grid, orientation_set] : RTUTIL.getTrackGridOrientationMap(planar_enlarged_rect, pa_box.get_box_track_axis())) {
      for (int32_t x : *grid.first) {
        for (int32_t y : *grid.second) {
          PANode& node = pa_node_map[x][y];
          for (const Orientation& orientation : orientation_set) {
            if (orientation == Orientation::kAbove || orientation == Orientation::kBelow) {
              continue;
            }
            PANode* neighbor_node = node.getNeighborNode(orientation);
            if (neighbor_node == nullptr) {
              continue;
            }
            updateNodeNetToGraph(node, change_type, net_shape.get_net_idx(), orientation, is_fixed);
            updateNodeNetToGraph(*neighbor_node, change_type, net_shape.get_net_idx(), RTUTIL.getOppositeOrientation(orientation), is_fixed);
          }
        }
      }
    }
  }
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 enclosure_half_span + spacing
    int32_t enlarged_x_size = enclosure_half_x_span + x_spacing;
    int32_t enlarged_y_size = enclosure_half_y_span + y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [grid, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, pa_box.get_box_track_axis())) {
      for (int32_t x : *grid.first) {
        for (int32_t y : *grid.second) {
          PANode& node = pa_node_map[x][y];
          for (const Orientation& orientation : orientation_set) {
            if (orientation == Orientation::kEast || orientation == Orientation::kWest || orientation == Orientation::kSouth
                || orientation == Orientation::kNorth) {
              continue;
            }
            PANode* neighbor_node = node.getNeighborNode(orientation);
            if (neighbor_node == nullptr) {
              continue;
            }
            updateNodeNetToGraph(node, change_type, net_shape.get_net_idx(), orientation, is_fixed);
            updateNodeNetToGraph(*neighbor_node, change_type, net_shape.get_net_idx(), RTUTIL.getOppositeOrientation(orientation), is_fixed);
          }
        }
      }
    }
  }
}

void PinAccessor::updateCutNetShapeToGraph(PABox& pa_box, ChangeType change_type, NetShape& net_shape, bool is_fixed)
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
        // prl
        spacing_pair_list.emplace_back(0, cut_layer.get_curr_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_curr_spacing() / RT_SQRT_2, cut_layer.get_curr_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_curr_prl(), cut_layer.get_curr_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_prl_spacing(), cut_layer.get_curr_prl());
        // eol
        spacing_pair_list.emplace_back(0, cut_layer.get_curr_eol_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_spacing() / RT_SQRT_2, cut_layer.get_curr_eol_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_prl(), cut_layer.get_curr_eol_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_prl_spacing(), cut_layer.get_curr_eol_prl());
        cut_spacing_map[curr_cut_layer_idx] = spacing_pair_list;
      }
    }
    int32_t below_cut_layer_idx = net_shape.get_layer_idx() - 1;
    if (0 <= below_cut_layer_idx && below_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[below_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        // prl
        spacing_pair_list.emplace_back(0, cut_layer.get_below_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_below_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_below_spacing() / RT_SQRT_2, cut_layer.get_below_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_below_prl(), cut_layer.get_below_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_below_prl_spacing(), cut_layer.get_below_prl());
        cut_spacing_map[below_cut_layer_idx] = spacing_pair_list;
      }
    }
    int32_t above_cut_layer_idx = net_shape.get_layer_idx() + 1;
    if (0 <= above_cut_layer_idx && above_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[above_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        // prl
        spacing_pair_list.emplace_back(0, cut_layer.get_above_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_above_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_above_spacing() / RT_SQRT_2, cut_layer.get_above_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_above_prl(), cut_layer.get_above_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_above_prl_spacing(), cut_layer.get_above_prl());
        cut_spacing_map[above_cut_layer_idx] = spacing_pair_list;
      }
    }
  }
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
      // 膨胀size为 cut_shape_half_span + spacing
      int32_t enlarged_x_size = cut_shape_half_x_span + x_spacing;
      int32_t enlarged_y_size = cut_shape_half_y_span + y_spacing;
      // 贴合的也不算违例
      enlarged_x_size -= 1;
      enlarged_y_size -= 1;
      PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
      for (auto& [grid, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, pa_box.get_box_track_axis())) {
        for (int32_t x : *grid.first) {
          for (int32_t y : *grid.second) {
            if (!RTUTIL.exist(orientation_set, Orientation::kAbove) && !RTUTIL.exist(orientation_set, Orientation::kBelow)) {
              continue;
            }
            PANode& below_node = layer_node_map[below_routing_layer_idx][x][y];
            if (below_node.hasNeighborNode(Orientation::kAbove)) {
              updateNodeNetToGraph(below_node, change_type, net_shape.get_net_idx(), Orientation::kAbove, is_fixed);
            }
            PANode& above_node = layer_node_map[above_routing_layer_idx][x][y];
            if (above_node.hasNeighborNode(Orientation::kBelow)) {
              updateNodeNetToGraph(above_node, change_type, net_shape.get_net_idx(), Orientation::kBelow, is_fixed);
            }
          }
        }
      }
    }
  }
}

void PinAccessor::updateFixedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(pa_box, net_shape)) {
    PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      pa_shadow.get_net_fixed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      pa_shadow.get_net_fixed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void PinAccessor::updateFixedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(pa_box, net_shape)) {
    PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      pa_shadow.get_net_fixed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      pa_shadow.get_net_fixed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void PinAccessor::updateFixedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
    if (!net_shape.get_is_routing()) {
      continue;
    }
    for (PlanarRect& shadow_shape : getShadowShape(pa_box, net_shape)) {
      PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
      if (change_type == ChangeType::kAdd) {
        pa_shadow.get_net_fixed_rect_map()[net_idx].insert(shadow_shape);
      } else if (change_type == ChangeType::kDel) {
        pa_shadow.get_net_fixed_rect_map()[net_idx].erase(shadow_shape);
      }
    }
  }
}

void PinAccessor::updateRoutedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(pa_box, net_shape)) {
    PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      pa_shadow.get_net_routed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      pa_shadow.get_net_routed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void PinAccessor::updateRoutedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, segment)) {
    if (!net_shape.get_is_routing()) {
      continue;
    }
    for (PlanarRect& shadow_shape : getShadowShape(pa_box, net_shape)) {
      PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
      if (change_type == ChangeType::kAdd) {
        pa_shadow.get_net_routed_rect_map()[net_idx].insert(shadow_shape);
      } else if (change_type == ChangeType::kDel) {
        pa_shadow.get_net_routed_rect_map()[net_idx].erase(shadow_shape);
      }
    }
  }
}

void PinAccessor::updateRoutedRectToShadow(PABox& pa_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, routed_rect.getRealLayerRect(), is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(pa_box, net_shape)) {
    PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      pa_shadow.get_net_routed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      pa_shadow.get_net_routed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void PinAccessor::addPatchViolationToShadow(PABox& pa_box, Violation& violation)
{
  EXTLayerRect& violation_shape = violation.get_violation_shape();

  PAShadow& pa_shadow = pa_box.get_layer_shadow_map()[violation_shape.get_layer_idx()];
  pa_shadow.get_violation_set().insert(violation_shape.get_real_rect());
}

std::vector<PlanarRect> PinAccessor::getShadowShape(PABox& pa_box, NetShape& net_shape)
{
  std::vector<PlanarRect> shadow_shape_list;
  if (net_shape.get_is_routing()) {
    shadow_shape_list = getRoutingShadowShapeList(pa_box, net_shape);
  } else {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  return shadow_shape_list;
}

std::vector<PlanarRect> PinAccessor::getRoutingShadowShapeList(PABox& pa_box, NetShape& net_shape)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
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
    int32_t max_eol_spacing = std::max(routing_layer.get_eol_spacing(), routing_layer.get_eol_ete());
    if (routing_layer.isPreferH()) {
      spacing_pair_list.emplace_back(max_eol_spacing, routing_layer.get_eol_within());
    } else {
      spacing_pair_list.emplace_back(routing_layer.get_eol_within(), max_eol_spacing);
    }
  }
  std::vector<PlanarRect> shadow_shape_list;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 spacing
    int32_t enlarged_x_size = x_spacing;
    int32_t enlarged_y_size = y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    shadow_shape_list.push_back(RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size));
  }
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 spacing
    int32_t enlarged_x_size = x_spacing;
    int32_t enlarged_y_size = y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    shadow_shape_list.push_back(RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size));
  }
  return shadow_shape_list;
}

#endif

#if 1  // get env

double PinAccessor::getFixedRectCost(PABox& pa_box, int32_t net_idx, EXTLayerRect& patch)
{
  double fixed_rect_unit = pa_box.get_pa_iter_param()->get_fixed_rect_unit();
  std::vector<PAShadow>& layer_shadow_map = pa_box.get_layer_shadow_map();

  double fixed_rect_cost = 0;
  for (auto& [graph_net_idx, fixed_rect_set] : layer_shadow_map[patch.get_layer_idx()].get_net_fixed_rect_map()) {
    if (net_idx == graph_net_idx) {
      continue;
    }
    for (const PlanarRect& fixed_rect : fixed_rect_set) {
      if (RTUTIL.isOpenOverlap(patch.get_real_rect(), fixed_rect)) {
        fixed_rect_cost += fixed_rect_unit;
      }
    }
  }
  return fixed_rect_cost;
}

double PinAccessor::getRoutedRectCost(PABox& pa_box, int32_t net_idx, EXTLayerRect& patch)
{
  double routed_rect_unit = pa_box.get_pa_iter_param()->get_routed_rect_unit();
  std::vector<PAShadow>& layer_shadow_map = pa_box.get_layer_shadow_map();

  double routed_rect_cost = 0;
  for (auto& [graph_net_idx, routed_rect_set] : layer_shadow_map[patch.get_layer_idx()].get_net_routed_rect_map()) {
    if (net_idx == graph_net_idx) {
      continue;
    }
    for (const PlanarRect& routed_rect : routed_rect_set) {
      if (RTUTIL.isOpenOverlap(patch.get_real_rect(), routed_rect)) {
        routed_rect_cost += routed_rect_unit;
      }
    }
  }
  return routed_rect_cost;
}

double PinAccessor::getViolationCost(PABox& pa_box, int32_t net_idx, EXTLayerRect& patch)
{
  double violation_unit = pa_box.get_pa_iter_param()->get_violation_unit();
  std::vector<PAShadow>& layer_shadow_map = pa_box.get_layer_shadow_map();

  double violation_cost = 0;
  for (const PlanarRect& violation : layer_shadow_map[patch.get_layer_idx()].get_violation_set()) {
    if (RTUTIL.isOpenOverlap(patch.get_real_rect(), violation)) {
      violation_cost += violation_unit;
    }
  }
  return violation_cost;
}

#endif

#if 1  // exhibit

void PinAccessor::updateSummary(PAModel& pa_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_pa_summary_map[pa_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_violation_num;

  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  routing_patch_num_map.clear();
  total_patch_num = 0;
  routing_violation_num_map.clear();
  total_violation_num = 0;

  for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        LayerCoord& first_coord = segment->get_first();
        int32_t first_layer_idx = first_coord.get_layer_idx();
        LayerCoord& second_coord = segment->get_second();
        int32_t second_layer_idx = second_coord.get_layer_idx();

        if (first_layer_idx == second_layer_idx) {
          double wire_length = RTUTIL.getManhattanDistance(first_coord, second_coord) / 1.0 / micron_dbu;
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
  }
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        routing_patch_num_map[patch->get_layer_idx()]++;
        total_patch_num++;
      }
    }
  }
  for (Violation& violation : pa_model.get_curr_route_violation_list()) {
    routing_violation_num_map[violation.get_violation_shape().get_layer_idx()]++;
    total_violation_num++;
  }
}

void PinAccessor::printSummary(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_pa_summary_map[pa_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_violation_num;

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
  fort::char_table routing_patch_num_map_table;
  {
    routing_patch_num_map_table.set_cell_text_align(fort::text_align::right);
    routing_patch_num_map_table << fort::header << "routing"
                                << "#patch"
                                << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_patch_num_map_table << routing_layer.get_layer_name() << routing_patch_num_map[routing_layer.get_layer_idx()]
                                  << RTUTIL.getPercentage(routing_patch_num_map[routing_layer.get_layer_idx()], total_patch_num) << fort::endr;
    }
    routing_patch_num_map_table << fort::header << "Total" << total_patch_num << RTUTIL.getPercentage(total_patch_num, total_patch_num) << fort::endr;
  }
  fort::char_table routing_violation_num_map_table;
  {
    routing_violation_num_map_table.set_cell_text_align(fort::text_align::right);
    routing_violation_num_map_table << fort::header << "routing"
                                    << "#violation"
                                    << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_violation_num_map_table << routing_layer.get_layer_name() << routing_violation_num_map[routing_layer.get_layer_idx()]
                                      << RTUTIL.getPercentage(routing_violation_num_map[routing_layer.get_layer_idx()], total_violation_num) << fort::endr;
    }
    routing_violation_num_map_table << fort::header << "Total" << total_violation_num << RTUTIL.getPercentage(total_violation_num, total_violation_num)
                                    << fort::endr;
  }
  RTUTIL.printTableList({routing_wire_length_map_table, cut_via_num_map_table, routing_patch_num_map_table});
  RTUTIL.printTableList({routing_violation_num_map_table});
}

void PinAccessor::outputNetCSV(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<GridMap<int32_t>> layer_net_map;
  layer_net_map.resize(routing_layer_list.size());
  for (GridMap<int32_t>& net_map : layer_net_map) {
    net_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<int32_t, std::set<int32_t>> net_layer_map;
      for (auto& [net_idx, pin_access_result_map] : gcell_map[x][y].get_net_pin_access_result_map()) {
        for (auto& [pin_idx, segment_set] : pin_access_result_map) {
          for (Segment<LayerCoord>* segment : segment_set) {
            int32_t first_layer_idx = segment->get_first().get_layer_idx();
            int32_t second_layer_idx = segment->get_second().get_layer_idx();
            RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
            for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
              net_layer_map[net_idx].insert(layer_idx);
            }
          }
        }
      }
      for (auto& [net_idx, pin_access_patch_map] : gcell_map[x][y].get_net_pin_access_patch_map()) {
        for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
          for (EXTLayerRect* patch : patch_set) {
            net_layer_map[net_idx].insert(patch->get_layer_idx());
          }
        }
      }
      for (auto& [net_idx, layer_set] : net_layer_map) {
        for (int32_t layer_idx : layer_set) {
          layer_net_map[layer_idx][x][y]++;
        }
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* net_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(pa_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", pa_model.get_iter(), ".csv"));
    GridMap<int32_t>& net_map = layer_net_map[routing_layer.get_layer_idx()];
    for (int32_t y = net_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < net_map.get_x_size(); x++) {
        RTUTIL.pushStream(net_csv_file, net_map[x][y], ",");
      }
      RTUTIL.pushStream(net_csv_file, "\n");
    }
    RTUTIL.closeFileStream(net_csv_file);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::outputViolationCSV(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<GridMap<int32_t>> layer_violation_map;
  layer_violation_map.resize(routing_layer_list.size());
  for (GridMap<int32_t>& violation_map : layer_violation_map) {
    violation_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (Violation* violation : gcell_map[x][y].get_violation_set()) {
        layer_violation_map[violation->get_violation_shape().get_layer_idx()][x][y]++;
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* violation_csv_file = RTUTIL.getOutputFileStream(
        RTUTIL.getString(pa_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), "_", pa_model.get_iter(), ".csv"));
    GridMap<int32_t>& violation_map = layer_violation_map[routing_layer.get_layer_idx()];
    for (int32_t y = violation_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < violation_map.get_x_size(); x++) {
        RTUTIL.pushStream(violation_csv_file, violation_map[x][y], ",");
      }
      RTUTIL.pushStream(violation_csv_file, "\n");
    }
    RTUTIL.closeFileStream(violation_csv_file);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinAccessor::outputJson(PAModel& pa_model)
{
  int32_t enable_notification = RTDM.getConfig().enable_notification;
  if (!enable_notification) {
    return;
  }
  std::map<std::string, std::string> json_path_map;
  json_path_map["net_map"] = outputNetJson(pa_model);
  json_path_map["violation_map"] = outputViolationJson(pa_model);
  json_path_map["summary"] = outputSummaryJson(pa_model);
  RTI.sendNotification("PA", pa_model.get_iter(), json_path_map);
}

std::string PinAccessor::outputNetJson(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;

  std::vector<nlohmann::json> net_json_list;
  {
    nlohmann::json result_shape_json;
    for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
      std::string net_name = net_list[net_idx].get_net_name();
      for (auto& [pin_idx, segment_set] : pin_access_result_map) {
        for (Segment<LayerCoord>* segment : segment_set) {
          for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
            std::string layer_name;
            if (net_shape.get_is_routing()) {
              layer_name = routing_layer_list[net_shape.get_layer_idx()].get_layer_name();
            } else {
              layer_name = cut_layer_list[net_shape.get_layer_idx()].get_layer_name();
            }
            result_shape_json["result_shape"][net_name]["path"].push_back(
                {net_shape.get_ll_x(), net_shape.get_ll_y(), net_shape.get_ur_x(), net_shape.get_ur_y(), layer_name});
          }
        }
      }
    }
    for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
      std::string net_name = net_list[net_idx].get_net_name();
      for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
        for (EXTLayerRect* patch : patch_set) {
          result_shape_json["result_shape"][net_name]["patch"].push_back({patch->get_real_ll_x(), patch->get_real_ll_y(), patch->get_real_ur_x(),
                                                                          patch->get_real_ur_y(), routing_layer_list[patch->get_layer_idx()].get_layer_name()});
        }
      }
    }
    net_json_list.push_back(result_shape_json);
  }
  std::string net_json_file_path = RTUTIL.getString(pa_temp_directory_path, "net_map_", pa_model.get_iter(), ".json");
  std::ofstream* net_json_file = RTUTIL.getOutputFileStream(net_json_file_path);
  (*net_json_file) << net_json_list;
  RTUTIL.closeFileStream(net_json_file);
  return net_json_file_path;
}

std::string PinAccessor::outputViolationJson(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;

  std::vector<nlohmann::json> violation_json_list;
  for (Violation& violation : pa_model.get_curr_route_violation_list()) {
    EXTLayerRect& violation_shape = violation.get_violation_shape();

    nlohmann::json violation_json;
    violation_json["type"] = GetViolationTypeName()(violation.get_violation_type());
    violation_json["shape"]
        = {violation_shape.get_real_rect().get_ll_x(), violation_shape.get_real_rect().get_ll_y(), violation_shape.get_real_rect().get_ur_x(),
           violation_shape.get_real_rect().get_ur_y(), routing_layer_list[violation_shape.get_layer_idx()].get_layer_name()};
    for (int32_t net_idx : violation.get_violation_net_set()) {
      if (net_idx != -1) {
        violation_json["net"].push_back(net_list[net_idx].get_net_name());
      } else {
        violation_json["net"].push_back("obs");
      }
    }
    violation_json_list.push_back(violation_json);
  }
  std::string violation_json_file_path = RTUTIL.getString(pa_temp_directory_path, "violation_map_", pa_model.get_iter(), ".json");
  std::ofstream* violation_json_file = RTUTIL.getOutputFileStream(violation_json_file_path);
  (*violation_json_file) << violation_json_list;
  RTUTIL.closeFileStream(violation_json_file);
  return violation_json_file_path;
}

std::string PinAccessor::outputSummaryJson(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_pa_summary_map[pa_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_pa_summary_map[pa_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_pa_summary_map[pa_model.get_iter()].total_violation_num;

  nlohmann::json summary_json;
  summary_json["iter"] = pa_model.get_iter();
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    summary_json["routing_wire_length_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = wire_length;
  }
  summary_json["total_wire_length"] = total_wire_length;
  for (auto& [cut_layer_idx, via_num] : cut_via_num_map) {
    summary_json["cut_via_num_map"][cut_layer_list[cut_layer_idx].get_layer_name()] = via_num;
  }
  summary_json["total_via_num"] = total_via_num;
  for (auto& [routing_layer_idx, patch_num] : routing_patch_num_map) {
    summary_json["routing_patch_num_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = patch_num;
  }
  summary_json["total_patch_num"] = total_patch_num;
  for (auto& [routing_layer_idx, violation_num] : routing_violation_num_map) {
    summary_json["routing_violation_num_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = violation_num;
  }
  summary_json["total_violation_num"] = total_violation_num;

  std::string summary_json_file_path = RTUTIL.getString(pa_temp_directory_path, "summary_", pa_model.get_iter(), ".json");
  std::ofstream* summary_json_file = RTUTIL.getOutputFileStream(summary_json_file_path);
  (*summary_json_file) << summary_json;
  RTUTIL.closeFileStream(summary_json_file);
  return summary_json_file_path;
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

  // base_region
  {
    GPStruct base_region_struct("base_region");
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(die.get_real_rect());
    base_region_struct.push(gp_boundary);
    gp_gds.addStruct(base_region_struct);
  }

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

  // access_point
  for (auto& [net_idx, access_point_set] : pa_model.get_curr_net_access_point_map()) {
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

  // access result
  for (auto& [net_idx, pin_access_result_map] : pa_model.get_curr_net_pin_access_result_map()) {
    GPStruct access_result_struct(RTUTIL.getString("access_result(net_", net_idx, ")"));
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kDetailedPath));
          gp_boundary.set_rect(net_shape.get_rect());
          if (net_shape.get_is_routing()) {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
          } else {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
          }
          access_result_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(access_result_struct);
  }

  // access patch
  for (auto& [net_idx, pin_access_patch_map] : pa_model.get_curr_net_pin_access_patch_map()) {
    GPStruct access_patch_struct(RTUTIL.getString("access_patch(net_", net_idx, ")"));
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatch));
        gp_boundary.set_rect(patch->get_real_rect());
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
        access_patch_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(access_patch_struct);
  }

  // violation
  {
    for (Violation& violation : pa_model.get_curr_route_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kRouteViolation));
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
        pa_node.forEachNeighborNode([&](Orientation orient, PANode* neighbor) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          PANode* opposite_neighbor = neighbor->getNeighborNode(opposite_orient);
          if (opposite_neighbor == nullptr) {
            RTLOG.error(Loc::current(), "The pa_node neighbor is not bidirectional!");
          }
          if (opposite_neighbor != &pa_node) {
            RTLOG.error(Loc::current(), "The pa_node neighbor is not bidirectional!");
          }
          if (RTUTIL.getOrientation(LayerCoord(pa_node), LayerCoord(*neighbor)) == orient) {
            return;
          }
          RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
        });
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
        if (!pa_node.hasAnyNeighborNode()) {
          RTLOG.error(Loc::current(), "The neighbor of group coord (", coord.get_x(), ",", coord.get_y(), ",", layer_idx, ") is empty in box(",
                      pa_box_id.get_x(), ",", pa_box_id.get_y(), ")");
        }
        if (RTUTIL.isInside(pa_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        RTLOG.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
  }
}

void PinAccessor::debugPlotPABox(PABox& pa_box, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& pa_temp_directory_path = RTDM.getConfig().pa_temp_directory_path;

  PlanarRect box_real_rect = pa_box.get_box_rect().get_real_rect();

  int32_t point_size = 5;

  GPGDS gp_gds;

  // base_region
  {
    GPStruct base_region_struct("base_region");
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(box_real_rect);
    base_region_struct.push(gp_boundary);
    gp_gds.addStruct(base_region_struct);
  }

  // gcell_axis
  {
    GPStruct gcell_axis_struct("gcell_axis");
    for (int32_t x : RTUTIL.getScaleList(box_real_rect.get_ll_x(), box_real_rect.get_ur_x(), gcell_axis.get_x_grid_list())) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(x, box_real_rect.get_ll_y(), x, box_real_rect.get_ur_y());
      gcell_axis_struct.push(gp_path);
    }
    for (int32_t y : RTUTIL.getScaleList(box_real_rect.get_ll_y(), box_real_rect.get_ur_y(), gcell_axis.get_y_grid_list())) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(box_real_rect.get_ll_x(), y, box_real_rect.get_ur_x(), y);
      gcell_axis_struct.push(gp_path);
    }
    gp_gds.addStruct(gcell_axis_struct);
  }

  // box_track_axis
  {
    GPStruct box_track_axis_struct("box_track_axis");
    PlanarCoord& real_ll = box_real_rect.get_ll();
    PlanarCoord& real_ur = box_real_rect.get_ur();
    ScaleAxis& box_track_axis = pa_box.get_box_track_axis();
    std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
    std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());
    for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(routing_layer_list.size()); layer_idx++) {
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

  // access_point
  for (auto& [net_idx, access_point_set] : pa_box.get_net_access_point_map()) {
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

  // access result
  for (auto& [net_idx, pin_access_result_map] : pa_box.get_net_pin_access_result_map()) {
    GPStruct access_result_struct(RTUTIL.getString("access_result(net_", net_idx, ")"));
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
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
    }
    gp_gds.addStruct(access_result_struct);
  }

  // access patch
  for (auto& [net_idx, pin_access_patch_map] : pa_box.get_net_pin_access_patch_map()) {
    GPStruct access_patch_struct(RTUTIL.getString("access_patch(net_", net_idx, ")"));
    for (auto& [pin_idx, patch_set] : pin_access_patch_map) {
      for (EXTLayerRect* patch : patch_set) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
        gp_boundary.set_rect(patch->get_real_rect());
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
        access_patch_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(access_patch_struct);
  }

  // layer_node_map
  {
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
            gp_text_node_real_coord.set_message(RTUTIL.getString("(", pa_node.get_x(), " , ", pa_node.get_y(), " , ", pa_node.get_layer_idx(), ")"));
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

            if (pa_node.hasOrientFixedRect()) {
              y -= y_reduced_span;
              GPText gp_text_orient_fixed_rect_map_info;
              gp_text_orient_fixed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_fixed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_fixed_rect_map_info_message = "--";
              for (auto& [orient, net_list] : pa_node.get_orient_fixed_rect_list()) {
                orient_fixed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
                for (int32_t net_idx : net_list) {
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

            if (pa_node.hasOrientRoutedRect()) {
              y -= y_reduced_span;
              GPText gp_text_orient_routed_rect_map_info;
              gp_text_orient_routed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_routed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_routed_rect_map_info_message = "--";
              for (auto& [orient, net_list] : pa_node.get_orient_routed_rect_list()) {
                orient_routed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
                for (int32_t net_idx : net_list) {
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

            if (pa_node.hasOrientViolationNumber()) {
              y -= y_reduced_span;
              GPText gp_text_orient_violation_number_map_info;
              gp_text_orient_violation_number_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_violation_number_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_violation_number_map_info_message = "--";
              pa_node.forEachViolationNumber([&](Orientation orient, int32_t violation_number) {
                orient_violation_number_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", violation_number != 0, ")");
              });
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

            pa_node.forEachNeighborNode([&](Orientation orientation, PANode*) {
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
            });
          }
        }
      }
      gp_gds.addStruct(neighbor_map_struct);
    }
  }

  // layer_shadow_map
  {
    std::vector<PAShadow>& layer_shadow_map = pa_box.get_layer_shadow_map();
    for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(routing_layer_list.size()); layer_idx++) {
      PAShadow& pa_shadow = layer_shadow_map[layer_idx];

      for (auto& [net_idx, rect_set] : pa_shadow.get_net_fixed_rect_map()) {
        GPStruct fixed_rect_struct(RTUTIL.getString("shadow_fixed_rect(net_", net_idx, ")"));
        for (const PlanarRect& rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShadow));
          gp_boundary.set_rect(rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          fixed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(fixed_rect_struct);
      }

      for (auto& [net_idx, rect_set] : pa_shadow.get_net_routed_rect_map()) {
        GPStruct routed_rect_struct(RTUTIL.getString("shadow_routed_rect(net_", net_idx, ")"));
        for (const PlanarRect& rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShadow));
          gp_boundary.set_rect(rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          routed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(routed_rect_struct);
      }

      GPStruct violation_struct(RTUTIL.getString("shadow_violation"));
      for (const PlanarRect& rect : pa_shadow.get_violation_set()) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShadow));
        gp_boundary.set_rect(rect);
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
        violation_struct.push(gp_boundary);
      }
      gp_gds.addStruct(violation_struct);
    }
  }

  // task
  for (PATask* pa_task : pa_box.get_pa_task_list()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", pa_task->get_net_idx(), ")"));

    for (PAGroup& pa_group : pa_task->get_pa_group_list()) {
      for (LayerCoord& coord : pa_group.get_coord_list()) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kKey));
        gp_boundary.set_rect(RTUTIL.getEnlargedRect(coord, point_size));
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(coord.get_layer_idx()));
        task_struct.push(gp_boundary);
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
    for (Segment<LayerCoord>& segment : pa_box.get_net_task_access_result_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(pa_task->get_net_idx(), segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kDetailedPath));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        task_struct.push(gp_boundary);
      }
    }
    for (EXTLayerRect& patch : pa_box.get_net_task_access_patch_map()[pa_task->get_net_idx()][pa_task->get_task_idx()]) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatch));
      gp_boundary.set_rect(patch.get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch.get_layer_idx()));
      task_struct.push(gp_boundary);
    }
    gp_gds.addStruct(task_struct);
  }

  // violation
  {
    for (Violation& violation : pa_box.get_route_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kRouteViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      violation_struct.push(gp_boundary);
      gp_gds.addStruct(violation_struct);
    }
    for (Violation& violation : pa_box.get_patch_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatchViolation));
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

  std::string gds_file_path
      = RTUTIL.getString(pa_temp_directory_path, flag, "_pa_box_", pa_box.get_pa_box_id().get_x(), "_", pa_box.get_pa_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
