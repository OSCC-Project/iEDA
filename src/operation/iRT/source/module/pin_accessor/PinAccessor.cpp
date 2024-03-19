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
    LOG_INST.error(Loc::current(), "The instance not initialized!");
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
  LOG_INST.info(Loc::current(), "Starting...");
  PAModel pa_model = initPAModel();
  initAccessPointList(pa_model);
  buildAccessPointList(pa_model);
  updateToGCellMap(pa_model);
  updateSummary(pa_model);
  printSummary(pa_model);
  writePinCSV(pa_model);
  LOG_INST.info(Loc::current(), "Completed", monitor.getStatsInfo());

  // debugPlotPAModel();
}

// private

PinAccessor* PinAccessor::_pa_instance = nullptr;

PAModel PinAccessor::initPAModel()
{
  PAModel pa_model;
  return pa_model;
}

void PinAccessor::initAccessPointList(PAModel& pa_model)
{
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

  std::vector<std::pair<int32_t, Pin*>> net_pin_pair_list;
  for (Net& net : net_list) {
    for (Pin& pin : net.get_pin_list()) {
      net_pin_pair_list.emplace_back(net.get_net_idx(), &pin);
    }
  }
#pragma omp parallel for
  for (std::pair<int32_t, Pin*>& net_pin_pair : net_pin_pair_list) {
    Pin* pin = net_pin_pair.second;
    std::vector<AccessPoint>& access_point_list = net_pin_pair.second->get_access_point_list();
    std::vector<LayerRect> legal_shape_list = getLegalShapeList(net_pin_pair.first, pin);
    for (auto getAccessPointList :
         {std::bind(&PinAccessor::getAccessPointListByPrefTrackGrid, this, std::placeholders::_1, std::placeholders::_2),
          std::bind(&PinAccessor::getAccessPointListByCurrTrackGrid, this, std::placeholders::_1, std::placeholders::_2),
          std::bind(&PinAccessor::getAccessPointListByTrackCenter, this, std::placeholders::_1, std::placeholders::_2),
          std::bind(&PinAccessor::getAccessPointListByShapeCenter, this, std::placeholders::_1, std::placeholders::_2)}) {
      for (AccessPoint& access_point : getAccessPointList(pin->get_pin_idx(), legal_shape_list)) {
        access_point_list.push_back(access_point);
      }
      if (!access_point_list.empty()) {
        break;
      }
    }
    if (access_point_list.empty()) {
      LOG_INST.error(Loc::current(), "No access point was generated!");
    }
  }
}

std::vector<LayerRect> PinAccessor::getLegalShapeList(int32_t net_idx, Pin* pin)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::map<int32_t, std::vector<EXTLayerRect>> layer_pin_shape_list;
  for (EXTLayerRect& routing_shape : pin->get_routing_shape_list()) {
    layer_pin_shape_list[routing_shape.get_layer_idx()].emplace_back(routing_shape);
  }
  std::vector<LayerRect> legal_rect_list;
  for (auto& [layer_idx, pin_shape_list] : layer_pin_shape_list) {
    std::vector<PlanarRect> planar_legal_rect_list = getPlanarLegalRectList(net_idx, pin_shape_list);
    // 对legal rect进行融合，prefer横就竖着切，prefer竖就横着切
    if (routing_layer_list[layer_idx].isPreferH()) {
      planar_legal_rect_list = RTUtil::mergeRectListByBoost(planar_legal_rect_list, Direction::kVertical);
    } else {
      planar_legal_rect_list = RTUtil::mergeRectListByBoost(planar_legal_rect_list, Direction::kHorizontal);
    }
    for (PlanarRect planar_legal_rect : planar_legal_rect_list) {
      legal_rect_list.emplace_back(planar_legal_rect, layer_idx);
    }
  }
  if (!legal_rect_list.empty()) {
    return legal_rect_list;
  }
  LOG_INST.warn(Loc::current(), "The pin ", pin->get_pin_name(), " without legal shape!");
  for (EXTLayerRect& routing_shape : pin->get_routing_shape_list()) {
    legal_rect_list.emplace_back(routing_shape.getRealLayerRect());
  }
  return legal_rect_list;
}

std::vector<PlanarRect> PinAccessor::getPlanarLegalRectList(int32_t curr_net_idx, std::vector<EXTLayerRect>& pin_shape_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  int32_t curr_layer_idx;
  {
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      if (pin_shape_list.front().get_layer_idx() != pin_shape.get_layer_idx()) {
        LOG_INST.error(Loc::current(), "The pin_shape_list is not on the same layer!");
      }
    }
    curr_layer_idx = pin_shape_list.front().get_layer_idx();
  }
  std::vector<EXTLayerRect> reduced_rect_list;
  {
    std::vector<PlanarRect> origin_pin_shape_list;
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      origin_pin_shape_list.push_back(pin_shape.get_real_rect());
    }
    // 当前层缩小后的结果
    int32_t reduced_size = routing_layer_list[curr_layer_idx].get_min_width() / 2;
    for (PlanarRect& real_rect : RTUtil::getClosedReducedRectListByBoost(origin_pin_shape_list, reduced_size)) {
      EXTLayerRect reduced_rect;
      reduced_rect.set_real_rect(real_rect);
      reduced_rect.set_grid_rect(RTUtil::getClosedGCellGridRect(reduced_rect.get_real_rect(), gcell_axis));
      reduced_rect_list.push_back(reduced_rect);
    }
  }
  // 要被剪裁的obstacle的集合 排序按照 本层 上层
  /**
   * 要被剪裁的obstacle的集合
   * 如果不是最顶层就往上取一层
   * 是最顶层就往下取一层
   */
  std::vector<int32_t> pin_layer_idx_list;
  if (curr_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
    pin_layer_idx_list = {curr_layer_idx, curr_layer_idx + 1};
  } else {
    pin_layer_idx_list = {curr_layer_idx, curr_layer_idx - 1};
  }
  std::vector<std::vector<PlanarRect>> routing_obs_shape_list_list;
  for (int32_t pin_layer_idx : pin_layer_idx_list) {
    RoutingLayer& routing_layer = routing_layer_list[pin_layer_idx];
    std::vector<PlanarRect> routing_obs_shape_list;
    for (EXTLayerRect& reduced_rect : reduced_rect_list) {
      for (auto& [is_routing, layer_net_fixed_rect_map] : DM_INST.getTypeLayerNetFixedRectMap(reduced_rect)) {
        if (!is_routing) {
          continue;
        }
        for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
          if (pin_layer_idx != layer_idx) {
            continue;
          }
          for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
            if (net_idx == curr_net_idx) {
              continue;
            }
            for (EXTLayerRect* fixed_rect : fixed_rect_set) {
              int32_t enlarged_size = routing_layer.getMinSpacing(fixed_rect->get_real_rect()) + (routing_layer.get_min_width() / 2);
              PlanarRect enlarged_rect = RTUtil::getEnlargedRect(fixed_rect->get_real_rect(), enlarged_size);
              if (RTUtil::isOpenOverlap(reduced_rect.get_real_rect(), enlarged_rect)) {
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
    std::vector<PlanarRect> legal_rect_list_temp = RTUtil::getClosedCuttingRectListByBoost(legal_rect_list, routing_obs_shape_list);
    if (!legal_rect_list_temp.empty()) {
      legal_rect_list = legal_rect_list_temp;
    } else {
      break;
    }
  }
  return legal_rect_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByPrefTrackGrid(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    int32_t other_layer_idx;
    if (curr_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
      other_layer_idx = curr_layer_idx + 1;
    } else {
      other_layer_idx = curr_layer_idx - 1;
    }
    // prefer track grid
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    RoutingLayer other_routing_layer = routing_layer_list[other_layer_idx];
    if (curr_routing_layer.isPreferH()) {
      for (int32_t x : RTUtil::getScaleList(ll_x, ur_x, other_routing_layer.getXTrackGridList())) {
        for (int32_t y : RTUtil::getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
    } else {
      for (int32_t x : RTUtil::getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
        for (int32_t y : RTUtil::getScaleList(ll_y, ur_y, other_routing_layer.getYTrackGridList())) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kPrefTrackGrid);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByCurrTrackGrid(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    // curr layer track grid
    for (int32_t x : RTUtil::getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
      for (int32_t y : RTUtil::getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kCurrTrackGrid);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByTrackCenter(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

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
    for (int32_t y : RTUtil::getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList())) {
      layer_coord_list.emplace_back(mid_x, y, curr_layer_idx);
    }
    for (int32_t x : RTUtil::getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList())) {
      layer_coord_list.emplace_back(x, mid_y, curr_layer_idx);
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(pin_idx, layer_coord, AccessPointType::kTrackCenter);
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
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

#pragma omp parallel for
  for (Net& net : net_list) {
    std::vector<PlanarCoord> coord_list;
    for (Pin& pin : net.get_pin_list()) {
      for (AccessPoint& access_point : pin.get_access_point_list()) {
        coord_list.push_back(access_point.get_real_coord());
      }
    }
    BoundingBox& bounding_box = net.get_bounding_box();
    bounding_box.set_real_rect(RTUtil::getBoundingBox(coord_list));
    bounding_box.set_grid_rect(RTUtil::getOpenGCellGridRect(bounding_box.get_real_rect(), gcell_axis));
    for (Pin& pin : net.get_pin_list()) {
      for (AccessPoint& access_point : pin.get_access_point_list()) {
        access_point.set_grid_coord(RTUtil::getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
      }
    }
  }
}

void PinAccessor::updateToGCellMap(PAModel& pa_model)
{
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

  // 更新到顶层
  for (Net& net : net_list) {
    for (Pin& pin : net.get_pin_list()) {
      for (AccessPoint& access_point : pin.get_access_point_list()) {
        DM_INST.updateAccessPointToGCellMap(ChangeType::kAdd, net.get_net_idx(), &access_point);
      }
    }
  }
}

#if 1  // debug

void PinAccessor::debugPlotPAModel(PAModel& pa_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = DM_INST.getConfig().pa_temp_directory_path;

  GPGDS gp_gds;

  // track_axis_struct
  GPStruct track_axis_struct("track_axis_struct");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<int32_t> x_list = RTUtil::getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), routing_layer.getXTrackGridList());
    std::vector<int32_t> y_list = RTUtil::getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), routing_layer.getYTrackGridList());
    for (int32_t x : x_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
      track_axis_struct.push(gp_path);
    }
    for (int32_t y : y_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
      track_axis_struct.push(gp_path);
    }
  }
  gp_gds.addStruct(track_axis_struct);

  // 整张版图的fixed_rect
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      GCell& gcell = gcell_map[x][y];
      for (auto& [is_routing, layer_net_fixed_rect_map] : gcell.get_type_layer_net_fixed_rect_map()) {
        for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
          for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
            GPStruct fixed_rect_struct(RTUtil::getString("fixed_rect(net_", net_idx, ")"));
            for (auto& fixed_rect : fixed_rect_set) {
              GPBoundary gp_boundary;
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
              gp_boundary.set_rect(fixed_rect->get_real_rect());
              if (is_routing) {
                gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
              } else {
                gp_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(layer_idx));
              }
              fixed_rect_struct.push(gp_boundary);
            }
            gp_gds.addStruct(fixed_rect_struct);
          }
        }
      }
    }
  }

  // access_point
  for (Net& net : net_list) {
    GPStruct access_point_struct(RTUtil::getString("access_point(net_", net.get_net_idx(), ")"));
    for (Pin& pin : net.get_pin_list()) {
      for (AccessPoint& access_point : pin.get_access_point_list()) {
        int32_t x = access_point.get_real_x();
        int32_t y = access_point.get_real_y();

        GPBoundary access_point_boundary;
        access_point_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(access_point.get_layer_idx()));
        access_point_boundary.set_data_type(static_cast<int32_t>(GPDataType::kAccessPoint));
        access_point_boundary.set_rect(x - 10, y - 10, x + 10, y + 10);
        access_point_struct.push(access_point_boundary);
      }
    }
    gp_gds.addStruct(access_point_struct);
  }

  std::string gds_file_path = RTUtil::getString(pa_temp_directory_path, "access_point.gds");
  GP_INST.plot(gp_gds, gds_file_path);
}

#endif

#if 1  // exhibit

void PinAccessor::updateSummary(PAModel& pa_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::map<int32_t, int32_t>& routing_access_point_num_map = DM_INST.getSummary().pa_summary.routing_access_point_num_map;
  std::map<AccessPointType, int32_t>& type_access_point_num_map = DM_INST.getSummary().pa_summary.type_access_point_num_map;
  int32_t& total_access_point_num = DM_INST.getSummary().pa_summary.total_access_point_num;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_access_point_num_map[routing_layer.get_layer_idx()] = 0;
  }
  type_access_point_num_map = {{AccessPointType::kNone, 0},
                               {AccessPointType::kPrefTrackGrid, 0},
                               {AccessPointType::kCurrTrackGrid, 0},
                               {AccessPointType::kTrackCenter, 0},
                               {AccessPointType::kShapeCenter, 0}};
  total_access_point_num = 0;

  for (auto& [net_idx, access_point_list] : DM_INST.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_list) {
      routing_access_point_num_map[access_point->get_layer_idx()]++;
      type_access_point_num_map[access_point->get_type()]++;
      total_access_point_num++;
    }
  }
}

void PinAccessor::printSummary(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::map<int32_t, int32_t>& routing_access_point_num_map = DM_INST.getSummary().pa_summary.routing_access_point_num_map;
  std::map<AccessPointType, int32_t>& type_access_point_num_map = DM_INST.getSummary().pa_summary.type_access_point_num_map;
  int32_t& total_access_point_num = DM_INST.getSummary().pa_summary.total_access_point_num;

  fort::char_table routing_access_point_num_map_table;
  {
    routing_access_point_num_map_table << fort::header << "routing_layer"
                                       << "access_point_num"
                                       << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_access_point_num_map_table << routing_layer.get_layer_name() << routing_access_point_num_map[routing_layer.get_layer_idx()]
                                         << RTUtil::getPercentage(routing_access_point_num_map[routing_layer.get_layer_idx()],
                                                                  total_access_point_num)
                                         << fort::endr;
    }
    routing_access_point_num_map_table << fort::header << "Total" << total_access_point_num
                                       << RTUtil::getPercentage(total_access_point_num, total_access_point_num) << fort::endr;
  }
  fort::char_table type_access_point_num_map_table;
  {
    type_access_point_num_map_table << fort::header << "type"
                                    << "access_point_num"
                                    << "proportion" << fort::endr;
    for (auto& [type, access_point_num] : type_access_point_num_map) {
      type_access_point_num_map_table << GetAccessPointTypeName()(type) << access_point_num
                                      << RTUtil::getPercentage(access_point_num, total_access_point_num) << fort::endr;
    }
    type_access_point_num_map_table << fort::header << "Total" << total_access_point_num
                                    << RTUtil::getPercentage(total_access_point_num, total_access_point_num) << fort::endr;
  }
  RTUtil::printTableList({routing_access_point_num_map_table, type_access_point_num_map_table});
}

void PinAccessor::writePinCSV(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string& pa_temp_directory_path = DM_INST.getConfig().pa_temp_directory_path;
  int32_t output_csv = DM_INST.getConfig().output_csv;
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
        = RTUtil::getOutputFileStream(RTUtil::getString(pa_temp_directory_path, "pin_map_", routing_layer.get_layer_name(), ".csv"));
    GridMap<int32_t>& pin_map = layer_pin_map[routing_layer.get_layer_idx()];
    for (int32_t y = pin_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < pin_map.get_x_size(); x++) {
        RTUtil::pushStream(pin_csv_file, pin_map[x][y], ",");
      }
      RTUtil::pushStream(pin_csv_file, "\n");
    }
    RTUtil::closeFileStream(pin_csv_file);
  }
}

#endif

}  // namespace irt
