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
#include "PAModel.hpp"
#include "PANet.hpp"

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

void PinAccessor::access(std::vector<Net>& net_list)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin accessing...");
  PAModel pa_model = initPAModel(net_list);
  initAccessPointList(pa_model);
  buildAccessPointList(pa_model);
  updatePAModel(pa_model);
  LOG_INST.info(Loc::current(), "End access", monitor.getStatsInfo());

  // plotPAModel(pa_model);
  // reportPAModel(pa_model);
}

// private

PinAccessor* PinAccessor::_pa_instance = nullptr;

PAModel PinAccessor::initPAModel(std::vector<Net>& net_list)
{
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
  std::vector<std::pair<int32_t, PAPin*>> net_pin_pair_list;
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      net_pin_pair_list.emplace_back(pa_net.get_net_idx(), &pa_pin);
    }
  }
#pragma omp parallel for
  for (std::pair<int32_t, PAPin*>& net_pin_pair : net_pin_pair_list) {
    std::vector<AccessPoint>& access_point_list = net_pin_pair.second->get_access_point_list();
    std::vector<LayerRect> legal_shape_list = getLegalShapeList(pa_model, net_pin_pair.first, net_pin_pair.second);
    for (auto getAccessPointList : {std::bind(&PinAccessor::getAccessPointListByPrefTrackGrid, this, std::placeholders::_1),
                                    std::bind(&PinAccessor::getAccessPointListByCurrTrackGrid, this, std::placeholders::_1),
                                    std::bind(&PinAccessor::getAccessPointListByTrackCenter, this, std::placeholders::_1),
                                    std::bind(&PinAccessor::getAccessPointListByShapeCenter, this, std::placeholders::_1)}) {
      for (AccessPoint& access_point : getAccessPointList(legal_shape_list)) {
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

std::vector<LayerRect> PinAccessor::getLegalShapeList(PAModel& pa_model, int32_t pa_net_idx, PAPin* pa_pin)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::map<int32_t, std::vector<EXTLayerRect>> layer_pin_shape_list;
  for (EXTLayerRect& routing_shape : pa_pin->get_routing_shape_list()) {
    layer_pin_shape_list[routing_shape.get_layer_idx()].emplace_back(routing_shape);
  }
  std::vector<LayerRect> legal_rect_list;
  for (auto& [layer_idx, pin_shpae_list] : layer_pin_shape_list) {
    std::vector<PlanarRect> planar_legal_rect_list = getPlanarLegalRectList(pa_model, pa_net_idx, pin_shpae_list);
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
  LOG_INST.warn(Loc::current(), "The pin ", pa_pin->get_pin_name(), " without legal shape!");
  for (EXTLayerRect& routing_shape : pa_pin->get_routing_shape_list()) {
    legal_rect_list.emplace_back(routing_shape.getRealLayerRect());
  }
  return legal_rect_list;
}

std::vector<PlanarRect> PinAccessor::getPlanarLegalRectList(PAModel& pa_model, int32_t pa_net_idx,
                                                            std::vector<EXTLayerRect>& pin_shape_list)
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
  // 要被剪裁的blockage的集合 排序按照 本层 上层
  /**
   * 要被剪裁的blockage的集合
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
  for (int32_t layer_idx : pin_layer_idx_list) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    std::vector<PlanarRect> routing_obs_shape_list;
    for (EXTLayerRect& reduced_rect : reduced_rect_list) {
      auto net_fixed_rect_map = DM_INST.getTypeLayerNetFixedRectMap(reduced_rect)[true][layer_idx];
      for (auto& [net_idx, rect_set] : net_fixed_rect_map) {
        if (net_idx == pa_net_idx) {
          continue;
        }
        for (EXTLayerRect* rect : rect_set) {
          int32_t enlarged_size = routing_layer.getMinSpacing(rect->get_real_rect()) + (routing_layer.get_min_width() / 2);
          PlanarRect enlarged_rect = RTUtil::getEnlargedRect(rect->get_real_rect(), enlarged_size);
          if (RTUtil::isOpenOverlap(reduced_rect.get_real_rect(), enlarged_rect)) {
            routing_obs_shape_list.push_back(enlarged_rect);
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

std::vector<AccessPoint> PinAccessor::getAccessPointListByPrefTrackGrid(std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t lb_x = legal_shape.get_lb_x();
    int32_t lb_y = legal_shape.get_lb_y();
    int32_t rt_x = legal_shape.get_rt_x();
    int32_t rt_y = legal_shape.get_rt_y();
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
      for (int32_t x : RTUtil::getScaleList(lb_x, rt_x, other_routing_layer.getXTrackGridList())) {
        for (int32_t y : RTUtil::getScaleList(lb_y, rt_y, curr_routing_layer.getYTrackGridList())) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
    } else {
      for (int32_t x : RTUtil::getScaleList(lb_x, rt_x, curr_routing_layer.getXTrackGridList())) {
        for (int32_t y : RTUtil::getScaleList(lb_y, rt_y, other_routing_layer.getYTrackGridList())) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(layer_coord.get_x(), layer_coord.get_y(), layer_coord.get_layer_idx(), AccessPointType::kPrefTrackGrid);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByCurrTrackGrid(std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t lb_x = legal_shape.get_lb_x();
    int32_t lb_y = legal_shape.get_lb_y();
    int32_t rt_x = legal_shape.get_rt_x();
    int32_t rt_y = legal_shape.get_rt_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    // curr layer track grid
    for (int32_t x : RTUtil::getScaleList(lb_x, rt_x, curr_routing_layer.getXTrackGridList())) {
      for (int32_t y : RTUtil::getScaleList(lb_y, rt_y, curr_routing_layer.getYTrackGridList())) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(layer_coord.get_x(), layer_coord.get_y(), layer_coord.get_layer_idx(), AccessPointType::kCurrTrackGrid);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByTrackCenter(std::vector<LayerRect>& legal_shape_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t lb_x = legal_shape.get_lb_x();
    int32_t lb_y = legal_shape.get_lb_y();
    int32_t rt_x = legal_shape.get_rt_x();
    int32_t rt_y = legal_shape.get_rt_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    RoutingLayer curr_routing_layer = routing_layer_list[curr_layer_idx];
    // on track
    int32_t mid_x = (lb_x + rt_x) / 2;
    int32_t mid_y = (lb_y + rt_y) / 2;
    for (int32_t y : RTUtil::getScaleList(lb_y, rt_y, curr_routing_layer.getYTrackGridList())) {
      layer_coord_list.emplace_back(mid_x, y, curr_layer_idx);
    }
    for (int32_t x : RTUtil::getScaleList(lb_x, rt_x, curr_routing_layer.getXTrackGridList())) {
      layer_coord_list.emplace_back(x, mid_y, curr_layer_idx);
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(layer_coord.get_x(), layer_coord.get_y(), layer_coord.get_layer_idx(), AccessPointType::kTrackCenter);
  }
  return access_point_list;
}

std::vector<AccessPoint> PinAccessor::getAccessPointListByShapeCenter(std::vector<LayerRect>& legal_shape_list)
{
  std::vector<LayerCoord> layer_coord_list;
  for (LayerRect& legal_shape : legal_shape_list) {
    int32_t lb_x = legal_shape.get_lb_x();
    int32_t lb_y = legal_shape.get_lb_y();
    int32_t rt_x = legal_shape.get_rt_x();
    int32_t rt_y = legal_shape.get_rt_y();
    int32_t curr_layer_idx = legal_shape.get_layer_idx();
    // on shape
    int32_t mid_x = (lb_x + rt_x) / 2;
    int32_t mid_y = (lb_y + rt_y) / 2;
    layer_coord_list.emplace_back(mid_x, mid_y, curr_layer_idx);
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());

  std::vector<AccessPoint> access_point_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    access_point_list.emplace_back(layer_coord.get_x(), layer_coord.get_y(), layer_coord.get_layer_idx(), AccessPointType::kShapeCenter);
  }
  return access_point_list;
}

void PinAccessor::buildAccessPointList(PAModel& pa_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();

#pragma omp parallel for
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    std::vector<PlanarCoord> coord_list;
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        coord_list.push_back(access_point.get_real_coord());
      }
    }
    BoundingBox& bounding_box = pa_net.get_bounding_box();
    bounding_box.set_real_rect(RTUtil::getBoundingBox(coord_list));
    bounding_box.set_grid_rect(RTUtil::getOpenGCellGridRect(bounding_box.get_real_rect(), gcell_axis));
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        access_point.set_grid_coord(RTUtil::getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
      }
    }
  }
}

void PinAccessor::updatePAModel(PAModel& pa_model)
{
  // 更新到顶层
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      Pin& origin_pin = pa_net.get_origin_net()->get_pin_list()[pa_pin.get_pin_idx()];
      if (origin_pin.get_pin_idx() != pa_pin.get_pin_idx()) {
        LOG_INST.error(Loc::current(), "The pin idx is not equal!");
      }
      origin_pin.set_access_point_list(pa_pin.get_access_point_list());
      for (AccessPoint& access_point : origin_pin.get_access_point_list()) {
        DM_INST.updateAccessPointToGCellMap(ChangeType::kAdd, pa_net.get_net_idx(), &access_point);
      }
    }
    pa_net.get_origin_net()->set_bounding_box(pa_net.get_bounding_box());
  }
}

#if 1  // exhibit

void PinAccessor::plotPAModel(PAModel& pa_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string pa_temp_directory_path = DM_INST.getConfig().pa_temp_directory_path;

  GPGDS gp_gds;

  // track_axis_struct
  GPStruct track_axis_struct("track_axis_struct");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<int32_t> x_list = RTUtil::getScaleList(die.get_real_lb_x(), die.get_real_rt_x(), routing_layer.getXTrackGridList());
    std::vector<int32_t> y_list = RTUtil::getScaleList(die.get_real_lb_y(), die.get_real_rt_y(), routing_layer.getYTrackGridList());
    for (int32_t x : x_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(x, die.get_real_lb_y(), x, die.get_real_rt_y());
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(routing_layer.get_layer_idx()));
      track_axis_struct.push(gp_path);
    }
    for (int32_t y : y_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(die.get_real_lb_x(), y, die.get_real_rt_x(), y);
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
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    GPStruct access_point_struct(RTUtil::getString("access_point(net_", pa_net.get_net_idx(), ")"));
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
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

void PinAccessor::reportPAModel(PAModel& pa_model)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin reporting...");
  reportSummary(pa_model);
  writePinCSV(pa_model);
  LOG_INST.info(Loc::current(), "End report", monitor.getStatsInfo());
}

void PinAccessor::reportSummary(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::map<int32_t, int32_t>& pa_routing_access_point_map = DM_INST.getSummary().pa_summary.routing_access_point_map;
  std::map<AccessPointType, int32_t>& pa_type_access_point_map = DM_INST.getSummary().pa_summary.type_access_point_map;
  int32_t& pa_total_access_point_num = DM_INST.getSummary().pa_summary.total_access_point_num;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    pa_routing_access_point_map[routing_layer.get_layer_idx()] = 0;
  }
  pa_type_access_point_map = {{AccessPointType::kNone, 0},
                              {AccessPointType::kPrefTrackGrid, 0},
                              {AccessPointType::kCurrTrackGrid, 0},
                              {AccessPointType::kTrackCenter, 0},
                              {AccessPointType::kShapeCenter, 0}};
  pa_total_access_point_num = 0;

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [net_idx, access_point_list] : gcell_map[x][y].get_net_access_point_map()) {
        for (AccessPoint* access_point : access_point_list) {
          pa_routing_access_point_map[access_point->get_layer_idx()]++;
          pa_type_access_point_map[access_point->get_type()]++;
          pa_total_access_point_num++;
        }
      }
    }
  }
}

void PinAccessor::writePinCSV(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string pa_temp_directory_path = DM_INST.getConfig().pa_temp_directory_path;

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
