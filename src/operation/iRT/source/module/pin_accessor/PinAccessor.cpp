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

#include <sstream>

#include "GDSPlotter.hpp"
#include "GPGDS.hpp"
#include "PAModel.hpp"
#include "PANet.hpp"
#include "RTAPI.hpp"

namespace irt {

// public

void PinAccessor::initInst(Config& config, Database& database)
{
  if (_pa_instance == nullptr) {
    _pa_instance = new PinAccessor(config, database);
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

void PinAccessor::access(std::vector<Net>& net_list)
{
  Monitor monitor;

  std::vector<PANet> pa_net_list = _pa_data_manager.convertToPANetList(net_list);
  accessPANetList(pa_net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kPinAccessor), " completed!", monitor.getStatsInfo());
}

// private

PinAccessor* PinAccessor::_pa_instance = nullptr;

void PinAccessor::init(Config& config, Database& database)
{
  _pa_data_manager.input(config, database);
}

void PinAccessor::accessPANetList(std::vector<PANet>& pa_net_list)
{
  PAModel pa_model = initPAModel(pa_net_list);
  buildPAModel(pa_model);
  accessPAModel(pa_model);
  checkPAModel(pa_model);
  updatePAModel(pa_model);
  countPAModel(pa_model);
  reportPAModel(pa_model);
}

#if 1  // build pa_model

PAModel PinAccessor::initPAModel(std::vector<PANet>& pa_net_list)
{
  Die& die = _pa_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _pa_data_manager.getDatabase().get_routing_layer_list();

  PAModel pa_model;
  std::vector<GridMap<PAGCell>>& layer_gcell_map = pa_model.get_layer_gcell_map();
  layer_gcell_map.resize(routing_layer_list.size());
  for (size_t layer_idx = 0; layer_idx < layer_gcell_map.size(); layer_idx++) {
    GridMap<PAGCell>& gcell_map = layer_gcell_map[layer_idx];
    gcell_map.init(die.getXSize(), die.getYSize());
    for (irt_int x = 0; x < die.getXSize(); x++) {
      for (irt_int y = 0; y < die.getYSize(); y++) {
        PAGCell& pa_gcell = gcell_map[x][y];
        pa_gcell.set_coord(x, y);
        pa_gcell.set_layer_idx(static_cast<irt_int>(layer_idx));
      }
    }
  }
  pa_model.set_pa_net_list(pa_net_list);

  return pa_model;
}

void PinAccessor::buildPAModel(PAModel& pa_model)
{
  initGCellRealRect(pa_model);
  addBlockageList(pa_model);
  cutBlockageList(pa_model);
}

void PinAccessor::initGCellRealRect(PAModel& pa_model)
{
  GCellAxis& gcell_axis = _pa_data_manager.getDatabase().get_gcell_axis();

  std::vector<GridMap<PAGCell>>& layer_gcell_map = pa_model.get_layer_gcell_map();
  for (size_t layer_idx = 0; layer_idx < layer_gcell_map.size(); layer_idx++) {
    GridMap<PAGCell>& node_map = layer_gcell_map[layer_idx];
    for (irt_int x = 0; x < node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < node_map.get_y_size(); y++) {
        PAGCell& pa_gcell = node_map[x][y];
        pa_gcell.set_real_rect(RTUtil::getRealRect(x, y, gcell_axis));
      }
    }
  }
}

void PinAccessor::addBlockageList(PAModel& pa_model)
{
  GCellAxis& gcell_axis = _pa_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _pa_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _pa_data_manager.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _pa_data_manager.getDatabase().get_routing_blockage_list();

  std::vector<GridMap<PAGCell>>& layer_gcell_map = pa_model.get_layer_gcell_map();
  for (Blockage& routing_blockage : routing_blockage_list) {
    irt_int layer_idx = routing_blockage.get_layer_idx();
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_blockage.get_real_rect());
    PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_blockage.get_real_rect(), min_spacing, die.get_real_rect());
    PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
    for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
      for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
        layer_gcell_map[layer_idx][x][y].get_net_blockage_map()[-1].push_back(enlarged_real_rect);
      }
    }
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
        irt_int layer_idx = routing_shape.get_layer_idx();
        irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_shape.get_real_rect());
        PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_shape.get_real_rect(), min_spacing, die.get_real_rect());
        PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
            layer_gcell_map[layer_idx][x][y].get_net_blockage_map()[pa_net.get_net_idx()].push_back(enlarged_real_rect);
          }
        }
      }
    }
  }
}

void PinAccessor::cutBlockageList(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _pa_data_manager.getDatabase().get_routing_layer_list();

  for (GridMap<PAGCell>& gcell_map : pa_model.get_layer_gcell_map()) {
    for (irt_int x = 0; x < gcell_map.get_x_size(); x++) {
      for (irt_int y = 0; y < gcell_map.get_y_size(); y++) {
        PAGCell& pa_gcell = gcell_map[x][y];
        RoutingLayer& routing_layer = routing_layer_list[pa_gcell.get_layer_idx()];
        std::map<irt_int, std::vector<PlanarRect>>& net_blockage_map = pa_gcell.get_net_blockage_map();

        std::vector<PlanarRect> new_blockage_list;
        new_blockage_list.reserve(net_blockage_map[-1].size());
        std::map<PlanarRect, std::vector<PlanarRect>, CmpPlanarRectByXASC> blockage_shape_list_map;

        for (PlanarRect& blockage : net_blockage_map[-1]) {
          bool is_cutting = false;
          for (auto& [net_idx, net_shape_list] : net_blockage_map) {
            if (net_idx == -1) {
              continue;
            }
            for (PlanarRect& net_shape : net_shape_list) {
              if (!RTUtil::isInside(blockage, net_shape)) {
                continue;
              }
              blockage_shape_list_map[blockage].push_back(net_shape);
              is_cutting = true;
            }
          }
          if (!is_cutting) {
            new_blockage_list.push_back(blockage);
          }
        }
        for (auto& [blockage, net_shape_list] : blockage_shape_list_map) {
          gtl::polygon_90_set_data<int> poly_set;
          poly_set += RTUtil::convertToGTLRect(blockage);
          for (PlanarRect& net_shape : net_shape_list) {
            irt_int enlarged_size = routing_layer.get_min_width() + routing_layer.getMinSpacing(net_shape);
            poly_set -= RTUtil::convertToGTLRect(RTUtil::getEnlargedRect(net_shape, enlarged_size));
          }
          std::vector<gtl::rectangle_data<int>> slicing_rect_list;
          gtl::get_rectangles(slicing_rect_list, poly_set);
          for (gtl::rectangle_data<int>& slicing_rect : slicing_rect_list) {
            new_blockage_list.push_back(RTUtil::convertToPlanarRect(slicing_rect));
          }
        }
        net_blockage_map[-1] = new_blockage_list;
      }
    }
  }
}

#endif

#if 1  // access pa_model

void PinAccessor::accessPAModel(PAModel& pa_model)
{
  Monitor monitor;

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();

  irt_int batch_size = RTUtil::getBatchSize(pa_net_list.size());

  Monitor stage_monitor;
#pragma omp parallel for
  for (size_t i = 0; i < pa_net_list.size(); i++) {
    accessPANet(pa_model, pa_net_list[i]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Processed ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Processed ", pa_net_list.size(), " nets", monitor.getStatsInfo());
  }
}

void PinAccessor::accessPANet(PAModel& pa_model, PANet& pa_net)
{
  initAccessPointList(pa_net);
  mergeAccessPointList(pa_net);
  selectAccessPointList(pa_net);
}

void PinAccessor::initAccessPointList(PANet& pa_net)
{
  std::vector<RoutingLayer>& routing_layer_list = _pa_data_manager.getDatabase().get_routing_layer_list();
  irt_int top_routing_layer_idx = _pa_data_manager.getConfig().top_routing_layer_idx;
  irt_int bottom_routing_layer_idx = _pa_data_manager.getConfig().bottom_routing_layer_idx;

  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::vector<AccessPoint>& access_point_list = pa_pin.get_access_point_list();
    for (LayerRect& aligned_pin_shape : getIntersectPinShapeList(pa_pin)) {
      irt_int lb_x = aligned_pin_shape.get_lb_x();
      irt_int lb_y = aligned_pin_shape.get_lb_y();
      irt_int rt_x = aligned_pin_shape.get_rt_x();
      irt_int rt_y = aligned_pin_shape.get_rt_y();
      irt_int pin_shape_layer_idx = aligned_pin_shape.get_layer_idx();

      // routing layer info
      std::vector<irt_int> layer_idx_list;
      irt_int mid_layer_idx = pin_shape_layer_idx;
      mid_layer_idx = std::min(mid_layer_idx, top_routing_layer_idx);
      mid_layer_idx = std::max(mid_layer_idx, bottom_routing_layer_idx);
      for (irt_int layer_idx : {mid_layer_idx - 1, mid_layer_idx, mid_layer_idx + 1}) {
        if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
          continue;
        }
        layer_idx_list.push_back(layer_idx);
      }
      // generate access point
      for (irt_int i = 0; i < static_cast<irt_int>(layer_idx_list.size()) - 1; i++) {
        // prefer track grid
        TrackGrid pref_x_track = routing_layer_list[layer_idx_list[i]].getPreferTrackGrid();
        TrackGrid pref_y_track = routing_layer_list[layer_idx_list[i + 1]].getPreferTrackGrid();
        if (routing_layer_list[layer_idx_list[i]].isPreferH()) {
          std::swap(pref_x_track, pref_y_track);
        }
        std::vector<irt_int> pref_x_list = RTUtil::getClosedScaleList(lb_x, rt_x, pref_x_track);
        std::vector<irt_int> pref_y_list = RTUtil::getClosedScaleList(lb_y, rt_y, pref_y_track);
        for (irt_int x : pref_x_list) {
          for (irt_int y : pref_y_list) {
            access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kPrefTrackGrid);
          }
        }
        irt_int shape_x_mid = (lb_x + rt_x) / 2;
        irt_int shape_y_mid = (lb_y + rt_y) / 2;
        // prefer track center
        for (irt_int x : pref_x_list) {
          access_point_list.emplace_back(x, shape_y_mid, pin_shape_layer_idx, AccessPointType::kPrefTrackCenter);
        }
        for (irt_int y : pref_y_list) {
          access_point_list.emplace_back(shape_x_mid, y, pin_shape_layer_idx, AccessPointType::kPrefTrackCenter);
        }
        // shape center
        access_point_list.emplace_back(shape_x_mid, shape_y_mid, pin_shape_layer_idx, AccessPointType::kShapeCenter);
      }
    }
  }
}

std::vector<LayerRect> PinAccessor::getIntersectPinShapeList(PAPin& pa_pin)
{
  std::map<irt_int, gtl::polygon_90_set_data<irt_int>> layer_polygon_map;
  for (EXTLayerRect& pin_shape : pa_pin.get_routing_shape_list()) {
    layer_polygon_map[pin_shape.get_layer_idx()] += RTUtil::convertToGTLRect(pin_shape.get_real_rect());
  }
  gtl::polygon_90_set_data<irt_int> intersect_polygon = layer_polygon_map.begin()->second;
  for (auto& [layer_idx, polygon] : layer_polygon_map) {
    intersect_polygon *= polygon;
  }
  std::vector<LayerRect> intersect_pin_shape_list;
  if (intersect_polygon.empty()) {
    for (EXTLayerRect& pin_shape : pa_pin.get_routing_shape_list()) {
      intersect_pin_shape_list.emplace_back(pin_shape.get_real_rect(), pin_shape.get_layer_idx());
    }
  } else {
    std::vector<gtl::rectangle_data<int>> h_slicing_rect_list;
    gtl::get_rectangles(h_slicing_rect_list, intersect_polygon, gtl::orientation_2d_enum::HORIZONTAL);
    std::vector<gtl::rectangle_data<int>> v_slicing_rect_list;
    gtl::get_rectangles(v_slicing_rect_list, intersect_polygon, gtl::orientation_2d_enum::VERTICAL);
    for (auto& [layer_idx, polygon] : layer_polygon_map) {
      for (gtl::rectangle_data<irt_int>& h_slicing_rect : h_slicing_rect_list) {
        intersect_pin_shape_list.emplace_back(RTUtil::convertToPlanarRect(h_slicing_rect), layer_idx);
      }
      for (gtl::rectangle_data<irt_int>& v_slicing_rect : v_slicing_rect_list) {
        intersect_pin_shape_list.emplace_back(RTUtil::convertToPlanarRect(v_slicing_rect), layer_idx);
      }
    }
  }
  return intersect_pin_shape_list;
}

void PinAccessor::mergeAccessPointList(PANet& pa_net)
{
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::map<LayerCoord, AccessPointType, CmpLayerCoordByLayerASC> coord_type_map;
    std::vector<AccessPoint>& access_point_list = pa_pin.get_access_point_list();
    for (AccessPoint& access_point : access_point_list) {
      LayerCoord coord(access_point.get_real_coord(), access_point.get_layer_idx());
      if (RTUtil::exist(coord_type_map, coord)) {
        coord_type_map[coord] = std::min(coord_type_map[coord], access_point.get_type());
      } else {
        coord_type_map[coord] = access_point.get_type();
      }
    }
    access_point_list.clear();
    for (auto& [layer_coord, type] : coord_type_map) {
      access_point_list.emplace_back(layer_coord.get_x(), layer_coord.get_y(), layer_coord.get_layer_idx(), type);
    }
    if (access_point_list.empty()) {
      LOG_INST.error(Loc::current(), "The pin idx ", pa_pin.get_pin_idx(), " access_point_list is empty!");
    }
  }
}

void PinAccessor::selectAccessPointList(PANet& pa_net)
{
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::map<AccessPointType, std::vector<AccessPoint>> type_point_map;
    for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
      type_point_map[access_point.get_type()].push_back(access_point);
    }
    for (AccessPointType access_point_type :
         {AccessPointType::kPrefTrackGrid, AccessPointType::kPrefTrackCenter, AccessPointType::kShapeCenter}) {
      std::vector<AccessPoint>& access_point_list = type_point_map[access_point_type];
      if (access_point_list.empty()) {
        continue;
      }
      pa_pin.set_access_point_list(access_point_list);
      break;
    }
  }
}

#endif

#if 1  // check pa_model

void PinAccessor::checkPAModel(PAModel& pa_model)
{
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      std::vector<AccessPoint>& access_point_list = pa_pin.get_access_point_list();
      if (access_point_list.empty()) {
        LOG_INST.error(Loc::current(), "The pin ", pa_pin.get_pin_idx(), " access point list is empty!");
      }

      for (AccessPoint& access_point : access_point_list) {
        if (access_point.get_type() == AccessPointType::kNone) {
          LOG_INST.error(Loc::current(), "The access point type is wrong!");
        }
        bool is_legal = false;
        for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
          if (routing_shape.get_layer_idx() == access_point.get_layer_idx()
              && RTUtil::isInside(routing_shape.get_real_rect(), access_point.get_real_coord())) {
            is_legal = true;
            break;
          }
        }
        if (!is_legal) {
          LOG_INST.error(Loc::current(), "The access point is not in routing shape!");
        }
      }
    }
  }
}

#endif

#if 1  // update pa_model

void PinAccessor::updatePAModel(PAModel& pa_model)
{
#pragma omp parallel for
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    buildBoundingBox(pa_net);
    buildAccessPointList(pa_net);
    buildDrivingPin(pa_net);
  }
  updateOriginPAResult(pa_model);
}

void PinAccessor::buildBoundingBox(PANet& pa_net)
{
  GCellAxis& gcell_axis = _pa_data_manager.getDatabase().get_gcell_axis();

  std::vector<PlanarCoord> coord_list;
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
      coord_list.push_back(access_point.get_real_coord());
    }
  }
  EXTPlanarRect& bounding_box = pa_net.get_bounding_box();
  bounding_box.set_real_rect(RTUtil::getBoundingBox(coord_list));
  bounding_box.set_grid_rect(RTUtil::getOpenGridRect(bounding_box.get_real_rect(), gcell_axis));
}

void PinAccessor::buildAccessPointList(PANet& pa_net)
{
  GCellAxis& gcell_axis = _pa_data_manager.getDatabase().get_gcell_axis();
  BoundingBox& bounding_box = pa_net.get_bounding_box();

  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
      access_point.set_grid_coord(RTUtil::getGridCoord(access_point.get_real_coord(), gcell_axis, bounding_box));
    }
  }
}

void PinAccessor::buildDrivingPin(PANet& pa_net)
{
  PAPin& pa_driving_pin = pa_net.get_pa_driving_pin();
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    if (pa_driving_pin.get_pin_idx() != pa_pin.get_pin_idx()) {
      continue;
    }
    pa_driving_pin = pa_pin;
    return;
  }
  LOG_INST.error(Loc::current(), "Unable to find a driving pin!");
}

void PinAccessor::updateOriginPAResult(PAModel& pa_model)
{
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    std::vector<Pin>& pin_list = pa_net.get_origin_net()->get_pin_list();
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      Pin& pin = pin_list[pa_pin.get_pin_idx()];
      if (pin.get_pin_idx() != pa_pin.get_pin_idx()) {
        LOG_INST.error(Loc::current(), "The pin idx is not equal!");
      }
      pin.set_access_point_list(pa_pin.get_access_point_list());
    }

    Pin& driving_pin = pa_net.get_origin_net()->get_driving_pin();
    PAPin& pa_driving_pin = pa_net.get_pa_driving_pin();
    if (driving_pin.get_pin_idx() != pa_driving_pin.get_pin_idx()) {
      LOG_INST.error(Loc::current(), "The pin idx is not equal!");
    }
    driving_pin.set_access_point_list(pa_driving_pin.get_access_point_list());

    pa_net.get_origin_net()->set_bounding_box(pa_net.get_bounding_box());
  }
}

#endif

#if 1  // report pa_model

void PinAccessor::countPAModel(PAModel& pa_model)
{
  PAModelStat& pa_mode_stat = pa_model.get_pa_mode_stat();

  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    std::map<irt_int, std::set<irt_int>> layer_port_set_map;
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      pa_mode_stat.addTotalPinNum(1);

      std::set<irt_int> port_layer_idx_set;
      for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
        port_layer_idx_set.insert(routing_shape.get_layer_idx());
      }
      for (irt_int port_layer_idx : port_layer_idx_set) {
        pa_mode_stat.addTotalPortNum(1);
        pa_mode_stat.get_layer_port_num_map()[port_layer_idx]++;
      }

      std::vector<AccessPoint>& access_point_list = pa_pin.get_access_point_list();
      std::sort(access_point_list.begin(), access_point_list.end(),
                [](AccessPoint& a, AccessPoint& b) { return a.get_type() < b.get_type(); });
      switch (access_point_list.front().get_type()) {
        case AccessPointType::kPrefTrackGrid:
          pa_mode_stat.addTrackGridPinNum(1);
          break;
        case AccessPointType::kPrefTrackCenter:
          pa_mode_stat.addTrackCenterPinNum(1);
          break;
        case AccessPointType::kShapeCenter:
          pa_mode_stat.addShapeCenterPinNum(1);
          break;
        default:
          LOG_INST.error(Loc::current(), "Type of access point is wrong!");
          break;
      }
      for (AccessPoint& access_point : access_point_list) {
        pa_mode_stat.addTotalAccessPointNum(1);
        pa_mode_stat.get_layer_access_point_num_map()[access_point.get_layer_idx()]++;
      }
    }
  }
}

void PinAccessor::reportPAModel(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _pa_data_manager.getDatabase().get_routing_layer_list();

  PAModelStat& pa_mode_stat = pa_model.get_pa_mode_stat();
  irt_int total_pin_num = pa_mode_stat.get_total_pin_num();
  irt_int track_grid_pin_num = pa_mode_stat.get_track_grid_pin_num();
  irt_int track_center_pin_num = pa_mode_stat.get_track_center_pin_num();
  irt_int shape_center_pin_num = pa_mode_stat.get_shape_center_pin_num();
  irt_int total_port_num = pa_mode_stat.get_total_port_num();
  std::map<irt_int, irt_int>& layer_port_num_map = pa_mode_stat.get_layer_port_num_map();
  irt_int total_access_point_num = pa_mode_stat.get_total_access_point_num();
  std::map<irt_int, irt_int>& layer_access_point_num_map = pa_mode_stat.get_layer_access_point_num_map();

  fort::char_table pin_table;
  pin_table.set_border_style(FT_SOLID_STYLE);
  pin_table << fort::header << "Access Type"
            << "Pin Number" << fort::endr;
  pin_table << "Track Grid" << RTUtil::getString(track_grid_pin_num, "(", RTUtil::getPercentage(track_grid_pin_num, total_pin_num), "%)")
            << fort::endr;
  pin_table << "Track Center"
            << RTUtil::getString(track_center_pin_num, "(", RTUtil::getPercentage(track_center_pin_num, total_pin_num), "%)") << fort::endr;
  pin_table << "Shape Center"
            << RTUtil::getString(shape_center_pin_num, "(", RTUtil::getPercentage(shape_center_pin_num, total_pin_num), "%)") << fort::endr;
  pin_table << fort::header << "Total" << total_pin_num << fort::endr;

  for (std::string table_str : RTUtil::splitString(pin_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }

  fort::char_table port_table;
  port_table.set_border_style(FT_SOLID_STYLE);
  port_table << fort::header << "Routing Layer"
             << "Port Number"
             << "Access Point Number" << fort::endr;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int port_num = layer_port_num_map[routing_layer.get_layer_idx()];
    irt_int access_point_num = layer_access_point_num_map[routing_layer.get_layer_idx()];
    port_table << routing_layer.get_layer_name() << RTUtil::getString(port_num, "(", RTUtil::getPercentage(port_num, total_port_num), "%)")
               << RTUtil::getString(access_point_num, "(", RTUtil::getPercentage(access_point_num, total_access_point_num), "%)") << fort::endr;
  }
  port_table << fort::header << "Total" << total_port_num << total_access_point_num << fort::endr;

  for (std::string table_str : RTUtil::splitString(port_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
