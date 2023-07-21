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

void PinAccessor::access(std::vector<Net>& net_list)
{
  Monitor monitor;

  accessNetList(net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kPinAccessor), " completed!", monitor.getStatsInfo());
}

// private

PinAccessor* PinAccessor::_pa_instance = nullptr;

void PinAccessor::accessNetList(std::vector<Net>& net_list)
{
  PAModel pa_model = init(net_list);
  iterative(pa_model);
  update(pa_model);
}

#if 1  // init

PAModel PinAccessor::init(std::vector<Net>& net_list)
{
  PAModel pa_model = initPAModel(net_list);
  buildPAModel(pa_model);
  checkPAModel(pa_model);
  return pa_model;
}

PAModel PinAccessor::initPAModel(std::vector<Net>& net_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  PAModel pa_model;
  GridMap<PAGCell>& pa_gcell_map = pa_model.get_pa_gcell_map();
  pa_gcell_map.init(die.getXSize(), die.getYSize());
  for (irt_int x = 0; x < die.getXSize(); x++) {
    for (irt_int y = 0; y < die.getYSize(); y++) {
      PAGCell& pa_gcell = pa_gcell_map[x][y];
      pa_gcell.set_base_region(RTUtil::getRealRect(x, y, gcell_axis));
      pa_gcell.set_top_layer_idx(routing_layer_list.back().get_layer_idx());
      pa_gcell.set_bottom_layer_idx(routing_layer_list.front().get_layer_idx());
      for (PASourceType pa_source_type : {PASourceType::kBlockAndPin, PASourceType::kEnclosure}) {
        pa_gcell.get_source_region_query_map()[pa_source_type] = RTAPI_INST.initRegionQuery();
      }
    }
  }
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
  pa_net.set_net_name(net.get_net_name());
  for (Pin& pin : net.get_pin_list()) {
    pa_net.get_pa_pin_list().push_back(PAPin(pin));
  }
  pa_net.set_pa_driving_pin(PAPin(net.get_driving_pin()));
  pa_net.set_bounding_box(net.get_bounding_box());
  return pa_net;
}

void PinAccessor::buildPAModel(PAModel& pa_model)
{
  updateNetRectMap(pa_model);
  cutBlockageList(pa_model);
}

void PinAccessor::updateNetRectMap(PAModel& pa_model)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = DM_INST.getDatabase().get_cut_blockage_list();

  for (Blockage& routing_blockage : routing_blockage_list) {
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), routing_blockage.get_layer_idx());
    addRectToEnv(pa_model, PASourceType::kBlockAndPin, -1, blockage_real_rect, true);
  }
  for (Blockage& cut_blockage : cut_blockage_list) {
    LayerRect blockage_real_rect(cut_blockage.get_real_rect(), cut_blockage.get_layer_idx());
    addRectToEnv(pa_model, PASourceType::kBlockAndPin, -1, blockage_real_rect, false);
  }
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        addRectToEnv(pa_model, PASourceType::kBlockAndPin, pa_net.get_net_idx(), shape_real_rect, true);
      }
      for (EXTLayerRect& cut_shape : pa_pin.get_cut_shape_list()) {
        LayerRect shape_real_rect(cut_shape.get_real_rect(), cut_shape.get_layer_idx());
        addRectToEnv(pa_model, PASourceType::kBlockAndPin, pa_net.get_net_idx(), shape_real_rect, false);
      }
    }
  }
}

void PinAccessor::addRectToEnv(PAModel& pa_model, PASourceType pa_source_type, irt_int net_idx, LayerRect real_rect, bool is_routing)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  GridMap<PAGCell>& pa_gcell_map = pa_model.get_pa_gcell_map();

  ids::DRCRect ids_drc_rect = RTAPI_INST.convertToIDSRect(net_idx, real_rect, is_routing);
  for (const LayerRect& max_scope_real_rect : RTAPI_INST.getMaxScope(ids_drc_rect)) {
    LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
    PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
    for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
      for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
        if (is_routing) {
          pa_gcell_map[x][y].get_source_routing_net_rect_map()[pa_source_type][real_rect.get_layer_idx()][net_idx].push_back(real_rect);
        } else {
          pa_gcell_map[x][y].get_source_cut_net_rect_map()[pa_source_type][real_rect.get_layer_idx()][net_idx].push_back(real_rect);
        }
        RTAPI_INST.addEnvRectList(pa_gcell_map[x][y].get_source_region_query_map()[pa_source_type], ids_drc_rect);
      }
    }
  }
}

void PinAccessor::cutBlockageList(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  GridMap<PAGCell>& pa_gcell_map = pa_model.get_pa_gcell_map();

  for (irt_int x = 0; x < pa_gcell_map.get_x_size(); x++) {
    for (irt_int y = 0; y < pa_gcell_map.get_y_size(); y++) {
      PAGCell& pa_gcell = pa_gcell_map[x][y];

      for (auto& [routing_layer_idx, net_rect_map] : pa_gcell.get_source_routing_net_rect_map()[PASourceType::kBlockAndPin]) {
        RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];

        std::vector<LayerRect> new_blockage_list;
        new_blockage_list.reserve(net_rect_map[-1].size());
        std::map<LayerRect, std::vector<PlanarRect>, CmpLayerRectByXASC> blockage_shape_list_map;

        for (LayerRect& blockage : net_rect_map[-1]) {
          bool is_cutting = false;
          for (auto& [net_idx, net_shape_list] : net_rect_map) {
            if (net_idx == -1) {
              continue;
            }
            for (LayerRect& net_shape : net_shape_list) {
              if (!RTUtil::isInside(blockage, net_shape)) {
                continue;
              }
              for (LayerRect& min_scope_net_shape : RTAPI_INST.getMinScope(RTAPI_INST.convertToIDSRect(net_idx, net_shape, true))) {
                PlanarRect enlarge_net_shape = RTUtil::getEnlargedRect(min_scope_net_shape, routing_layer.get_min_width());
                blockage_shape_list_map[blockage].push_back(enlarge_net_shape);
              }
              is_cutting = true;
            }
          }
          if (!is_cutting) {
            new_blockage_list.push_back(blockage);
          }
        }
        for (auto& [blockage, enlarge_net_shape_list] : blockage_shape_list_map) {
          for (PlanarRect& cutting_rect : RTUtil::getCuttingRectList(blockage, enlarge_net_shape_list)) {
            new_blockage_list.emplace_back(cutting_rect, blockage.get_layer_idx());
          }
        }
        net_rect_map[-1] = new_blockage_list;
      }
    }
  }
}

void PinAccessor::checkPAModel(PAModel& pa_model)
{
  GridMap<PAGCell>& pa_gcell_map = pa_model.get_pa_gcell_map();
  for (irt_int x_idx = 0; x_idx < pa_gcell_map.get_x_size(); x_idx++) {
    for (irt_int y_idx = 0; y_idx < pa_gcell_map.get_y_size(); y_idx++) {
      PAGCell& pa_gcell = pa_gcell_map[x_idx][y_idx];
      for (auto& [source, routing_net_rect_map] : pa_gcell.get_source_routing_net_rect_map()) {
        for (auto& [layer_idx, net_rect_map] : routing_net_rect_map) {
          for (auto& [net_idx, rect_list] : net_rect_map) {
            for (LayerRect& rect : rect_list) {
              if (layer_idx == rect.get_layer_idx()) {
                continue;
              }
              LOG_INST.error(Loc::current(), "The layer of source routing net rect is different!");
            }
          }
        }
      }
      for (auto& [source, cut_net_rect_map] : pa_gcell.get_source_cut_net_rect_map()) {
        for (auto& [layer_idx, net_rect_map] : cut_net_rect_map) {
          for (auto& [net_idx, rect_list] : net_rect_map) {
            for (LayerRect& rect : rect_list) {
              if (layer_idx == rect.get_layer_idx()) {
                continue;
              }
              LOG_INST.error(Loc::current(), "The layer of source cut net rect is different!");
            }
          }
        }
      }
    }
  }
}

#endif

#if 1  // iterative

void PinAccessor::iterative(PAModel& pa_model)
{
  irt_int pa_iter_num = 1;
  for (irt_int iter = 1; iter <= pa_iter_num; iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Iteration(", iter, "/", pa_iter_num, ") ******");

    accessPAModel(pa_model);
    processPAModel(pa_model);
    reportPAModel(pa_model);

    LOG_INST.info(Loc::current(), "****** End Iteration(", iter, "/", pa_iter_num, ")", iter_monitor.getStatsInfo(), " ******");
  }
}

void PinAccessor::accessPAModel(PAModel& pa_model)
{
  accessPANetList(pa_model);
  updateNetEnclosureMap(pa_model);
  eliminateViaConflict(pa_model);
}

void PinAccessor::accessPANetList(PAModel& pa_model)
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
  initAccessPointList(pa_model, pa_net);
  mergeAccessPointList(pa_net);
  selectAccessPointList(pa_net);
  eliminateDRCViolation(pa_model, pa_net);
}

void PinAccessor::initAccessPointList(PAModel& pa_model, PANet& pa_net)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;

  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::vector<AccessPoint>& access_point_list = pa_pin.get_access_point_list();
    for (LayerRect& aligned_pin_shape : getLegalPinShapeList(pa_model, pa_net.get_net_idx(), pa_pin)) {
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
      // track grid
      for (irt_int layer_idx : layer_idx_list) {
        for (irt_int x : RTUtil::getClosedScaleList(lb_x, rt_x, routing_layer_list[layer_idx].getXTrackGridList())) {
          for (irt_int y : RTUtil::getClosedScaleList(lb_y, rt_y, routing_layer_list[layer_idx].getYTrackGridList())) {
            access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kTrackGrid);
          }
        }
      }
      for (irt_int i = 0; i < static_cast<irt_int>(layer_idx_list.size()) - 1; i++) {
        RoutingLayer curr_routing_layer = routing_layer_list[layer_idx_list[i]];
        RoutingLayer adj_routing_layer = routing_layer_list[layer_idx_list[i + 1]];
        for (irt_int x : RTUtil::getClosedScaleList(lb_x, rt_x, curr_routing_layer.getXTrackGridList())) {
          for (irt_int y : RTUtil::getClosedScaleList(lb_y, rt_y, adj_routing_layer.getYTrackGridList())) {
            access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kTrackGrid);
          }
        }
        for (irt_int y : RTUtil::getClosedScaleList(lb_y, rt_y, curr_routing_layer.getYTrackGridList())) {
          for (irt_int x : RTUtil::getClosedScaleList(lb_x, rt_x, adj_routing_layer.getXTrackGridList())) {
            access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kTrackGrid);
          }
        }
      }
      // on track
      irt_int mid_x = (lb_x + rt_x) / 2;
      irt_int mid_y = (lb_y + rt_y) / 2;
      for (irt_int layer_idx : layer_idx_list) {
        for (irt_int x : RTUtil::getClosedScaleList(lb_x, rt_x, routing_layer_list[layer_idx].getXTrackGridList())) {
          for (irt_int y : {lb_y, mid_y, rt_y}) {
            access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kOnTrack);
          }
        }
        for (irt_int y : RTUtil::getClosedScaleList(lb_y, rt_y, routing_layer_list[layer_idx].getYTrackGridList())) {
          for (irt_int x : {lb_x, mid_x, rt_x}) {
            access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kOnTrack);
          }
        }
      }
      // on shape
      for (irt_int x : {lb_x, mid_x, rt_x}) {
        for (irt_int y : {lb_y, mid_y, rt_y}) {
          access_point_list.emplace_back(x, y, pin_shape_layer_idx, AccessPointType::kOnShape);
        }
      }
    }
    if (access_point_list.empty()) {
      LOG_INST.error(Loc::current(), "No access point was generated!");
    }
  }
}

std::vector<LayerRect> PinAccessor::getLegalPinShapeList(PAModel& pa_model, irt_int pa_net_idx, PAPin& pa_pin)
{
  std::map<irt_int, std::vector<EXTLayerRect>> layer_pin_shape_list;
  for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
    layer_pin_shape_list[routing_shape.get_layer_idx()].emplace_back(routing_shape);
  }
  std::vector<LayerRect> legal_rect_list;
  for (auto& [layer_idx, pin_shpae_list] : layer_pin_shape_list) {
    std::vector<PlanarRect> planar_legal_rect_list;
    std::vector<PlanarRect> up_via_legal_rect_list = getViaLegalRectList(pa_model, pa_net_idx, layer_idx, pin_shpae_list);
    planar_legal_rect_list.insert(planar_legal_rect_list.end(), up_via_legal_rect_list.begin(), up_via_legal_rect_list.end());
    std::vector<PlanarRect> down_via_legal_rect_list = getViaLegalRectList(pa_model, pa_net_idx, layer_idx - 1, pin_shpae_list);
    planar_legal_rect_list.insert(planar_legal_rect_list.end(), down_via_legal_rect_list.begin(), down_via_legal_rect_list.end());

    for (Direction direction : {Direction::kHorizontal, Direction::kVertical}) {
      for (PlanarRect legal_rect : RTUtil::getMergeRectList(planar_legal_rect_list, direction)) {
        legal_rect_list.emplace_back(legal_rect, layer_idx);
      }
    }
  }
  if (legal_rect_list.empty()) {
    LOG_INST.warning(Loc::current(), "The pin ", pa_pin.get_pin_name(), " has no legal pin shape!");
    for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
      legal_rect_list.emplace_back(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
    }
  }
  return legal_rect_list;
}

std::vector<PlanarRect> PinAccessor::getViaLegalRectList(PAModel& pa_model, irt_int pa_net_idx, irt_int via_below_layer_idx,
                                                         std::vector<EXTLayerRect>& pin_shape_list)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  GridMap<PAGCell>& pa_gcell_map = pa_model.get_pa_gcell_map();

  if (via_below_layer_idx < routing_layer_list.front().get_layer_idx()
      || routing_layer_list.back().get_layer_idx() <= via_below_layer_idx) {
    return {};
  }
  ViaMaster& via_master = layer_via_master_list[via_below_layer_idx].front();

  // pin_shape 往里缩小的的形状
  std::vector<PlanarRect> reduced_rect_list;
  {
    irt_int half_x_span = -1;
    irt_int half_y_span = -1;
    if (via_master.get_above_enclosure().get_layer_idx() == pin_shape_list.front().get_layer_idx()) {
      half_x_span = via_master.get_above_enclosure().getXSpan() / 2;
      half_y_span = via_master.get_above_enclosure().getYSpan() / 2;
    } else {
      half_x_span = via_master.get_below_enclosure().getXSpan() / 2;
      half_y_span = via_master.get_below_enclosure().getYSpan() / 2;
    }
    // 当前层缩小后的结果
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      if (!RTUtil::hasReducedRect(pin_shape.get_real_rect(), half_x_span, half_y_span, half_x_span, half_y_span)) {
        continue;
      }
      PlanarRect reduced_rect = RTUtil::getReducedRect(pin_shape.get_real_rect(), half_x_span, half_y_span, half_x_span, half_y_span);
      reduced_rect_list.push_back(reduced_rect);
    }
  }
  // pin_shape 原始的形状
  std::vector<PlanarRect> origin_rect_list;
  for (EXTLayerRect& pin_shape : pin_shape_list) {
    origin_rect_list.push_back(pin_shape.get_real_rect());
  }
  // pin_shape 由于blockage要被剪裁的形状
  std::vector<PlanarRect> cutting_rect_list;
  for (LayerRect enclosure : {via_master.get_above_enclosure(), via_master.get_below_enclosure()}) {
    irt_int half_x_span = enclosure.getXSpan() / 2;
    irt_int half_y_span = enclosure.getYSpan() / 2;

    for (EXTLayerRect& pin_shape : pin_shape_list) {
      for (irt_int x = pin_shape.get_grid_lb_x(); x <= pin_shape.get_grid_rt_x(); x++) {
        for (irt_int y = pin_shape.get_grid_lb_y(); y <= pin_shape.get_grid_rt_y(); y++) {
          for (auto& [curr_net_idx, net_rect_list] :
               pa_gcell_map[x][y].get_source_routing_net_rect_map()[PASourceType::kBlockAndPin][enclosure.get_layer_idx()]) {
            if (pa_net_idx == curr_net_idx) {
              continue;
            }
            for (LayerRect& net_rect : net_rect_list) {
              for (LayerRect& min_scope_blockage : RTAPI_INST.getMinScope(RTAPI_INST.convertToIDSRect(curr_net_idx, net_rect, true))) {
                PlanarRect enlarged_rect = RTUtil::getEnlargedRect(min_scope_blockage, half_x_span, half_y_span, half_x_span, half_y_span);
                if (!RTUtil::isOpenOverlap(pin_shape.get_real_rect(), enlarged_rect)) {
                  continue;
                }
                cutting_rect_list.push_back(enlarged_rect);
              }
            }
          }
        }
      }
    }
  }
  std::vector<PlanarRect> via_legal_rect_list = RTUtil::getCuttingRectList(origin_rect_list, cutting_rect_list);
  std::vector<PlanarRect> reduced_legal_rect_list = RTUtil::getOverlap(via_legal_rect_list, reduced_rect_list);
  if (!reduced_legal_rect_list.empty()) {
    via_legal_rect_list = reduced_legal_rect_list;
  }
  return via_legal_rect_list;
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
  selectAccessPointType(pa_net);
  buildBoundingBox(pa_net);
  buildAccessPointList(pa_net);
  selectGCellAccessPoint(pa_net);
}

void PinAccessor::selectAccessPointType(PANet& pa_net)
{
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::vector<AccessPoint>& pin_access_point_list = pa_pin.get_access_point_list();
    std::map<irt_int, std::map<AccessPointType, std::vector<AccessPoint>>> layer_access_point_map;
    for (AccessPoint& access_point : pin_access_point_list) {
      layer_access_point_map[access_point.get_layer_idx()][access_point.get_type()].push_back(access_point);
    }
    pin_access_point_list.clear();
    for (auto& [layer_idx, type_point_map] : layer_access_point_map) {
      for (AccessPointType access_point_type : {AccessPointType::kTrackGrid, AccessPointType::kOnTrack, AccessPointType::kOnShape}) {
        std::vector<AccessPoint>& candidate_access_point_list = type_point_map[access_point_type];
        if (candidate_access_point_list.empty()) {
          continue;
        }
        for (AccessPoint& access_point : candidate_access_point_list) {
          pin_access_point_list.push_back(access_point);
        }
        break;
      }
    }
  }
}

void PinAccessor::buildBoundingBox(PANet& pa_net)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();

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
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  BoundingBox& bounding_box = pa_net.get_bounding_box();

  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
      access_point.set_grid_coord(RTUtil::getGridCoord(access_point.get_real_coord(), gcell_axis, bounding_box));
    }
  }
}

void PinAccessor::selectGCellAccessPoint(PANet& pa_net)
{
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::vector<AccessPoint>& pin_access_point_list = pa_pin.get_access_point_list();
    std::map<irt_int, std::map<PlanarCoord, std::vector<AccessPoint>, CmpPlanarCoordByXASC>> layer_access_point_map;
    for (AccessPoint& access_point : pin_access_point_list) {
      layer_access_point_map[access_point.get_layer_idx()][access_point.get_grid_coord()].push_back(access_point);
    }
    pin_access_point_list.clear();
    for (auto& [layer_idx, grid_access_point_map] : layer_access_point_map) {
      std::vector<AccessPoint> candidate_access_point_list;
      for (auto& [grid_coord, access_point_list] : grid_access_point_map) {
        if (candidate_access_point_list.size() < access_point_list.size()) {
          candidate_access_point_list = access_point_list;
        }
      }
      for (AccessPoint& access_point : candidate_access_point_list) {
        pin_access_point_list.push_back(access_point);
      }
    }
  }
}

void PinAccessor::eliminateDRCViolation(PAModel& pa_model, PANet& pa_net)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::vector<AccessPoint> legal_access_point_list;
    for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
      irt_int access_point_layer_idx = access_point.get_layer_idx();
      for (irt_int via_below_layer_idx : {access_point_layer_idx - 1, access_point_layer_idx}) {
        if (routing_layer_list.back().get_layer_idx() <= via_below_layer_idx
            || via_below_layer_idx < routing_layer_list.front().get_layer_idx()) {
          continue;
        }
        std::vector<Segment<LayerCoord>> segment_list;
        segment_list.emplace_back(LayerCoord(access_point.get_real_coord(), via_below_layer_idx),
                                  LayerCoord(access_point.get_real_coord(), via_below_layer_idx + 1));
        if (!hasViolation(pa_model, PASourceType::kBlockAndPin, pa_net.get_net_idx(), segment_list)) {
          legal_access_point_list.push_back(access_point);
          break;
        }
      }
    }
    if (!legal_access_point_list.empty()) {
      pa_pin.set_access_point_list(legal_access_point_list);
    }
  }
}

bool PinAccessor::hasViolation(PAModel& pa_model, PASourceType pa_source_type, irt_int net_idx,
                               std::vector<Segment<LayerCoord>>& segment_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  GridMap<PAGCell>& pa_gcell_map = pa_model.get_pa_gcell_map();

  std::vector<ids::DRCRect> ids_drc_rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx != second_layer_idx) {
      RTUtil::sortASC(first_layer_idx, second_layer_idx);
      for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        ViaMaster& via_master = layer_via_master_list[layer_idx].front();

        LayerRect& above_enclosure = via_master.get_above_enclosure();
        LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
        ids_drc_rect_list.push_back(RTAPI_INST.convertToIDSRect(net_idx, offset_above_enclosure, true));

        LayerRect& below_enclosure = via_master.get_below_enclosure();
        LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
        ids_drc_rect_list.push_back(RTAPI_INST.convertToIDSRect(net_idx, offset_below_enclosure, true));

        for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
          LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
          ids_drc_rect_list.push_back(RTAPI_INST.convertToIDSRect(net_idx, offset_cut_shape, false));
        }
      }
    } else {
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
      LayerRect wire_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width), first_layer_idx);
      ids_drc_rect_list.push_back(RTAPI_INST.convertToIDSRect(net_idx, wire_rect, true));
    }
  }
  std::set<PlanarCoord, CmpPlanarCoordByXASC> grid_coord_set;
  for (ids::DRCRect& ids_drc_rect : ids_drc_rect_list) {
    for (const LayerRect& max_scope_real_rect : RTAPI_INST.getMaxScope(ids_drc_rect)) {
      PlanarRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
      PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
      for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
        for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
          grid_coord_set.insert(PlanarCoord(x, y));
        }
      }
    }
  }
  bool has_violation = false;
  for (const PlanarCoord& grid_coord : grid_coord_set) {
    PAGCell& pa_gcell = pa_gcell_map[grid_coord.get_x()][grid_coord.get_y()];
    if (RTAPI_INST.hasViolation(pa_gcell.get_source_region_query_map()[pa_source_type], ids_drc_rect_list)) {
      has_violation = true;
      break;
    }
  }
  return has_violation;
}

void PinAccessor::updateNetEnclosureMap(PAModel& pa_model)
{
  // check access point
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

  // update enclosure
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    std::vector<LayerCoord> real_coord_list;
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (LayerCoord& real_coord : pa_pin.getRealCoordList()) {
        real_coord_list.push_back(real_coord);
      }
    }
    std::vector<LayerRect> real_rect_list;
    for (LayerCoord& real_coord : real_coord_list) {
      irt_int layer_idx = real_coord.get_layer_idx();
      for (irt_int via_below_layer_idx : RTUtil::getViaBelowLayerIdxList(layer_idx, bottom_routing_layer_idx, top_routing_layer_idx)) {
        std::vector<Segment<LayerCoord>> segment_list;
        segment_list.emplace_back(LayerCoord(real_coord.get_planar_coord(), via_below_layer_idx),
                                  LayerCoord(real_coord.get_planar_coord(), via_below_layer_idx + 1));
        addRectToEnv(pa_model, PASourceType::kEnclosure, pa_net.get_net_idx(), segment_list);
      }
    }
  }
}

void PinAccessor::addRectToEnv(PAModel& pa_model, PASourceType pa_source_type, irt_int net_idx,
                               std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<ids::DRCRect> ids_drc_rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx != second_layer_idx) {
      RTUtil::sortASC(first_layer_idx, second_layer_idx);
      for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        ViaMaster& via_master = layer_via_master_list[layer_idx].front();

        LayerRect& above_enclosure = via_master.get_above_enclosure();
        LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
        addRectToEnv(pa_model, pa_source_type, net_idx, offset_above_enclosure, true);

        LayerRect& below_enclosure = via_master.get_below_enclosure();
        LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
        addRectToEnv(pa_model, pa_source_type, net_idx, offset_below_enclosure, true);

        for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
          LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
          addRectToEnv(pa_model, pa_source_type, net_idx, offset_cut_shape, false);
        }
      }
    } else {
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
      LayerRect wire_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width), first_layer_idx);
      addRectToEnv(pa_model, pa_source_type, net_idx, wire_rect, true);
    }
  }
}

void PinAccessor::eliminateViaConflict(PAModel& pa_model)
{
  Monitor monitor;

  std::vector<PANet>& pa_net_list = pa_model.get_pa_net_list();

  irt_int batch_size = RTUtil::getBatchSize(pa_net_list.size());

  Monitor stage_monitor;

#pragma omp parallel for
  for (size_t i = 0; i < pa_net_list.size(); i++) {
    selectByViaNumber(pa_net_list[i], pa_model);
    selectByNetDistance(pa_net_list[i]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Eliminate conflict ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Eliminate conflict ", pa_net_list.size(), " nets", monitor.getStatsInfo());
  }
}

void PinAccessor::selectByViaNumber(PANet& pa_net, PAModel& pa_model)
{
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    std::vector<AccessPoint>& pin_access_point_list = pa_pin.get_access_point_list();
    // via_num_access_point_map
    std::map<irt_int, std::vector<AccessPoint>, std::greater<irt_int>> via_num_access_point_map;
    for (AccessPoint& access_point : pin_access_point_list) {
      irt_int via_num = 0;
      for (std::vector<irt_int> via_below_layer_idx_list :
           RTUtil::getLevelViaBelowLayerIdxList(access_point.get_layer_idx(), bottom_routing_layer_idx, top_routing_layer_idx)) {
        for (irt_int via_below_layer_idx : via_below_layer_idx_list) {
          std::vector<Segment<LayerCoord>> segment_list;
          segment_list.emplace_back(LayerCoord(access_point.get_real_coord(), via_below_layer_idx),
                                    LayerCoord(access_point.get_real_coord(), via_below_layer_idx + 1));
          if (hasViolation(pa_model, PASourceType::kEnclosure, pa_net.get_net_idx(), segment_list)) {
            break;
          } else {
            via_num++;
          }
        }
      }
      via_num_access_point_map[via_num].push_back(access_point);
    }
    pin_access_point_list = via_num_access_point_map.begin()->second;
  }
}

void PinAccessor::selectByNetDistance(PANet& pa_net)
{
  for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
    LayerCoord balance_coord = RTUtil::getBalanceCoord(pa_pin.getRealCoordList());
    std::map<irt_int, std::vector<AccessPoint>> distance_access_point_map;
    std::vector<AccessPoint>& pin_access_point_list = pa_pin.get_access_point_list();
    for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
      LayerCoord real_coord(access_point.get_real_coord(), access_point.get_layer_idx());
      distance_access_point_map[RTUtil::getManhattanDistance(balance_coord, real_coord)].push_back(access_point);
    }
    pin_access_point_list.clear();
    pin_access_point_list.push_back(distance_access_point_map.begin()->second.front());
  }
}

void PinAccessor::processPAModel(PAModel& pa_model)
{
#pragma omp parallel for
  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    buildBoundingBox(pa_net);
    buildAccessPointList(pa_net);
    buildDrivingPin(pa_net);
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

void PinAccessor::reportPAModel(PAModel& pa_model)
{
  countPAModel(pa_model);
  reportTable(pa_model);
}

void PinAccessor::countPAModel(PAModel& pa_model)
{
  PAModelStat pa_mode_stat;

  std::map<AccessPointType, irt_int>& type_pin_num_map = pa_mode_stat.get_type_pin_num_map();
  std::map<irt_int, irt_int>& routing_port_num_map = pa_mode_stat.get_routing_port_num_map();
  std::map<irt_int, irt_int>& routing_access_point_num_map = pa_mode_stat.get_routing_access_point_num_map();

  for (PANet& pa_net : pa_model.get_pa_net_list()) {
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      AccessPointType access_point_type = pa_pin.get_access_point_list().front().get_type();
      type_pin_num_map[access_point_type]++;
    }
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      std::set<irt_int> routing_layer_idx_set;
      for (EXTLayerRect& routing_shape : pa_pin.get_routing_shape_list()) {
        routing_layer_idx_set.insert(routing_shape.get_layer_idx());
      }
      for (irt_int routing_layer_idx : routing_layer_idx_set) {
        routing_port_num_map[routing_layer_idx]++;
      }
    }
    for (PAPin& pa_pin : pa_net.get_pa_pin_list()) {
      for (AccessPoint& access_point : pa_pin.get_access_point_list()) {
        routing_access_point_num_map[access_point.get_layer_idx()]++;
      }
    }
  }
  irt_int total_pin_num = 0;
  irt_int total_port_num = 0;
  irt_int total_access_point_num = 0;
  for (auto& [type, pin_num] : type_pin_num_map) {
    total_pin_num += pin_num;
  }
  for (auto& [routing_layer_idx, port_num] : routing_port_num_map) {
    total_port_num += port_num;
  }
  for (auto& [routing_layer_idx, access_point_num] : routing_access_point_num_map) {
    total_access_point_num += access_point_num;
  }
  pa_mode_stat.set_total_pin_num(total_pin_num);
  pa_mode_stat.set_total_port_num(total_port_num);
  pa_mode_stat.set_total_access_point_num(total_access_point_num);

  pa_model.set_pa_mode_stat(pa_mode_stat);
}

void PinAccessor::reportTable(PAModel& pa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  PAModelStat& pa_mode_stat = pa_model.get_pa_mode_stat();
  std::map<AccessPointType, irt_int>& type_pin_num_map = pa_mode_stat.get_type_pin_num_map();
  irt_int total_pin_num = pa_mode_stat.get_total_pin_num();
  std::map<irt_int, irt_int>& routing_port_num_map = pa_mode_stat.get_routing_port_num_map();
  irt_int total_port_num = pa_mode_stat.get_total_port_num();
  std::map<irt_int, irt_int>& routing_access_point_num_map = pa_mode_stat.get_routing_access_point_num_map();
  irt_int total_access_point_num = pa_mode_stat.get_total_access_point_num();

  fort::char_table pin_table;
  pin_table.set_border_style(FT_SOLID_STYLE);
  pin_table << fort::header << "Access Type"
            << "Pin Number" << fort::endr;
  pin_table << "Track Grid"
            << RTUtil::getString(type_pin_num_map[AccessPointType::kTrackGrid], "(",
                                 RTUtil::getPercentage(type_pin_num_map[AccessPointType::kTrackGrid], total_pin_num), "%)")
            << fort::endr;
  pin_table << "On Track"
            << RTUtil::getString(type_pin_num_map[AccessPointType::kOnTrack], "(",
                                 RTUtil::getPercentage(type_pin_num_map[AccessPointType::kOnTrack], total_pin_num), "%)")
            << fort::endr;
  pin_table << "On Shape"
            << RTUtil::getString(type_pin_num_map[AccessPointType::kOnShape], "(",
                                 RTUtil::getPercentage(type_pin_num_map[AccessPointType::kOnShape], total_pin_num), "%)")
            << fort::endr;
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
    irt_int routing_layer_idx = routing_layer.get_layer_idx();
    port_table << routing_layer.get_layer_name()
               << RTUtil::getString(routing_port_num_map[routing_layer_idx], "(",
                                    RTUtil::getPercentage(routing_port_num_map[routing_layer_idx], total_port_num), "%)")
               << RTUtil::getString(routing_access_point_num_map[routing_layer_idx], "(",
                                    RTUtil::getPercentage(routing_access_point_num_map[routing_layer_idx], total_access_point_num), "%)")
               << fort::endr;
  }
  port_table << fort::header << "Total" << total_port_num << total_access_point_num << fort::endr;
  for (std::string table_str : RTUtil::splitString(port_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

#if 1  // update

void PinAccessor::update(PAModel& pa_model)
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
}  // namespace irt
