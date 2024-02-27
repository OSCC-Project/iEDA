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
// MERCHANTABILITY OR FIT FOR A SARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "SupplyAnalyzer.hpp"

#include "GDSPlotter.hpp"
#include "GPGDS.hpp"
#include "SAModel.hpp"

namespace irt {

// public

void SupplyAnalyzer::initInst()
{
  if (_sa_instance == nullptr) {
    _sa_instance = new SupplyAnalyzer();
  }
}

SupplyAnalyzer& SupplyAnalyzer::getInst()
{
  if (_sa_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_sa_instance;
}

void SupplyAnalyzer::destroyInst()
{
  if (_sa_instance != nullptr) {
    delete _sa_instance;
    _sa_instance = nullptr;
  }
}

// function

void SupplyAnalyzer::analyze(std::vector<Net>& net_list)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin analyzing...");
  SAModel sa_model = initSAModel(net_list);
  buildLayerNodeMap(sa_model);
  buildSupplySchedule(sa_model);
  analyzeSupply(sa_model);
  updateSAModel(sa_model);
  LOG_INST.info(Loc::current(), "End analyze", monitor.getStatsInfo());

  // plotSAModel(sa_model);
  // reportSAModel(sa_model);
}

// private

SupplyAnalyzer* SupplyAnalyzer::_sa_instance = nullptr;

SAModel SupplyAnalyzer::initSAModel(std::vector<Net>& net_list)
{
  SAModel sa_model;
  return sa_model;
}

void SupplyAnalyzer::buildLayerNodeMap(SAModel& sa_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();

  std::vector<GridMap<SANode>>& layer_node_map = sa_model.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());

  for (size_t layer_idx = 0; layer_idx < layer_node_map.size(); layer_idx++) {
    GridMap<SANode>& sa_node_map = layer_node_map[layer_idx];
    sa_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());

    for (int32_t x = 0; x < sa_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sa_node_map.get_y_size(); y++) {
        SANode& sa_node = sa_node_map[x][y];
        sa_node.set_shape(RTUtil::getRealRectByGCell(x, y, gcell_axis));
        sa_node.set_net_fixed_rect_map(gcell_map[x][y].get_type_layer_net_fixed_rect_map()[true][layer_idx]);
        if (routing_layer_list[layer_idx].isPreferH()) {
          sa_node.get_orien_supply_map()[Orientation::kEast] = 0;
          sa_node.get_orien_supply_map()[Orientation::kWest] = 0;
        } else {
          sa_node.get_orien_supply_map()[Orientation::kSouth] = 0;
          sa_node.get_orien_supply_map()[Orientation::kNorth] = 0;
        }
      }
    }
  }
}

void SupplyAnalyzer::buildSupplySchedule(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<SANode>>& layer_node_map = sa_model.get_layer_node_map();

  std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>> grid_pair_list_list;
  // 仅需要分为两批
  grid_pair_list_list.resize(2);
  for (size_t layer_idx = 0; layer_idx < layer_node_map.size(); layer_idx++) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    GridMap<SANode>& node_map = layer_node_map[layer_idx];
    if (routing_layer.isPreferH()) {
      for (int32_t y = 0; y < node_map.get_y_size(); y++) {
        for (int32_t begin_x = 1; begin_x <= 2; begin_x++) {
          for (int32_t x = begin_x; x < node_map.get_x_size(); x += 2) {
            grid_pair_list_list[x % 2].emplace_back(LayerCoord(x - 1, y, layer_idx), LayerCoord(x, y, layer_idx));
          }
        }
      }
    } else {
      for (int32_t x = 0; x < node_map.get_x_size(); x++) {
        for (int32_t begin_y = 1; begin_y <= 2; begin_y++) {
          for (int32_t y = begin_y; y < node_map.get_y_size(); y += 2) {
            grid_pair_list_list[y % 2].emplace_back(LayerCoord(x, y - 1, layer_idx), LayerCoord(x, y, layer_idx));
          }
        }
      }
    }
  }
  sa_model.set_grid_pair_list_list(grid_pair_list_list);
}

void SupplyAnalyzer::analyzeSupply(SAModel& sa_model)
{
  std::vector<GridMap<SANode>>& layer_node_map = sa_model.get_layer_node_map();

  size_t total_pair_num = 0;
  for (std::vector<std::pair<LayerCoord, LayerCoord>>& grid_pair_list : sa_model.get_grid_pair_list_list()) {
    total_pair_num += grid_pair_list.size();
  }

  size_t analyzed_pair_num = 0;
  for (std::vector<std::pair<LayerCoord, LayerCoord>>& grid_pair_list : sa_model.get_grid_pair_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (std::pair<LayerCoord, LayerCoord>& grid_pair : grid_pair_list) {
      LayerCoord first_coord = grid_pair.first;
      LayerCoord second_coord = grid_pair.second;
      if (first_coord.get_layer_idx() != second_coord.get_layer_idx()) {
        LOG_INST.error(Loc::current(), "The grid_pair layer_idx is not equal!");
      }
      int32_t layer_idx = first_coord.get_layer_idx();

      Orientation first_orientation = RTUtil::getOrientation(first_coord, second_coord);
      Orientation second_orientation = RTUtil::getOppositeOrientation(first_orientation);

      SANode& first_node = layer_node_map[layer_idx][first_coord.get_x()][first_coord.get_y()];
      SANode& second_node = layer_node_map[layer_idx][second_coord.get_x()][second_coord.get_y()];
      for (LayerRect& wire : getCrossingWireList(layer_idx, first_node, second_node)) {
        if (isAccess(wire, first_node, second_node)) {
          first_node.get_orien_supply_map()[first_orientation]++;
          second_node.get_orien_supply_map()[second_orientation]++;
        }
      }
    }
    analyzed_pair_num += grid_pair_list.size();
    LOG_INST.info(Loc::current(), "Analyzed ", analyzed_pair_num, "/", total_pair_num, " grid pairs", stage_monitor.getStatsInfo());
  }
}

std::vector<LayerRect> SupplyAnalyzer::getCrossingWireList(int32_t layer_idx, SANode& first_node, SANode& second_node)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[layer_idx];

  PlanarRect base_rect = RTUtil::getBoundingBox({first_node.get_shape(), second_node.get_shape()});
  int32_t real_lb_x = base_rect.get_lb_x();
  int32_t real_lb_y = base_rect.get_lb_y();
  int32_t real_rt_x = base_rect.get_rt_x();
  int32_t real_rt_y = base_rect.get_rt_y();
  int32_t half_width = routing_layer.get_min_width() / 2;

  std::vector<LayerRect> wire_list;
  if (routing_layer.isPreferH()) {
    for (int32_t y : RTUtil::getOpenScaleList(real_lb_y, real_rt_y, routing_layer.getYTrackGridList())) {
      wire_list.emplace_back(real_lb_x, y - half_width, real_rt_x, y + half_width, layer_idx);
    }
  } else {
    for (int32_t x : RTUtil::getOpenScaleList(real_lb_x, real_rt_x, routing_layer.getXTrackGridList())) {
      wire_list.emplace_back(x - half_width, real_lb_y, x + half_width, real_rt_y, layer_idx);
    }
  }
  return wire_list;
}

bool SupplyAnalyzer::isAccess(LayerRect& wire, SANode& first_node, SANode& second_node)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[wire.get_layer_idx()];

  for (SANode* node : {&first_node, &second_node}) {
    for (auto& [net_idx, fixed_rect_set] : node->get_net_fixed_rect_map()) {
      for (auto& fixed_rect : fixed_rect_set) {
        int32_t enlarged_size = routing_layer.getMinSpacing(fixed_rect->get_real_rect());
        PlanarRect enlarged_rect = RTUtil::getEnlargedRect(fixed_rect->get_real_rect(), enlarged_size);
        if (RTUtil::isOpenOverlap(enlarged_rect, wire)) {
          // 阻塞
          return false;
        }
      }
    }
  }
  return true;
}

void SupplyAnalyzer::updateSAModel(SAModel& sa_model)
{
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();

  std::vector<GridMap<SANode>>& layer_node_map = sa_model.get_layer_node_map();

  for (size_t layer_idx = 0; layer_idx < layer_node_map.size(); layer_idx++) {
    GridMap<SANode>& sa_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < sa_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < sa_node_map.get_y_size(); y++) {
        gcell_map[x][y].get_routing_orien_supply_map()[layer_idx] = sa_node_map[x][y].get_orien_supply_map();
      }
    }
  }
}

#if 1  // exhibit

void SupplyAnalyzer::plotSAModel(SAModel& sa_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string sa_temp_directory_path = DM_INST.getConfig().sa_temp_directory_path;

  GPGDS gp_gds;

  // track_axis_struct
  GPStruct track_axis_struct("track_axis_struct");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<int32_t> x_list = RTUtil::getClosedScaleList(die.get_real_lb_x(), die.get_real_rt_x(), routing_layer.getXTrackGridList());
    std::vector<int32_t> y_list = RTUtil::getClosedScaleList(die.get_real_lb_y(), die.get_real_rt_y(), routing_layer.getYTrackGridList());
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

  // gcell_axis
  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<int32_t> gcell_x_list = RTUtil::getClosedScaleList(die.get_real_lb_x(), die.get_real_rt_x(), gcell_axis.get_x_grid_list());
  std::vector<int32_t> gcell_y_list = RTUtil::getClosedScaleList(die.get_real_lb_y(), die.get_real_rt_y(), gcell_axis.get_y_grid_list());
  for (int32_t x : gcell_x_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, die.get_real_lb_y(), x, die.get_real_rt_y());
    gcell_axis_struct.push(gp_path);
  }
  for (int32_t y : gcell_y_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(die.get_real_lb_x(), y, die.get_real_rt_x(), y);
    gcell_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(gcell_axis_struct);

  // sa_node_map
  GPStruct sa_node_map_struct("sa_node_map");
  std::vector<GridMap<SANode>>& layer_node_map = sa_model.get_layer_node_map();
  for (size_t layer_idx = 0; layer_idx < layer_node_map.size(); layer_idx++) {
    GridMap<SANode>& sa_node_map = layer_node_map[layer_idx];
    for (int32_t grid_x = 0; grid_x < sa_node_map.get_x_size(); grid_x++) {
      for (int32_t grid_y = 0; grid_y < sa_node_map.get_y_size(); grid_y++) {
        SANode& sa_node = sa_node_map[grid_x][grid_y];
        PlanarRect shape = sa_node.get_shape();
        int32_t y_reduced_span = shape.getYSpan() / 25;
        int32_t y = shape.get_rt_y();

        if (!sa_node.get_orien_supply_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_supply_map_info;
          gp_text_orien_supply_map_info.set_coord(shape.get_lb_x(), y);
          gp_text_orien_supply_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          std::string orien_supply_map_message = "--";
          for (auto& [orientation, supply] : sa_node.get_orien_supply_map()) {
            orien_supply_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", supply, ")");
          }
          gp_text_orien_supply_map_info.set_message(orien_supply_map_message);
          gp_text_orien_supply_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(static_cast<int32_t>(layer_idx)));
          gp_text_orien_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          sa_node_map_struct.push(gp_text_orien_supply_map_info);
        }
      }
    }
  }
  gp_gds.addStruct(sa_node_map_struct);

  std::string gds_file_path = RTUtil::getString(sa_temp_directory_path, "supply.gds");
  GP_INST.plot(gp_gds, gds_file_path);
}

void SupplyAnalyzer::reportSAModel(SAModel& sa_model)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin reporting...");
  reportSummary(sa_model);
  writeSupplyCSV(sa_model);
  LOG_INST.info(Loc::current(), "End report", monitor.getStatsInfo());
}

void SupplyAnalyzer::reportSummary(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::map<int32_t, int32_t>& sa_routing_supply_map = DM_INST.getSummary().sa_summary.routing_supply_map;
  int32_t& sa_total_supply_num = DM_INST.getSummary().sa_summary.total_supply_num;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    sa_routing_supply_map[routing_layer.get_layer_idx()] = 0;
  }
  sa_total_supply_num = 0;

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [routing_layer_idx, orien_supply_map] : gcell_map[x][y].get_routing_orien_supply_map()) {
        for (auto& [orien, supply] : orien_supply_map) {
          sa_routing_supply_map[routing_layer_idx] += supply;
          sa_total_supply_num += supply;
        }
      }
    }
  }
}

void SupplyAnalyzer::writeSupplyCSV(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string sa_temp_directory_path = DM_INST.getConfig().sa_temp_directory_path;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* supply_csv_file
        = RTUtil::getOutputFileStream(RTUtil::getString(sa_temp_directory_path, "supply_map_", routing_layer.get_layer_name(), ".csv"));
    for (int32_t y = gcell_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
        int32_t total_supply = 0;
        for (auto& [orien, supply] : gcell_map[x][y].get_routing_orien_supply_map()[routing_layer.get_layer_idx()]) {
          total_supply += supply;
        }
        RTUtil::pushStream(supply_csv_file, total_supply, ",");
      }
      RTUtil::pushStream(supply_csv_file, "\n");
    }
    RTUtil::closeFileStream(supply_csv_file);
  }
}

#endif

}  // namespace irt
