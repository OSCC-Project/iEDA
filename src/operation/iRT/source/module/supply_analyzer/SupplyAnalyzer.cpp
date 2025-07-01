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
#include "SupplyAnalyzer.hpp"

#include "GDSPlotter.hpp"
#include "GPGDS.hpp"
#include "Monitor.hpp"
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
    RTLOG.error(Loc::current(), "The instance not initialized!");
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

void SupplyAnalyzer::analyze()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  SAModel sa_model = initSAModel();
  setSAComParam(sa_model);
  buildSupplySchedule(sa_model);
  analyzeSupply(sa_model);
  replenishPinSupply(sa_model);
  analyzeDemandUnit(sa_model);
  // debugPlotSAModel(sa_model);
  updateSummary(sa_model);
  printSummary(sa_model);
  outputPlanarSupplyCSV(sa_model);
  outputLayerSupplyCSV(sa_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

SupplyAnalyzer* SupplyAnalyzer::_sa_instance = nullptr;

SAModel SupplyAnalyzer::initSAModel()
{
  SAModel sa_model;
  return sa_model;
}

void SupplyAnalyzer::setSAComParam(SAModel& sa_model)
{
  double supply_reduction = 0.2;
  double boundary_wire_unit = 1;
  double internal_wire_unit = 1;
  double internal_via_unit = 0.5;
  /**
   * supply_reduction, boundary_wire_unit, internal_wire_unit, internal_via_unit
   */
  // clang-format off
  SAComParam sa_com_param(supply_reduction, boundary_wire_unit, internal_wire_unit, internal_via_unit);
  // clang-format on
  RTLOG.info(Loc::current(), "supply_reduction: ", sa_com_param.get_supply_reduction());
  RTLOG.info(Loc::current(), "boundary_wire_unit: ", sa_com_param.get_boundary_wire_unit());
  RTLOG.info(Loc::current(), "internal_wire_unit: ", sa_com_param.get_internal_wire_unit());
  RTLOG.info(Loc::current(), "internal_via_unit: ", sa_com_param.get_internal_via_unit());
  sa_model.set_sa_com_param(sa_com_param);
}

void SupplyAnalyzer::buildSupplySchedule(SAModel& sa_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    if (routing_layer.get_layer_idx() < bottom_routing_layer_idx || top_routing_layer_idx < routing_layer.get_layer_idx()) {
      continue;
    }
    if (routing_layer.isPreferH()) {
      for (int32_t begin_x = 1; begin_x <= 2; begin_x++) {
        std::vector<std::pair<LayerCoord, LayerCoord>> grid_pair_list;
        for (int32_t y = 0; y < die.getYSize(); y++) {
          for (int32_t x = begin_x; x < die.getXSize(); x += 2) {
            grid_pair_list.emplace_back(LayerCoord(x - 1, y, routing_layer.get_layer_idx()), LayerCoord(x, y, routing_layer.get_layer_idx()));
          }
        }
        sa_model.get_grid_pair_list_list().push_back(grid_pair_list);
      }
    } else {
      for (int32_t begin_y = 1; begin_y <= 2; begin_y++) {
        std::vector<std::pair<LayerCoord, LayerCoord>> grid_pair_list;
        for (int32_t x = 0; x < die.getXSize(); x++) {
          for (int32_t y = begin_y; y < die.getYSize(); y += 2) {
            grid_pair_list.emplace_back(LayerCoord(x, y - 1, routing_layer.get_layer_idx()), LayerCoord(x, y, routing_layer.get_layer_idx()));
          }
        }
        sa_model.get_grid_pair_list_list().push_back(grid_pair_list);
      }
    }
  }
}

void SupplyAnalyzer::analyzeSupply(SAModel& sa_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  double supply_reduction = sa_model.get_sa_com_param().get_supply_reduction();

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
      EXTLayerRect search_rect = getSearchRect(first_coord, second_coord);

      std::map<Orientation, int32_t>& first_orient_supply_map
          = gcell_map[first_coord.get_x()][first_coord.get_y()].get_routing_orient_supply_map()[search_rect.get_layer_idx()];
      std::map<Orientation, int32_t>& second_orient_supply_map
          = gcell_map[second_coord.get_x()][second_coord.get_y()].get_routing_orient_supply_map()[search_rect.get_layer_idx()];

      Orientation first_orientation = RTUTIL.getOrientation(first_coord, second_coord);
      Orientation second_orientation = RTUTIL.getOppositeOrientation(first_orientation);

      std::vector<PlanarRect> obs_rect_list;
      {
        for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(search_rect)) {
          if (!is_routing) {
            continue;
          }
          for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
            if (search_rect.get_layer_idx() != layer_idx) {
              continue;
            }
            for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
              for (EXTLayerRect* fixed_rect : fixed_rect_set) {
                obs_rect_list.push_back(fixed_rect->get_real_rect());
              }
            }
          }
        }
        for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(search_rect)) {
          for (Segment<LayerCoord>* segment : segment_set) {
            for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
              if (!net_shape.get_is_routing()) {
                continue;
              }
              if (search_rect.get_layer_idx() != net_shape.get_layer_idx()) {
                continue;
              }
              obs_rect_list.push_back(net_shape);
            }
          }
        }
        for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(search_rect)) {
          for (EXTLayerRect* patch : patch_set) {
            if (search_rect.get_layer_idx() != patch->get_layer_idx()) {
              continue;
            }
            obs_rect_list.push_back(patch->get_real_rect());
          }
        }
      }
      std::vector<LayerRect> wire_list = getCrossingWireList(search_rect);
      for (LayerRect& wire : wire_list) {
        if (isAccess(wire, obs_rect_list)) {
          first_orient_supply_map[first_orientation]++;
          second_orient_supply_map[second_orientation]++;
        }
      }
      int32_t max_supply = std::max(0, static_cast<int32_t>(std::round(static_cast<double>(wire_list.size()) * (1 - supply_reduction))));
      for (auto& [orient, supply] : first_orient_supply_map) {
        supply = std::min(supply, max_supply);
      }
      for (auto& [orient, supply] : second_orient_supply_map) {
        supply = std::min(supply, max_supply);
      }
    }
    analyzed_pair_num += grid_pair_list.size();
    RTLOG.info(Loc::current(), "Analyzed ", analyzed_pair_num, "/", total_pair_num, "(", RTUTIL.getPercentage(analyzed_pair_num, total_pair_num),
               ") grid pairs", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

EXTLayerRect SupplyAnalyzer::getSearchRect(LayerCoord& first_coord, LayerCoord& second_coord)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  if (first_coord.get_layer_idx() != second_coord.get_layer_idx()) {
    RTLOG.error(Loc::current(), "The grid_pair layer_idx is not equal!");
  }
  EXTLayerRect search_rect;
  search_rect.set_real_rect(RTUTIL.getBoundingBox({RTUTIL.getRealRectByGCell(first_coord, gcell_axis), RTUTIL.getRealRectByGCell(second_coord, gcell_axis)}));
  search_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(search_rect.get_real_rect(), gcell_axis));
  search_rect.set_layer_idx(first_coord.get_layer_idx());
  return search_rect;
}

std::vector<LayerRect> SupplyAnalyzer::getCrossingWireList(EXTLayerRect& search_rect)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  RoutingLayer& routing_layer = routing_layer_list[search_rect.get_layer_idx()];
  int32_t half_wire_width = routing_layer.get_min_width() / 2;

  int32_t real_ll_x = search_rect.get_real_ll_x();
  int32_t real_ll_y = search_rect.get_real_ll_y();
  int32_t real_ur_x = search_rect.get_real_ur_x();
  int32_t real_ur_y = search_rect.get_real_ur_y();

  std::vector<LayerRect> wire_list;
  if (routing_layer.isPreferH()) {
    for (int32_t y : RTUTIL.getScaleList(real_ll_y, real_ur_y, routing_layer.getYTrackGridList())) {
      wire_list.emplace_back(real_ll_x, y - half_wire_width, real_ur_x, y + half_wire_width, search_rect.get_layer_idx());
    }
  } else {
    for (int32_t x : RTUTIL.getScaleList(real_ll_x, real_ur_x, routing_layer.getXTrackGridList())) {
      wire_list.emplace_back(x - half_wire_width, real_ll_y, x + half_wire_width, real_ur_y, search_rect.get_layer_idx());
    }
  }
  return wire_list;
}

bool SupplyAnalyzer::isAccess(LayerRect& wire, std::vector<PlanarRect>& obs_rect_list)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[wire.get_layer_idx()];

  for (PlanarRect& obs_rect : obs_rect_list) {
    int32_t enlarged_size = routing_layer.getPRLSpacing(obs_rect);
    PlanarRect enlarged_rect = RTUTIL.getEnlargedRect(obs_rect, enlarged_size);
    if (RTUTIL.isOpenOverlap(enlarged_rect, wire)) {
      // 阻塞
      return false;
    }
  }
  return true;
}

void SupplyAnalyzer::replenishPinSupply(SAModel& sa_model)
{
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::map<PlanarCoord, std::map<int32_t, std::set<int32_t>>, CmpPlanarCoordByXASC> coord_routing_net_map;
  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      coord_routing_net_map[access_point->get_grid_coord()][access_point->get_layer_idx()].insert(net_idx);
    }
  }
  for (auto& [grid_coord, routing_net_map] : coord_routing_net_map) {
    std::map<int32_t, int32_t> routing_min_supply_map;
    {
      std::set<int32_t> bottom_net_idx_set;
      for (int32_t routing_layer_idx = 0; routing_layer_idx <= bottom_routing_layer_idx; routing_layer_idx++) {
        if (RTUTIL.exist(routing_net_map, routing_layer_idx)) {
          bottom_net_idx_set.insert(routing_net_map[routing_layer_idx].begin(), routing_net_map[routing_layer_idx].end());
        }
        routing_min_supply_map[routing_layer_idx] = static_cast<int32_t>(bottom_net_idx_set.size());
      }
      std::set<int32_t> top_net_idx_set;
      for (int32_t routing_layer_idx = static_cast<int32_t>(routing_layer_list.size()) - 1; top_routing_layer_idx <= routing_layer_idx; routing_layer_idx--) {
        if (RTUTIL.exist(routing_net_map, routing_layer_idx)) {
          top_net_idx_set.insert(routing_net_map[routing_layer_idx].begin(), routing_net_map[routing_layer_idx].end());
        }
        routing_min_supply_map[routing_layer_idx] = static_cast<int32_t>(top_net_idx_set.size());
      }
      for (int32_t routing_layer_idx = bottom_routing_layer_idx + 1; routing_layer_idx < top_routing_layer_idx; routing_layer_idx++) {
        if (RTUTIL.exist(routing_net_map, routing_layer_idx)) {
          routing_min_supply_map[routing_layer_idx] = static_cast<int32_t>(routing_net_map[routing_layer_idx].size());
        }
      }
    }
    GCell& gcell = gcell_map[grid_coord.get_x()][grid_coord.get_y()];
    for (auto& [routing_layer_idx, min_supply] : routing_min_supply_map) {
      if (min_supply <= 0) {
        continue;
      }
      if (routing_layer_list[routing_layer_idx].isPreferH()) {
        for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
          int32_t& supply = gcell.get_routing_orient_supply_map()[routing_layer_idx][orientation];
          supply = std::max(supply, min_supply);
        }
      } else {
        for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
          int32_t& supply = gcell.get_routing_orient_supply_map()[routing_layer_idx][orientation];
          supply = std::max(supply, min_supply);
        }
      }
    }
  }
}

void SupplyAnalyzer::analyzeDemandUnit(SAModel& sa_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  double boundary_wire_unit = sa_model.get_sa_com_param().get_boundary_wire_unit();
  double internal_wire_unit = sa_model.get_sa_com_param().get_internal_wire_unit();
  double internal_via_unit = sa_model.get_sa_com_param().get_internal_via_unit();

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      GCell& gcell = gcell_map[x][y];
      gcell.set_boundary_wire_unit(boundary_wire_unit);
      gcell.set_internal_wire_unit(internal_wire_unit);
      gcell.set_internal_via_unit(internal_via_unit);
    }
  }
}

#if 1  // exhibit

void SupplyAnalyzer::updateSummary(SAModel& sa_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, int32_t>& routing_supply_map = summary.sa_summary.routing_supply_map;
  int32_t& total_supply = summary.sa_summary.total_supply;

  routing_supply_map.clear();
  total_supply = 0;

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [routing_layer_idx, orient_supply_map] : gcell_map[x][y].get_routing_orient_supply_map()) {
        for (auto& [orient, supply] : orient_supply_map) {
          routing_supply_map[routing_layer_idx] += supply;
          total_supply += supply;
        }
      }
    }
  }
}

void SupplyAnalyzer::printSummary(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, int32_t>& routing_supply_map = summary.sa_summary.routing_supply_map;
  int32_t& total_supply = summary.sa_summary.total_supply;

  fort::char_table routing_supply_map_table;
  {
    routing_supply_map_table.set_cell_text_align(fort::text_align::right);
    routing_supply_map_table << fort::header << "routing"
                             << "supply"
                             << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_supply_map_table << routing_layer.get_layer_name() << routing_supply_map[routing_layer.get_layer_idx()]
                               << RTUTIL.getPercentage(routing_supply_map[routing_layer.get_layer_idx()], total_supply) << fort::endr;
    }
    routing_supply_map_table << fort::header << "Total" << total_supply << RTUTIL.getPercentage(total_supply, total_supply) << fort::endr;
  }
  RTUTIL.printTableList({routing_supply_map_table});
}

void SupplyAnalyzer::outputPlanarSupplyCSV(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& sa_temp_directory_path = RTDM.getConfig().sa_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::ofstream* supply_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(sa_temp_directory_path, "supply_map_planar.csv"));
  for (int32_t y = gcell_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      int32_t total_supply = 0;
      for (RoutingLayer& routing_layer : routing_layer_list) {
        for (auto& [orient, supply] : gcell_map[x][y].get_routing_orient_supply_map()[routing_layer.get_layer_idx()]) {
          total_supply += supply;
        }
      }
      RTUTIL.pushStream(supply_csv_file, total_supply, ",");
    }
    RTUTIL.pushStream(supply_csv_file, "\n");
  }
  RTUTIL.closeFileStream(supply_csv_file);
}

void SupplyAnalyzer::outputLayerSupplyCSV(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& sa_temp_directory_path = RTDM.getConfig().sa_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* supply_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(sa_temp_directory_path, "supply_map_", routing_layer.get_layer_name(), ".csv"));
    for (int32_t y = gcell_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
        int32_t total_supply = 0;
        for (auto& [orient, supply] : gcell_map[x][y].get_routing_orient_supply_map()[routing_layer.get_layer_idx()]) {
          total_supply += supply;
        }
        RTUTIL.pushStream(supply_csv_file, total_supply, ",");
      }
      RTUTIL.pushStream(supply_csv_file, "\n");
    }
    RTUTIL.closeFileStream(supply_csv_file);
  }
}

#endif

#if 1  // debug

void SupplyAnalyzer::debugPlotSAModel(SAModel& sa_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& sa_temp_directory_path = RTDM.getConfig().sa_temp_directory_path;

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

  // routing result
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    GPStruct detailed_result_struct(RTUTIL.getString("detailed_result(net_", net_idx, ")"));
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
        detailed_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(detailed_result_struct);
  }

  // routing patch
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    GPStruct detailed_patch_struct(RTUTIL.getString("detailed_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      detailed_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(detailed_patch_struct);
  }

  // supply_map
  {
    GPStruct supply_map_struct("supply_map");
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
        PlanarRect shape = RTUTIL.getRealRectByGCell(x, y, gcell_axis);
        for (auto& [layer_idx, orient_supply_map] : gcell_map[x][y].get_routing_orient_supply_map()) {
          int32_t y_reduced_span = shape.getYSpan() / 25;
          int32_t y = shape.get_ur_y();

          if (!orient_supply_map.empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_supply_map_info;
            gp_text_orient_supply_map_info.set_coord(shape.get_ll_x(), y);
            gp_text_orient_supply_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_supply_map_message = "--";
            for (auto& [orientation, supply] : orient_supply_map) {
              orient_supply_map_message += RTUTIL.getString("(", GetOrientationName()(orientation), ":", supply, ")");
            }
            gp_text_orient_supply_map_info.set_message(orient_supply_map_message);
            gp_text_orient_supply_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(static_cast<int32_t>(layer_idx)));
            gp_text_orient_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            supply_map_struct.push(gp_text_orient_supply_map_info);
          }
        }
      }
    }
    gp_gds.addStruct(supply_map_struct);
  }

  std::string gds_file_path = RTUTIL.getString(sa_temp_directory_path, "supply.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
