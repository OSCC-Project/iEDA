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
  reduceSupply(sa_model);
  buildIgnoreNet(sa_model);
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
  int32_t supply_reduction = 0;
  double boundary_wire_unit = 1;
  double internal_wire_unit = 1;
  double internal_via_unit = 1;
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
  PlanarRect search_real_rect;
  {
    PlanarRect first_real_rect = RTUTIL.getRealRectByGCell(first_coord, gcell_axis);
    PlanarCoord first_mid_coord = first_real_rect.getMidPoint();
    PlanarRect second_real_rect = RTUTIL.getRealRectByGCell(second_coord, gcell_axis);
    PlanarCoord second_mid_coord = second_real_rect.getMidPoint();
    if (RTUTIL.isHorizontal(first_coord, second_coord)) {
      std::vector<PlanarCoord> coord_list;
      coord_list.emplace_back(first_mid_coord.get_x(), first_real_rect.get_ll_y());
      coord_list.emplace_back(first_mid_coord.get_x(), first_real_rect.get_ur_y());
      coord_list.emplace_back(second_mid_coord.get_x(), second_real_rect.get_ll_y());
      coord_list.emplace_back(second_mid_coord.get_x(), second_real_rect.get_ur_y());
      search_real_rect = RTUTIL.getBoundingBox(coord_list);
    } else if (RTUTIL.isVertical(first_coord, second_coord)) {
      std::vector<PlanarCoord> coord_list;
      coord_list.emplace_back(first_real_rect.get_ll_x(), first_mid_coord.get_y());
      coord_list.emplace_back(first_real_rect.get_ur_x(), first_mid_coord.get_y());
      coord_list.emplace_back(second_real_rect.get_ll_x(), second_mid_coord.get_y());
      coord_list.emplace_back(second_real_rect.get_ur_x(), second_mid_coord.get_y());
      search_real_rect = RTUTIL.getBoundingBox(coord_list);
    }
  }
  EXTLayerRect search_rect;
  search_rect.set_real_rect(search_real_rect);
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

void SupplyAnalyzer::reduceSupply(SAModel& sa_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t supply_reduction = sa_model.get_sa_com_param().get_supply_reduction();

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [routing_layer_idx, orient_supply_map] : gcell_map[x][y].get_routing_orient_supply_map()) {
        for (auto& [orient, supply] : orient_supply_map) {
          supply = std::max(0, supply - supply_reduction);
        }
      }
    }
  }
}

void SupplyAnalyzer::buildIgnoreNet(SAModel& sa_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<int32_t, std::map<int32_t, std::set<Orientation>>> routing_ignore_net_orient_map;
      for (auto& [is_routing, layer_net_fixed_rect_map] : gcell_map[x][y].get_type_layer_net_fixed_rect_map()) {
        if (!is_routing) {
          continue;
        }
        for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
          for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
            if (net_idx == -1) {
              continue;
            }
            for (EXTLayerRect* fixed_rect : fixed_rect_set) {
              if (RTUTIL.isClosedOverlap(gcell_map[x][y], fixed_rect->get_real_rect())) {
                routing_ignore_net_orient_map[layer_idx][net_idx] = {};
              }
            }
          }
        }
      }
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_detailed_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
            if (!net_shape.get_is_routing()) {
              continue;
            }
            if (RTUTIL.isClosedOverlap(gcell_map[x][y], net_shape.get_rect())) {
              routing_ignore_net_orient_map[net_shape.get_layer_idx()][net_idx] = {};
            }
          }
        }
      }
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_detailed_patch_map()) {
        for (EXTLayerRect* patch : patch_set) {
          if (RTUTIL.isClosedOverlap(gcell_map[x][y], patch->get_real_rect())) {
            routing_ignore_net_orient_map[patch->get_layer_idx()][net_idx] = {};
          }
        }
      }
      for (auto& [routing_layer_idx, ignore_net_orient_map] : routing_ignore_net_orient_map) {
        std::set<Orientation> ignore_orient_set;
        ignore_orient_set.insert(Orientation::kAbove);
        ignore_orient_set.insert(Orientation::kBelow);
        if (bottom_routing_layer_idx <= routing_layer_idx && routing_layer_idx <= top_routing_layer_idx) {
          if (routing_layer_list[routing_layer_idx].isPreferH()) {
            ignore_orient_set.insert(Orientation::kWest);
            ignore_orient_set.insert(Orientation::kEast);
          } else {
            ignore_orient_set.insert(Orientation::kSouth);
            ignore_orient_set.insert(Orientation::kNorth);
          }
        }
        for (auto& [net_idx, orient_set] : ignore_net_orient_map) {
          orient_set = ignore_orient_set;
        }
      }
      gcell_map[x][y].set_routing_ignore_net_orient_map(routing_ignore_net_orient_map);
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

  // gcell_map
  {
    GPStruct gcell_map_struct("gcell_map");
    for (RoutingLayer& routing_layer : routing_layer_list) {
      for (int32_t grid_x = 0; grid_x < gcell_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < gcell_map.get_y_size(); grid_y++) {
          GCell& gcell = gcell_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getRealRectByGCell(grid_x, grid_y, gcell_axis);
          int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
          int32_t y = real_rect.get_ur_y();

          y -= y_reduced_span;
          GPText gp_text_node_grid_coord;
          gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", y, " , ", routing_layer.get_layer_idx(), ")"));
          gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
          gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          gcell_map_struct.push(gp_text_node_grid_coord);

          if (RTUTIL.exist(gcell.get_routing_orient_supply_map(), routing_layer.get_layer_idx())) {
            y -= y_reduced_span;
            GPText gp_text_orient_supply_map;
            gp_text_orient_supply_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_supply_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_orient_supply_map.set_message("orient_supply_map: ");
            gp_text_orient_supply_map.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
            gp_text_orient_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
            gcell_map_struct.push(gp_text_orient_supply_map);

            y -= y_reduced_span;
            GPText gp_text_orient_supply_map_info;
            gp_text_orient_supply_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_supply_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_supply_map_info_message = "--";
            for (auto& [orient, supply] : gcell.get_routing_orient_supply_map()[routing_layer.get_layer_idx()]) {
              orient_supply_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", supply, ")");
            }
            gp_text_orient_supply_map_info.set_message(orient_supply_map_info_message);
            gp_text_orient_supply_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
            gp_text_orient_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            gcell_map_struct.push(gp_text_orient_supply_map_info);
          }

          if (RTUTIL.exist(gcell.get_routing_ignore_net_orient_map(), routing_layer.get_layer_idx())) {
            y -= y_reduced_span;
            GPText gp_text_ignore_net_map;
            gp_text_ignore_net_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_ignore_net_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_ignore_net_map.set_message("ignore_net_map: ");
            gp_text_ignore_net_map.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
            gp_text_ignore_net_map.set_presentation(GPTextPresentation::kLeftMiddle);
            gcell_map_struct.push(gp_text_ignore_net_map);

            y -= y_reduced_span;
            GPText gp_text_ignore_net_map_info;
            gp_text_ignore_net_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_ignore_net_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string ignore_net_map_info_message = "--";
            for (auto& [net_idx, orient_set] : gcell.get_routing_ignore_net_orient_map()[routing_layer.get_layer_idx()]) {
              ignore_net_map_info_message += RTUTIL.getString("(", net_idx);
              for (Orientation orient : orient_set) {
                ignore_net_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
              }
              ignore_net_map_info_message += RTUTIL.getString(")");
            }
            gp_text_ignore_net_map_info.set_message(ignore_net_map_info_message);
            gp_text_ignore_net_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
            gp_text_ignore_net_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            gcell_map_struct.push(gp_text_ignore_net_map_info);
          }
        }
      }
    }
    gp_gds.addStruct(gcell_map_struct);
  }

  std::string gds_file_path = RTUTIL.getString(sa_temp_directory_path, "supply.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
