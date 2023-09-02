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
#include "RTAPI.hpp"

#include "CongTile.hpp"
#include "DRCChecker.hpp"
#include "DataManager.hpp"
#include "DetailedRouter.hpp"
#include "DrcAPI.hpp"
#include "DrcRect.h"
#include "EarlyGlobalRouter.hpp"
#include "GDSPlotter.hpp"
#include "GlobalRouter.hpp"
#include "Monitor.hpp"
#include "PinAccessor.hpp"
#include "RegionQuery.hpp"
#include "ResourceAllocator.hpp"
#include "Stage.hpp"
#include "TimingEval.hpp"
#include "TrackAssigner.hpp"
#include "ViolationRepairer.hpp"
#include "builder.h"
#include "flow_config.h"
#include "icts_fm/file_cts.h"
#include "icts_io.h"
#include "idm.h"

namespace irt {

// public

RTAPI& RTAPI::getInst()
{
  if (_rt_api_instance == nullptr) {
    _rt_api_instance = new RTAPI();
  }
  return *_rt_api_instance;
}

void RTAPI::destroyInst()
{
  if (_rt_api_instance != nullptr) {
    delete _rt_api_instance;
    _rt_api_instance = nullptr;
  }
}

// RT

void RTAPI::initRT(std::map<std::string, std::any> config_map)
{
  Logger::initInst();
  // clang-format off
  LOG_INST.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  LOG_INST.info(Loc::current(), "_____ ________ ________     _______________________ ________ ________  ");
  LOG_INST.info(Loc::current(), "___(_)___  __ \\___  __/     __  ___/___  __/___    |___  __ \\___  __/");
  LOG_INST.info(Loc::current(), "__  / __  /_/ /__  /        _____ \\ __  /   __  /| |__  /_/ /__  /    ");
  LOG_INST.info(Loc::current(), "_  /  _  _, _/ _  /         ____/ / _  /    _  ___ |_  _, _/ _  /      ");
  LOG_INST.info(Loc::current(), "/_/   /_/ |_|  /_/          /____/  /_/     /_/  |_|/_/ |_|  /_/       ");
  LOG_INST.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  LOG_INST.printLogFilePath();
  DataManager::initInst();
  DRCChecker::initInst();
  DM_INST.input(config_map, dmInst->get_idb_builder());
  GDSPlotter::initInst();
}

void RTAPI::runRT(std::vector<Tool> tool_list)
{
  std::set<Stage> stage_set;
  for (Tool tool : tool_list) {
    stage_set.insert(convertToStage(tool));
  }
  std::vector<Stage> stage_list = {Stage::kNone,          Stage::kPinAccessor,    Stage::kResourceAllocator, Stage::kGlobalRouter,
                                   Stage::kTrackAssigner, Stage::kDetailedRouter, Stage::kViolationRepairer, Stage::kNone};
  irt_int stage_idx = 1;
  while (!RTUtil::exist(stage_set, stage_list[stage_idx])) {
    stage_idx++;
  }
  if (stage_list[stage_idx - 1] != Stage::kNone) {
    DM_INST.load(stage_list[stage_idx - 1]);
  }

  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

  while (RTUtil::exist(stage_set, stage_list[stage_idx])) {
    switch (stage_list[stage_idx]) {
      case Stage::kPinAccessor:
        PinAccessor::initInst();
        PA_INST.access(net_list);
        PinAccessor::destroyInst();
        break;
      case Stage::kResourceAllocator:
        ResourceAllocator::initInst();
        RA_INST.allocate(net_list);
        ResourceAllocator::destroyInst();
        break;
      case Stage::kGlobalRouter:
        GlobalRouter::initInst();
        GR_INST.route(net_list);
        GlobalRouter::destroyInst();
        break;
      case Stage::kTrackAssigner:
        TrackAssigner::initInst();
        TA_INST.assign(net_list);
        TrackAssigner::destroyInst();
        break;
      case Stage::kDetailedRouter:
        DetailedRouter::initInst();
        DR_INST.route(net_list);
        DetailedRouter::destroyInst();
        break;
      case Stage::kViolationRepairer:
        ViolationRepairer::initInst();
        VR_INST.repair(net_list);
        ViolationRepairer::destroyInst();
        break;
      default:
        break;
    }
    if (DM_INST.getConfig().enable_output_gds_files == 1) {
      GP_INST.plot(net_list, stage_list[stage_idx], true, false);
    }
    DM_INST.save(stage_list[stage_idx]);
    stage_idx++;
  }
}

Stage RTAPI::convertToStage(Tool tool)
{
  Stage stage = Stage::kNone;
  switch (tool) {
    case Tool::kPinAccessor:
      stage = Stage::kPinAccessor;
      break;
    case Tool::kResourceAllocator:
      stage = Stage::kResourceAllocator;
      break;
    case Tool::kGlobalRouter:
      stage = Stage::kGlobalRouter;
      break;
    case Tool::kTrackAssigner:
      stage = Stage::kTrackAssigner;
      break;
    case Tool::kDetailedRouter:
      stage = Stage::kDetailedRouter;
      break;
    case Tool::kViolationRepairer:
      stage = Stage::kViolationRepairer;
      break;
  }
  return stage;
}

void RTAPI::destroyRT()
{
  GDSPlotter::destroyInst();
  DM_INST.output(dmInst->get_idb_builder());
  DRCChecker::destroyInst();
  DataManager::destroyInst();
  LOG_INST.printLogFilePath();
  // clang-format off
  LOG_INST.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  LOG_INST.info(Loc::current(), "_____ ________ ________     _______________________   ________________________  __  ");
  LOG_INST.info(Loc::current(), "___(_)___  __ \\___  __/     ___  ____/____  _/___  | / /____  _/__  ___/___  / / / ");
  LOG_INST.info(Loc::current(), "__  / __  /_/ /__  /        __  /_     __  /  __   |/ /  __  /  _____ \\ __  /_/ /  ");
  LOG_INST.info(Loc::current(), "_  /  _  _, _/ _  /         _  __/    __/ /   _  /|  /  __/ /   ____/ / _  __  /    ");
  LOG_INST.info(Loc::current(), "/_/   /_/ |_|  /_/          /_/       /___/   /_/ |_/   /___/   /____/  /_/ /_/     ");
  LOG_INST.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  Logger::destroyInst();
}

// EGR

void RTAPI::runEGR(std::map<std::string, std::any> config_map)
{
  Monitor egr_monitor;

  EarlyGlobalRouter::initInst(config_map, dmInst->get_idb_builder());
  EGR_INST.route();
  // EGR_INST.plotCongstLoc();
  EarlyGlobalRouter::destroyInst();

  LOG_INST.info(Loc::current(), "Run EGR completed!", egr_monitor.getStatsInfo());
}

// EVAL

eval::TileGrid* RTAPI::getCongestonMap(std::map<std::string, std::any> config_map)
{
  Monitor egr_monitor;

  EarlyGlobalRouter::initInst(config_map, dmInst->get_idb_builder());
  EGR_INST.route();

  eval::TileGrid* eval_tile_grid = new eval::TileGrid();
  irt_int cell_width = EGR_INST.getDataManager().getConfig().cell_width;
  irt_int cell_height = EGR_INST.getDataManager().getConfig().cell_height;
  Die& die = EGR_INST.getDataManager().getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = EGR_INST.getDataManager().getDatabase().get_routing_layer_list();
  std::vector<GridMap<EGRNode>>& layer_resource_map = EGR_INST.getDataManager().getDatabase().get_layer_resource_map();

  if (layer_resource_map.empty()) {
    LOG_INST.error(Loc::current(), "The size of space resource map is empty!");
  }

  // init eval_tile_grid
  eval_tile_grid->set_lx(die.get_real_lb_x());
  eval_tile_grid->set_ly(die.get_real_lb_y());
  eval_tile_grid->set_tileCnt(die.getXSize(), die.getYSize());
  eval_tile_grid->set_tileSize(cell_width, cell_height);
  eval_tile_grid->set_num_routing_layers(static_cast<int>(layer_resource_map.size()));  // single layer

  // init eval_tiles
  std::vector<eval::Tile*>& eval_tiles = eval_tile_grid->get_tiles();
  for (size_t layer_idx = 0; layer_idx < layer_resource_map.size(); layer_idx++) {
    GridMap<EGRNode>& resource_map = layer_resource_map[layer_idx];
    for (int x = 0; x < resource_map.get_x_size(); x++) {
      for (int y = 0; y < resource_map.get_y_size(); y++) {
        EGRNode& egr_node = resource_map[x][y];
        eval::Tile* tile = new eval::Tile(x, y, egr_node.get_lb_x(), egr_node.get_lb_y(), egr_node.get_rt_x(), egr_node.get_rt_y(),
                                          static_cast<irt_int>(layer_idx));
        tile->set_direction(routing_layer_list[layer_idx].get_direction() == Direction::kHorizontal);

        tile->set_east_cap(static_cast<irt_int>(std::round(egr_node.get_east_supply())));
        tile->set_north_cap(static_cast<irt_int>(std::round(egr_node.get_north_supply())));
        tile->set_south_cap(static_cast<irt_int>(std::round(egr_node.get_south_supply())));
        tile->set_west_cap(static_cast<irt_int>(std::round(egr_node.get_west_supply())));
        tile->set_track_cap(static_cast<irt_int>(std::round(egr_node.get_track_supply())));

        tile->set_east_use(static_cast<irt_int>(std::round(egr_node.get_east_demand())));
        tile->set_north_use(static_cast<irt_int>(std::round(egr_node.get_north_demand())));
        tile->set_south_use(static_cast<irt_int>(std::round(egr_node.get_south_demand())));
        tile->set_west_use(static_cast<irt_int>(std::round(egr_node.get_west_demand())));
        tile->set_track_use(static_cast<irt_int>(std::round(egr_node.get_track_demand())));

        eval_tiles.push_back(tile);
      }
    }
  }
  EarlyGlobalRouter::destroyInst();
  return eval_tile_grid;
}

std::vector<double> RTAPI::getWireLengthAndViaNum(std::map<std::string, std::any> config_map)
{
  std::vector<double> wire_length_via_num;
  EarlyGlobalRouter::initInst(config_map, dmInst->get_idb_builder());
  EGR_INST.route();
  wire_length_via_num.push_back(EGR_INST.getDataManager().getEGRStat().get_total_wire_length());
  wire_length_via_num.push_back(EGR_INST.getDataManager().getEGRStat().get_total_via_num());
  EarlyGlobalRouter::destroyInst();
  return wire_length_via_num;
}

// DRC

void* RTAPI::initRegionQuery()
{
  return idrc::DrcAPIInst.init();
}

void RTAPI::destroyRegionQuery(void* region_query)
{
  if (region_query != nullptr) {
    idrc::DrcAPIInst.destroy(static_cast<idrc::RegionQuery*>(region_query));
    region_query = nullptr;
  }
}

void RTAPI::addEnvRectList(void* region_query, const ids::DRCRect& env_rect)
{
  std::vector<ids::DRCRect> env_rect_list{env_rect};
  addEnvRectList(region_query, env_rect_list);
}

void RTAPI::addEnvRectList(void* region_query, const std::vector<ids::DRCRect>& env_rect_list)
{
  std::vector<idrc::DrcRect*> idrc_env_rect_list;
  for (ids::DRCRect env_rect : env_rect_list) {
    idrc_env_rect_list.push_back(idrc::DrcAPIInst.getDrcRect(env_rect));
  }
  idrc::DrcAPIInst.add(static_cast<idrc::RegionQuery*>(region_query), idrc_env_rect_list);
}

void RTAPI::delEnvRectList(void* region_query, const ids::DRCRect& env_rect)
{
  std::vector<ids::DRCRect> env_rect_list{env_rect};
  delEnvRectList(region_query, env_rect_list);
}

void RTAPI::delEnvRectList(void* region_query, const std::vector<ids::DRCRect>& env_rect_list)
{
  std::vector<idrc::DrcRect*> idrc_env_rect_list;
  for (ids::DRCRect env_rect : env_rect_list) {
    idrc_env_rect_list.push_back(idrc::DrcAPIInst.getDrcRect(env_rect));
  }
  idrc::DrcAPIInst.del(static_cast<idrc::RegionQuery*>(region_query), idrc_env_rect_list);
}

bool RTAPI::hasViolation(void* region_query, const ids::DRCRect& drc_rect)
{
  std::vector<ids::DRCRect> drc_rect_list = {drc_rect};
  return hasViolation(region_query, drc_rect_list);
}

bool RTAPI::hasViolation(void* region_query, const std::vector<ids::DRCRect>& drc_rect_list)
{
  std::vector<idrc::DrcRect*> idrc_drc_rect_list;
  for (const ids::DRCRect& drc_rect : drc_rect_list) {
    idrc_drc_rect_list.push_back(idrc::DrcAPIInst.getDrcRect(drc_rect));
  }
  return idrc::DrcAPIInst.check(static_cast<idrc::RegionQuery*>(region_query), idrc_drc_rect_list);
}

std::map<std::string, int> RTAPI::getViolation(void* region_query, const std::vector<ids::DRCRect>& drc_rect_list)
{
  std::map<std::string, irt_int> violation_name_num_map;
  violation_name_num_map.insert(std::make_pair("Cut EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Enclosure", 0));
  violation_name_num_map.insert(std::make_pair("Metal EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Short", 0));
  violation_name_num_map.insert(std::make_pair("Metal Parallel Run Length Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Notch Spacing", 0));
  violation_name_num_map.insert(std::make_pair("MinStep", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Area", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Corner Fill Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Hole Area", 0));

  addEnvRectList(region_query, drc_rect_list);
  violation_name_num_map = getViolation(region_query);
  delEnvRectList(region_query, drc_rect_list);
  return violation_name_num_map;
}

std::map<std::string, int> RTAPI::getViolation(void* region_query)
{
  std::map<std::string, irt_int> violation_name_num_map;
  violation_name_num_map.insert(std::make_pair("Cut EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Cut Enclosure", 0));
  violation_name_num_map.insert(std::make_pair("Metal EOL Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Short", 0));
  violation_name_num_map.insert(std::make_pair("Metal Parallel Run Length Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Notch Spacing", 0));
  violation_name_num_map.insert(std::make_pair("MinStep", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Area", 0));
  violation_name_num_map.insert(std::make_pair("Cut Diff Layer Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Metal Corner Fill Spacing", 0));
  violation_name_num_map.insert(std::make_pair("Minimal Hole Area", 0));

  for (auto [rule_name, violation_list] : idrc::DrcAPIInst.check(static_cast<idrc::RegionQuery*>(region_query))) {
    violation_name_num_map[rule_name] = static_cast<irt_int>(violation_list.size());
  }
  return violation_name_num_map;
}

std::vector<LayerRect> RTAPI::getMaxScope(const ids::DRCRect& drc_rect)
{
  std::vector<ids::DRCRect> drc_rect_list = {drc_rect};
  return getMaxScope(drc_rect_list);
}

std::vector<LayerRect> RTAPI::getMaxScope(const std::vector<ids::DRCRect>& drc_rect_list)
{
  std::vector<idrc::DrcRect*> drc_rect_ptr_list;
  for (const ids::DRCRect& drc_rect : drc_rect_list) {
    drc_rect_ptr_list.push_back(idrc::DrcAPIInst.getDrcRect(drc_rect));
  }

  std::vector<LayerRect> max_scope_list;
  for (idrc::DrcRect* max_scope : idrc::DrcAPIInst.getMaxScope(drc_rect_ptr_list)) {
    max_scope_list.push_back(convertToLayerRect(idrc::DrcAPIInst.getDrcRect(max_scope)));
  }
  return max_scope_list;
}

std::vector<LayerRect> RTAPI::getMinScope(const ids::DRCRect& drc_rect)
{
  std::vector<ids::DRCRect> drc_rect_list = {drc_rect};
  return getMinScope(drc_rect_list);
}

std::vector<LayerRect> RTAPI::getMinScope(const std::vector<ids::DRCRect>& drc_rect_list)
{
  std::vector<idrc::DrcRect*> drc_rect_ptr_list;
  for (const ids::DRCRect& drc_rect : drc_rect_list) {
    drc_rect_ptr_list.push_back(idrc::DrcAPIInst.getDrcRect(drc_rect));
  }

  std::vector<LayerRect> min_scope_list;
  for (idrc::DrcRect* min_scope : idrc::DrcAPIInst.getMinScope(drc_rect_ptr_list)) {
    min_scope_list.push_back(convertToLayerRect(idrc::DrcAPIInst.getDrcRect(min_scope)));
  }
  return min_scope_list;
}

LayerRect RTAPI::convertToLayerRect(ids::DRCRect ids_rect)
{
  std::map<std::string, irt_int>& routing_layer_name_to_idx_map = DM_INST.getHelper().get_routing_layer_name_to_idx_map();
  std::map<std::string, irt_int>& cut_layer_name_to_idx_map = DM_INST.getHelper().get_cut_layer_name_to_idx_map();

  LayerRect rt_rect;
  rt_rect.set_rect(ids_rect.lb_x, ids_rect.lb_y, ids_rect.rt_x, ids_rect.rt_y);

  if (RTUtil::exist(routing_layer_name_to_idx_map, ids_rect.layer_name)) {
    rt_rect.set_layer_idx(routing_layer_name_to_idx_map[ids_rect.layer_name]);
  } else if (RTUtil::exist(cut_layer_name_to_idx_map, ids_rect.layer_name)) {
    rt_rect.set_layer_idx(routing_layer_name_to_idx_map[ids_rect.layer_name]);
  }

  return rt_rect;
}

ids::DRCRect RTAPI::convertToIDSRect(int net_idx, LayerRect rt_rect, bool is_routing)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  ids::DRCRect ids_rect;

  ids_rect.so_id = net_idx;

  ids_rect.lb_x = rt_rect.get_lb_x();
  ids_rect.lb_y = rt_rect.get_lb_y();
  ids_rect.rt_x = rt_rect.get_rt_x();
  ids_rect.rt_y = rt_rect.get_rt_y();

  if (is_routing) {
    ids_rect.layer_name = routing_layer_list[rt_rect.get_layer_idx()].get_layer_name();
  } else {
    ids_rect.layer_name = cut_layer_list[rt_rect.get_layer_idx()].get_layer_name();
  }

  ids_rect.is_artificial = false;

  return ids_rect;
}

// CTS

std::vector<ids::PHYNode> RTAPI::getPHYNodeList(std::vector<ids::Segment> segment_list)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  Helper& helper = DM_INST.getHelper();

  std::vector<ids::PHYNode> phy_node_list;

  for (ids::Segment segment : segment_list) {
    if (segment.first_x == segment.second_x && segment.first_y == segment.second_y
        && segment.first_layer_name == segment.second_layer_name) {
      continue;
    }

    if (segment.first_layer_name == segment.second_layer_name) {
      // wire
      ids::PHYNode phy_node;
      phy_node.type = ids::PHYNodeType::kWire;
      phy_node.wire.first_x = segment.first_x;
      phy_node.wire.first_y = segment.first_y;
      phy_node.wire.second_x = segment.second_x;
      phy_node.wire.second_y = segment.second_y;
      phy_node.wire.layer_name = segment.first_layer_name;
      phy_node_list.push_back(phy_node);
    } else {
      // via
      irt_int first_layer_idx = helper.getRoutingLayerIdxByName(segment.first_layer_name);
      irt_int second_layer_idx = helper.getRoutingLayerIdxByName(segment.second_layer_name);
      if (first_layer_idx > second_layer_idx) {
        std::swap(first_layer_idx, second_layer_idx);
      }
      for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
        ids::PHYNode phy_node;
        phy_node.type = ids::PHYNodeType::kVia;
        phy_node.via.via_name = layer_via_master_list[layer_idx].front().get_via_name();
        phy_node.via.x = segment.first_x;
        phy_node.via.y = segment.first_y;
        phy_node_list.push_back(phy_node);
      }
    }
  }
  return phy_node_list;
}

// private

RTAPI* RTAPI::_rt_api_instance = nullptr;

}  // namespace irt
