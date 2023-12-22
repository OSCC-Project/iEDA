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
#include "builder.h"
#include "flow_config.h"
#include "icts_fm/file_cts.h"
#include "icts_io.h"
#include "idm.h"
#include "idrc_api.h"

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
  Monitor monitor;

  std::set<Stage> stage_set;
  for (Tool tool : tool_list) {
    stage_set.insert(convertToStage(tool));
  }
  std::vector<Stage> stage_list
      = {Stage::kNone,           Stage::kPinAccessor, Stage::kResourceAllocator, Stage::kGlobalRouter, Stage::kTrackAssigner,
         Stage::kDetailedRouter, Stage::kNone};
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
        // ResourceAllocator::initInst();
        // RA_INST.allocate(net_list);
        // ResourceAllocator::destroyInst();
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
      default:
        break;
    }
    if (DM_INST.getConfig().enable_output_gds_files == 1) {
      GP_INST.plot(net_list, stage_list[stage_idx], true, false);
    }
    DM_INST.save(stage_list[stage_idx]);
    stage_idx++;
  }

  LOG_INST.info(Loc::current(), "The RT completed!", monitor.getStatsInfo());
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
        tile->set_direction(routing_layer_list[layer_idx].get_prefer_direction() == Direction::kHorizontal);

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

std::vector<Violation> RTAPI::getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                               std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_idb_segment_map)
{
  std::vector<Violation> violation_list;
  return violation_list;
}

// STA

void RTAPI::reportGRTiming()
{
}

void RTAPI::reportDRTiming()
{
  // //////////////////////////////////////////////////////////////
  // //////////////////////////////////////////////////////////////
  // struct RCPin
  // {
  //   RCPin() = default;
  //   RCPin(PlanarCoord coord, bool is_real_pin, std::string pin_name)
  //   {
  //     _coord = coord;
  //     _is_real_pin = is_real_pin;
  //     _pin_name = pin_name;
  //   }
  //   RCPin(PlanarCoord coord, bool is_real_pin, irt_int fake_pin_id)
  //   {
  //     _coord = coord;
  //     _is_real_pin = is_real_pin;
  //     _fake_pin_id = fake_pin_id;
  //   }
  //   ~RCPin() = default;

  //   PlanarCoord _coord;
  //   bool _is_real_pin = false;
  //   std::string _pin_name;
  //   irt_int _fake_pin_id;
  // };
  // auto getRCSegmentList = [](std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC>& coord_real_pin_map,
  //                            std::vector<Segment<PlanarCoord>>& routing_segment_list) {
  //   std::vector<Segment<RCPin>> rc_segment_list;
  //   // 生成线长为0的线段
  //   for (auto& [coord, real_pin_list] : coord_real_pin_map) {
  //     for (size_t i = 1; i < real_pin_list.size(); i++) {
  //       RCPin first_rc_pin(coord, true, real_pin_list[i - 1]);
  //       RCPin second_rc_pin(coord, true, real_pin_list[i]);
  //       rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
  //     }
  //   }
  //   // 构建coord_fake_pin_map
  //   std::map<PlanarCoord, irt_int, CmpPlanarCoordByXASC> coord_fake_pin_map;
  //   irt_int fake_id = 0;
  //   for (Segment<PlanarCoord>& routing_segment : routing_segment_list) {
  //     PlanarCoord& first_coord = routing_segment.get_first();
  //     PlanarCoord& second_coord = routing_segment.get_second();

  //     if (!RTUtil::exist(coord_real_pin_map, first_coord) && !RTUtil::exist(coord_fake_pin_map, first_coord)) {
  //       coord_fake_pin_map[first_coord] = fake_id++;
  //     }
  //     if (!RTUtil::exist(coord_real_pin_map, second_coord) && !RTUtil::exist(coord_fake_pin_map, second_coord)) {
  //       coord_fake_pin_map[second_coord] = fake_id++;
  //     }
  //   }
  //   // 将routing_segment_list生成rc_segment_list
  //   for (Segment<PlanarCoord>& routing_segment : routing_segment_list) {
  //     PlanarCoord& first_coord = routing_segment.get_first();
  //     PlanarCoord& second_coord = routing_segment.get_second();

  //     RCPin first_rc_pin;
  //     if (RTUtil::exist(coord_real_pin_map, first_coord)) {
  //       first_rc_pin = RCPin(first_coord, true, coord_real_pin_map[first_coord].front());
  //     } else if (RTUtil::exist(coord_fake_pin_map, first_coord)) {
  //       first_rc_pin = RCPin(first_coord, false, coord_fake_pin_map[first_coord]);
  //     } else {
  //       LOG_INST.error(Loc::current(), "The coord is not exist!");
  //     }
  //     RCPin second_rc_pin;
  //     if (RTUtil::exist(coord_real_pin_map, second_coord)) {
  //       second_rc_pin = RCPin(second_coord, true, coord_real_pin_map[second_coord].front());
  //     } else if (RTUtil::exist(coord_fake_pin_map, second_coord)) {
  //       second_rc_pin = RCPin(second_coord, false, coord_fake_pin_map[second_coord]);
  //     } else {
  //       LOG_INST.error(Loc::current(), "The coord is not exist!");
  //     }
  //     rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
  //   }
  //   return rc_segment_list;
  // };
  // /////////////////////////////////////////////////////////////
  // /////////////////////////////////////////////////////////////
  // ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  // timing_engine->set_num_threads(40);
  // timing_engine->buildGraph();
  // timing_engine->initRcTree();

  // ista::Netlist* sta_netlist = timing_engine->get_netlist();

  // for (Net& net : DM_INST.getDatabase().get_net_list()) {
  //   // coord_real_pin_map
  //   std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC> coord_real_pin_map;
  //   for (Pin& pin : net.get_pin_list()) {
  //     coord_real_pin_map[pin.get_protected_access_point().getRealLayerCoord()].push_back(pin.get_pin_name());
  //   }
  //   // routing_segment_list
  //   std::vector<Segment<PlanarCoord>> routing_segment_list;
  //   for (irt::TNode<irt::PHYNode>* phy_node_node : RTUtil::getNodeList(net.get_vr_result_tree())) {
  //     PHYNode& phy_node = phy_node_node->value();
  //     if (phy_node.isType<PinNode>() || phy_node.isType<ViaNode>() || phy_node.isType<PatchNode>()) {
  //       continue;
  //     }
  //     if (phy_node.isType<WireNode>()) {
  //       WireNode& wire_node = phy_node.getNode<WireNode>();
  //       routing_segment_list.emplace_back(wire_node.get_first(), wire_node.get_second());
  //     } else {
  //       LOG_INST.error(Loc::current(), "The phy node is incorrect type!");
  //     }
  //   }
  //   // 构建RC-tree
  //   ista::Net* ista_net = sta_netlist->findNet(net.get_net_name().c_str());
  //   for (Segment<RCPin>& segment : getRCSegmentList(coord_real_pin_map, routing_segment_list)) {
  //     auto getRctNode = [timing_engine, sta_netlist, ista_net](RCPin& rc_pin) {
  //       ista::RctNode* rct_node = nullptr;
  //       if (rc_pin._is_real_pin) {
  //         ista::DesignObject* pin_port = nullptr;
  //         auto pin_port_list = sta_netlist->findPin(rc_pin._pin_name.c_str(), false, false);
  //         if (!pin_port_list.empty()) {
  //           pin_port = pin_port_list.front();
  //         } else {
  //           pin_port = sta_netlist->findPort(rc_pin._pin_name.c_str());
  //         }
  //         rct_node = timing_engine->makeOrFindRCTreeNode(pin_port);
  //       } else {
  //         rct_node = timing_engine->makeOrFindRCTreeNode(ista_net, rc_pin._fake_pin_id);
  //       }
  //       return rct_node;
  //     };
  //     RCPin& first_rc_pin = segment.get_first();
  //     RCPin& second_rc_pin = segment.get_second();

  //     irt_int distance = RTUtil::getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
  //     int32_t unit = dmInst->get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu();
  //     std::optional<double> width = std::nullopt;
  //     double cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getCapacitance(1, distance / 1.0 / unit,
  //     width); double res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getResistance(1, distance / 1.0 /
  //     unit, width);

  //     ista::RctNode* first_node = getRctNode(first_rc_pin);
  //     ista::RctNode* second_node = getRctNode(second_rc_pin);
  //     timing_engine->makeResistor(ista_net, first_node, second_node, res);
  //     timing_engine->incrCap(first_node, cap / 2);
  //     timing_engine->incrCap(second_node, cap / 2);
  //   }
  //   timing_engine->updateRCTreeInfo(ista_net);
  // }
  // timing_engine->updateTiming();
  // timing_engine->reportTiming();
}

// other

void RTAPI::runOther()
{
  idb::IdbBuilder* idb_builder = dmInst->get_idb_builder();

  IdbNetList* idb_net_list = idb_builder->get_def_service()->get_design()->get_net_list();

  //////////////////////////////////////////
  // 删除net内所有的virtual
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    for (idb::IdbRegularWire* wire : idb_net->get_wire_list()->get_wire_list()) {
      std::vector<idb::IdbRegularWireSegment*>& segment_list = wire->get_segment_list();
      std::sort(segment_list.begin(), segment_list.end(), [](idb::IdbRegularWireSegment* a, idb::IdbRegularWireSegment* b) {
        bool a_is_virtual = a->is_virtual(a->get_point_second());
        bool b_is_virtual = b->is_virtual(b->get_point_second());
        if (a_is_virtual == false && b_is_virtual == true) {
          return true;
        }
        return false;
      });
    }
  }
  // 删除net内所有的virtual
  //////////////////////////////////////////
}

#if 0

void RTAPI::runOther()
{
  idb::IdbBuilder* idb_builder = dmInst->get_idb_builder();

  idb::IdbPins* idb_pin_list = idb_builder->get_def_service()->get_design()->get_io_pin_list();
  IdbNetList* idb_net_list = idb_builder->get_def_service()->get_design()->get_net_list();

  //////////////////////////////////////////
  // 删除net内所有的wire
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    idb_net->clear_wire_list();
  }
  // 删除net内所有的wire
  //////////////////////////////////////////

  //////////////////////////////////////////
  // 删除net: 虚拟的io_pin与io_cell连接的PAD
  std::vector<std::string> remove_net_list;
  for (idb::IdbNet* idb_net : idb_net_list->get_net_list()) {
    bool has_io_pin = false;
    if (idb_net->get_io_pin() != nullptr) {
      has_io_pin = true;
    }
    bool has_io_cell = false;
    for (idb::IdbInstance* instance : idb_net->get_instance_list()->get_instance_list()) {
      if (instance->get_cell_master()->is_pad()) {
        has_io_cell = true;
        break;
      }
    }
    if (has_io_pin && has_io_cell) {
      LOG_INST.info(Loc::current(), "The net '", idb_net->get_net_name(), "' connects PAD and io_pin! removing...");
      remove_net_list.push_back(idb_net->get_net_name());
    }
  }
  for (std::string remove_net : remove_net_list) {
    idb_net_list->remove_net(remove_net);
  }
  // 删除net: 虚拟的io_pin与io_cell连接的PAD
  //////////////////////////////////////////

  //////////////////////////////////////////
  // 删除虚空的io_pin
  std::vector<idb::IdbPin*> remove_pin_list;
  for (idb::IdbPin* io_pin : idb_pin_list->get_pin_list()) {
    if (io_pin->get_port_box_list().empty()) {
      std::cout << io_pin->get_pin_name() << std::endl;
      remove_pin_list.push_back(io_pin);
    }
  }
  for (idb::IdbPin* io_pin : remove_pin_list) {
    idb_pin_list->remove_pin(io_pin);
  }
  // 删除虚空的io_pin
  //////////////////////////////////////////
}

#endif

// private

RTAPI* RTAPI::_rt_api_instance = nullptr;

}  // namespace irt
