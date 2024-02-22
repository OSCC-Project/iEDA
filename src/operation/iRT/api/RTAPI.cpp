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
#include "DataManager.hpp"
#include "DetailedRouter.hpp"
#include "DrcRect.h"
#include "EarlyGlobalRouter.hpp"
#include "GDSPlotter.hpp"
#include "GlobalRouter.hpp"
#include "InitialRouter.hpp"
#include "Monitor.hpp"
#include "PinAccessor.hpp"
#include "SupplyAnalyzer.hpp"
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
  DM_INST.input(config_map, dmInst->get_idb_builder());
  GDSPlotter::initInst();
}

void RTAPI::runRT()
{
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

  PinAccessor::initInst();
  PA_INST.access(net_list);
  PinAccessor::destroyInst();

  SupplyAnalyzer::initInst();
  SA_INST.analyze(net_list);
  SupplyAnalyzer::destroyInst();

  InitialRouter::initInst();
  IR_INST.route(net_list);
  InitialRouter::destroyInst();

  GlobalRouter::initInst();
  GR_INST.route(net_list);
  GlobalRouter::destroyInst();

  TrackAssigner::initInst();
  TA_INST.assign(net_list);
  TrackAssigner::destroyInst();

  DetailedRouter::initInst();
  DR_INST.route(net_list);
  DetailedRouter::destroyInst();
}

void RTAPI::destroyRT()
{
  GDSPlotter::destroyInst();
  DM_INST.output(dmInst->get_idb_builder());
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
                                               std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
                                               std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_result_map)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Helper& helper = DM_INST.getHelper();

  std::vector<Violation> violation_list;
  idrc::DrcApi drc_api;
  drc_api.init();
  for (auto& [type, idrc_violation_list] : drc_api.check(env_shape_list, net_pin_shape_map, net_result_map)) {
    for (idrc::DrcViolation* idrc_violation : idrc_violation_list) {
      if (idrc_violation->get_net_ids().size() < 2) {
        continue;
      }

      idb::IdbLayer* idb_layer = idrc_violation->get_layer();

      EXTLayerRect ext_layer_rect;
      if (idrc_violation->is_rect()) {
        idrc::DrcViolationRect* idrc_violation_rect = static_cast<idrc::DrcViolationRect*>(idrc_violation);
        ext_layer_rect.set_real_lb(idrc_violation_rect->get_llx(), idrc_violation_rect->get_lly());
        ext_layer_rect.set_real_rt(idrc_violation_rect->get_urx(), idrc_violation_rect->get_ury());
      } else {
        LOG_INST.error(Loc::current(), "Type not supported!");
      }
      ext_layer_rect.set_grid_rect(RTUtil::getClosedGCellGridRect(ext_layer_rect.get_real_rect(), gcell_axis));
      ext_layer_rect.set_layer_idx(idb_layer->is_routing() ? helper.getRoutingLayerIdxByName(idb_layer->get_name())
                                                           : helper.getCutLayerIdxByName(idb_layer->get_name()));

      Violation violation;
      violation.set_violation_shape(ext_layer_rect);
      violation.set_is_routing(idb_layer->is_routing());
      violation.set_violation_net_set(idrc_violation->get_net_ids());
      violation_list.push_back(violation);
    }
  }
  return violation_list;
}

// STA

void RTAPI::reportGRTiming()
{
  // #if 1  // 数据结构定义
  //   struct RCPin
  //   {
  //     RCPin() = default;
  //     RCPin(PlanarCoord coord, bool is_real_pin, std::string pin_name)
  //     {
  //       _coord = coord;
  //       _is_real_pin = is_real_pin;
  //       _pin_name = pin_name;
  //     }
  //     RCPin(PlanarCoord coord, bool is_real_pin, irt_int fake_pin_id)
  //     {
  //       _coord = coord;
  //       _is_real_pin = is_real_pin;
  //       _fake_pin_id = fake_pin_id;
  //     }
  //     ~RCPin() = default;

  //     PlanarCoord _coord;
  //     bool _is_real_pin = false;
  //     std::string _pin_name;
  //     irt_int _fake_pin_id;
  //   };
  //   auto getRCSegmentList = [](std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC>& coord_real_pin_map,
  //                              std::vector<Segment<PlanarCoord>>& routing_segment_list) {
  //     std::vector<Segment<RCPin>> rc_segment_list;
  //     // 生成线长为0的线段
  //     for (auto& [coord, real_pin_list] : coord_real_pin_map) {
  //       for (size_t i = 1; i < real_pin_list.size(); i++) {
  //         RCPin first_rc_pin(coord, true, real_pin_list[i - 1]);
  //         RCPin second_rc_pin(coord, true, real_pin_list[i]);
  //         rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
  //       }
  //     }
  //     // 构建coord_fake_pin_map
  //     std::map<PlanarCoord, irt_int, CmpPlanarCoordByXASC> coord_fake_pin_map;
  //     irt_int fake_id = 0;
  //     for (Segment<PlanarCoord>& routing_segment : routing_segment_list) {
  //       PlanarCoord& first_coord = routing_segment.get_first();
  //       PlanarCoord& second_coord = routing_segment.get_second();

  //       if (!RTUtil::exist(coord_real_pin_map, first_coord) && !RTUtil::exist(coord_fake_pin_map, first_coord)) {
  //         coord_fake_pin_map[first_coord] = fake_id++;
  //       }
  //       if (!RTUtil::exist(coord_real_pin_map, second_coord) && !RTUtil::exist(coord_fake_pin_map, second_coord)) {
  //         coord_fake_pin_map[second_coord] = fake_id++;
  //       }
  //     }
  //     // 将routing_segment_list生成rc_segment_list
  //     for (Segment<PlanarCoord>& routing_segment : routing_segment_list) {
  //       PlanarCoord& first_coord = routing_segment.get_first();
  //       PlanarCoord& second_coord = routing_segment.get_second();

  //       RCPin first_rc_pin;
  //       if (RTUtil::exist(coord_real_pin_map, first_coord)) {
  //         first_rc_pin = RCPin(first_coord, true, coord_real_pin_map[first_coord].front());
  //       } else if (RTUtil::exist(coord_fake_pin_map, first_coord)) {
  //         first_rc_pin = RCPin(first_coord, false, coord_fake_pin_map[first_coord]);
  //       } else {
  //         LOG_INST.error(Loc::current(), "The coord is not exist!");
  //       }
  //       RCPin second_rc_pin;
  //       if (RTUtil::exist(coord_real_pin_map, second_coord)) {
  //         second_rc_pin = RCPin(second_coord, true, coord_real_pin_map[second_coord].front());
  //       } else if (RTUtil::exist(coord_fake_pin_map, second_coord)) {
  //         second_rc_pin = RCPin(second_coord, false, coord_fake_pin_map[second_coord]);
  //       } else {
  //         LOG_INST.error(Loc::current(), "The coord is not exist!");
  //       }
  //       rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
  //     }
  //     return rc_segment_list;
  //   };
  // #endif

  // #if 1  // 生成net_coord_real_pin_map和net_segment_map
  //   std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

  //   std::map<irt_int, std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC>> net_key_coord_pin_map;
  //   for (Net& net : net_list) {
  //     for (Pin& pin : net.get_pin_list()) {
  //       auto pin_name = pin.get_pin_name();
  //       pin_name.erase(std::remove(pin_name.begin(), pin_name.end(), '\\'), pin_name.end());
  //       net_key_coord_pin_map[net.get_net_idx()][pin.get_protected_access_point().getGridLayerCoord()].push_back(pin_name);
  //     }
  //   }

  //   std::map<irt_int, std::vector<Segment<PlanarCoord>>> net_segment_map;
  //   for (Net& net : net_list) {
  //     for (auto seg_node : RTUtil::getSegListByTree(net.get_gr_result_tree())) {
  //       net_segment_map[net.get_net_idx()].emplace_back(seg_node.get_first()->value().get_grid_coord().get_planar_coord(),
  //                                                       seg_node.get_second()->value().get_grid_coord().get_planar_coord());
  //     }
  //   }
  //   ///////////////////////////////////////去环
  //   for (auto& [net_id, segment_list] : net_segment_map) {
  //     // layer_segment_list
  //     std::vector<Segment<LayerCoord>> layer_segment_list;
  //     for (Segment<PlanarCoord>& segment : segment_list) {
  //       layer_segment_list.emplace_back(segment.get_first(), segment.get_second());
  //     }
  //     // driving_grid_coord_list
  //     // key_coord_pin_map
  //     std::vector<Pin>& pin_list = net_list[net_id].get_pin_list();
  //     std::vector<LayerCoord> driving_grid_coord_list;
  //     std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  //     for (size_t i = 0; i < pin_list.size(); i++) {
  //       driving_grid_coord_list.push_back(pin_list[i].get_protected_access_point().get_grid_coord());
  //       key_coord_pin_map[pin_list[i].get_protected_access_point().get_grid_coord()].insert(static_cast<irt_int>(i));
  //     }
  //     MTree<LayerCoord> coord_tree = RTUtil::getTreeByFullFlow(driving_grid_coord_list, layer_segment_list, key_coord_pin_map);

  //     segment_list.clear();
  //     for (auto& seg_node : RTUtil::getSegListByTree(coord_tree)) {
  //       segment_list.emplace_back(seg_node.get_first()->value(), seg_node.get_second()->value());
  //     }
  //   }
  // #endif

  // #if 1  // 主流程
  //   ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  //   auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(timing_engine->get_ista());
  //   db_adapter->set_idb(DM_INST.getHelper().get_idb_builder());
  //   db_adapter->convertDBToTimingNetlist();
  //   timing_engine->set_db_adapter(std::move(db_adapter));
  //   timing_engine->set_num_threads(40);
  //   timing_engine->buildGraph();
  //   timing_engine->initRcTree();

  //   ista::Netlist* sta_netlist = timing_engine->get_netlist();
  //   for (Net& net : DM_INST.getDatabase().get_net_list()) {
  //     // coord_real_pin_map
  //     std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC> coord_real_pin_map =
  //     net_key_coord_pin_map[net.get_net_idx()];
  //     // routing_segment_list
  //     std::vector<Segment<PlanarCoord>> routing_segment_list = net_segment_map[net.get_net_idx()];
  //     // 构建RC-tree
  //     auto net_name = net.get_net_name();
  //     net_name.erase(std::remove(net_name.begin(), net_name.end(), '\\'), net_name.end());
  //     ista::Net* ista_net = sta_netlist->findNet(net_name.c_str());
  //     for (Segment<RCPin>& segment : getRCSegmentList(coord_real_pin_map, routing_segment_list)) {
  //       auto getRctNode = [timing_engine, sta_netlist, ista_net](RCPin& rc_pin) {
  //         ista::RctNode* rct_node = nullptr;
  //         if (rc_pin._is_real_pin) {
  //           ista::DesignObject* pin_port = nullptr;
  //           auto pin_port_list = sta_netlist->findPin(rc_pin._pin_name.c_str(), false, false);
  //           if (!pin_port_list.empty()) {
  //             pin_port = pin_port_list.front();
  //           } else {
  //             pin_port = sta_netlist->findPort(rc_pin._pin_name.c_str());
  //           }
  //           rct_node = timing_engine->makeOrFindRCTreeNode(pin_port);
  //         } else {
  //           rct_node = timing_engine->makeOrFindRCTreeNode(ista_net, rc_pin._fake_pin_id);
  //         }
  //         return rct_node;
  //       };
  //       RCPin& first_rc_pin = segment.get_first();
  //       RCPin& second_rc_pin = segment.get_second();

  //       irt_int distance = RTUtil::getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
  //       int32_t unit = dmInst->get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu();
  //       std::optional<double> width = std::nullopt;
  //       double cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getCapacitance(1, distance / 1.0 / unit,
  //       width); double res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getResistance(1, distance / 1.0 /
  //       unit, width);

  //       ista::RctNode* first_node = getRctNode(first_rc_pin);
  //       ista::RctNode* second_node = getRctNode(second_rc_pin);
  //       timing_engine->makeResistor(ista_net, first_node, second_node, res);
  //       timing_engine->incrCap(first_node, cap / 2);
  //       timing_engine->incrCap(second_node, cap / 2);
  //     }
  //     timing_engine->updateRCTreeInfo(ista_net);

  //     // auto* rc_tree = timing_engine->get_ista()->getRcNet(ista_net)->rct();
  //     // rc_tree->printGraphViz();
  //   }
  //   Monitor monitor;
  //   timing_engine->updateTiming();
  //   LOG_INST.info(Loc::current(), "[temp_report] gr sta updateTiming time : ", monitor.getStatsInfo());
  //   timing_engine->reportTiming();

  //   auto clk_list = timing_engine->getClockList();
  //   std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
  //     auto clk_name = clk->get_clock_name();
  //     auto setup_tns = timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
  //     auto setup_wns = timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
  //     auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);

  //     LOG_INST.info(Loc::current(), "[temp_report] gr tns : ", setup_tns);
  //     LOG_INST.info(Loc::current(), "[temp_report] gr wns : ", setup_wns);
  //     LOG_INST.info(Loc::current(), "[temp_report] gr suggest freq : ", suggest_freq);
  //   });

  // #endif
}

void RTAPI::reportDRTiming()
{
  // #if 1  // 数据结构定义
  //   struct RCPin
  //   {
  //     RCPin() = default;
  //     RCPin(PlanarCoord coord, bool is_real_pin, std::string pin_name)
  //     {
  //       _coord = coord;
  //       _is_real_pin = is_real_pin;
  //       _pin_name = pin_name;
  //     }
  //     RCPin(PlanarCoord coord, bool is_real_pin, irt_int fake_pin_id)
  //     {
  //       _coord = coord;
  //       _is_real_pin = is_real_pin;
  //       _fake_pin_id = fake_pin_id;
  //     }
  //     ~RCPin() = default;

  //     PlanarCoord _coord;
  //     bool _is_real_pin = false;
  //     std::string _pin_name;
  //     irt_int _fake_pin_id;
  //   };
  //   auto getRCSegmentList = [](std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC>& coord_real_pin_map,
  //                              std::vector<Segment<PlanarCoord>>& routing_segment_list) {
  //     std::vector<Segment<RCPin>> rc_segment_list;
  //     // 生成线长为0的线段
  //     for (auto& [coord, real_pin_list] : coord_real_pin_map) {
  //       for (size_t i = 1; i < real_pin_list.size(); i++) {
  //         RCPin first_rc_pin(coord, true, real_pin_list[i - 1]);
  //         RCPin second_rc_pin(coord, true, real_pin_list[i]);
  //         rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
  //       }
  //     }
  //     // 构建coord_fake_pin_map
  //     std::map<PlanarCoord, irt_int, CmpPlanarCoordByXASC> coord_fake_pin_map;
  //     irt_int fake_id = 0;
  //     for (Segment<PlanarCoord>& routing_segment : routing_segment_list) {
  //       PlanarCoord& first_coord = routing_segment.get_first();
  //       PlanarCoord& second_coord = routing_segment.get_second();

  //       if (!RTUtil::exist(coord_real_pin_map, first_coord) && !RTUtil::exist(coord_fake_pin_map, first_coord)) {
  //         coord_fake_pin_map[first_coord] = fake_id++;
  //       }
  //       if (!RTUtil::exist(coord_real_pin_map, second_coord) && !RTUtil::exist(coord_fake_pin_map, second_coord)) {
  //         coord_fake_pin_map[second_coord] = fake_id++;
  //       }
  //     }
  //     // 将routing_segment_list生成rc_segment_list
  //     for (Segment<PlanarCoord>& routing_segment : routing_segment_list) {
  //       PlanarCoord& first_coord = routing_segment.get_first();
  //       PlanarCoord& second_coord = routing_segment.get_second();

  //       RCPin first_rc_pin;
  //       if (RTUtil::exist(coord_real_pin_map, first_coord)) {
  //         first_rc_pin = RCPin(first_coord, true, coord_real_pin_map[first_coord].front());
  //       } else if (RTUtil::exist(coord_fake_pin_map, first_coord)) {
  //         first_rc_pin = RCPin(first_coord, false, coord_fake_pin_map[first_coord]);
  //       } else {
  //         LOG_INST.error(Loc::current(), "The coord is not exist!");
  //       }
  //       RCPin second_rc_pin;
  //       if (RTUtil::exist(coord_real_pin_map, second_coord)) {
  //         second_rc_pin = RCPin(second_coord, true, coord_real_pin_map[second_coord].front());
  //       } else if (RTUtil::exist(coord_fake_pin_map, second_coord)) {
  //         second_rc_pin = RCPin(second_coord, false, coord_fake_pin_map[second_coord]);
  //       } else {
  //         LOG_INST.error(Loc::current(), "The coord is not exist!");
  //       }
  //       rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
  //     }
  //     return rc_segment_list;
  //   };
  // #endif

  // #if 1  // 生成net_coord_real_pin_map和net_segment_map
  //   std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();
  //   GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();

  //   std::map<irt_int, std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC>> net_key_coord_pin_map;
  //   for (Net& net : net_list) {
  //     for (Pin& pin : net.get_pin_list()) {
  //       auto pin_name = pin.get_pin_name();
  //       pin_name.erase(std::remove(pin_name.begin(), pin_name.end(), '\\'), pin_name.end());
  //       net_key_coord_pin_map[net.get_net_idx()][pin.get_protected_access_point().getRealLayerCoord()].push_back(pin_name);
  //     }
  //   }

  //   std::map<irt_int, std::vector<Segment<PlanarCoord>>> net_segment_map;
  //   for (irt_int x = 0; x < gcell_map.get_x_size(); x++) {
  //     for (irt_int y = 0; y < gcell_map.get_y_size(); y++) {
  //       for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_result_map()) {
  //         for (Segment<LayerCoord>* segment : segment_set) {
  //           net_segment_map[net_idx].emplace_back(segment->get_first(), segment->get_second());
  //         }
  //       }
  //     }
  //   }
  //   ///////////////////////////////////////去环
  //   for (auto& [net_id, segment_list] : net_segment_map) {
  //     // layer_segment_list
  //     std::vector<Segment<LayerCoord>> layer_segment_list;
  //     for (Segment<PlanarCoord>& segment : segment_list) {
  //       layer_segment_list.emplace_back(segment.get_first(), segment.get_second());
  //     }
  //     // driving_grid_coord_list
  //     // key_coord_pin_map
  //     std::vector<Pin>& pin_list = net_list[net_id].get_pin_list();
  //     std::vector<LayerCoord> driving_grid_coord_list;
  //     std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  //     for (size_t i = 0; i < pin_list.size(); i++) {
  //       driving_grid_coord_list.push_back(pin_list[i].get_protected_access_point().get_real_coord());
  //       key_coord_pin_map[pin_list[i].get_protected_access_point().get_real_coord()].insert(static_cast<irt_int>(i));
  //     }
  //     MTree<LayerCoord> coord_tree = RTUtil::getTreeByFullFlow(driving_grid_coord_list, layer_segment_list, key_coord_pin_map);

  //     segment_list.clear();
  //     for (auto& seg_node : RTUtil::getSegListByTree(coord_tree)) {
  //       segment_list.emplace_back(seg_node.get_first()->value(), seg_node.get_second()->value());
  //     }
  //   }
  // #endif

  // #if 1  // 主流程
  //   ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  //   auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(timing_engine->get_ista());
  //   db_adapter->set_idb(DM_INST.getHelper().get_idb_builder());
  //   db_adapter->convertDBToTimingNetlist();
  //   timing_engine->set_db_adapter(std::move(db_adapter));
  //   timing_engine->set_num_threads(40);
  //   timing_engine->buildGraph();
  //   timing_engine->initRcTree();

  //   ista::Netlist* sta_netlist = timing_engine->get_netlist();
  //   for (Net& net : DM_INST.getDatabase().get_net_list()) {
  //     // coord_real_pin_map
  //     std::map<PlanarCoord, std::vector<std::string>, CmpPlanarCoordByXASC> coord_real_pin_map =
  //     net_key_coord_pin_map[net.get_net_idx()];
  //     // routing_segment_list
  //     std::vector<Segment<PlanarCoord>> routing_segment_list = net_segment_map[net.get_net_idx()];
  //     // 构建RC-tree
  //     auto net_name = net.get_net_name();
  //     net_name.erase(std::remove(net_name.begin(), net_name.end(), '\\'), net_name.end());
  //     ista::Net* ista_net = sta_netlist->findNet(net_name.c_str());
  //     for (Segment<RCPin>& segment : getRCSegmentList(coord_real_pin_map, routing_segment_list)) {
  //       auto getRctNode = [timing_engine, sta_netlist, ista_net](RCPin& rc_pin) {
  //         ista::RctNode* rct_node = nullptr;
  //         if (rc_pin._is_real_pin) {
  //           ista::DesignObject* pin_port = nullptr;
  //           auto pin_port_list = sta_netlist->findPin(rc_pin._pin_name.c_str(), false, false);
  //           if (!pin_port_list.empty()) {
  //             pin_port = pin_port_list.front();
  //           } else {
  //             pin_port = sta_netlist->findPort(rc_pin._pin_name.c_str());
  //           }
  //           rct_node = timing_engine->makeOrFindRCTreeNode(pin_port);
  //         } else {
  //           rct_node = timing_engine->makeOrFindRCTreeNode(ista_net, rc_pin._fake_pin_id);
  //         }
  //         return rct_node;
  //       };
  //       RCPin& first_rc_pin = segment.get_first();
  //       RCPin& second_rc_pin = segment.get_second();

  //       irt_int distance = RTUtil::getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
  //       int32_t unit = dmInst->get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu();
  //       std::optional<double> width = std::nullopt;
  //       double cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getCapacitance(1, distance / 1.0 / unit,
  //       width); double res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())->getResistance(1, distance / 1.0 /
  //       unit, width);

  //       ista::RctNode* first_node = getRctNode(first_rc_pin);
  //       ista::RctNode* second_node = getRctNode(second_rc_pin);
  //       timing_engine->makeResistor(ista_net, first_node, second_node, res);
  //       timing_engine->incrCap(first_node, cap / 2);
  //       timing_engine->incrCap(second_node, cap / 2);
  //     }
  //     timing_engine->updateRCTreeInfo(ista_net);

  //     // auto* rc_tree = timing_engine->get_ista()->getRcNet(ista_net)->rct();
  //     // rc_tree->printGraphViz();
  //   }
  //   Monitor monitor;
  //   timing_engine->updateTiming();
  //   LOG_INST.info(Loc::current(), "[temp_report] dr sta updateTiming time: ", monitor.getStatsInfo());
  //   timing_engine->reportTiming();

  //   auto clk_list = timing_engine->getClockList();
  //   std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
  //     auto clk_name = clk->get_clock_name();
  //     auto setup_tns = timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
  //     auto setup_wns = timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
  //     auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);

  //     LOG_INST.info(Loc::current(), "[temp_report] dr tns : ", setup_tns);
  //     LOG_INST.info(Loc::current(), "[temp_report] dr wns : ", setup_wns);
  //     LOG_INST.info(Loc::current(), "[temp_report] dr suggest freq : ", suggest_freq);
  //   });

  // #endif
}

// other

void RTAPI::clearDef()
{
  idb::IdbBuilder* idb_builder = dmInst->get_idb_builder();
  idb::IdbPins* idb_pin_list = idb_builder->get_def_service()->get_design()->get_io_pin_list();
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

// private

RTAPI* RTAPI::_rt_api_instance = nullptr;

}  // namespace irt
