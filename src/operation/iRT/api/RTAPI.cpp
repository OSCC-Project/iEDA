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

#if 1  // 外部调用RT的API

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
  DM_INST.prepare(config_map, dmInst->get_idb_builder());
  GDSPlotter::initInst();
}

void RTAPI::runEGR()
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Starting...");

  PinAccessor::initInst();
  PA_INST.access();
  PinAccessor::destroyInst();

  SupplyAnalyzer::initInst();
  SA_INST.analyze();
  SupplyAnalyzer::destroyInst();

  InitialRouter::initInst();
  IR_INST.route();
  InitialRouter::destroyInst();

  LOG_INST.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTAPI::runRT()
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Starting...");

  PinAccessor::initInst();
  PA_INST.access();
  PinAccessor::destroyInst();

  SupplyAnalyzer::initInst();
  SA_INST.analyze();
  SupplyAnalyzer::destroyInst();

  InitialRouter::initInst();
  IR_INST.route();
  InitialRouter::destroyInst();

  GlobalRouter::initInst();
  GR_INST.route();
  GlobalRouter::destroyInst();

  TrackAssigner::initInst();
  TA_INST.assign();
  TrackAssigner::destroyInst();

  DetailedRouter::initInst();
  DR_INST.route();
  DetailedRouter::destroyInst();

  LOG_INST.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTAPI::destroyRT()
{
  GDSPlotter::destroyInst();
  DM_INST.clean();
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
    if (idb_net->get_io_pins() != nullptr) {
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

eval::TileGrid* RTAPI::getCongestionMap(std::map<std::string, std::any> config_map, double& wire_length)
{
  Monitor egr_monitor;

  EarlyGlobalRouter::initInst(config_map, dmInst->get_idb_builder());
  EGR_INST.route();
  wire_length = EGR_INST.getDataManager().getEGRStat().get_total_wire_length();

  eval::TileGrid* eval_tile_grid = new eval::TileGrid();
  int32_t cell_width = EGR_INST.getDataManager().getConfig().cell_width;
  int32_t cell_height = EGR_INST.getDataManager().getConfig().cell_height;
  Die& die = EGR_INST.getDataManager().getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = EGR_INST.getDataManager().getDatabase().get_routing_layer_list();
  std::vector<GridMap<EGRNode>>& layer_resource_map = EGR_INST.getDataManager().getDatabase().get_layer_resource_map();

  if (layer_resource_map.empty()) {
    LOG_INST.error(Loc::current(), "The size of space resource map is empty!");
  }

  // init eval_tile_grid
  eval_tile_grid->set_lx(die.get_real_ll_x());
  eval_tile_grid->set_ly(die.get_real_ll_y());
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
        eval::Tile* tile = new eval::Tile(x, y, egr_node.get_ll_x(), egr_node.get_ll_y(), egr_node.get_ur_x(), egr_node.get_ur_y(),
                                          static_cast<int32_t>(layer_idx));
        tile->set_direction(routing_layer_list[layer_idx].get_prefer_direction() == Direction::kHorizontal);

        tile->set_east_cap(static_cast<int32_t>(std::round(egr_node.get_east_supply())));
        tile->set_north_cap(static_cast<int32_t>(std::round(egr_node.get_north_supply())));
        tile->set_south_cap(static_cast<int32_t>(std::round(egr_node.get_south_supply())));
        tile->set_west_cap(static_cast<int32_t>(std::round(egr_node.get_west_supply())));
        tile->set_track_cap(static_cast<int32_t>(std::round(egr_node.get_track_supply())));

        tile->set_east_use(static_cast<int32_t>(std::round(egr_node.get_east_demand())));
        tile->set_north_use(static_cast<int32_t>(std::round(egr_node.get_north_demand())));
        tile->set_south_use(static_cast<int32_t>(std::round(egr_node.get_south_demand())));
        tile->set_west_use(static_cast<int32_t>(std::round(egr_node.get_west_demand())));
        tile->set_track_use(static_cast<int32_t>(std::round(egr_node.get_track_demand())));

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

#endif

#if 1  // RT调用外部的API

std::vector<Violation> RTAPI::getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                               std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
                                               std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_wire_via_map)
{
  /**
   * env_shape_list 存储 obstacle obs pin_shape
   * net_idb_segment_map 存储 wire via patch
   */
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Helper& helper = DM_INST.getHelper();

  std::vector<Violation> violation_list;
  idrc::DrcApi drc_api;
  drc_api.init();
  for (auto& [type, idrc_violation_list] : drc_api.check(env_shape_list, net_pin_shape_map, net_wire_via_map)) {
    for (idrc::DrcViolation* idrc_violation : idrc_violation_list) {
      if (idrc_violation->get_net_ids().size() < 2) {
        continue;
      }

      idb::IdbLayer* idb_layer = idrc_violation->get_layer();

      EXTLayerRect ext_layer_rect;
      if (idrc_violation->is_rect()) {
        idrc::DrcViolationRect* idrc_violation_rect = static_cast<idrc::DrcViolationRect*>(idrc_violation);
        ext_layer_rect.set_real_ll(idrc_violation_rect->get_llx(), idrc_violation_rect->get_lly());
        ext_layer_rect.set_real_ur(idrc_violation_rect->get_urx(), idrc_violation_rect->get_ury());
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

std::map<std::string, std::vector<double>> RTAPI::getTiming(
    std::map<int32_t, std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>>& net_coord_real_pin_map,
    std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_segment_map)
{
#if 1  // 数据结构定义
  struct RCPin
  {
    RCPin() = default;
    RCPin(LayerCoord coord, bool is_real_pin, std::string pin_name)
    {
      _coord = coord;
      _is_real_pin = is_real_pin;
      _pin_name = pin_name;
    }
    RCPin(LayerCoord coord, bool is_real_pin, int32_t fake_pin_id)
    {
      _coord = coord;
      _is_real_pin = is_real_pin;
      _fake_pin_id = fake_pin_id;
    }
    ~RCPin() = default;

    LayerCoord _coord;
    bool _is_real_pin = false;
    std::string _pin_name;
    int32_t _fake_pin_id;
  };
  auto getRCSegmentList = [](std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>& coord_real_pin_map,
                             std::vector<Segment<LayerCoord>>& routing_segment_list) {
    // 预处理 对名字去重
    for (auto& [coord, real_pin_list] : coord_real_pin_map) {
      std::sort(real_pin_list.begin(), real_pin_list.end());
      real_pin_list.erase(std::unique(real_pin_list.begin(), real_pin_list.end()), real_pin_list.end());
    }
    // 构建coord_fake_pin_map
    std::map<LayerCoord, int32_t, CmpLayerCoordByXASC> coord_fake_pin_map;
    {
      int32_t fake_id = 0;
      for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
        LayerCoord& first_coord = routing_segment.get_first();
        LayerCoord& second_coord = routing_segment.get_second();

        if (!RTUtil::exist(coord_real_pin_map, first_coord) && !RTUtil::exist(coord_fake_pin_map, first_coord)) {
          coord_fake_pin_map[first_coord] = fake_id++;
        }
        if (!RTUtil::exist(coord_real_pin_map, second_coord) && !RTUtil::exist(coord_fake_pin_map, second_coord)) {
          coord_fake_pin_map[second_coord] = fake_id++;
        }
      }
    }
    std::vector<Segment<RCPin>> rc_segment_list;
    {
      // 生成线长为0的线段
      for (auto& [coord, real_pin_list] : coord_real_pin_map) {
        for (size_t i = 1; i < real_pin_list.size(); i++) {
          RCPin first_rc_pin(coord, true, RTUtil::escapeBackslash(real_pin_list[i - 1]));
          RCPin second_rc_pin(coord, true, RTUtil::escapeBackslash(real_pin_list[i]));
          rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
        }
      }
      // 生成线长大于0的线段
      for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
        auto getRCPin = [&](LayerCoord& coord) {
          RCPin rc_pin;
          if (RTUtil::exist(coord_real_pin_map, coord)) {
            rc_pin = RCPin(coord, true, RTUtil::escapeBackslash(coord_real_pin_map[coord].front()));
          } else if (RTUtil::exist(coord_fake_pin_map, coord)) {
            rc_pin = RCPin(coord, false, coord_fake_pin_map[coord]);
          } else {
            LOG_INST.error(Loc::current(), "The coord is not exist!");
          }
          return rc_pin;
        };
        rc_segment_list.emplace_back(getRCPin(routing_segment.get_first()), getRCPin(routing_segment.get_second()));
      }
    }
    return rc_segment_list;
  };
#endif

#if 1  // 主流程
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();
  int32_t thread_number = DM_INST.getConfig().thread_number;

  ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(timing_engine->get_ista());
  db_adapter->set_idb(DM_INST.getHelper().get_idb_builder());
  db_adapter->convertDBToTimingNetlist();
  timing_engine->set_db_adapter(std::move(db_adapter));
  timing_engine->set_num_threads(thread_number);
  timing_engine->buildGraph();
  timing_engine->initRcTree();

  ista::Netlist* sta_net_list = timing_engine->get_netlist();
  for (auto& [net_idx, coord_real_pin_map] : net_coord_real_pin_map) {
    std::vector<Segment<LayerCoord>>& routing_segment_list = net_routing_segment_map[net_idx];
    // 构建RC-tree
    ista::Net* ista_net = sta_net_list->findNet(RTUtil::escapeBackslash(net_list[net_idx].get_net_name()).c_str());
    for (Segment<RCPin>& segment : getRCSegmentList(coord_real_pin_map, routing_segment_list)) {
      RCPin& first_rc_pin = segment.get_first();
      RCPin& second_rc_pin = segment.get_second();

      double cap = 0;
      double res = 0;
      if (first_rc_pin._coord.get_layer_idx() == second_rc_pin._coord.get_layer_idx()) {
        int32_t distance = RTUtil::getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
        int32_t unit = DM_INST.getHelper().get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu();
        std::optional<double> width = std::nullopt;
        cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())
                  ->getCapacitance(first_rc_pin._coord.get_layer_idx() + 1, distance / 1.0 / unit, width);
        res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())
                  ->getResistance(first_rc_pin._coord.get_layer_idx() + 1, distance / 1.0 / unit, width);
      }

      auto getRctNode = [timing_engine, sta_net_list, ista_net](RCPin& rc_pin) {
        ista::RctNode* rct_node = nullptr;
        if (rc_pin._is_real_pin) {
          ista::DesignObject* pin_port = nullptr;
          auto pin_port_list = sta_net_list->findPin(rc_pin._pin_name.c_str(), false, false);
          if (!pin_port_list.empty()) {
            pin_port = pin_port_list.front();
          } else {
            pin_port = sta_net_list->findPort(rc_pin._pin_name.c_str());
          }
          rct_node = timing_engine->makeOrFindRCTreeNode(pin_port);
        } else {
          rct_node = timing_engine->makeOrFindRCTreeNode(ista_net, rc_pin._fake_pin_id);
        }
        return rct_node;
      };
      ista::RctNode* first_node = getRctNode(first_rc_pin);
      ista::RctNode* second_node = getRctNode(second_rc_pin);
      timing_engine->makeResistor(ista_net, first_node, second_node, res);
      timing_engine->incrCap(first_node, cap / 2);
      timing_engine->incrCap(second_node, cap / 2);
    }
    timing_engine->updateRCTreeInfo(ista_net);

    // auto* rc_tree = timing_engine->get_ista()->getRcNet(ista_net)->rct();
    // rc_tree->printGraphViz();
    // dot -Tpdf tree.dot -o tree.pdf
  }
  timing_engine->updateTiming();
  timing_engine->reportTiming();

  std::map<std::string, std::vector<double>> timing_map;
  auto clk_list = timing_engine->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    timing_map[clk_name] = {setup_tns, setup_wns, suggest_freq};
  });
  return timing_map;
#endif
}

void RTAPI::outputDef(std::string output_def_file_path)
{
  DM_INST.outputToIDB();
  dmInst->saveDef(output_def_file_path);
}

void RTAPI::outputSummary()
{
  Summary& rt_summary = DM_INST.getSummary();
  idb::RTSummary& top_rt_summary = dmInst->get_feature_summary().getRTSummary();

  top_rt_summary.pa_summary.routing_access_point_num_map = rt_summary.pa_summary.routing_access_point_num_map;
  for (auto& [type, access_point_num] : rt_summary.pa_summary.type_access_point_num_map) {
    top_rt_summary.pa_summary.type_access_point_num_map[GetAccessPointTypeName()(type)] = access_point_num;
  }
  top_rt_summary.pa_summary.total_access_point_num = rt_summary.pa_summary.total_access_point_num;

  top_rt_summary.sa_summary.routing_supply_map = rt_summary.sa_summary.routing_supply_map;
  top_rt_summary.sa_summary.total_supply = rt_summary.sa_summary.total_supply;

  top_rt_summary.ir_summary.routing_demand_map = rt_summary.ir_summary.routing_demand_map;
  top_rt_summary.ir_summary.total_demand = rt_summary.ir_summary.total_demand;
  top_rt_summary.ir_summary.routing_overflow_map = rt_summary.ir_summary.routing_overflow_map;
  top_rt_summary.ir_summary.total_overflow = rt_summary.ir_summary.total_overflow;
  top_rt_summary.ir_summary.routing_wire_length_map = rt_summary.ir_summary.routing_wire_length_map;
  top_rt_summary.ir_summary.total_wire_length = rt_summary.ir_summary.total_wire_length;
  top_rt_summary.ir_summary.cut_via_num_map = rt_summary.ir_summary.cut_via_num_map;
  top_rt_summary.ir_summary.total_via_num = rt_summary.ir_summary.total_via_num;
  top_rt_summary.ir_summary.timing = rt_summary.ir_summary.timing;

  for (auto& [iter, gr_summary] : rt_summary.iter_gr_summary_map) {
    idb::GRSummary& top_gr_summary = top_rt_summary.iter_gr_summary_map[iter];
    top_gr_summary.routing_demand_map = gr_summary.routing_demand_map;
    top_gr_summary.total_demand = gr_summary.total_demand;
    top_gr_summary.routing_overflow_map = gr_summary.routing_overflow_map;
    top_gr_summary.total_overflow = gr_summary.total_overflow;
    top_gr_summary.routing_wire_length_map = gr_summary.routing_wire_length_map;
    top_gr_summary.total_wire_length = gr_summary.total_wire_length;
    top_gr_summary.cut_via_num_map = gr_summary.cut_via_num_map;
    top_gr_summary.total_via_num = gr_summary.total_via_num;
    top_gr_summary.timing = gr_summary.timing;
  }

  top_rt_summary.ta_summary.routing_wire_length_map = rt_summary.ta_summary.routing_wire_length_map;
  top_rt_summary.ta_summary.total_wire_length = rt_summary.ta_summary.total_wire_length;
  top_rt_summary.ta_summary.routing_violation_num_map = rt_summary.ta_summary.routing_violation_num_map;
  top_rt_summary.ta_summary.total_violation_num = rt_summary.ta_summary.total_violation_num;

  for (auto& [iter, dr_summary] : rt_summary.iter_dr_summary_map) {
    idb::DRSummary& top_dr_summary = top_rt_summary.iter_dr_summary_map[iter];
    top_dr_summary.routing_wire_length_map = dr_summary.routing_wire_length_map;
    top_dr_summary.total_wire_length = dr_summary.total_wire_length;
    top_dr_summary.cut_via_num_map = dr_summary.cut_via_num_map;
    top_dr_summary.total_via_num = dr_summary.total_via_num;
    top_dr_summary.routing_patch_num_map = dr_summary.routing_patch_num_map;
    top_dr_summary.total_patch_num = dr_summary.total_patch_num;
    top_dr_summary.routing_violation_num_map = dr_summary.routing_violation_num_map;
    top_dr_summary.total_violation_num = dr_summary.total_violation_num;
    top_dr_summary.timing = dr_summary.timing;
  }
}

#endif

// private

RTAPI* RTAPI::_rt_api_instance = nullptr;

}  // namespace irt
