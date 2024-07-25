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
#include "RTInterface.hpp"

#include "CongTile.hpp"
#include "DataManager.hpp"
#include "DetailedRouter.hpp"
#include "GDSPlotter.hpp"
#include "GlobalRouter.hpp"
#include "InitialRouter.hpp"
#include "Monitor.hpp"
#include "PinAccessor.hpp"
#include "SupplyAnalyzer.hpp"
#include "TimingEval.hpp"
#include "TopologyGenerator.hpp"
#include "TrackAssigner.hpp"
#include "builder.h"
#include "feature_irt.h"
#include "feature_ista.h"
#include "flow_config.h"
#include "icts_fm/file_cts.h"
#include "icts_io.h"
#include "idm.h"
#include "idrc_api.h"

namespace irt {

// public

RTInterface& RTInterface::getInst()
{
  if (_rt_interface_instance == nullptr) {
    _rt_interface_instance = new RTInterface();
  }
  return *_rt_interface_instance;
}

void RTInterface::destroyInst()
{
  if (_rt_interface_instance != nullptr) {
    delete _rt_interface_instance;
    _rt_interface_instance = nullptr;
  }
}

#if 1  // 外部调用RT的API

void RTInterface::initRT(std::map<std::string, std::any> config_map)
{
  Logger::initInst();
  // clang-format off
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  RTLOG.info(Loc::current(), "_____ ________ ________     _______________________ ________ ________  ");
  RTLOG.info(Loc::current(), "___(_)___  __ \\___  __/     __  ___/___  __/___    |___  __ \\___  __/");
  RTLOG.info(Loc::current(), "__  / __  /_/ /__  /        _____ \\ __  /   __  /| |__  /_/ /__  /    ");
  RTLOG.info(Loc::current(), "_  /  _  _, _/ _  /         ____/ / _  /    _  ___ |_  _, _/ _  /      ");
  RTLOG.info(Loc::current(), "/_/   /_/ |_|  /_/          /____/  /_/     /_/  |_|/_/ |_|  /_/       ");
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  RTLOG.printLogFilePath();
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  DataManager::initInst();
  RTDM.input(config_map, dmInst->get_idb_builder());
  GDSPlotter::initInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::runEGR()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  PinAccessor::initInst();
  RTPA.access();
  PinAccessor::destroyInst();

  SupplyAnalyzer::initInst();
  RTSA.analyze();
  SupplyAnalyzer::destroyInst();

  TopologyGenerator::initInst();
  RTTG.generate();
  TopologyGenerator::destroyInst();

  InitialRouter::initInst();
  RTIR.route();
  InitialRouter::destroyInst();

  GlobalRouter::initInst();
  RTGR.route();
  GlobalRouter::destroyInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::runRT()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  PinAccessor::initInst();
  RTPA.access();
  PinAccessor::destroyInst();

  SupplyAnalyzer::initInst();
  RTSA.analyze();
  SupplyAnalyzer::destroyInst();

  TopologyGenerator::initInst();
  RTTG.generate();
  TopologyGenerator::destroyInst();

  InitialRouter::initInst();
  RTIR.route();
  InitialRouter::destroyInst();

  GlobalRouter::initInst();
  RTGR.route();
  GlobalRouter::destroyInst();

  TrackAssigner::initInst();
  RTTA.assign();
  TrackAssigner::destroyInst();

  DetailedRouter::initInst();
  RTDR.route();
  DetailedRouter::destroyInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void RTInterface::destroyRT()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GDSPlotter::destroyInst();
  RTDM.output();
  DataManager::destroyInst();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  RTLOG.printLogFilePath();
  // clang-format off
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  RTLOG.info(Loc::current(), "_____ ________ ________     _______________________   ________________________  __  ");
  RTLOG.info(Loc::current(), "___(_)___  __ \\___  __/     ___  ____/____  _/___  | / /____  _/__  ___/___  / / / ");
  RTLOG.info(Loc::current(), "__  / __  /_/ /__  /        __  /_     __  /  __   |/ /  __  /  _____ \\ __  /_/ /  ");
  RTLOG.info(Loc::current(), "_  /  _  _, _/ _  /         _  __/    __/ /   _  /|  /  __/ /   ____/ / _  __  /    ");
  RTLOG.info(Loc::current(), "/_/   /_/ |_|  /_/          /_/       /___/   /_/ |_/   /___/   /____/  /_/ /_/     ");
  RTLOG.info(Loc::current(), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
  // clang-format on
  Logger::destroyInst();
}

void RTInterface::clearDef()
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
      RTLOG.info(Loc::current(), "The net '", idb_net->get_net_name(), "' connects PAD and io_pin! removing...");
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
      RTLOG.info(Loc::current(), io_pin->get_pin_name());
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

#if 1  // RT调用外部的API

std::vector<Violation> RTInterface::getViolationList(std::vector<idb::IdbLayerShape*>& env_shape_list,
                                                     std::map<int32_t, std::vector<idb::IdbLayerShape*>>& net_pin_shape_map,
                                                     std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& net_wire_via_map,
                                                     std::string stage)
{
  std::set<idrc::ViolationEnumType> check_select;
  if (stage == "TA") {
    check_select.insert(idrc::ViolationEnumType::kShort);
  } else if (stage == "DR") {
    check_select.insert(idrc::ViolationEnumType::kShort);
    check_select.insert(idrc::ViolationEnumType::kDefaultSpacing);
  } else {
    RTLOG.error(Loc::current(), "Currently not supporting other stages");
  }
  /**
   * env_shape_list 存储 obstacle obs pin_shape
   * net_idb_segment_map 存储 wire via patch
   */
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();
  std::map<std::string, int32_t>& cut_layer_name_to_idx_map = RTDM.getDatabase().get_cut_layer_name_to_idx_map();

  std::vector<Violation> violation_list;
  idrc::DrcApi drc_api;
  drc_api.init();
  for (auto& [type, idrc_violation_list] : drc_api.check(env_shape_list, net_pin_shape_map, net_wire_via_map, check_select)) {
    for (idrc::DrcViolation* idrc_violation : idrc_violation_list) {
      // self的drc违例先过滤
      if (idrc_violation->get_net_ids().size() < 2) {
        continue;
      }
      // 由于pin_shape之间的drc违例存在，第一布线层的drc违例先过滤
      idb::IdbLayer* idb_layer = idrc_violation->get_layer();
      if (idb_layer->is_routing()) {
        if (routing_layer_name_to_idx_map[idb_layer->get_name()] == 0) {
          continue;
        }
      }
      EXTLayerRect ext_layer_rect;
      if (idrc_violation->is_rect()) {
        idrc::DrcViolationRect* idrc_violation_rect = static_cast<idrc::DrcViolationRect*>(idrc_violation);
        ext_layer_rect.set_real_ll(idrc_violation_rect->get_llx(), idrc_violation_rect->get_lly());
        ext_layer_rect.set_real_ur(idrc_violation_rect->get_urx(), idrc_violation_rect->get_ury());
      } else {
        RTLOG.error(Loc::current(), "Type not supported!");
      }
      ext_layer_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(ext_layer_rect.get_real_rect(), gcell_axis));
      ext_layer_rect.set_layer_idx(idb_layer->is_routing() ? routing_layer_name_to_idx_map[idb_layer->get_name()]
                                                           : cut_layer_name_to_idx_map[idb_layer->get_name()]);

      Violation violation;
      violation.set_violation_shape(ext_layer_rect);
      violation.set_is_routing(idb_layer->is_routing());
      violation.set_violation_net_set(idrc_violation->get_net_ids());
      violation_list.push_back(violation);
    }
  }
  return violation_list;
}

std::map<std::string, std::vector<double>> RTInterface::getTiming(
    std::vector<std::map<std::string, std::vector<LayerCoord>>>& real_pin_coord_map_list,
    std::vector<std::vector<Segment<LayerCoord>>>& routing_segment_list_list)
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
#endif

#if 1  // 函数定义
  auto initTimingEngine = [](idb::IdbBuilder* idb_builder, int32_t thread_number) {
    ista::TimingEngine* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
    if (!timing_engine->get_db_adapter()) {
      auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(timing_engine->get_ista());
      db_adapter->set_idb(idb_builder);
      db_adapter->convertDBToTimingNetlist();
      timing_engine->set_db_adapter(std::move(db_adapter));
    }
    timing_engine->set_num_threads(thread_number);
    timing_engine->buildGraph();
    timing_engine->initRcTree();
    return timing_engine;
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

        if (!RTUTIL.exist(coord_real_pin_map, first_coord) && !RTUTIL.exist(coord_fake_pin_map, first_coord)) {
          coord_fake_pin_map[first_coord] = fake_id++;
        }
        if (!RTUTIL.exist(coord_real_pin_map, second_coord) && !RTUTIL.exist(coord_fake_pin_map, second_coord)) {
          coord_fake_pin_map[second_coord] = fake_id++;
        }
      }
    }
    std::vector<Segment<RCPin>> rc_segment_list;
    {
      // 生成线长为0的线段
      for (auto& [coord, real_pin_list] : coord_real_pin_map) {
        for (size_t i = 1; i < real_pin_list.size(); i++) {
          RCPin first_rc_pin(coord, true, RTUTIL.escapeBackslash(real_pin_list[i - 1]));
          RCPin second_rc_pin(coord, true, RTUTIL.escapeBackslash(real_pin_list[i]));
          rc_segment_list.emplace_back(first_rc_pin, second_rc_pin);
        }
      }
      // 生成线长大于0的线段
      for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
        auto getRCPin = [&](LayerCoord& coord) {
          RCPin rc_pin;
          if (RTUTIL.exist(coord_real_pin_map, coord)) {
            rc_pin = RCPin(coord, true, RTUTIL.escapeBackslash(coord_real_pin_map[coord].front()));
          } else if (RTUTIL.exist(coord_fake_pin_map, coord)) {
            rc_pin = RCPin(coord, false, coord_fake_pin_map[coord]);
          } else {
            RTLOG.error(Loc::current(), "The coord is not exist!");
          }
          return rc_pin;
        };
        rc_segment_list.emplace_back(getRCPin(routing_segment.get_first()), getRCPin(routing_segment.get_second()));
      }
    }
    return rc_segment_list;
  };
  auto getRctNode = [](ista::TimingEngine* timing_engine, ista::Netlist* sta_net_list, ista::Net* ista_net, RCPin& rc_pin) {
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
#endif

#if 1  // 预处理流程
  // 每个pin只留一个连通的坐标
  for (size_t i = 0; i < real_pin_coord_map_list.size(); i++) {
    std::vector<Segment<LayerCoord>>& routing_segment_list = routing_segment_list_list[i];
    for (auto& [pin_name, coord_list] : real_pin_coord_map_list[i]) {
      if (coord_list.size() < 2) {
        continue;
      }
      if (routing_segment_list.empty()) {
        coord_list.erase(coord_list.begin() + 1, coord_list.end());
      } else {
        for (LayerCoord& coord : coord_list) {
          bool is_exist = false;
          for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
            if (coord == routing_segment.get_first() || coord == routing_segment.get_second()) {
              is_exist = true;
              break;
            }
          }
          if (is_exist) {
            coord_list[0] = coord;
            coord_list.erase(coord_list.begin() + 1, coord_list.end());
            break;
          }
        }
      }
      if (coord_list.size() > 2) {
        RTLOG.error(Loc::current(), "The pin ", pin_name, " is not in segment_list");
      }
    }
  }
  // coord_real_pin_map_list
  std::vector<std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>> coord_real_pin_map_list;
  coord_real_pin_map_list.resize(real_pin_coord_map_list.size());
  for (size_t i = 0; i < real_pin_coord_map_list.size(); i++) {
    for (auto& [real_pin, coord_list] : real_pin_coord_map_list[i]) {
      for (LayerCoord& coord : coord_list) {
        coord_real_pin_map_list[i][coord].push_back(real_pin);
      }
    }
  }
#endif

#if 1  // 主流程
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  int32_t thread_number = RTDM.getConfig().thread_number;

  ista::TimingEngine* timing_engine = initTimingEngine(RTDM.getDatabase().get_idb_builder(), thread_number);
  ista::Netlist* sta_net_list = timing_engine->get_netlist();

  for (size_t net_idx = 0; net_idx < coord_real_pin_map_list.size(); net_idx++) {
    ista::Net* ista_net = sta_net_list->findNet(RTUTIL.escapeBackslash(net_list[net_idx].get_net_name()).c_str());
    for (Segment<RCPin>& segment : getRCSegmentList(coord_real_pin_map_list[net_idx], routing_segment_list_list[net_idx])) {
      RCPin& first_rc_pin = segment.get_first();
      RCPin& second_rc_pin = segment.get_second();

      double cap = 0;
      double res = 0;
      if (first_rc_pin._coord.get_layer_idx() == second_rc_pin._coord.get_layer_idx()) {
        int32_t distance = RTUTIL.getManhattanDistance(first_rc_pin._coord, second_rc_pin._coord);
        int32_t unit = RTDM.getDatabase().get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu();
        std::optional<double> width = std::nullopt;
        cap = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())
                  ->getCapacitance(first_rc_pin._coord.get_layer_idx() + 1, distance / 1.0 / unit, width);
        res = dynamic_cast<ista::TimingIDBAdapter*>(timing_engine->get_db_adapter())
                  ->getResistance(first_rc_pin._coord.get_layer_idx() + 1, distance / 1.0 / unit, width);
      }

      ista::RctNode* first_node = getRctNode(timing_engine, sta_net_list, ista_net, first_rc_pin);
      ista::RctNode* second_node = getRctNode(timing_engine, sta_net_list, ista_net, second_rc_pin);
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
    auto setup_tns = timing_engine->getTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = timing_engine->getWNS(clk_name, AnalysisMode::kMax);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    timing_map[clk_name] = {setup_tns, setup_wns, suggest_freq};
  });
  return timing_map;
#endif
}

ieda_feature::RTSummary RTInterface::outputSummary()
{
  ieda_feature::RTSummary top_rt_summary;

  Summary& rt_summary = RTDM.getSummary();

  // pa_summary
  top_rt_summary.pa_summary.routing_access_point_num_map = rt_summary.pa_summary.routing_access_point_num_map;
  for (auto& [type, access_point_num] : rt_summary.pa_summary.type_access_point_num_map) {
    top_rt_summary.pa_summary.type_access_point_num_map[GetAccessPointTypeName()(type)] = access_point_num;
  }
  top_rt_summary.pa_summary.total_access_point_num = rt_summary.pa_summary.total_access_point_num;
  // sa_summary
  top_rt_summary.sa_summary.routing_supply_map = rt_summary.sa_summary.routing_supply_map;
  top_rt_summary.sa_summary.total_supply = rt_summary.sa_summary.total_supply;
  // ir_summary
  top_rt_summary.ir_summary.routing_demand_map = rt_summary.ir_summary.routing_demand_map;
  top_rt_summary.ir_summary.total_demand = rt_summary.ir_summary.total_demand;
  top_rt_summary.ir_summary.routing_overflow_map = rt_summary.ir_summary.routing_overflow_map;
  top_rt_summary.ir_summary.total_overflow = rt_summary.ir_summary.total_overflow;
  top_rt_summary.ir_summary.routing_wire_length_map = rt_summary.ir_summary.routing_wire_length_map;
  top_rt_summary.ir_summary.total_wire_length = rt_summary.ir_summary.total_wire_length;
  top_rt_summary.ir_summary.cut_via_num_map = rt_summary.ir_summary.cut_via_num_map;
  top_rt_summary.ir_summary.total_via_num = rt_summary.ir_summary.total_via_num;

  for (auto timing : rt_summary.ir_summary.timing) {
    ieda_feature::NetTiming net_timing;
    net_timing.net_name = timing.first;
    auto timing_array = timing.second;
    net_timing.setup_tns = timing_array[0];
    net_timing.setup_wns = timing_array[1];
    net_timing.suggest_freq = timing_array[2];
    top_rt_summary.ir_summary.nets_timing.push_back(net_timing);
  }
  // gr_summary
  for (auto& [iter, gr_summary] : rt_summary.iter_gr_summary_map) {
    ieda_feature::GRSummary& top_gr_summary = top_rt_summary.iter_gr_summary_map[iter];
    top_gr_summary.routing_demand_map = gr_summary.routing_demand_map;
    top_gr_summary.total_demand = gr_summary.total_demand;
    top_gr_summary.routing_overflow_map = gr_summary.routing_overflow_map;
    top_gr_summary.total_overflow = gr_summary.total_overflow;
    top_gr_summary.routing_wire_length_map = gr_summary.routing_wire_length_map;
    top_gr_summary.total_wire_length = gr_summary.total_wire_length;
    top_gr_summary.cut_via_num_map = gr_summary.cut_via_num_map;
    top_gr_summary.total_via_num = gr_summary.total_via_num;

    for (auto timing : gr_summary.timing) {
      ieda_feature::NetTiming net_timing;
      net_timing.net_name = timing.first;
      auto timing_array = timing.second;
      net_timing.setup_tns = timing_array[0];
      net_timing.setup_wns = timing_array[1];
      net_timing.suggest_freq = timing_array[2];
      top_gr_summary.nets_timing.push_back(net_timing);
    }
  }
  // ta_summary
  top_rt_summary.ta_summary.routing_wire_length_map = rt_summary.ta_summary.routing_wire_length_map;
  top_rt_summary.ta_summary.total_wire_length = rt_summary.ta_summary.total_wire_length;
  top_rt_summary.ta_summary.routing_violation_num_map = rt_summary.ta_summary.routing_violation_num_map;
  top_rt_summary.ta_summary.total_violation_num = rt_summary.ta_summary.total_violation_num;
  // dr_summary
  for (auto& [iter, dr_summary] : rt_summary.iter_dr_summary_map) {
    ieda_feature::DRSummary& top_dr_summary = top_rt_summary.iter_dr_summary_map[iter];
    top_dr_summary.routing_wire_length_map = dr_summary.routing_wire_length_map;
    top_dr_summary.total_wire_length = dr_summary.total_wire_length;
    top_dr_summary.cut_via_num_map = dr_summary.cut_via_num_map;
    top_dr_summary.total_via_num = dr_summary.total_via_num;
    top_dr_summary.routing_patch_num_map = dr_summary.routing_patch_num_map;
    top_dr_summary.total_patch_num = dr_summary.total_patch_num;
    top_dr_summary.routing_violation_num_map = dr_summary.routing_violation_num_map;
    top_dr_summary.total_violation_num = dr_summary.total_violation_num;

    for (auto timing : dr_summary.timing) {
      ieda_feature::NetTiming net_timing;
      net_timing.net_name = timing.first;
      auto timing_array = timing.second;
      net_timing.setup_tns = timing_array[0];
      net_timing.setup_wns = timing_array[1];
      net_timing.suggest_freq = timing_array[2];
      top_dr_summary.nets_timing.push_back(net_timing);
    }
  }

  return top_rt_summary;
}

#endif

// private

RTInterface* RTInterface::_rt_interface_instance = nullptr;

}  // namespace irt
