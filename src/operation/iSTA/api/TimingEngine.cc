// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file TimingEngine.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-08-20
 */

#include "TimingEngine.hh"

#include <iostream>
#include <optional>

#include "FlatSet.hh"
#include "TimingIDBAdapter.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/Instance.hh"
#include "netlist/Netlist.hh"
#include "sdc-cmd/Cmd.hh"
#include "sta/Sta.hh"
#include "sta/StaAnalyze.hh"
#include "sta/StaApplySdc.hh"
#include "sta/StaBuildGraph.hh"
#include "sta/StaBuildRCTree.hh"
#include "sta/StaClockPropagation.hh"
#include "sta/StaClockTree.hh"
#include "sta/StaConstPropagation.hh"
#include "sta/StaDataPropagation.hh"
#include "sta/StaDelayPropagation.hh"
#include "sta/StaDump.hh"
#include "sta/StaGraph.hh"
#include "sta/StaLevelization.hh"
#include "sta/StaPathData.hh"
#include "sta/StaSlewPropagation.hh"
#include "sta/StaVertex.hh"
#include "tcl/ScriptEngine.hh"

namespace ista {

TimingEngine* TimingEngine::_timing_engine = nullptr;

TimingEngine::TimingEngine() { _ista = Sta::getOrCreateSta(); }

TimingEngine::~TimingEngine() { Sta::destroySta(); }

/**
 * @brief Get the TimingEngine instance, if not, create one.
 *
 * @return TimingEngine*
 */
TimingEngine* TimingEngine::getOrCreateTimingEngine() {
  static std::mutex mt;
  if (_timing_engine == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_timing_engine == nullptr) {
      _timing_engine = new TimingEngine();
    }
  }
  return _timing_engine;
}

/**
 * @brief Destory the TimingEngine.
 *
 */
void TimingEngine::destroyTimingEngine() {
  delete _timing_engine;
  _timing_engine = nullptr;
}

/**
 * @brief set the numbers of threads
 *
 * @param num_thread
 * @return TimingEngine&.
 */
TimingEngine& TimingEngine::set_num_threads(unsigned num_thread) {
  auto* ista = _ista;
  ista->set_num_threads(num_thread);
  return *this;
}

void TimingEngine::set_db_adapter(std::unique_ptr<TimingDBAdapter> db_adapter) {
  _db_adapter = std::move(db_adapter);
}

/**
 * @brief read def design for construct netlist db.
 * 
 * @param def_file 
 * @param lef_files 
 * @return TimingEngine& 
 */
TimingEngine& TimingEngine::readDefDesign(std::string def_file,
                                          std::vector<std::string>& lef_files) {
  auto* db_builder = new idb::IdbBuilder();
  db_builder->buildLef(lef_files);
  db_builder->buildDef(def_file);

  auto db_adapter = std::make_unique<TimingIDBAdapter>(get_ista());
  db_adapter->set_idb(db_builder);
  db_adapter->convertDBToTimingNetlist();

  set_db_adapter(std::move(db_adapter));

  return *this;
}

/**
 * @brief set the db builder which has read def design.
 * 
 * @param db_builder 
 * @return TimingEngine& 
 */
TimingEngine& TimingEngine::setDefDesignBuilder(void* db_builder) {
  auto db_adapter = std::make_unique<TimingIDBAdapter>(get_ista());

  db_adapter->set_idb(static_cast<idb::IdbBuilder*>(db_builder));
  db_adapter->convertDBToTimingNetlist();

  set_db_adapter(std::move(db_adapter));

  return *this;
}

/**
 * @brief get the LibTable of a cell.
 * table.
 *
 * @param cell_name
 * @param TableType(kCellRise = 0,kCellFall = 1,kRiseTransition =
 * 2,kFallTransition = 3)
 * @return LibTable*
 */
LibTable* TimingEngine::getCellLibertyTable(const char* cell_name,
                                            LibTable::TableType table_type) {
  LibCell* lib_cell = _ista->findLibertyCell(cell_name);
  LOG_FATAL_IF(!lib_cell) << cell_name << " lib cell is not found.";
  const char* from_port_name = nullptr;
  const char* to_port_name = nullptr;

  auto& str2ports = lib_cell->get_str2ports();
  for (auto& [port_name, port] : str2ports) {
    if (port->isInput()) {
      from_port_name = port_name;
    } else {
      to_port_name = port_name;
    }
  }
  std::optional<LibArcSet*> timing_arc_set =
      lib_cell->findLibertyArcSet(from_port_name, to_port_name);
  LibArc* the_timing_arc = nullptr;
  if (timing_arc_set) {
    the_timing_arc = (*timing_arc_set)->front();

    LOG_FATAL_IF(!the_timing_arc);
    LibTableModel* table_model = the_timing_arc->get_table_model();
    LibTable* table = dynamic_cast<LibDelayTableModel*>(table_model)
                          ->getTable(int(table_type));
    return table;
  }
  return nullptr;
}

/**
 * @brief get the LibTable of a cell.
 * table.
 *
 * @param cell_name
 * @param from_port_name
 * @param to_port_name
 * @param TableType(kCellRise = 0,kCellFall = 1,kRiseTransition =
 * 2,kFallTransition = 3)
 * @return LibTable*
 */
LibTable* TimingEngine::getCellLibertyTable(const char* cell_name,
                                            const char* from_port_name,
                                            const char* to_port_name,
                                            LibTable::TableType table_type) {
  LibCell* lib_cell = _ista->findLibertyCell(cell_name);

  std::optional<LibArcSet*> timing_arc_set =
      lib_cell->findLibertyArcSet(from_port_name, to_port_name);
  LibArc* the_timing_arc = nullptr;
  if (timing_arc_set) {
    the_timing_arc = (*timing_arc_set)->front();

    LOG_FATAL_IF(!the_timing_arc);
    LibTableModel* table_model = the_timing_arc->get_table_model();
    LibTable* table = dynamic_cast<LibDelayTableModel*>(table_model)
                          ->getTable(int(table_type));
    return table;
  }
  return nullptr;
}

/**
 * @brief get the LibTable of a cell.
 * table.
 *
 * @param cell_name
 * @param timing_type
 * timing_type(TimingType::kRisingEdge,TimingType::kFallingEdge,TimingType::kComb)
 * @param TableType(kCellRise = 0,kCellFall = 1,kRiseTransition =
 * 2,kFallTransition = 3)
 * @return LibTable*
 */
LibTable* TimingEngine::getCellLibertyTable(const char* cell_name,
                                            LibArc::TimingType timing_type,
                                            LibTable::TableType table_type) {
  LibCell* lib_cell = _ista->findLibertyCell(cell_name);
  const char* from_port_name = nullptr;
  const char* to_port_name = nullptr;

  auto& str2ports = lib_cell->get_str2ports();
  LOG_FATAL_IF(str2ports.size() != 2)
      << cell_name << " may be not a standard cell.";
  for (auto& [port_name, port] : str2ports) {
    if (port->isInput()) {
      from_port_name = port_name;
    } else {
      to_port_name = port_name;
    }
  }

  std::optional<LibArcSet*> timing_arc_set = lib_cell->findLibertyArcSet(
      from_port_name, to_port_name, LibArc::TimingType::kComb);
  LibArc* the_timing_arc = nullptr;
  if (timing_arc_set) {
    the_timing_arc = (*timing_arc_set)->front();

    LOG_FATAL_IF(!the_timing_arc);
    LibTableModel* table_model = the_timing_arc->get_table_model();
    LibTable* table = dynamic_cast<LibDelayTableModel*>(table_model)
                          ->getTable(int(table_type));
    return table;
  }
  return nullptr;
}

/**
 * @brief find the end/start pins in the given start/ pin in the
 * timing path. (after running the step:updateTiming(StaClockPropagation))
 *
 * @param pin_name
 * @param is_find_end
 * @return std::set<std::string>
 */
std::set<std::string> TimingEngine::findStartOrEnd(const char* pin_name) {
  auto* the_vertex = _ista->findVertex(pin_name);
  LOG_FATAL_IF(!the_vertex) << pin_name << " vertex is not found.";
  if (!the_vertex->is_start() && !the_vertex->is_end()) {
    return {};
  }
  bool is_find_end = the_vertex->is_start();
  std::set<std::string> pin_names =
      _ista->findStartOrEnd(the_vertex, is_find_end);
  return pin_names;
}

/**
 * @brief obtain the start2end pairs of the all timing path.
 *
 * @return std::map<std::string, std::string>
 */
std::map<std::string, std::string> TimingEngine::getStartEndPairs() {
  StaGraph* the_graph = &(_ista->get_graph());
  std::map<std::string, std::string> start2end;
  StaVertex* vertex;
  FOREACH_END_VERTEX(the_graph, vertex) {
    std::string end_pin_name = vertex->getName();
    auto& start_vertexes = vertex->get_fanin_start_vertexes();
    for (auto& start_vertex : start_vertexes) {
      std::string start_pin_name = start_vertex->getName();
      start2end[start_pin_name] = end_pin_name;
    }
  }
  FOREACH_START_VERTEX(the_graph, vertex) {
    std::string start_pin_name = vertex->getName();
    auto& end_vertexes = vertex->get_fanout_end_vertexes();
    for (auto& end_vertex : end_vertexes) {
      std::string end_pin_name = end_vertex->getName();
      start2end[start_pin_name] = end_pin_name;
    }
  }

  return start2end;
}

/**
 * @brief find the clock pin name according to the instance name.
 *
 * @param inst_name
 * @return const char*
 */
std::string TimingEngine::findClockPinName(const char* inst_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  Instance* inst = design_netlist->findInstance(inst_name);
  LOG_FATAL_IF(!inst);
  auto& the_graph = ista->get_graph();
  Pin* pin;
  std::string the_clock_pin_name;
  FOREACH_INSTANCE_PIN(inst, pin) {
    auto the_vertex = the_graph.findVertex(pin);
    LOG_FATAL_IF(!the_vertex);
    if ((*the_vertex)->is_clock()) {
      the_clock_pin_name = (*the_vertex)->getName();
      break;
    }
  }

  return the_clock_pin_name;
}

/**
 * @brief Build RC tree accord !spef! file.
 *
 * @param design_work_space
 * @param spef_file
 * @param kmethod
 * @return TimingEngine&
 */
TimingEngine& TimingEngine::buildRCTree(const char* spef_file,
                                        DelayCalcMethod kmethod) {
  StaBuildRCTree build_rc_tree(spef_file, kmethod);

  StaGraph& the_graph = _ista->get_graph();
  build_rc_tree(&the_graph);

  return *this;
}

/**
 * @brief init one rc tree.
 *
 * @param net
 */
void TimingEngine::initRcTree(Net* net) {
  StaBuildRCTree build_rc_tree;
  auto rc_net = build_rc_tree.createRcNet(net);
  _timing_engine->get_ista()->addRcNet(net, std::move(rc_net));
  _timing_engine->updateRCTreeInfo(net);
}

/**
 * @brief init all net rc tree.
 *
 */
void TimingEngine::initRcTree() {
  Netlist* design_nl = _timing_engine->get_netlist();
  Net* net;
  FOREACH_NET(design_nl, net) { initRcTree(net); }
}

/**
 * @brief reset rc tree to nullptr.
 *
 * @param net
 */
void TimingEngine::resetRcTree(Net* net) {
  _timing_engine->get_ista()->resetRcNet(net);
}

/**
 * @brief Make RC tree internal node.
 *
 * @param net
 * @param id
 * @return RctNode*
 */
RctNode* TimingEngine::makeOrFindRCTreeNode(Net* net, int64_t id) {
  StaBuildRCTree build_rc_tree;
  auto* rc_net = _timing_engine->get_ista()->getRcNet(net);
  if (!rc_net) {
    auto created_rc_net = build_rc_tree.createRcNet(net);
    rc_net = created_rc_net.get();
    created_rc_net->makeRct();
    _timing_engine->get_ista()->addRcNet(net, std::move(created_rc_net));
  } else {
    if (!rc_net->rct()) {
      rc_net->makeRct();
    }
  }

  auto* rc_tree = rc_net->rct();
  std::string node_name = Str::printf("%s:%lld", net->get_name(), id);

  auto* node = rc_tree->node(node_name);
  if (!node) {
    return rc_tree->insertNode(node_name);
  }

  return node;
}

/**
 * @brief Make RC tree pin node.
 *
 * @param pin_or_port
 * @return RctNode*
 */
RctNode* TimingEngine::makeOrFindRCTreeNode(DesignObject* pin_or_port) {
  StaBuildRCTree build_rc_tree;
  auto* net = pin_or_port->get_net();
  LOG_FATAL_IF(!net);
  auto* rc_net = _timing_engine->get_ista()->getRcNet(net);
  if (!rc_net) {
    auto created_rc_net = build_rc_tree.createRcNet(net);
    rc_net = created_rc_net.get();
    created_rc_net->makeRct();
    _timing_engine->get_ista()->addRcNet(net, std::move(created_rc_net));
  } else {
    if (!rc_net->rct()) {
      rc_net->makeRct();
    }
  }

  auto* rc_tree = rc_net->rct();
  std::string node_name = pin_or_port->getFullName();

  auto* node = rc_tree->node(node_name);
  if (!node) {
    return rc_tree->insertNode(node_name);
  }

  return node;
}

/**
 * @brief find the exist rc tree node.
 * 
 * @param net 
 * @param node_name 
 * @return RctNode* 
 */
RctNode* TimingEngine::findRCTreeNode(Net *net, std::string& node_name) {
  auto* rc_net = _timing_engine->get_ista()->getRcNet(net);
  LOG_FATAL_IF(!rc_net) << net->get_name() << " rc net is not found.";
  auto* rc_tree = rc_net->rct();

  return rc_tree->node(node_name);
}

void TimingEngine::incrCap(RctNode* node, double cap, bool is_incremental) {
  is_incremental ? node->incrCap(cap) : node->setCap(cap);
}

/**
 * @brief Make resistor edge of rc tree.
 *
 * @param node1
 * @param node2
 * @param res
 */
void TimingEngine::makeResistor(Net* net, RctNode* from_node, RctNode* to_node,
                                double res) {
  auto* rc_net = _timing_engine->get_ista()->getRcNet(net);
  if (!rc_net) {
    return;
  }
  auto* rc_tree = rc_net->rct();

  auto from_fanouts = from_node->get_fanout();

  // judge whether the edge is already exist.
  auto found = std::ranges::find_if(from_fanouts, [&](auto* edge) {
    return &(edge->get_to()) == to_node;
  });

  if (found != from_fanouts.end()) {
    return;
  }

  rc_tree->insertEdge(from_node, to_node, res, true);
  rc_tree->insertEdge(to_node, from_node, res, false);
}

/**
 * @brief update rc info after make rc tree.
 *
 * @param net
 */
void TimingEngine::updateRCTreeInfo(Net* net) {
  auto* rc_net = _timing_engine->get_ista()->getRcNet(net);
  
  if (rc_net) {
    rc_net->updateRcTreeInfo();
    auto* rct = rc_net->rct();
    if (rct) {
      // check and break loop.
      rc_net->checkLoop();
      rct->updateRcTiming();
    }
  }
}


/**
 * @brief update all rc tree elmore delay use gpu speedup.
 * 
 */
void TimingEngine::updateAllRCTree() {
#if CUDA_DELAY
  auto all_rc_nets = _timing_engine->get_ista()->getAllRcNet();
  calc_rc_timing(all_rc_nets);
#endif
}


/**
 * @brief build balanced rc tree of the net and update rc tree info.
 *
 * @param net_name
 * @param loadname2wl
 */
void TimingEngine::buildRcTreeAndUpdateRcTreeInfo(
    const char* net_name, std::map<std::string, double>& loadname2wl) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* net = design_netlist->findNet(net_name);

  auto* driver = net->getDriver();
  auto driver_node = makeOrFindRCTreeNode(driver);
  auto loads = net->getLoads();
  auto* db_adapter = get_db_adapter();
  std::optional<double> width = std::nullopt;
  double unit_res =
      dynamic_cast<TimingIDBAdapter*>(db_adapter)->getAverageResistance(width);
  double unit_cap =
      dynamic_cast<TimingIDBAdapter*>(db_adapter)->getAverageCapacitance(width);

  for (const auto& load : loads) {
    auto load_node = makeOrFindRCTreeNode(load);
    std::string load_name = load->get_name();
    double load_net_wl = loadname2wl[load_name];
    double cap = unit_cap * load_net_wl;
    double res = unit_res * load_net_wl;
    makeResistor(net, driver_node, load_node, res / loads.size());
    bool is_incremental = true;
    incrCap(driver_node, cap / (2 * loads.size()), is_incremental);
    incrCap(load_node, cap / (2 * loads.size()), is_incremental);
  }
  updateRCTreeInfo(net);
}

/**
 * @brief Get all node slew of virtual rc tree.
 *
 * @param rc_tree_name
 * @param driver_slew
 * @return std::map<std::string, double>
 */
std::map<std::string, double> TimingEngine::getVirtualRCTreeAllNodeSlew(
    const char* rc_tree_name, double driver_slew, TransType trans_type) {
  if (!_virtual_rc_trees.contains(rc_tree_name)) {
    LOG_FATAL << "virtual RC tree " << rc_tree_name << " does not exist!";
  }

  auto& virtual_rc_tree = _virtual_rc_trees[rc_tree_name];

  std::map<std::string, double> all_node_slews;
  all_node_slews = virtual_rc_tree.getAllNodeSlew(driver_slew, AnalysisMode::kMax, trans_type);

  return all_node_slews;
}

/**
 * @brief get all node delay of virtual rc tree.
 *
 * @param rc_tree_name
 * @return std::map<std::string, double>
 */
std::map<std::string, double> TimingEngine::getVirtualRCTreeAllNodeDelay(
    const char* rc_tree_name) {
  if (!_virtual_rc_trees.contains(rc_tree_name)) {
    LOG_FATAL << "virtual RC tree " << rc_tree_name << " does not exist!";
  }
  auto& virtual_rc_tree = _virtual_rc_trees[rc_tree_name];

  std::map<std::string, double> all_node_delays;

  for (auto& [node_name, node] : virtual_rc_tree.get_nodes()) {
    all_node_delays[node_name] = node.delay();
  }

  return all_node_delays;
}

/**
 * @brief incremental propagation to update timing data.
 *
 * @return TimingEngine&
 */
TimingEngine& TimingEngine::incrUpdateTiming() {
  resetPathData();

  _incr_func.applyFwdQueue();

  StaGraph& the_graph = _ista->get_graph();

  the_graph.exec([](StaGraph* the_graph) -> unsigned {
    StaAnalyze analyze_path;
    unsigned is_ok = analyze_path(the_graph);

    StaApplySdc apply_sdc_post_analyze(
        StaApplySdc::PropType::kApplySdcPostProp);
    is_ok &= apply_sdc_post_analyze(the_graph);

    return is_ok;
  });

  _incr_func.applyBwdQueue();

  return *this;
}

/**
 * @brief get the clock domain.
 *
 * @return std::vector<StaClock*>
 */
std::vector<StaClock*> TimingEngine::getClockList() {
  std::vector<StaClock*> clock_list;
  auto& clocks = _ista->get_clocks();
  for (auto& clock : clocks) {
    clock_list.push_back(clock.get());
  }
  return clock_list;
}
/**
 * @brief set clock type to propgated clock.
 *
 * @param clock_name
 */
void TimingEngine::setPropagatedClock(const char* clock_name) {
  auto* the_clock = _ista->findClock(clock_name);
  the_clock->setPropagateClock();
}

/**
 * @brief judge whether the clock is propagated clock.
 *
 * @param clock_name
 * @return true
 * @return false
 */
bool TimingEngine::isPropagatedClock(const char* clock_name) {
  auto* the_clock = _ista->findClock(clock_name);
  return !the_clock->isIdealClockNetwork();
}

/**
 * @brief get one prop clock of clock net.
 *
 * @param clock_net
 * @return StaClock*
 */
StaClock* TimingEngine::getPropClockOfNet(Net* clock_net) {
  auto* driver = clock_net->getDriver();
  auto* driver_vertex = _ista->findVertex(driver);
  if (driver->isInout()) {
    driver_vertex = _ista->get_graph().getAssistant(driver_vertex);
  }
  auto* prop_clock = driver_vertex->getPropClock();
  return prop_clock;
}

/**
 * @brief get all clocks of the clock net.
 *
 * @param clock_net
 * @return std::unordered_set<StaClock*>
 */
std::unordered_set<StaClock*> TimingEngine::getPropClocksOfNet(Net* clock_net) {
  auto* driver = clock_net->getDriver();
  auto* driver_vertex = _ista->findVertex(driver);
  if (driver->isInout()) {
    driver_vertex = _ista->get_graph().getAssistant(driver_vertex);
  }

  auto analysis_mode = get_ista()->get_analysis_mode();
  if (analysis_mode == AnalysisMode::kMaxMin) {
    analysis_mode = AnalysisMode::kMin;  // choose any one.
  }
  auto prop_clocks =
      driver_vertex->getPropagatedClock(analysis_mode, TransType::kRise, false);
  return prop_clocks;
}

/**
 * @brief get master clock of generate clock.
 *
 * @param master_clock
 * @return std::string
 */
std::string TimingEngine::getMasterClockOfGenerateClock(
    const std::string& generate_clock) {
  auto& constrains = _ista->get_constrains();
  SdcClock* sdc_clock = constrains->findClock(generate_clock.c_str());
  std::string master_clock_name = sdc_clock->get_clock_name();

  do {
    auto* sdc_generate_clock = dynamic_cast<SdcGenerateCLock*>(sdc_clock);
    if (sdc_generate_clock) {
      master_clock_name = sdc_generate_clock->get_source_name();
      sdc_clock = constrains->findClock(master_clock_name.c_str());
    } else {
      break;
    }
  } while (true);

  return master_clock_name;
}

/**
 * @brief get the name of master clock, random one if more than one clock when
 * have clock mux.
 *
 * @param clock_net
 * @return std::string(master_clock_name)
 */
std::string TimingEngine::getMasterClockOfNet(Net* clock_net) {
  auto* clock = getPropClockOfNet(clock_net);
  const char* clock_name = clock->get_clock_name();

  return getMasterClockOfGenerateClock(clock_name);
}

/**
 * @brief get all master clocks of propagated clocks, may be more than one clock
 * when have clock mux.
 *
 * @param clock_net
 * @return std::vector<std::string>
 */
std::vector<std::string> TimingEngine::getMasterClocksOfNet(Net* clock_net) {
  std::vector<std::string> master_clocks;
  auto propagated_clocks = getPropClocksOfNet(clock_net);
  for (auto* propagated_clock : propagated_clocks) {
    std::string master_clock_name =
        getMasterClockOfGenerateClock(propagated_clock->get_clock_name());
    master_clocks.emplace_back(std::move(master_clock_name));
  }

  return master_clocks;
}

/**
 * @brief get all the clock net name list in design netlist.
 *
 * @return std::vector<std::string> clock_net_name_list
 */
std::vector<std::string> TimingEngine::getClockNetNameList() {
  std::vector<std::string> clock_net_name_list;

  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  Net* the_net;

  FOREACH_NET(design_netlist, the_net) {
    if (the_net->isClockNet()) {
      const char* net_name = the_net->get_name();
      clock_net_name_list.emplace_back(net_name);
    }
  }
  return clock_net_name_list;
}

/**
 * @brief check if the net is a clock net.
 *
 * @param net_name
 * @return true
 * @return false
 */
bool TimingEngine::isClockNet(const char* net_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* net = design_netlist->findNet(net_name);
  bool is_clock_net = net->isClockNet();
  return is_clock_net;
}

Net* TimingEngine::findNet(const char* net_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* net = design_netlist->findNet(net_name);
  return net;
}

/**
 * @brief Insert buffer need to change the netlist.
 *
 * @param instance_name
 */
void TimingEngine::insertBuffer(const char* instance_name) {
  auto* ista = _ista;

  if (!ista->isBuildGraph()) {
    return;
  }

  auto* design_netlist = ista->get_netlist();
  auto* instance = design_netlist->findInstance(instance_name);
  LOG_FATAL_IF(!instance);

  auto& the_graph = ista->get_graph();

  StaBuildGraph build_graph;
  build_graph.buildInst(&the_graph, instance);

  FlatSet<StaArc*> to_be_removed_arcs;
  Vector<Net*> buffer_nets;
  Pin* pin;
  FOREACH_INSTANCE_PIN(instance, pin) {
    auto* net = pin->get_net();
    if (!net) {
      continue;
    }
    if (pin->isOutput()) {
      auto load_pins = net->getLoads();
      for (auto* load_pin : load_pins) {
        auto load_vertex = the_graph.findVertex(load_pin);
        LOG_FATAL_IF(!load_vertex);

        auto& snk_arcs = (*load_vertex)->get_snk_arcs();
        for (auto* snk_arc : snk_arcs) {
          if (snk_arc->isNetArc()) {
            to_be_removed_arcs.insert(snk_arc);
          }
        }
      }
    } else {
      auto* driver_pin = net->getDriver();
      auto driver_vertex = the_graph.findVertex(driver_pin);
      LOG_FATAL_IF(!driver_vertex);

      auto& src_arcs = (*driver_vertex)->get_src_arcs();
      for (auto* src_arc : src_arcs) {
        if (src_arc->isNetArc()) {
          to_be_removed_arcs.insert(src_arc);
        }
      }
    }

    buffer_nets.push_back(net);
  }

  /*remove old arc*/
  for (auto* to_be_removed_arc : to_be_removed_arcs) {
    // LOG_INFO << "removed arc : " << Str::printf("%p", to_be_removed_arc);
    to_be_removed_arc->get_src()->removeSrcArc(to_be_removed_arc);
    to_be_removed_arc->get_snk()->removeSnkArc(to_be_removed_arc);
    the_graph.removeArc(to_be_removed_arc);
  }

  /*create new arc*/
  for (auto* net : buffer_nets) {
    build_graph.buildNet(&the_graph, net);
  }
}

/**
 * @brief Remove buffer need to change the netlist.
 *
 * @param instance_name
 */
void TimingEngine::removeBuffer(const char* instance_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* instance = design_netlist->findInstance(instance_name);
  LOG_FATAL_IF(!instance);
  auto& the_graph = ista->get_graph();

  FlatSet<StaArc*> to_be_changed_arcs;
  StaVertex* buffer_driver_vertex = nullptr;
  Net* buffer_driver_net = nullptr;

  Pin* pin;
  FOREACH_INSTANCE_PIN(instance, pin) {
    auto the_vertex = the_graph.findVertex(pin);
    LOG_FATAL_IF(!the_vertex);

    auto* net = pin->get_net();
    if (pin->isOutput()) {
      auto load_pins = net->getLoads();
      for (auto* load_pin : load_pins) {
        auto load_vertex = the_graph.findVertex(load_pin);
        LOG_FATAL_IF(!load_vertex);

        auto& snk_arcs = (*load_vertex)->get_snk_arcs();
        for (auto* snk_arc : snk_arcs) {
          if (snk_arc->isNetArc()) {
            to_be_changed_arcs.insert(snk_arc);
          }
        }
      }
      get_ista()->removeRcNet(net);
    } else {
      auto* driver_pin = net->getDriver();
      auto driver_vertex = the_graph.findVertex(driver_pin);
      LOG_FATAL_IF(!driver_vertex);
      buffer_driver_vertex = *driver_vertex;
      buffer_driver_net = net;

      auto to_be_removed_arcs = (*driver_vertex)->getSnkArc(*the_vertex);
      auto* to_be_removed_arc = to_be_removed_arcs.front();  // only one.

      to_be_removed_arc->get_src()->removeSrcArc(to_be_removed_arc);
      to_be_removed_arc->get_snk()->removeSnkArc(to_be_removed_arc);
      the_graph.removeArc(to_be_removed_arc);
      initRcTree(net);
    }

    the_graph.removePinVertex(pin, *the_vertex);
  }

  /*change buffer load arc and net*/
  for (auto* to_be_changed_arc : to_be_changed_arcs) {
    to_be_changed_arc->set_src(buffer_driver_vertex);
    buffer_driver_vertex->addSrcArc(to_be_changed_arc);
    dynamic_cast<StaNetArc*>(to_be_changed_arc)->set_net(buffer_driver_net);
  }
}

/**
 * @brief Change the size or level of an existing instance, e.g., NAND2_X2 to
 * NAND2_X3. The instance's logic function and topology is guaranteed to be
 * the same, along with the currently-connected nets. However, the pin
 * capacitances of the new cell type might be different.
 *
 */
void TimingEngine::repowerInstance(const char* instance_name,
                                   const char* cell_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* instance = design_netlist->findInstance(instance_name);
  auto& the_graph = ista->get_graph();

  auto* inst_liberty_cell = ista->findLibertyCell(cell_name);

  Pin* pin;
  FOREACH_INSTANCE_PIN(instance, pin) {
    auto* liberty_port =
        inst_liberty_cell->get_cell_port_or_port_bus(pin->get_name());
    LOG_FATAL_IF(liberty_port->isLibertyPortBus());
    pin->set_cell_port(dynamic_cast<LibPort*>(liberty_port));
    if (pin->isInput()) {
      auto the_vertex = the_graph.findVertex(pin);
      LOG_FATAL_IF(!the_vertex);
      FOREACH_SRC_ARC((*the_vertex), the_arc) {
        auto* the_inst_arc = dynamic_cast<StaInstArc*>(the_arc);
        if (the_inst_arc) {
          auto* origin_lib_arc = the_inst_arc->get_lib_arc();

          auto lib_arc_set = inst_liberty_cell->findLibertyArcSet(
              origin_lib_arc->get_src_port(), origin_lib_arc->get_snk_port(),
              the_inst_arc->getTimingType());

          LOG_FATAL_IF(!lib_arc_set);

          auto* new_lib_arc = (*lib_arc_set)->front();
          the_inst_arc->set_lib_arc(new_lib_arc);
        }
      }
    }
  }

  instance->set_inst_cell(inst_liberty_cell);
}

/**
 * @brief move the instance to a new location.
 *
 * @param instance_name the moved instance name.
 * @param update_level the propgate end level minus current prop start level.
 * @param prop_type bwd or fwd or both incr update.
 */
void TimingEngine::moveInstance(const char* instance_name,
                                std::optional<unsigned> update_level,
                                PropType prop_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* instance = design_netlist->findInstance(instance_name);
  auto& the_graph = ista->get_graph();

  Pin* pin;
  FOREACH_INSTANCE_PIN(instance, pin) {
    auto the_vertex = the_graph.findVertex(pin);
    LOG_FATAL_IF(!the_vertex);
    if (pin->isInput()) {
      if (prop_type == PropType::kFwd || prop_type == PropType::kFwdAndBwd) {
        FOREACH_SNK_ARC((*the_vertex), the_arc) {
          auto* src_vertex = the_arc->get_src();
          _incr_func.insertFwdQueue(src_vertex);

          StaResetPropagation reset_fwd_prop;
          reset_fwd_prop.set_incr_func(&_incr_func);
          if (update_level) {
            reset_fwd_prop.set_max_min_level((*the_vertex)->get_level() +
                                             (*update_level));
          }
          src_vertex->exec(reset_fwd_prop);
        }
      }
    } else {
      if (prop_type == PropType::kBwd || prop_type == PropType::kFwdAndBwd) {
        FOREACH_SRC_ARC((*the_vertex), the_arc) {
          auto* snk_vertex = the_arc->get_snk();
          _incr_func.insertBwdQueue(snk_vertex);

          StaResetPropagation reset_bwd_prop;
          reset_bwd_prop.set_is_bwd();
          reset_bwd_prop.set_incr_func(&_incr_func);
          if (update_level && ((*the_vertex)->get_level() > (*update_level))) {
            reset_bwd_prop.set_max_min_level((*the_vertex)->get_level() -
                                             (*update_level));
          }
          snk_vertex->exec(reset_bwd_prop);
        }
      }
    }
  }
}

/**
 * @brief set net delay(fs).
 *
 * @param wl
 * @param ucap
 * @param net_name
 * @param load_pin_name
 * @param mode_trans
 */
void TimingEngine::setNetDelay(double wl, double ucap, const char* net_name,
                               const char* load_pin_name,
                               ModeTransPair mode_trans) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* net = design_netlist->findNet(net_name);
  auto& the_graph = ista->get_graph();

  double net_linear_delay = wl * ucap;  // fs

  DesignObject* pin_port = net->get_pin_port(load_pin_name);
  auto the_vertex = the_graph.findVertex(pin_port);
  LOG_FATAL_IF(!the_vertex);

  LOG_INFO << "set net delay(fs) start";
  FOREACH_SNK_ARC((*the_vertex), the_arc) {
    auto* net_arc = dynamic_cast<StaNetArc*>(the_arc);
    if (Str::equal(net_arc->get_net()->get_name(), net_name)) {
      net_arc->resetArcDelayBucket();
      auto* arc_delay =
          new StaArcDelayData(mode_trans.first, mode_trans.second, net_arc,
                              static_cast<int>(net_linear_delay));
      net_arc->addData(arc_delay);
    }
  }
  LOG_INFO << "set net delay(fs) end";
}

/**
 * @brief report instance delay.
 *
 * @param inst_name
 * @param src_port_name
 * @param snk_port_name
 * @param mode
 * @param trans_type
 * @return double instance delay.
 */
double TimingEngine::getInstDelay(const char* inst_name,
                                  const char* src_port_name,
                                  const char* snk_port_name, AnalysisMode mode,
                                  TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* instance = design_netlist->findInstance(inst_name);

  auto& the_graph = ista->get_graph();

  Pin* pin;
  int arc_delay = 0;
  FOREACH_INSTANCE_PIN(instance, pin) {
    if (pin->isInput()) {
      auto the_vertex = the_graph.findVertex(pin);
      LOG_FATAL_IF(!the_vertex);
      FOREACH_SRC_ARC((*the_vertex), the_arc) {
        auto* src_vertex = the_arc->get_src();
        auto* snk_vertex = the_arc->get_snk();
        std::string src_vertex_name = src_vertex->getName();
        std::string snk_vertex_name = snk_vertex->getName();
        auto* instance_arc = dynamic_cast<StaInstArc*>(the_arc);

        if (src_vertex_name == src_port_name &&
            snk_vertex_name == snk_port_name) {
          arc_delay = instance_arc->get_arc_delay(mode, trans_type);
          break;
        }
      }
    }
  }
  return FS_TO_NS(arc_delay);
}

/**
 * @brief the worst arc delay for the specified instance.
 *
 * @param inst_name
 * @param mode
 * @param trans_type
 * @return double (the worst arc delay for the specified instance.)
 */
double TimingEngine::getInstWorstArcDelay(const char* inst_name,
                                          AnalysisMode mode,
                                          TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* instance = design_netlist->findInstance(inst_name);

  auto& the_graph = ista->get_graph();

  Pin* pin;
  int worst_arc_delay = 0;
  FOREACH_INSTANCE_PIN(instance, pin) {
    if (pin->isInput()) {
      auto the_vertex = the_graph.findVertex(pin);
      LOG_FATAL_IF(!the_vertex);
      FOREACH_SRC_ARC((*the_vertex), the_arc) {
        auto* instance_arc = dynamic_cast<StaInstArc*>(the_arc);
        int arc_delay = instance_arc->get_arc_delay(mode, trans_type);
        if (worst_arc_delay < arc_delay) {
          worst_arc_delay = arc_delay;
        }
      }
    }
  }
  return FS_TO_NS(worst_arc_delay);
}
/**
 * @brief report net delay.
 *
 * @param net_name
 * @param load_pin_name
 * @param mode
 * @param trans_type
 * @return double
 */
double TimingEngine::getNetDelay(const char* net_name,
                                 const char* load_pin_name, AnalysisMode mode,
                                 TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  auto* net = design_netlist->findNet(net_name);
  Vector<DesignObject*> pin_ports = net->get_pin_ports();
  auto& the_graph = ista->get_graph();

  DesignObject* pin_port = net->get_pin_port(load_pin_name);
  auto the_vertex = the_graph.findVertex(pin_port);
  LOG_FATAL_IF(!the_vertex);

  int net_delay = 0;
  FOREACH_SNK_ARC((*the_vertex), the_arc) {
    auto* net_arc = dynamic_cast<StaNetArc*>(the_arc);
    if (Str::equal(net_arc->get_net()->get_name(), net_name)) {
      net_delay = net_arc->get_arc_delay(mode, trans_type);
      break;
    }
  }
  return FS_TO_NS(net_delay);
}

/**
 * @brief report slew.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return double slew.
 */
double TimingEngine::getSlew(const char* pin_name, AnalysisMode mode,
                             TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  auto* pin = match_pins.front();

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(pin);
  LOG_FATAL_IF(!the_vertex);

  auto slew = (*the_vertex)->getSlewNs(mode, trans_type);
  return slew ? *slew : 0.0;
}

/**
 * @brief report arrive time.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return double arrive_time.
 */
std::optional<double> TimingEngine::getAT(const char* pin_name,
                                          AnalysisMode mode,
                                          TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();

  // sjc change.
  DesignObject* port_or_pin;
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  if (match_pins.empty()) {  // port case.
    port_or_pin = design_netlist->findPort(pin_name);
  } else {
    port_or_pin = match_pins.front();
  }

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(port_or_pin);
  LOG_FATAL_IF(!the_vertex);

  auto arrive_time = (*the_vertex)->getArriveTime(mode, trans_type);

  return arrive_time ? std::optional<double>(FS_TO_NS(*arrive_time))
                     : std::nullopt;
}

/**
 * @brief report clock arrive time.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return double clock_arrive_time.
 */
std::optional<double> TimingEngine::getClockAT(
    const char* pin_name, AnalysisMode mode, TransType trans_type,
    std::optional<std::string> clock_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();

  DesignObject* port_or_pin;
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  if (match_pins.empty()) {  // port case.
    port_or_pin = design_netlist->findPort(pin_name);
  } else {
    port_or_pin = match_pins.front();
  }

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(port_or_pin);
  LOG_FATAL_IF(!the_vertex);

  auto clock_arrive_time =
      (*the_vertex)
          ->getClockArriveTime(mode, trans_type, std::move(clock_name));

  return clock_arrive_time ? std::optional<double>(FS_TO_NS(*clock_arrive_time))
                           : std::nullopt;
}

/**
 * @brief report required time.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return double
 */
std::optional<double> TimingEngine::getRT(const char* pin_name,
                                          AnalysisMode mode,
                                          TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();

  // sjc change.
  DesignObject* port_or_pin;
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  if (match_pins.empty()) {  // port case.
    port_or_pin = design_netlist->findPort(pin_name);
  } else {
    port_or_pin = match_pins.front();
  }

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(port_or_pin);
  LOG_FATAL_IF(!the_vertex);

  auto req_time = (*the_vertex)->getReqTime(mode, trans_type);
  if (req_time) {
    return FS_TO_NS(*req_time);
  }
  return std::nullopt;
}

/**
 * @brief get the clock of the pin.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return StaClock*
 */
StaClock* TimingEngine::getPropClock(const char* pin_name, AnalysisMode mode,
                                     TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  auto* pin = match_pins.front();

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(pin);
  LOG_FATAL_IF(!the_vertex);

  auto* prop_clock = (*the_vertex)->getPropClock(mode, trans_type);

  return prop_clock;
}

/**
 * @brief  Get the violated StaSeqPathDatas of the two specified sinks that form
 * the timing path.
 *
 * @param pin1_name.(end vertex)
 * @param pin2_name.(start vertex)
 * @param mode
 * @return std::priority_queue<StaSeqPathData*, std::vector<StaSeqPathData*>,
 * decltype(cmp)>
 */
std::priority_queue<StaSeqPathData*, std::vector<StaSeqPathData*>,
                    decltype(seq_data_cmp)>
TimingEngine::getViolatedSeqPathsBetweenTwoSinks(const char* pin1_name,
                                                 const char* pin2_name,
                                                 AnalysisMode mode) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins1 =
      design_netlist->findObj(pin1_name, false, false);
  auto* pin1 = match_pins1.front();
  auto& the_graph = ista->get_graph();

  auto the_vertex1 = the_graph.findVertex(pin1);
  LOG_FATAL_IF(!the_vertex1);

  std::vector<DesignObject*> match_pins2 =
      design_netlist->findObj(pin2_name, false, false);
  auto* pin2 = match_pins2.front();

  auto the_vertex2 = the_graph.findVertex(pin2);
  LOG_FATAL_IF(!the_vertex2);

  return _ista->getViolatedSeqPathsBetweenTwoSinks(*the_vertex1, *the_vertex2,
                                                   mode);
}

/**
 * @brief Get the slack of the two specified sinks that form the timing
 * path.
 *
 * @param clock_pin1_name.(end vertex)
 * @param clock_pin2_name.(start vertex)
 * @param mode
 * @return double(ns)
 */
std::optional<double> TimingEngine::getWorstSlackBetweenTwoSinks(
    const char* clock_pin1_name, const char* clock_pin2_name,
    AnalysisMode mode) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins1 =
      design_netlist->findObj(clock_pin1_name, false, false);
  auto* pin1 = match_pins1.front();
  auto& the_graph = ista->get_graph();

  auto the_vertex1 = the_graph.findVertex(pin1);
  LOG_FATAL_IF(!the_vertex1);

  std::vector<DesignObject*> match_pins2 =
      design_netlist->findObj(clock_pin2_name, false, false);
  auto* pin2 = match_pins2.front();

  auto the_vertex2 = the_graph.findVertex(pin2);
  LOG_FATAL_IF(!the_vertex2);

  return _ista->getWorstSlackBetweenTwoSinks(*the_vertex1, *the_vertex2, mode);
}

/**
 * @brief report slack.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return double
 */
std::optional<double> TimingEngine::getSlack(const char* pin_name,
                                             AnalysisMode mode,
                                             TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();

  // sjc change.
  DesignObject* port_or_pin;
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  if (match_pins.empty()) {  // port case.
    port_or_pin = design_netlist->findPort(pin_name);
  } else {
    port_or_pin = match_pins.front();
  }

  // auto* pin = match_pins.front();

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(port_or_pin);
  LOG_FATAL_IF(!the_vertex);

  auto slack = (*the_vertex)->getSlackNs(mode, trans_type);

  return slack;
}

/**
 * @brief report the worst slack from all end vertexs.
 *
 * @param mode
 * @param trans_type
 * @return worst_vertex
 * @return worst_slack
 */
void TimingEngine::getWorstSlack(AnalysisMode mode, TransType trans_type,
                                 StaVertex*& worst_vertex,
                                 std::optional<double>& worst_slack) {
  auto* ista = _ista;
  StaGraph* the_graph = &(ista->get_graph());
  StaVertex* vertex;
  FOREACH_END_VERTEX(the_graph, vertex) {
    if (auto vertex_slack = vertex->getSlack(mode, trans_type); vertex_slack) {
      auto slack = FS_TO_NS(*vertex_slack);

      if (!worst_slack || slack < *worst_slack) {
        worst_slack = slack;
        worst_vertex = vertex;
      }
    }
  }
}

/**
 * @brief report network latency.
 *
 * @param clock_pin_name
 * @param mode
 * @param trans_type
 * @return double network latency.
 */
double TimingEngine::getClockNetworkLatency(const char* clock_pin_name,
                                            AnalysisMode mode,
                                            TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_clock_pins =
      design_netlist->findPin(clock_pin_name, false, false);
  auto* clock_pin = match_clock_pins.front();

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(clock_pin);
  LOG_FATAL_IF(!the_vertex);

  int64_t network_latency = (*the_vertex)->getNetworkLatency(mode, trans_type);
  return FS_TO_NS(network_latency);
}

/**
 * @brief report skew.
 *
 * @param src_clock_pin_name
 * @param snk_clock_pin_name
 * @param mode
 * @param trans_type
 * @return double skew.
 */
double TimingEngine::getClockSkew(const char* src_clock_pin_name,
                                  const char* snk_clock_pin_name,
                                  AnalysisMode mode, TransType trans_type) {
  double src_network_latency =
      getClockNetworkLatency(src_clock_pin_name, mode, trans_type);
  double snk_network_latency =
      getClockNetworkLatency(snk_clock_pin_name, mode, trans_type);
  double skew = snk_network_latency - src_network_latency;
  return skew;
}

/**
 * @brief report the inst's pin capacitance (inst_name:pin_name).
 *
 * @param pin_name
 * @return double
 */
double TimingEngine::getInstPinCapacitance(const char* pin_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  auto* pin = match_pins.front();
  auto* pin1 = dynamic_cast<Pin*>(pin);
  double cap = pin1->cap();
  return cap;
}

/**
 * @brief report the inst's pin capacitance by the specified mode and
 * trans_type (inst_name:pin_name).
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return double
 */
double TimingEngine::getInstPinCapacitance(const char* pin_name,
                                           AnalysisMode mode,
                                           TransType trans_type) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  auto* pin = match_pins.front();
  auto* pin1 = dynamic_cast<Pin*>(pin);
  double cap = pin1->cap(mode, trans_type);
  return cap;
}

/**
 * @brief report the liberty cell's pin capacitance
 * (cell_name:pin_name).
 *
 * @param cell_pin_name
 * @return double
 */
double TimingEngine::getLibertyCellPinCapacitance(const char* cell_pin_name) {
  const char* sep = "/:";
  auto [cell_name, pin_name] = Str::splitTwoPart(cell_pin_name, sep);
  LibCell* lib_cell = _ista->findLibertyCell(cell_name.c_str());
  auto& cell_ports = lib_cell->get_cell_ports();

  double cap = 0;
  for (auto& cell_port : cell_ports) {
    auto port_name = cell_port->get_port_name();
    if (Str::equal(port_name, pin_name.c_str())) {
      cap = cell_port->get_port_cap();
    }
  }

  return cap;
}

/**
 * @brief get the input pins' name of the liberty cell.
 *
 * @param cell_name
 * @return std::vector<std::string>
 */
std::vector<std::string> TimingEngine::getLibertyCellInputpin(
    const char* cell_name) {
  LibCell* lib_cell = _ista->findLibertyCell(cell_name);
  auto& cell_ports = lib_cell->get_cell_ports();

  std::vector<std::string> input_pin_names;
  for (auto& cell_port : cell_ports) {
    if (cell_port->isInput()) {
      auto pin_name = cell_port->get_port_name();
      input_pin_names.emplace_back(pin_name);
    }
  }

  return input_pin_names;
}

/**
 * @brief  get the staclock.
 *
 * @param clock_pin_name
 * @return StaClock*
 */
StaClock* TimingEngine::getPropClock(const char* clock_pin_name) {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_clock_pins =
      design_netlist->findPin(clock_pin_name, false, false);
  auto* clock_pin = match_clock_pins.front();

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(clock_pin);
  LOG_FATAL_IF(!the_vertex);

  auto* sta_clock = (*the_vertex)->getPropClock();

  return sta_clock;
}

/**
 * @brief Get the path all driver vertex and load delay.
 *
 * @param path_data
 * @return std::vector<PathNet>
 */
std::vector<TimingEngine::PathNet> TimingEngine::getPathDriverVertexs(
    StaSeqPathData* path_data) {
  auto path_datas = path_data->getPathDelayData();
  std::vector<PathNet> path_driver_vertexs;

  while (!path_datas.empty()) {
    auto* path_data = path_datas.top();
    auto* path_vertex = path_data->get_own_vertex();
    auto* the_obj = path_vertex->get_design_obj();
    if (the_obj->get_net()->getDriver() == the_obj) {
      path_datas.pop();
      auto* load_path_data = path_datas.top();
      auto* load_vertex = load_path_data->get_own_vertex();
      auto delay =
          path_data->get_arrive_time() - load_path_data->get_arrive_time();
      path_driver_vertexs.emplace_back(
          PathNet{path_vertex, load_vertex, FS_TO_NS(delay)});
    }
    path_datas.pop();
  }

  return path_driver_vertexs;
}

/**
 * @brief Get the fanout num of driver vertex.
 *
 * @param driver_vertex
 * @return int
 */
int TimingEngine::getFanoutNumOfDriverVertex(StaVertex* driver_vertex) {
  return static_cast<int>(
      driver_vertex->get_design_obj()->get_net()->getFanouts());
}

/**
 * @brief Get the fanout vertex of driver.
 *
 * @param driver_vertex
 * @return std::vector<StaVertex*>
 */
std::vector<StaVertex*> TimingEngine::getFanoutVertexs(
    StaVertex* driver_vertex) {
  std::vector<StaVertex*> fanout_vertexs;

  for (auto* src_arc : driver_vertex->get_src_arcs()) {
    fanout_vertexs.push_back(src_arc->get_snk());
  }
  return fanout_vertexs;
}

/**
 * @brief judege whether the instance is sequential cell.
 *
 * @param instance_name
 * @return unsigned
 */
unsigned TimingEngine::isSequentialCell(const char* instance_name) {
  auto* design_netlist = _ista->get_netlist();
  auto* design_instance = design_netlist->findInstance(instance_name);

  Pin* design_pin;
  FOREACH_INSTANCE_PIN(design_instance, design_pin) {
    auto* the_vertex = _ista->findVertex(design_pin);
    if (the_vertex->is_clock()) {
      return 1;
    }
  }

  return 0;
}

/**
 * @brief get the cell type of the cell.
 *
 * @param cell_name
 * @return std::string
 */
std::string TimingEngine::getCellType(const char* cell_name) {
  std::string cell_type = "Others";
  auto* liberty_cell = _ista->findLibertyCell(cell_name);
  //   LOG_FATAL_IF(!liberty_cell);
  if (liberty_cell == nullptr) {
    LOG_WARNING << "Can not find cell name = " << cell_name;
  } else {
    if (liberty_cell->isBuffer()) {
      cell_type = "Buffer";
    } else if (liberty_cell->isInverter()) {
      cell_type = "Inverter";
    } else if (liberty_cell->isICG()) {
      cell_type = "ICG";
    } else {
      cell_type = "Others";
    }
  }

  return cell_type;
}

/**
 * @brief get the cell area of the cell.
 *
 * @param cell_name
 * @return double(unit:um2)
 */
double TimingEngine::getCellArea(const char* cell_name) {
  auto* liberty_cell = _ista->findLibertyCell(cell_name);
  if (liberty_cell == nullptr) {
    LOG_WARNING << "Can not find cell name = " << cell_name;
    return 0;
  }
  double cell_area = liberty_cell->get_cell_area();
  return cell_area;
}

/**
 * @brief judge whether the pin is clock pin.
 *
 * @param pin_name
 * @return unsigned
 */
unsigned TimingEngine::isClock(const char* pin_name) const {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_pins =
      design_netlist->findPin(pin_name, false, false);
  auto* pin = match_pins.front();

  auto& the_graph = ista->get_graph();

  auto the_vertex = the_graph.findVertex(pin);
  LOG_FATAL_IF(!the_vertex);

  auto is_clock = (*the_vertex)->is_clock();
  return is_clock;
}

/**
 * @brief judge whether the pin is load.
 *
 * @param pin_name
 * @return unsigned
 */
unsigned TimingEngine::isLoad(const char* pin_name) const {
  auto* ista = _ista;
  auto* design_netlist = ista->get_netlist();
  std::vector<DesignObject*> match_objs =
      design_netlist->findObj(pin_name, false, false);
  auto* obj = match_objs.front();

  unsigned isLoad = 0;
  if (obj->isPort()) {
    auto* port = dynamic_cast<Port*>(obj);
    if (port->isOutput()) {
      isLoad = 1;
    }
  } else {  // for pin.
    auto* pin = dynamic_cast<Pin*>(obj);
    if (pin->isInput()) {
      isLoad = 1;
    }
  }

  return isLoad;
}

/**
 * @brief check the port cap whether meet the limit.
 *
 * @param pin_name
 * @param mode
 * @param trans_type
 * @param capacitance
 * @param limit may be has not limit.
 * @param slack
 */
void TimingEngine::validateCapacitance(const char* pin_name, AnalysisMode mode,
                                       TransType trans_type,
                                       double& capacitance,
                                       std::optional<double>& limit,
                                       double& slack) {
  auto* the_vertex = findVertex(pin_name);
  if (!the_vertex) {
    return;
  }

  // only consider driver cap.
  capacitance = _ista->getVertexCapacitance(the_vertex, mode, trans_type);
  limit = _ista->getVertexCapacitanceLimit(the_vertex, mode, trans_type);

  if (limit) {
    slack = *limit - capacitance;
  }
}

/**
 * @brief check the real fanout nums and the limit fanout nums in liberty,
 * calculate the fanout slack.
 *
 * @param pin_name
 * @param mode
 * @return fanout
 * @return limit
 * @return slack
 */
void TimingEngine::validateFanout(const char* pin_name, AnalysisMode mode,
                                  double& fanout, std::optional<double>& limit,
                                  double& slack) {
  auto* the_vertex = findVertex(pin_name);
  if (!the_vertex) {
    return;
  }

  // only consider driver pin.
  if (auto* obj = the_vertex->get_design_obj();
      !obj->isPin() || !obj->isOutput()) {
    return;
  }

  fanout = the_vertex->get_src_arcs().size();
  limit = _ista->getDriverVertexFanoutLimit(the_vertex, mode);

  if (limit) {
    slack = *limit - fanout;
  }
}

/**
 * @brief check the real slew and the limit slew in liberty,
 * calculate the slew slack.
 * @param pin_name
 * @param mode
 * @param trans_type
 * @return slew
 * @return limit
 * @return slack
 */
void TimingEngine::validateSlew(const char* pin_name, AnalysisMode mode,
                                TransType trans_type, double& slew,
                                std::optional<double>& limit, double& slack) {
  auto* the_vertex = findVertex(pin_name);
  if (!the_vertex) {
    return;
  }

  auto vertex_slew = the_vertex->getSlewNs(mode, trans_type);
  slew = vertex_slew ? *vertex_slew : 0.0;
  limit = _ista->getVertexSlewLimit(the_vertex, mode, trans_type);

  if (limit && vertex_slew) {
    slack = *limit - slew;
  }
}

}  // namespace ista