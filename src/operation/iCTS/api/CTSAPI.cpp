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
#include "CTSAPI.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <ranges>
#include <unordered_map>

// #include "Balancer.h"
#include "BEAT.hh"
#include "CtsCellLib.hh"
#include "CtsConfig.h"
#include "CtsDBWrapper.h"
#include "CtsDesign.hh"
#include "CtsReport.h"
#include "Evaluator.h"
#include "GDSPloter.h"
#include "JsonParser.h"
#include "Node.hh"
#include "Operator.h"
// #include "Optimizer.h"
#include "Pin.hh"
// #include "RTAPI.hpp"
#include "Router.h"
// #include "Synthesis.h"
#include "TimingPropagator.hh"
#include "ToApi.hpp"
#include "TreeBuilder.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "idm.h"
#include "log/Log.hh"
#include "model/ModelFactory.h"
#include "model/mplHelper/MplHelper.h"
#include "model/python/PyToolBase.h"
#include "salt/base/flute.h"
#include "salt/salt.h"
#ifdef PY_MODEL
#include "PyModel.h"
#endif
namespace icts {
#define DBCONFIG (dmInst->get_config())

CTSAPI* CTSAPI::_cts_api_instance = nullptr;

CTSAPI& CTSAPI::getInst()
{
  if (_cts_api_instance == nullptr) {
    _cts_api_instance = new CTSAPI();
  }
  return *_cts_api_instance;
}
void CTSAPI::destroyInst()
{
  if (_cts_api_instance != nullptr) {
    delete _cts_api_instance;
    _cts_api_instance = nullptr;
  }
}
// open API

void CTSAPI::runCTS()
{
  ieda::Stats stats;
  readData();
  routing();
  // synthesis();
  evaluate();
  // balance();
  // optimize();
  LOG_INFO << "Flow memory usage " << stats.memoryDelta() << "MB";
  LOG_INFO << "Flow elapsed time " << stats.elapsedRunTime() << "s";
  writeGDS();
  // writeDB();
}

void CTSAPI::writeDB()
{
  _db_wrapper->writeDef();
}

void CTSAPI::writeGDS()
{
  GDSPloter plotter;
  plotter.plotDesign();
  plotter.plotFlyLine();
}

void CTSAPI::report(const std::string& save_dir)
{
  if (_timing_engine == nullptr) {
    startDbSta();
  }
  if (_evaluator == nullptr) {
    _evaluator = new Evaluator();
    _evaluator->init();
    _evaluator->evaluate();
  }
  _evaluator->statistics(save_dir);
}

// flow API
void CTSAPI::resetAPI()
{
  _config = nullptr;
  _design = nullptr;
  _db_wrapper = nullptr;
  _report = nullptr;
  _log_ofs = nullptr;
  _libs = nullptr;
  // _synth = nullptr;
  _evaluator = nullptr;
  // _balancer = nullptr;
  _model_factory = nullptr;
  _timing_engine = nullptr;
}

void CTSAPI::init(const std::string& config_file)
{
  resetAPI();
  _config = new CtsConfig();
  JsonParser::getInstance().parse(config_file, _config);

  _design = new CtsDesign();
  if (dmInst->get_idb_builder()) {
    _db_wrapper = new CtsDBWrapper(dmInst->get_idb_builder());
  } else {
    LOG_FATAL << "idb builder is null";
  }
  _report = new CtsReportTable("iCTS");
  _log_ofs = new std::ofstream(_config->get_log_file(), std::ios::out | std::ios::trunc);
  _libs = new CtsLibs();

  // _synth = new Synthesis();
  _evaluator = new Evaluator();
  // _balancer = new Balancer();
  _model_factory = new ModelFactory();
#if (defined PY_MODEL) && (defined USE_EXTERNAL_MODEL)
  auto external_models = _config->get_external_models();
  for (auto [net_name, model_path] : external_models) {
    auto* model = _model_factory->pyLoad(model_path);
    _libs->insertModel(net_name, model);
  }
#endif
  startDbSta();
  TimingPropagator::init();
}

void CTSAPI::readData()
{
  if (_config->is_use_netlist()) {
    auto& net_list = _config->get_clock_netlist();
    for (auto& net : net_list) {
      _design->addClockNetName(net.first, net.second);
    }
  } else {
    readClockNetNames();
  }

  _db_wrapper->read();
}

void CTSAPI::routing()
{
  ieda::Stats stats;
  Router router;
  router.init();
  router.build();
  router.update();
  LOG_INFO << "Routing memory usage " << stats.memoryDelta() << "MB";
  LOG_INFO << "Routing elapsed time " << stats.elapsedRunTime() << "s";
}

// void CTSAPI::synthesis()
// {
//   ieda::Stats stats;
//   _synth->init();
//   _synth->insertCtsNetlist();
//   _synth->update();

//   LOG_INFO << "Synthesis memory usage " << stats.memoryDelta() << "MB";
//   LOG_INFO << "Synthesis elapsed time " << stats.elapsedRunTime() << "s";
// }

void CTSAPI::evaluate()
{
  ieda::Stats stats;
  if (_timing_engine == nullptr) {
    startDbSta();
  }

  _evaluator->init();
  // _evaluator->plotNet("sdram_clk_o", "sdram_clk_o.gds");
  _evaluator->evaluate();
  // _evaluator->plotPath("u0_soc_top/u0_sdram_axi/u_core/sample_data0_q_reg_0_");
  LOG_INFO << "Evaluate memory usage " << stats.memoryDelta() << "MB";
  LOG_INFO << "Evaluate elapsed time " << stats.elapsedRunTime() << "s";
}

// void CTSAPI::balance()
// {
//   auto* config = CTSAPIInst.get_config();
//   auto router_type = config->get_router_type();
//   // if (router_type != "SlewAware") {
//   //   // TBD
//   //   LOG_INFO << "Balance is only supported for SlewAware";
//   //   return;
//   // }
//   if (router_type != "SlewAware") {
//     // TBD
//     LOG_INFO << "Balance is only supported for SlewAware";
//     return;
//   }
//   ieda::Stats stats;
//   _balancer->init();
//   _balancer->balance();
//   _synth->incrementalInsertCtsNetlist();
//   _synth->update();
//   _evaluator->init();
//   _evaluator->evaluate();
//   LOG_INFO << "Balance memory usage " << stats.memoryDelta() << "MB";
//   LOG_INFO << "Balance elapsed time " << stats.elapsedRunTime() << "s";
// }

void CTSAPI::optimize()
{
  // ieda::Stats stats;
  // Optimizer optimizer;
  // auto clk_nets = _design->get_nets();
  // ToApiInst.initCTSDesignViolation(dmInst->get_idb_builder(), _timing_engine);
  // optimizer.optimize(clk_nets.begin(), clk_nets.end());
  // optimizer.update();
  // LOG_INFO << "Optimizer memory usage " << stats.memoryDelta() << "MB";
  // LOG_INFO << "Optimizer elapsed time " << stats.elapsedRunTime() << "s";
}

// iSTA
void CTSAPI::dumpVertexData(const std::vector<std::string>& vertex_names) const
{
  _timing_engine->get_ista()->dumpVertexData(vertex_names);
}

double CTSAPI::getClockUnitCap(const std::optional<icts::LayerPattern>& layer_pattern) const
{
  std::optional<double> width = std::nullopt;
  auto pattern = layer_pattern.value_or(icts::LayerPattern::kNone);
  switch (pattern) {
    case icts::LayerPattern::kH:
      return getStaDbAdapter()->getCapacitance(_config->get_h_layer(), 1.0, width);
      break;
    case icts::LayerPattern::kV:
      return getStaDbAdapter()->getCapacitance(_config->get_v_layer(), 1.0, width);
      break;
    case icts::LayerPattern::kNone:
      return getStaDbAdapter()->getCapacitance(_config->get_routing_layers().back(), 1.0, width);
    default:
      LOG_ERROR << "Unknown layer pattern";
      break;
  }
  return 0.0;
}

double CTSAPI::getClockUnitRes(const std::optional<icts::LayerPattern>& layer_pattern) const
{
  std::optional<double> width = std::nullopt;
  auto pattern = layer_pattern.value_or(icts::LayerPattern::kNone);
  switch (pattern) {
    case icts::LayerPattern::kH:
      return getStaDbAdapter()->getResistance(_config->get_h_layer(), 1.0, width);
      break;
    case icts::LayerPattern::kV:
      return getStaDbAdapter()->getResistance(_config->get_v_layer(), 1.0, width);
      break;
    case icts::LayerPattern::kNone:
      return getStaDbAdapter()->getResistance(_config->get_routing_layers().back(), 1.0, width);
    default:
      LOG_ERROR << "Unknown layer pattern";
      break;
  }
  return 0.0;
}

double CTSAPI::getSinkCap(icts::CtsInstance* sink) const
{
  auto* load_pin = sink->get_load_pin();
  return getSinkCap(load_pin->get_full_name());
}

double CTSAPI::getSinkCap(const std::string& load_pin_full_name) const
{
  return _timing_engine->reportInstPinCapacitance(load_pin_full_name.c_str());
}

bool CTSAPI::isFlipFlop(const std::string& inst_name) const
{
  return _timing_engine->isSequentialCell(inst_name.c_str());
}

bool CTSAPI::isClockNet(const std::string& net_name) const
{
  return findStaNet(net_name)->isClockNet();
}

void CTSAPI::startDbSta()
{
  _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(_timing_engine->get_ista());
  db_adapter->set_idb(_db_wrapper->get_idb());
  _timing_engine->set_db_adapter(std::move(db_adapter));
  readSTAFile();
  _timing_engine->get_ista()->set_n_worst_path_per_clock(10);
  // convertDBToTimingEngine();
  // setPropagateClock();
}

void CTSAPI::readClockNetNames() const
{
  _timing_engine->updateTiming();
  auto* netlist = _timing_engine->get_netlist();
  ista::Net* sta_net = nullptr;
  FOREACH_NET(netlist, sta_net)
  {
    if (sta_net->isClockNet()) {
      auto* sta_clock = _timing_engine->getPropClockOfNet(sta_net);
      // HARD CODE debug
      if (std::string(sta_clock->get_clock_name()) == "CLK_spi_clk") {
        continue;
      }
      _design->addClockNetName(sta_clock->get_clock_name(), sta_net->get_name());
      LOG_INFO << "Clock: " << sta_clock->get_clock_name() << " have net: " << sta_net->get_name();
      CTSAPIInst.saveToLog("Clock: ", sta_clock->get_clock_name(), " have net: ", sta_net->get_name());
    }
  }
}

void CTSAPI::setPropagateClock()
{
  auto* ista = _timing_engine->get_ista();
  auto& the_constrain = ista->get_constrains();
  auto& the_sdc_clocks = the_constrain->get_sdc_clocks();
  for (auto& the_sdc_clock : the_sdc_clocks) {
    the_sdc_clock.second->set_is_propagated();
  }
}

void CTSAPI::convertDBToTimingEngine()
{
  _timing_engine->resetNetlist();
  _timing_engine->resetGraph();
  _timing_engine->get_db_adapter()->convertDBToTimingNetlist();
  const char* sdc_path = DBCONFIG.get_sdc_path().c_str();
  _timing_engine->readSdc(sdc_path);
  _timing_engine->buildGraph();
}

void CTSAPI::reportTiming() const
{
  _timing_engine->updateTiming();
  _timing_engine->reportTiming({}, true, true);
}

void CTSAPI::refresh()
{
  // _timing_engine->get_db_adapter()->convertDBToTimingNetlist();
  _timing_engine->updateTiming();
}

icts::CtsPin* CTSAPI::findDriverPin(icts::CtsNet* net)
{
  auto* sta_net = findStaNet(net->get_net_name());
  if (sta_net == nullptr) {
    return nullptr;
  }
  auto driver_pin_name = sta_net->getDriver()->get_name();
  return net->findPin(driver_pin_name);
}

std::map<std::string, double> CTSAPI::elmoreDelay(const icts::EvalNet& eval_net)
{
  auto* sta_net = getStaDbAdapter()->makeNet(eval_net.get_name().c_str(), nullptr);
  buildRCTree(eval_net);
  auto* rc_net = _timing_engine->get_ista()->getRcNet(sta_net);
  auto* rc_tree = rc_net->rct();
  std::map<std::string, double> delay_map;
  for (auto pin : eval_net.get_pins()) {
    if (pin == eval_net.get_driver_pin()) {
      continue;
    }
    auto pin_name = pin->get_full_name();
    auto delay = rc_tree->delay(pin_name);
    delay_map[pin->get_instance()->get_name()] = delay;
  }
  getStaDbAdapter()->removeNet(sta_net);
  return delay_map;
}

std::vector<std::vector<double>> CTSAPI::queryCellLibIndex(const std::string& cell_master, const std::string& query_field,
                                                           const std::string& from_port, const std::string& to_port)
{
  std::vector<std::vector<double>> index_list;
  ista::LibertyTable::TableType table_type;
  if (query_field == "cell_rise") {
    table_type = ista::LibertyTable::TableType::kCellRise;
  } else if (query_field == "cell_fall") {
    table_type = ista::LibertyTable::TableType::kCellFall;
  } else if (query_field == "rise_transition") {
    table_type = ista::LibertyTable::TableType::kRiseTransition;
  } else if (query_field == "fall_transition") {
    table_type = ista::LibertyTable::TableType::kFallTransition;
  } else {
    LOG_FATAL << "buffer lib query field not supported";
  }
  ista::LibertyTable* table = nullptr;
  if (from_port.empty() && to_port.empty()) {
    table = _timing_engine->getCellLibertyTable(cell_master.c_str(), table_type);
  } else {
    table = _timing_engine->getCellLibertyTable(cell_master.c_str(), from_port.c_str(), to_port.c_str(), table_type);
  }
  auto& axes = table->get_axes();
  for (auto& axis : axes) {
    auto& axis_values = axis.get()->get_axis_values();
    std::vector<double> index;
    for (auto& axis_value : axis_values) {
      index.push_back(axis_value.get()->getFloatValue());
    }
    index_list.push_back(index);
  }
  return index_list;
}

std::vector<double> CTSAPI::queryCellLibValue(const std::string& cell_master, const std::string& query_field, const std::string& from_port,
                                              const std::string& to_port)
{
  std::vector<double> values;
  ista::LibertyTable::TableType table_type;
  if (query_field == "cell_rise") {
    table_type = ista::LibertyTable::TableType::kCellRise;
  } else if (query_field == "cell_fall") {
    table_type = ista::LibertyTable::TableType::kCellFall;
  } else if (query_field == "rise_transition") {
    table_type = ista::LibertyTable::TableType::kRiseTransition;
  } else if (query_field == "fall_transition") {
    table_type = ista::LibertyTable::TableType::kFallTransition;
  } else {
    LOG_FATAL << "buffer lib query field not supported";
  }
  ista::LibertyTable* table = nullptr;
  if (from_port.empty() && to_port.empty()) {
    table = _timing_engine->getCellLibertyTable(cell_master.c_str(), table_type);
  } else {
    table = _timing_engine->getCellLibertyTable(cell_master.c_str(), from_port.c_str(), to_port.c_str(), table_type);
  }
  auto& table_values = table->get_table_values();

  for (auto& table_value : table_values) {
    values.push_back(table_value.get()->getFloatValue());
  }
  return values;
}

icts::CtsCellLib* CTSAPI::getCellLib(const std::string& cell_master, const std::string& from_port, const std::string& to_port,
                                     const bool& use_work_value)
{
  CtsCellLib* lib = _libs->findLib(cell_master);
  if (lib) {
    return lib;
  }
  auto index_list = queryCellLibIndex(cell_master, "cell_rise", from_port, to_port);
  std::vector<double> rise_delay = queryCellLibValue(cell_master, "cell_rise", from_port, to_port);
  std::vector<double> fall_delay = queryCellLibValue(cell_master, "cell_fall", from_port, to_port);
  std::vector<double> rise_slew = queryCellLibValue(cell_master, "rise_transition", from_port, to_port);
  std::vector<double> fall_slew = queryCellLibValue(cell_master, "fall_transition", from_port, to_port);

  auto calc_mid_value = [](const std::vector<double>& rise_values, const std::vector<double>& fall_values) {
    std::vector<double> mid_values;
    for (size_t i = 0; i < rise_values.size(); ++i) {
      mid_values.emplace_back((rise_values[i] + fall_values[i]) / 2);
    }
    return mid_values;
  };

  auto delay_mid_value = calc_mid_value(rise_delay, fall_delay);
  auto slew_mid_value = calc_mid_value(rise_slew, fall_slew);

  lib = new CtsCellLib(cell_master, index_list, delay_mid_value, slew_mid_value);
  // set init cap by liberty
  auto init_cap = getCellCap(cell_master);
  lib->set_init_cap(init_cap);
  // fit linear coef
  auto slew_in = index_list[0];
  auto cap_out = index_list[1];

  std::vector<double> x_slew_in;
  std::vector<double> x_cap_out;
  std::vector<double> y_delay;
  std::vector<double> y_slew;

  for (size_t i = 0; i < slew_in.size(); ++i) {
    auto work_slew = slew_in[i];
    if (work_slew > _config->get_max_buf_tran() && use_work_value) {
      break;
    }
    for (size_t j = 0; j < cap_out.size(); ++j) {
      auto work_cap = cap_out[j];
      if (work_cap > _config->get_max_cap() && use_work_value) {
        break;
      }
      x_slew_in.emplace_back(work_slew);
      x_cap_out.emplace_back(work_cap);
      y_delay.emplace_back(delay_mid_value[i * cap_out.size() + j]);
      y_slew.emplace_back(slew_mid_value[i * cap_out.size() + j]);
    }
  }
  LOG_FATAL_IF(x_slew_in.empty() || x_cap_out.empty() || y_delay.empty() || y_slew.empty())
      << "No feasible work value, please check "
         "the config parameter: \"max_buf_tran\", \"max_sink_tran\" and "
         "\"max_cap\" with the liberty "
      << cell_master;
  std::vector<std::vector<double>> x_delay = {x_slew_in, x_cap_out};
  lib->set_delay_coef(_model_factory->cppLinearModel(x_delay, y_delay));

  std::vector<std::vector<double>> x_slew = {x_cap_out};
  lib->set_slew_coef(_model_factory->cppLinearModel(x_slew, y_slew));

#ifdef PY_MODEL
  auto* delay_lib_model = _model_factory->pyFit(x_delay, y_delay, icts::FitType::kCatBoost);
  lib->set_delay_lib_model(delay_lib_model);
  auto* slew_lib_model = _model_factory->pyFit(x_slew, y_slew, icts::FitType::kCatBoost);
  lib->set_slew_lib_model(slew_lib_model);
#endif
  _libs->insertLib(cell_master, lib);
  return lib;
}

std::vector<icts::CtsCellLib*> CTSAPI::getAllBufferLibs()
{
  auto buffer_types = _config->get_buffer_types();
  std::vector<icts::CtsCellLib*> all_buf_libs;
  for (auto buf_cell : buffer_types) {
    auto* buf_lib = getCellLib(buf_cell);
    all_buf_libs.emplace_back(buf_lib);
  }
  auto cmp = [](CtsCellLib* lib_1, CtsCellLib* lib_2) { return lib_1->getDelayIntercept() < lib_1->getDelayIntercept(); };
  std::sort(all_buf_libs.begin(), all_buf_libs.end(), cmp);
  return all_buf_libs;
}

std::vector<std::string> CTSAPI::getMasterClocks(icts::CtsNet* net) const
{
  auto* sta_net = findStaNet(net->get_net_name());
  return _timing_engine->getMasterClocksOfNet(sta_net);
}

double CTSAPI::getClockAT(const std::string& pin_name, const std::string& belong_clock_name) const
{
  auto clk_at = _timing_engine->reportClockAT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise, belong_clock_name);
  if (clk_at == std::nullopt) {
    LOG_WARNING << "get " << pin_name << " clock arrival time failed, which belong clock " << belong_clock_name;
    return 0.0;
  }
  return clk_at.value();
}

std::string CTSAPI::getCellType(const std::string& cell_master) const
{
  return _timing_engine->getCellType(cell_master.c_str());
}

double CTSAPI::getCellArea(const std::string& cell_master) const
{
  return _timing_engine->getCellArea(cell_master.c_str());
}

double CTSAPI::getCellCap(const std::string& cell_master) const
{
  auto input_pin_names = _timing_engine->getLibertyCellInputpin(cell_master.c_str());
  auto cell_pin_name = CTSAPIInst.toString(cell_master.c_str(), ":", input_pin_names[0].c_str());
  auto init_cap = _timing_engine->reportLibertyCellPinCapacitance(cell_pin_name.c_str());
  return init_cap;
}

double CTSAPI::getSlewIn(const std::string& pin_name) const
{
  return _timing_engine->reportSlew(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
}

double CTSAPI::getCapOut(const std::string& pin_name) const
{
  return _timing_engine->reportInstPinCapacitance(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
}

std::vector<double> CTSAPI::solvePolynomialRealRoots(const std::vector<double>& coeffs)
{
  return _model_factory->solvePolynomialRealRoots(coeffs);
}

// iRT
// void CTSAPI::iRTinit()
// {
//   std::map<std::string, std::any> config_map;
//   RTAPI_INST.initRT(config_map);
// }

// void CTSAPI::routingWire(icts::CtsNet* net)
// {
//   // HARD CODE
//   auto layer_1 = "M" + std::to_string(_config->get_routing_layers().front());
//   auto layer_2 = "M" + std::to_string(_config->get_routing_layers().back());
//   struct NondeTree
//   {
//     icts::CtsInstance* inst;
//     icts::Point point;
//     NondeTree* parent;
//     std::vector<NondeTree*> children;
//   };
//   std::map<std::string, NondeTree*> wire_node_map;
//   auto signal_wires = net->get_signal_wires();
//   for (auto signal_wire : signal_wires) {
//     auto first = signal_wire.get_first();
//     auto second = signal_wire.get_second();
//     auto first_name = DataTraits<Endpoint>::getId(first);
//     auto second_name = DataTraits<Endpoint>::getId(second);
//     auto first_inst = net->findInstance(first_name);
//     auto second_inst = net->findInstance(second_name);
//     auto* first_node = wire_node_map.find(first_name) != wire_node_map.end() ? wire_node_map[first_name] : nullptr;
//     auto* second_node = wire_node_map.find(second_name) != wire_node_map.end() ? wire_node_map[second_name] : nullptr;
//     if (first_node == nullptr) {
//       first_node = new NondeTree();
//       first_node->inst = first_inst;
//       if (first_inst) {
//         auto driver_pin = net->get_driver_pin();
//         first_node->point = _db_wrapper->getPinLoc(driver_pin);
//       } else {
//         first_node->point = DataTraits<Endpoint>::getPoint(first);
//       }
//       wire_node_map[first_name] = first_node;
//     }
//     if (second_node == nullptr) {
//       second_node = new NondeTree();
//       second_node->inst = second_inst;
//       if (second_inst) {
//         auto* inst = net->findInstance(second_name);
//         auto* load_pin = inst->get_load_pin();
//         second_node->point = _db_wrapper->getPinLoc(load_pin);
//       } else {
//         second_node->point = DataTraits<Endpoint>::getPoint(second);
//       }
//       wire_node_map[second_name] = second_node;
//     }
//     if (pgl::rectilinear(first_node->point, second_node->point)) {
//       first_node->children.emplace_back(second_node);
//       second_node->parent = first_node;
//     } else {
//       auto trunk_node = new NondeTree();
//       trunk_node->inst = nullptr;
//       trunk_node->point = icts::Point(first_node->point.x(), second_node->point.y());
//       first_node->children.emplace_back(trunk_node);
//       trunk_node->parent = first_node;
//       trunk_node->children.emplace_back(second_node);
//       second_node->parent = trunk_node;
//     }
//   }
//   auto* root = net->get_driver_inst();
//   auto wire_tree_root = wire_node_map[root->get_name()];
//   std::vector<ids::Segment> segs;
//   std::queue<NondeTree*> wire_tree_queue;
//   wire_tree_queue.push(wire_tree_root);
//   while (!wire_tree_queue.empty()) {
//     auto* node = wire_tree_queue.front();
//     wire_tree_queue.pop();
//     if (node->inst) {
//       auto driver_pin = net->get_driver_pin();
//       auto inst_layer = _db_wrapper->getPinLayer(driver_pin);
//       ids::Segment via{node->point.x(), node->point.y(), inst_layer, node->point.x(), node->point.y(), layer_2};
//       segs.emplace_back(via);
//     }
//     for (auto* child : node->children) {
//       auto clock_layer = node->point.x() == child->point.x() ? layer_2 : layer_1;
//       ids::Segment seg{node->point.x(), node->point.y(), clock_layer, child->point.x(), child->point.y(), clock_layer};
//       segs.emplace_back(seg);
//       if (child->inst) {
//         auto load_pin = child->inst->get_load_pin();
//         auto inst_layer = _db_wrapper->getPinLayer(load_pin);
//         ids::Segment via{child->point.x(), child->point.y(), clock_layer, child->point.x(), child->point.y(), inst_layer};
//         segs.emplace_back(via);
//       }
//       wire_tree_queue.push(child);
//     }
//     if (node->parent) {
//       auto father_wire = icts::Segment(node->point, node->parent->point);
//       bool need_via = false;
//       for (auto* child : node->parent->children) {
//         auto seg = icts::Segment(node->point, child->point);
//         if ((pgl::vertical(seg) && pgl::horizontal(father_wire)) || (pgl::horizontal(seg) && pgl::vertical(father_wire))) {
//           need_via = true;
//           break;
//         }
//       }
//       if (need_via) {
//         ids::Segment via{node->point.x(), node->point.y(), layer_1, node->point.x(), node->point.y(), layer_2};
//         segs.emplace_back(via);
//       }
//     }
//   }
//   auto* idb_net = _db_wrapper->ctsToIdb(net);
//   auto ph_nodes = RTAPI_INST.getPHYNodeList(segs);
//   auto* idb_layer_list = dmInst->get_idb_builder()->get_def_service()->get_layout()->get_layers();
//   auto* lef_via_list = dmInst->get_idb_builder()->get_lef_service()->get_layout()->get_via_list();
//   idb_net->clear_wire_list();
//   auto* idb_wire_list = idb_net->get_wire_list();
//   auto* idb_wire = idb_wire_list->add_wire();
//   idb_wire->set_wire_state(idb::IdbWiringStatement::kRouted);
//   bool print_new = false;
//   for (auto node : ph_nodes) {
//     auto* idb_segment = idb_wire->add_segment();
//     if (node.type == ids::PHYNodeType::kWire) {
//       auto wire = node.wire;
//       auto* idb_layer = idb_layer_list->find_layer(wire.layer_name);
//       idb_segment->set_layer(idb_layer);
//       idb_segment->add_point(wire.first_x, wire.first_y);
//       idb_segment->add_point(wire.second_x, wire.second_y);
//     } else if (node.type == ids::PHYNodeType::kVia) {
//       auto via = node.via;
//       auto* idb_via = lef_via_list->find_via(via.via_name);
//       auto* idb_layer_top = idb_via->get_instance()->get_top_layer_shape()->get_layer();
//       idb_segment->set_layer(idb_layer_top);
//       idb_segment->set_is_via(true);
//       idb_segment->add_point(via.x, via.y);
//       auto* idb_via_new = idb_segment->copy_via(idb_via);
//       idb_via_new->set_coordinate(via.x, via.y);
//     } else {
//       LOG_FATAL << "Unlegal PHY node type";
//     }
//     if (print_new == false) {
//       idb_segment->set_layer_as_new();
//       print_new = true;
//     }
//   }
// }

// void CTSAPI::iRTdestroy()
// {
//   RTAPI_INST.destroyRT();
// }

// // iTO
// std::vector<idb::IdbNet*> CTSAPI::fix(const icts::OptiNet& opti_net)
// {
//   ito::Tree* topo = ToApiInst.get_tree(static_cast<int>(opti_net.get_signal_wires().size() + 1));

//   makeTopo(topo, opti_net);
//   auto* idb_net = _db_wrapper->ctsToIdb(opti_net.get_clk_net());
//   return ToApiInst.optimizeCTSDesignViolation(idb_net, topo);
// }

// void CTSAPI::makeTopo(ito::Tree* topo, const icts::OptiNet& opti_net) const
// {
//   auto signal_wires = opti_net.get_signal_wires();
//   for (const auto& signal_wire : signal_wires) {
//     auto first_id = opti_net.getId(signal_wire.get_first()._name);
//     auto second_id = opti_net.getId(signal_wire.get_second()._name);
//     auto p1 = signal_wire.get_first()._point;
//     auto p2 = signal_wire.get_second()._point;
//     ToApiInst.addTopoEdge(topo, first_id, second_id, p1.x(), p1.y(), p2.x(), p2.y());
//   }

//   auto load_insts = opti_net.get_loads();
//   for (auto* load_inst : load_insts) {
//     auto* sta_pin = findStaPin(load_inst->get_load_pin());
//     ToApiInst.topoIdToDesignObject(topo, opti_net.getId(load_inst->get_name()), sta_pin);
//   }
//   auto* driver_inst = opti_net.get_driver();
//   auto driver_id = opti_net.getId(driver_inst->get_name());
//   ToApiInst.topoIdToDesignObject(topo, driver_id, findStaPin(driver_inst->get_out_pin()));
//   ToApiInst.topoSetDriverId(topo, driver_id);
// }

// synthesis
int32_t CTSAPI::getDbUnit() const
{
  auto* idb = dmInst->get_idb_builder();
  auto* idb_design = idb->get_def_service()->get_design();
  return idb_design->get_units()->get_micron_dbu();
}

bool CTSAPI::isInDie(const icts::Point& point) const
{
  auto die = _db_wrapper->get_core_bounding_box();
  return die.is_in(point);
}

// void CTSAPI::placeInstance(icts::CtsInstance* inst)
// {
//   _synth->place(inst);
// }

// void CTSAPI::cancelPlaceInstance(icts::CtsInstance* inst)
// {
//   _synth->cancelPlace(inst);
// }

idb::IdbInstance* CTSAPI::makeIdbInstance(const std::string& inst_name, const std::string& cell_master)
{
  auto sta_inst = getStaDbAdapter()->makeInstance(_timing_engine->findLibertyCell(cell_master.c_str()), inst_name.c_str());
  auto idb_inst = getStaDbAdapter()->staToDb(sta_inst);
  return idb_inst;
}

idb::IdbNet* CTSAPI::makeIdbNet(const std::string& net_name)
{
  auto sta_net = getStaDbAdapter()->makeNet(net_name.c_str(), nullptr);
  auto idb_net = getStaDbAdapter()->staToDb(sta_net);
  return idb_net;
}

void CTSAPI::linkIdbNetToSta(idb::IdbNet* idb_net)
{
  auto sta_net = getStaDbAdapter()->makeNet(idb_net->get_net_name().c_str(), nullptr);
  getStaDbAdapter()->crossRef(sta_net, idb_net);
}

void CTSAPI::disconnect(idb::IdbPin* pin)
{
  auto db_adapter = getStaDbAdapter();
  auto sta_pin = db_adapter->dbToStaPin(pin);
  db_adapter->disconnectPin(sta_pin);
}

void CTSAPI::connect(idb::IdbInstance* idb_inst, const std::string& pin_name, idb::IdbNet* net)
{
  auto db_adapter = getStaDbAdapter();
  auto sta_inst = _timing_engine->get_netlist()->findInstance(idb_inst->get_name().c_str());
  auto sta_net = db_adapter->dbToSta(net);
  db_adapter->connect(sta_inst, pin_name.c_str(), sta_net);
}

void CTSAPI::insertBuffer(const std::string& name)
{
  _timing_engine->insertBuffer(name.c_str());
}

void CTSAPI::resetId()
{
  _design->resetId();
}

int CTSAPI::genId()
{
  return _design->nextId();
}

void CTSAPI::genFluteTree(const std::string& net_name, icts::Pin* driver, const std::vector<icts::Pin*>& loads)
{
  std::vector<icts::Pin*> pins{driver};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;

  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net net;
  net.init(0, net_name, salt_pins);

  salt::Tree tree;
  salt::FluteBuilder flute_builder;
  flute_builder.Run(net, tree);
  tree.UpdateId();
  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  auto source = tree.source;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
}

void CTSAPI::genShallowLightTree(const std::string& net_name, icts::Pin* driver, const std::vector<icts::Pin*>& loads)
{
  std::vector<icts::Pin*> pins{driver};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;

  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net net;
  net.init(0, net_name, salt_pins);

  salt::Tree tree;
  salt::SaltBuilder salt_builder;
  salt_builder.Run(net, tree, 0);

  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  auto source = tree.source;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
}

icts::Inst* CTSAPI::genBeatSaltTree(const std::string& net_name, const std::vector<icts::Pin*>& loads,
                                    const std::optional<double>& skew_bound, const std::optional<icts::Point>& guide_loc,
                                    const TopoType& topo_type)
{
  // build BST
  auto* buf = icts::TreeBuilder::boundSkewTree(net_name, loads, skew_bound, guide_loc, topo_type);
  auto* driver_pin = buf->get_driver_pin();
  std::vector<Pin*> pins{driver_pin};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::unordered_map<Pin*, std::shared_ptr<salt::Pin>> salt_pin_map;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  // init
  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pin_map[pin] = salt_pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net salt_net;
  salt_net.init(0, net_name, salt_pins);
  // convert bound skew tree to salt data structure
  std::unordered_map<Node*, std::shared_ptr<salt::TreeNode>> salt_node_map;
  int id = pins.size();
  driver_pin->preOrder([&](Node* node) {
    auto loc = salt::Point(node->get_location().x(), node->get_location().y());
    std::shared_ptr<salt::TreeNode> salt_node;
    if (node->isPin()) {
      auto salt_pin = salt_pin_map[dynamic_cast<Pin*>(node)];
      salt_node = std::make_shared<salt::TreeNode>(loc, salt_pin, salt_pin->id);
    } else {
      salt_node = std::make_shared<salt::TreeNode>(loc, nullptr, id++);
    }
    salt_node_map[node] = salt_node;
    if (node->get_parent()) {
      auto salt_parent = salt_node_map[node->get_parent()];
      salt_node->parent = salt_parent;
      salt_parent->children.push_back(salt_node);
    }
  });
  salt::Tree bound_skew_tree(salt_node_map[driver_pin], &salt_net);
  // BEAT Salt
  icts::BeatSaltBuilder builder;
  builder.run(salt_net, bound_skew_tree, 0);

  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  icts::TreeBuilder::recoverNet(driver_pin->get_net());
  auto source = bound_skew_tree.source;
  buf = icts::TreeBuilder::genBufInst(net_name, icts::Point(source->loc.x, source->loc.y));
  driver_pin = buf->get_driver_pin();
  id_to_node[0] = driver_pin;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
  auto* net = icts::TimingPropagator::genNet(net_name, driver_pin, loads);
  icts::TimingPropagator::update(net);
  return buf;
}

icts::Inst* CTSAPI::genBeatTree(const std::string& net_name, const std::vector<icts::Pin*>& loads, const std::optional<double>& skew_bound,
                                const std::optional<icts::Point>& guide_loc, const TopoType& topo_type)
{
  // build BST
  auto* buf = icts::TreeBuilder::boundSkewTree(net_name, loads, skew_bound, guide_loc);
  auto* driver_pin = buf->get_driver_pin();
  std::vector<Pin*> pins{driver_pin};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::unordered_map<Pin*, std::shared_ptr<salt::Pin>> salt_pin_map;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  // init
  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pin_map[pin] = salt_pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net salt_net;
  salt_net.init(0, net_name, salt_pins);
  // convert bound skew tree to salt data structure
  std::unordered_map<Node*, std::shared_ptr<salt::TreeNode>> salt_node_map;
  int id = pins.size();
  driver_pin->preOrder([&](Node* node) {
    auto loc = salt::Point(node->get_location().x(), node->get_location().y());
    std::shared_ptr<salt::TreeNode> salt_node;
    if (node->isPin()) {
      auto salt_pin = salt_pin_map[dynamic_cast<Pin*>(node)];
      salt_node = std::make_shared<salt::TreeNode>(loc, salt_pin, salt_pin->id);
    } else {
      salt_node = std::make_shared<salt::TreeNode>(loc, nullptr, id++);
    }
    salt_node_map[node] = salt_node;
    if (node->get_parent()) {
      auto salt_parent = salt_node_map[node->get_parent()];
      salt_node->parent = salt_parent;
      salt_parent->children.push_back(salt_node);
    }
  });
  salt::Tree bound_skew_tree(salt_node_map[driver_pin], &salt_net);
  // BEAT Salt
  icts::BeatBuilder beat_builder;
  beat_builder.run(salt_net, bound_skew_tree, 0);

  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  icts::TreeBuilder::recoverNet(driver_pin->get_net());
  auto source = bound_skew_tree.source;
  buf = icts::TreeBuilder::genBufInst(net_name, icts::Point(source->loc.x, source->loc.y));
  driver_pin = buf->get_driver_pin();
  id_to_node[0] = driver_pin;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
  auto* net = icts::TimingPropagator::genNet(net_name, driver_pin, loads);
  icts::TimingPropagator::update(net);
  return buf;
}

// evaluate
bool CTSAPI::isTop(const std::string& net_name) const
{
  return _design->isClockTopNet(net_name);
}

void CTSAPI::buildRCTree(const std::vector<icts::EvalNet>& eval_nets)
{
  for (auto& eval_net : eval_nets) {
    buildRCTree(eval_net);
  }
}

void CTSAPI::buildRCTree(const icts::EvalNet& eval_net)
{
  auto net_name = eval_net.get_name();
  LOG_INFO << "Evaluate: " << net_name;
  resetRCTree(net_name);
  auto* sta_net = findStaNet(eval_net);
  auto layer_id = _config->get_routing_layers().back();
  auto* solver_net = _design->findSolverNet(net_name);
  if (!solver_net) {
    LOG_WARNING << "Can't find solver net: " << net_name;
  }
  auto* driver_pin = solver_net->get_driver_pin();
  driver_pin->preOrder([&](Node* node) {
    auto* parent = node->get_parent();
    if (parent == nullptr) {
      return;
    }
    auto parent_name = parent->isPin() ? dynamic_cast<Pin*>(parent)->get_inst()->get_name() : parent->get_name();
    auto child_name = node->isPin() ? dynamic_cast<Pin*>(node)->get_inst()->get_name() : node->get_name();
    ista::RctNode* front_node = makeRCTreeNode(eval_net, parent_name);
    ista::RctNode* back_node = makeRCTreeNode(eval_net, child_name);
    double len = TimingPropagator::calcLen(parent, node);
    auto res = getResistance(len, layer_id);
    auto cap = getCapacitance(len, layer_id);
    _timing_engine->makeResistor(sta_net, front_node, back_node, res);
    _timing_engine->incrCap(front_node, cap / 2, true);
    _timing_engine->incrCap(back_node, cap / 2, true);
  });

  _timing_engine->updateRCTreeInfo(sta_net);
}

void CTSAPI::resetRCTree(const std::string& net_name)
{
  auto* sta_net = findStaNet(net_name);
  _timing_engine->resetRcTree(sta_net);
}

// log
void CTSAPI::checkFile(const std::string& dir, const std::string& file_name, const std::string& suffix) const
{
  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  std::string origin_file_name = Str::printf("%s/%s%s", dir.c_str(), file_name.c_str(), suffix.c_str());
  std::string copy_design_work_space = Str::printf("%s/%s_%s_%s%s", dir.c_str(), file_name.c_str(), tmp.c_str(), "_backup", suffix.c_str());

  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
    return;
  }
  if (std::filesystem::exists(origin_file_name)) {
    std::filesystem::copy_file(origin_file_name, copy_design_work_space);
  }
}

// debug

void CTSAPI::writeVerilog() const
{
  _timing_engine->writeVerilog("cts_debug.v");
}

void CTSAPI::toPyArray(const icts::Point& point, const std::string& label)
{
  CTSAPIInst.saveToLog(label, "=[[", point.x(), ",", point.y(), "]]");
}

void CTSAPI::toPyArray(const icts::Segment& segment, const std::string& label)
{
  CTSAPIInst.saveToLog(label, "=[[", segment.low().x(), ",", segment.low().y(), "], [", segment.high().x(), ",", segment.high().y(), "]]");
}

void CTSAPI::toPyArray(const icts::Polygon& polygon, const std::string& label)
{
  auto message = label + "=" + "[";
  for (auto point : polygon.get_points()) {
    message += "[" + std::to_string(point.x()) + "," + std::to_string(point.y()) + "], ";
  }
  message += "]";
  CTSAPIInst.saveToLog(message);
}

void CTSAPI::toPyArray(const icts::CtsPolygon<int64_t>& polygon, const std::string& label)
{
  auto message = label + "=" + "[";
  for (auto point : polygon.get_points()) {
    message += "[" + std::to_string(point.x()) + "," + std::to_string(point.y()) + "], ";
  }
  message += "]";
  CTSAPIInst.saveToLog(message);
}

// python API
#ifdef PY_MODEL

#ifdef USE_EXTERNAL_MODEL
icts::ModelBase* CTSAPI::findExternalModel(const std::string& net_name)
{
  return _libs->findModel(net_name);
}
#endif

/**
 * @brief Python interface for plot
 *
 * @param x
 * @param y
 */
icts::ModelBase* CTSAPI::fitPyModel(const std::vector<std::vector<double>>& x, const std::vector<double>& y, const icts::FitType& fit_type)
{
  return _model_factory->pyFit(x, y, fit_type);
}

#endif

// function
std::vector<std::string> CTSAPI::splitString(std::string str, const char split)
{
  std::vector<std::string> string_list;

  std::istringstream iss(str);
  std::string token;
  while (getline(iss, token, split)) {
    string_list.push_back(token);
  }
  return string_list;
}

// private STA
void CTSAPI::readSTAFile()
{
  const char* sta_workspace = _config->get_sta_workspace().c_str();
  std::vector<const char*> lib_paths;
  for (auto& lib_path : DBCONFIG.get_lib_paths()) {
    lib_paths.push_back(lib_path.c_str());
  }
  _timing_engine->set_num_threads(80);
  _timing_engine->set_design_work_space(sta_workspace);
  _timing_engine->readLiberty(lib_paths);
  convertDBToTimingEngine();

  _timing_engine->initRcTree();
}

ista::RctNode* CTSAPI::makeRCTreeNode(const icts::EvalNet& eval_net, const std::string& name)
{
  auto* sta_net = findStaNet(eval_net);
  auto* inst = eval_net.get_instance(name);
  if (inst == nullptr) {
    std::vector<std::string> string_list = splitString(name, '_');
    if (string_list.size() == 2 && (string_list[0] == "steiner" || string_list[0] == "Salt")) {
      return _timing_engine->makeOrFindRCTreeNode(sta_net, std::stoi(string_list[1]));
    } else {
      LOG_FATAL << "Unknown pin name: " << name;
    }
  } else {
    for (auto pin : eval_net.get_pins()) {
      if (pin->get_instance() == inst) {
        return makeLogicRCTreeNode(pin);
      }
    }
  }
  return nullptr;
}

ista::RctNode* CTSAPI::makeLogicRCTreeNode(icts::CtsPin* pin)
{
  auto* pin_port = findStaPin(pin->is_io() ? pin->get_pin_name() : pin->get_full_name());
  return _timing_engine->makeOrFindRCTreeNode(pin_port);
}

ista::DesignObject* CTSAPI::findStaPin(icts::CtsPin* pin) const
{
  return findStaPin(pin->get_full_name());
}

ista::DesignObject* CTSAPI::findStaPin(const std::string& pin_full_name) const
{
  return _timing_engine->get_netlist()->findObj(pin_full_name.c_str(), false, false).front();
}

ista::Net* CTSAPI::findStaNet(const icts::EvalNet& eval_net) const
{
  return findStaNet(eval_net.get_clk_net()->get_net_name());
}

ista::Net* CTSAPI::findStaNet(const std::string& name) const
{
  return _timing_engine->get_netlist()->findNet(name.c_str());
}

double CTSAPI::getUnitCap() const
{
  std::optional<double> width = std::nullopt;
  return getStaDbAdapter()->getCapacitance(_config->get_routing_layers().back(), 1.0, width);
}

double CTSAPI::getUnitRes() const
{
  std::optional<double> width = std::nullopt;
  return getStaDbAdapter()->getResistance(_config->get_routing_layers().back(), 1.0, width);
}

double CTSAPI::getCapacitance(const double& wire_length, const int& level) const
{
  std::optional<double> width = std::nullopt;
  return getStaDbAdapter()->getCapacitance(level, wire_length, width);
}

double CTSAPI::getResistance(const double& wire_length, const int& level) const
{
  std::optional<double> width = std::nullopt;
  return getStaDbAdapter()->getResistance(level, wire_length, width);
}

double CTSAPI::getCapacitance(const icts::CtsSignalWire& signal_wire, const int& level) const
{
  return getCapacitance(getWireLength(signal_wire), level);
}

double CTSAPI::getResistance(const icts::CtsSignalWire& signal_wire, const int& level) const
{
  return getResistance(getWireLength(signal_wire), level);
}

double CTSAPI::getWireLength(const icts::CtsSignalWire& signal_wire) const
{
  return static_cast<double>(1.0 * signal_wire.getWireLength() / getDbUnit());
}

ista::TimingIDBAdapter* CTSAPI::getStaDbAdapter() const
{
  return dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter());
}

}  // namespace icts