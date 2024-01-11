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
/**
 * @file CTSAPI.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CTSAPI.hh"

#include <chrono>
#include <filesystem>
#include <functional>
#include <ranges>
#include <unordered_map>

#include "CBS.hh"
#include "CtsCellLib.hh"
#include "CtsConfig.hh"
#include "CtsDBWrapper.hh"
#include "CtsDesign.hh"
#include "Evaluator.hh"
#include "GDSPloter.hh"
#include "JsonParser.hh"
#include "Node.hh"
#include "Pin.hh"
#include "Router.hh"
#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"
#include "idm.h"
#include "log/Log.hh"
#include "model/ModelFactory.hh"
#include "model/mplHelper/MplHelper.hh"
#include "model/python/PyToolBase.hh"
#include "report/CtsReport.hh"
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
  evaluate();
  writeGDS();
  LOG_INFO << "Flow memory usage " << stats.memoryDelta() << "MB";
  LOG_INFO << "Flow elapsed time " << stats.elapsedRunTime() << "s";

  CTSAPIInst.saveToLog("\n\n##iCTS Run Time Log##");
  CTSAPIInst.saveToLog("Flow memory usage: ", stats.memoryDelta(), "MB");
  CTSAPIInst.saveToLog("Flow elapsed time: ", stats.elapsedRunTime(), "s");
  // writeDB();
}

void CTSAPI::writeDB()
{
  _db_wrapper->writeDef();
}

void CTSAPI::writeGDS()
{
  GDSPloter::plotDesign();
  GDSPloter::plotFlyLine();
  GDSPloter::writePyDesign();
  GDSPloter::writePyFlyLine();
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
  _timing_engine->destroyTimingEngine();
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
  _evaluator = nullptr;
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

  _evaluator = new Evaluator();
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

// iSTA
void CTSAPI::dumpVertexData(const std::vector<std::string>& vertex_names) const
{
  _timing_engine->get_ista()->dumpVertexData(vertex_names);
}

double CTSAPI::getClockUnitCap(const std::optional<icts::LayerPattern>& layer_pattern) const
{
  auto pattern = layer_pattern.value_or(icts::LayerPattern::kNone);
  auto* db_adapter = getStaDbAdapter();
  auto layer = _config->get_routing_layers().back();
  switch (pattern) {
    case icts::LayerPattern::kH:
      layer = _config->get_h_layer();
      break;
    case icts::LayerPattern::kV:
      layer = _config->get_v_layer();
      break;
    case icts::LayerPattern::kNone:
      layer = _config->get_routing_layers().back();
      break;
    default:
      LOG_ERROR << "Unknown layer pattern";
      break;
  }
  std::optional<double> width = std::nullopt;
  auto max_len = _config->get_max_length();
  return db_adapter->getCapacitance(layer, max_len, width) / max_len;
}

double CTSAPI::getClockUnitRes(const std::optional<icts::LayerPattern>& layer_pattern) const
{
  auto pattern = layer_pattern.value_or(icts::LayerPattern::kNone);
  auto* db_adapter = getStaDbAdapter();
  auto layer = _config->get_routing_layers().back();
  switch (pattern) {
    case icts::LayerPattern::kH:
      layer = _config->get_h_layer();
      break;
    case icts::LayerPattern::kV:
      layer = _config->get_v_layer();
      break;
    case icts::LayerPattern::kNone:
      layer = _config->get_routing_layers().back();
      break;
    default:
      LOG_ERROR << "Unknown layer pattern";
      break;
  }
  std::optional<double> width = std::nullopt;
  auto max_len = _config->get_max_length();
  return db_adapter->getResistance(layer, max_len, width) / max_len / 2;
}

double CTSAPI::getSinkCap(icts::CtsInstance* sink) const
{
  auto* load_pin = sink->get_load_pin();
  return getSinkCap(load_pin->get_full_name());
}

double CTSAPI::getSinkCap(const std::string& load_pin_full_name) const
{
  // remove all "\" in inst_name
  auto name = load_pin_full_name;
  name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
  return _timing_engine->reportInstPinCapacitance(name.c_str());
}

bool CTSAPI::isFlipFlop(const std::string& inst_name) const
{
  // remove all "\" in inst_name
  auto name = inst_name;
  name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
  return _timing_engine->isSequentialCell(name.c_str());
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
  ieda::Stats stats;
  _timing_engine->updateTiming();
  CTSAPIInst.saveToLog("\n\n##iSTA Run Time Log##");
  CTSAPIInst.saveToLog("update timing elapsed time: ", stats.elapsedRunTime(), "s");
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
  auto* db_adapter = getStaDbAdapter();
  auto* sta_net = db_adapter->makeNet(eval_net.get_name().c_str(), nullptr);
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
  db_adapter->removeNet(sta_net);
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
  std::ranges::sort(all_buf_libs, cmp);
  return all_buf_libs;
}

icts::CtsCellLib* CTSAPI::getRootBufferLib()
{
  auto root_buffer_type = _config->get_root_buffer_type();
  auto* buf_lib = getCellLib(root_buffer_type);
  return buf_lib;
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

// synthesis
int32_t CTSAPI::getDbUnit() const
{
  auto* idb = dmInst->get_idb_builder();
  auto* idb_design = idb->get_def_service()->get_design();
  return idb_design->get_units()->get_micron_dbu();
}

bool CTSAPI::isInDie(const icts::Point& point) const
{
  auto* die = _db_wrapper->get_core_bounding_box();
  auto pt = _db_wrapper->ctsToIdb(point);
  return die->containPoint(pt);
}

idb::IdbInstance* CTSAPI::makeIdbInstance(const std::string& inst_name, const std::string& cell_master)
{
  auto* db_adapter = getStaDbAdapter();
  auto sta_inst = db_adapter->makeInstance(_timing_engine->findLibertyCell(cell_master.c_str()), inst_name.c_str());
  auto idb_inst = db_adapter->staToDb(sta_inst);
  return idb_inst;
}

idb::IdbNet* CTSAPI::makeIdbNet(const std::string& net_name)
{
  auto* db_adapter = getStaDbAdapter();
  auto sta_net = db_adapter->makeNet(net_name.c_str(), nullptr);
  auto idb_net = db_adapter->staToDb(sta_net);
  return idb_net;
}

void CTSAPI::linkIdbNetToSta(idb::IdbNet* idb_net)
{
  auto* db_adapter = getStaDbAdapter();
  auto sta_net = db_adapter->makeNet(idb_net->get_net_name().c_str(), nullptr);
  db_adapter->crossRef(sta_net, idb_net);
}

void CTSAPI::disconnect(idb::IdbPin* pin)
{
  auto* db_adapter = getStaDbAdapter();
  auto sta_pin = db_adapter->dbToStaPin(pin);
  db_adapter->disconnectPin(sta_pin);
}

void CTSAPI::connect(idb::IdbInstance* idb_inst, const std::string& pin_name, idb::IdbNet* net)
{
  auto* db_adapter = getStaDbAdapter();
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
  TreeBuilder::fluteTree(net_name, driver, loads);
}

void CTSAPI::genShallowLightTree(const std::string& net_name, icts::Pin* driver, const std::vector<icts::Pin*>& loads)
{
  TreeBuilder::shallowLightTree(net_name, driver, loads);
}

icts::Inst* CTSAPI::genBoundSkewTree(const std::string& net_name, const std::vector<icts::Pin*>& loads,
                                     const std::optional<double>& skew_bound, const std::optional<icts::Point>& guide_loc,
                                     const TopoType& topo_type)
{
  return TreeBuilder::boundSkewTree(net_name, loads, skew_bound, guide_loc, topo_type);
}

icts::Inst* CTSAPI::genBstSaltTree(const std::string& net_name, const std::vector<icts::Pin*>& loads,
                                   const std::optional<double>& skew_bound, const std::optional<icts::Point>& guide_loc,
                                   const TopoType& topo_type)
{
  return TreeBuilder::bstSaltTree(net_name, loads, skew_bound, guide_loc, topo_type);
}

icts::Inst* CTSAPI::genCBSTree(const std::string& net_name, const std::vector<icts::Pin*>& loads, const std::optional<double>& skew_bound,
                               const std::optional<icts::Point>& guide_loc, const TopoType& topo_type)
{
  return TreeBuilder::cbsTree(net_name, loads, skew_bound, guide_loc, topo_type);
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
#ifdef DEBUG_ICTS_EVALUATOR
  LOG_INFO << "Evaluate: " << net_name;
#endif
  resetRCTree(net_name);
  auto* sta_net = findStaNet(eval_net);
  auto layer_id = _config->get_routing_layers().back();
  auto* solver_net = _design->findSolverNet(net_name);
  if (!solver_net) {
    LOG_WARNING << "Can't find solver net: " << net_name << ", It may be a pin-port(s) net";
    // buildPinPortsRCTree(eval_net);
    return;
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

void CTSAPI::buildPinPortsRCTree(const icts::EvalNet& eval_net)
{
  auto* sta_net = findStaNet(eval_net);
  auto net_name = eval_net.get_name();
  LOG_FATAL_IF(!sta_net) << "Can't find sta net: " << net_name;
  auto pins = sta_net->get_pin_ports();
  ista::DesignObject* driver_pin = nullptr;
  for (auto* pin : pins) {
    if (pin->isPin()) {
      driver_pin = pin;
      break;
    }
  }
  LOG_FATAL_IF(!driver_pin) << "Can't find driver pin of sta net: " << net_name;
  auto* driver_node = _timing_engine->makeOrFindRCTreeNode(driver_pin);
  auto* db_adapter = getStaDbAdapter();
  auto driver_loc = db_adapter->idbLocation(driver_pin);
  auto pt_dist = [](idb::IdbCoordinate<int32_t>* p1, idb::IdbCoordinate<int32_t>* p2) {
    return std::abs(p1->get_x() - p2->get_x()) + std::abs(p1->get_y() - p2->get_y());
  };
  std::ranges::for_each(pins, [&](ista::DesignObject* pin) {
    if (pin == driver_pin) {
      return;
    }
    auto* load_node = _timing_engine->makeOrFindRCTreeNode(pin);
    auto load_loc = db_adapter->idbLocation(pin);
    auto dist = pt_dist(driver_loc, load_loc);
    auto res = getResistance(1.0 * dist / TimingPropagator::getDbUnit(), _config->get_routing_layers().back());
    auto cap = getCapacitance(1.0 * dist / TimingPropagator::getDbUnit(), _config->get_routing_layers().back());
    _timing_engine->makeResistor(sta_net, driver_node, load_node, res);
    _timing_engine->incrCap(driver_node, cap / 2, true);
    _timing_engine->incrCap(load_node, cap / 2, true);
  });
  _timing_engine->updateRCTreeInfo(sta_net);
}

void CTSAPI::resetRCTree(const std::string& net_name)
{
  auto* sta_net = findStaNet(net_name);
  _timing_engine->resetRcTree(sta_net);
}

void CTSAPI::utilizationLog() const
{
  CTSAPIInst.saveToLog("\n\n##Utilization Log##");
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_layout = dmInst->get_idb_layout();
  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();
  auto* idb_die = idb_layout->get_die();
  auto die_width = ((double) idb_die->get_width()) / dbu;
  auto die_height = ((double) idb_die->get_height()) / dbu;

  auto idb_core_box = idb_layout->get_core()->get_bounding_box();
  auto core_width = ((double) idb_core_box->get_width()) / dbu;
  auto core_height = ((double) idb_core_box->get_height()) / dbu;
  CTSAPIInst.saveToLog("\nDIE Area ( um^2 ): ", ieda::Str::printf("%f = %03f * %03f", die_width * die_height, die_width, die_height));
  CTSAPIInst.saveToLog("DIE Usage: ", dmInst->dieUtilization());
  CTSAPIInst.saveToLog("CORE Area ( um^2 ): ", ieda::Str::printf("%f = %03f * %03f", core_width * core_height, core_width, core_height));
  CTSAPIInst.saveToLog("CORE Usage: ", dmInst->coreUtilization());
}

void CTSAPI::latencySkewLog() const
{
  CTSAPIInst.saveToLog("\n\n##Latency Skew Log##");

  auto fix_point_str = [](double data) { return ieda::Str::printf("%.3f", data); };
  auto cmp = [](ista::StaPathData* left, ista::StaPathData* right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };
  std::vector<std::pair<std::string, ista::AnalysisMode>> mode_list
      = {{"Setup", ista::AnalysisMode::kMax}, {"Hold", ista::AnalysisMode::kMin}};
  for (const auto& [clk, seq_path_group] : _timing_engine->get_ista()->get_clock_groups()) {
    CTSAPIInst.saveToLog("\nClock: ", clk->get_clock_name());
    for (const auto& [mode_str, mode] : mode_list) {
      CTSAPIInst.saveToLog("\n[", mode_str, " Mode]");
      std::priority_queue<ista::StaPathData*, std::vector<ista::StaPathData*>, decltype(cmp)> seq_data_queue(cmp);

      ista::StaPathEnd* path_end;
      ista::StaPathData* path_data;
      FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
      FOREACH_PATH_END_DATA(path_end, mode, path_data) { seq_data_queue.push(path_data); }
      auto* worst_seq_data = seq_data_queue.top();
      auto* launch_clock_data = worst_seq_data->get_launch_clock_data();
      auto* capture_clock_data = worst_seq_data->get_capture_clock_data();

      auto* launch_clock_vertex = launch_clock_data->get_own_vertex();
      auto* capture_clock_vertex = capture_clock_data->get_own_vertex();

      CTSAPIInst.saveToLog("Launch Latency: ", fix_point_str(FS_TO_NS(launch_clock_data->get_arrive_time())), " From ",
                           launch_clock_vertex->getNameWithCellName());
      CTSAPIInst.saveToLog("Capture Latency: ", fix_point_str(FS_TO_NS(capture_clock_data->get_arrive_time())), " From ",
                           capture_clock_vertex->getNameWithCellName());
      CTSAPIInst.saveToLog("Skew: ", fix_point_str(FS_TO_NS(worst_seq_data->getSkew())));
      // calc avg skew
      double total_skew = 0.0;
      int path_count = 0;
      FOREACH_PATH_END_DATA(path_end, mode, path_data)
      {
        total_skew += path_data->getSkew();
        ++path_count;
      }
      CTSAPIInst.saveToLog("Avg Skew = ", fix_point_str(FS_TO_NS(total_skew)), " / ", path_count, " = ",
                           fix_point_str(FS_TO_NS(total_skew / path_count)));
    }
  }
}

void CTSAPI::slackLog() const
{
  CTSAPIInst.saveToLog("\n\n##Slack Log##");
  auto clk_list = _timing_engine->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto hold_tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMin);
    auto hold_wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMin);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    CTSAPIInst.saveToLog("\nClk name: ", clk_name);
    CTSAPIInst.saveToLog("Setup (Max) WNS: ", setup_wns);
    CTSAPIInst.saveToLog("Setup (Max) TNS: ", setup_tns);
    CTSAPIInst.saveToLog("Hold (Min) WNS: ", hold_wns);
    CTSAPIInst.saveToLog("Hold (Min) TNS: ", hold_tns);
    CTSAPIInst.saveToLog("Suggest Freq: ", suggest_freq);
  });
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

// debug

void CTSAPI::writeVerilog() const
{
  _timing_engine->writeVerilog("cts_debug.v");
}

void CTSAPI::toPyArray(const icts::Point& point, const std::string& label)
{
  CTSAPIInst.saveToLog(label, "=[[", point.x(), ",", point.y(), "]]");
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
    if (string_list.size() == 2 && (string_list[0] == "steiner")) {
      return _timing_engine->makeOrFindRCTreeNode(sta_net, std::stoi(string_list[1]));
    } else {
      LOG_FATAL << "Unknown pin name: " << name;
    }
  } else {
    for (auto pin : eval_net.get_pins()) {
      if (pin->get_instance() == inst) {
        return makePinRCTreeNode(pin);
      }
    }
  }
  return nullptr;
}

ista::RctNode* CTSAPI::makePinRCTreeNode(icts::CtsPin* pin)
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
  // remove all "\" in inst_name
  auto name = pin_full_name;
  name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
  return _timing_engine->get_netlist()->findObj(name.c_str(), false, false).front();
}

ista::Net* CTSAPI::findStaNet(const icts::EvalNet& eval_net) const
{
  return findStaNet(eval_net.get_clk_net()->get_net_name());
}

ista::Net* CTSAPI::findStaNet(const std::string& name) const
{
  return _timing_engine->get_netlist()->findNet(name.c_str());
}

double CTSAPI::getCapacitance(const double& wire_length, const int& level) const
{
  std::optional<double> width = std::nullopt;
  return getStaDbAdapter()->getCapacitance(level, wire_length, width);
}

double CTSAPI::getResistance(const double& wire_length, const int& level) const
{
  std::optional<double> width = std::nullopt;
  return getStaDbAdapter()->getResistance(level, wire_length, width) / 2;
}

ista::TimingIDBAdapter* CTSAPI::getStaDbAdapter() const
{
  return dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter());
}

}  // namespace icts