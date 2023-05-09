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
#include "TimingEval.hpp"

#include <float.h>

#include <map>

#include "EvalUtil.hpp"

namespace eval {

TimingEval::TimingEval(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, std::vector<const char*> lib_file_path_list,
                       const char* sdc_file_path)
{
  _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  _timing_engine->set_num_threads(40);
  _timing_engine->set_design_work_space(sta_workspace_path);
  _timing_engine->readLiberty(lib_file_path_list);
  auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(_timing_engine->get_ista());
  db_adapter->set_idb(idb_builder);
  db_adapter->convertDBToTimingNetlist();
  _timing_engine->set_db_adapter(std::move(db_adapter));
  _timing_engine->readSdc(sdc_file_path);
  _timing_engine->initRcTree();
  _timing_engine->buildGraph();
  _timing_engine->updateTiming();
}

TimingNet* TimingEval::add_timing_net(const std::string& name)
{
  TimingNet* timing_net = new TimingNet();
  timing_net->set_name(name);
  _timing_net_list.push_back(timing_net);
  return timing_net;
}

void TimingEval::add_timing_net(const std::string& name, const std::vector<std::pair<TimingPin*, TimingPin*>>& pin_pair_list)
{
  TimingNet* timing_net = new TimingNet();
  timing_net->set_name(name);
  for (auto& pin : pin_pair_list) {
    timing_net->add_pin_pair(pin.first, pin.second);
  }
  _timing_net_list.push_back(timing_net);
}

// public
void TimingEval::estimateDelay(std::vector<std::string> lef_file_path_list, std::string def_file_path, const char* sta_workspace_path,
                               std::vector<const char*> lib_file_path_list, const char* sdc_file_path)
{
  idb::IdbBuilder* idb_builder = new idb::IdbBuilder();
  idb_builder->buildLef(lef_file_path_list);
  idb_builder->buildDef(def_file_path);
  estimateDelay(idb_builder, sta_workspace_path, lib_file_path_list, sdc_file_path);
}

void TimingEval::estimateDelay(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, std::vector<const char*> lib_file_path_list,
                               const char* sdc_file_path)
{
  // start TimingEngine
  _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  _timing_engine->set_num_threads(40);
  _timing_engine->set_design_work_space(sta_workspace_path);
  _timing_engine->readLiberty(lib_file_path_list);
  auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(_timing_engine->get_ista());
  db_adapter->set_idb(idb_builder);
  db_adapter->convertDBToTimingNetlist();
  _timing_engine->set_db_adapter(std::move(db_adapter));
  _timing_engine->readSdc(sdc_file_path);
  _timing_engine->initRcTree();
  _timing_engine->buildGraph();
  _timing_engine->updateTiming();

  // get sta_netlist
  auto netlist = _timing_engine->get_netlist();
  for (auto& eval_net : _timing_net_list) {
    // for debug
    double cap_sum = 0.0;
    double res_sum = 0.0;
    double wl_sum = 0.0;

    ista::Net* ista_net = netlist->findNet(eval_net->get_name().c_str());

    std::vector<std::pair<TimingPin*, TimingPin*>> pin_pair_list = eval_net->get_pin_pair_list();

    for (auto pin_pair : pin_pair_list) {
      TimingPin* first_pin = pin_pair.first;
      TimingPin* second_pin = pin_pair.second;

      ista::RctNode* first_node = nullptr;
      ista::RctNode* second_node = nullptr;

      if (first_pin->isRealPin()) {
        auto pin_port = netlist->findPin(first_pin->get_name().c_str(), false, false).front();
        first_node = _timing_engine->makeOrFindRCTreeNode(pin_port);
      } else {
        first_node = _timing_engine->makeOrFindRCTreeNode(ista_net, first_pin->get_id());
      }

      if (second_pin->isRealPin()) {
        auto pin_port = netlist->findPin(second_pin->get_name().c_str(), false, false).front();
        second_node = _timing_engine->makeOrFindRCTreeNode(pin_port);
      } else {
        second_node = _timing_engine->makeOrFindRCTreeNode(ista_net, second_pin->get_id());
      }

      // consider cross-layer wirelength
      int64_t wire_length = 0;
      int first_pin_layer_id = first_pin->get_layer_id();
      int second_pin_layer_id = second_pin->get_layer_id();

      if (first_pin->get_coord() == second_pin->get_coord()) {
        idb::IdbLayers* layers = idb_builder->get_lef_service()->get_layout()->get_layers();
        auto first_pin_layer = dynamic_cast<idb::IdbLayerRouting*>(layers->find_routing_layer(first_pin_layer_id));
        auto second_pin_layer = dynamic_cast<idb::IdbLayerRouting*>(layers->find_routing_layer(second_pin_layer_id));
        auto first_pin_layer_height = first_pin_layer->get_thickness();
        auto second_pin_layer_height = second_pin_layer->get_thickness();

        if (first_pin_layer_id > second_pin_layer_id) {
          wire_length = second_pin_layer_height * 2;
        } else {
          wire_length = first_pin_layer_height * 2;
        }
      } else {
        wire_length = first_pin->get_coord().computeDist(second_pin->get_coord());
      }

      std::optional<double> width = std::nullopt;
      int32_t unit = idb_builder->get_def_service()->get_design()->get_units()->get_micron_dbu();
      double cap = dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter())
                       ->getCapacitance(first_pin_layer_id, wire_length / 1.0 / unit, width);
      double res = dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter())
                       ->getResistance(first_pin_layer_id, wire_length / 1.0 / unit, width);

      // for debug
      cap_sum += cap;
      res_sum += res;
      wl_sum += wire_length;

      _timing_engine->makeResistor(ista_net, first_node, second_node, res);
      _timing_engine->incrCap(first_node, cap / 2);
      _timing_engine->incrCap(second_node, cap / 2);

      // for debug
      if (eval_net->get_name() == "hold_net_225" || eval_net->get_name() == "hold_net_221" || eval_net->get_name() == "clk_peri_net_1_57"
          || eval_net->get_name() == "clk_peri_net_1_50") {
        std::cout << "found net" << std::endl;
      }
    }
    _timing_engine->updateRCTreeInfo(ista_net);
  }
  _timing_engine->updateTiming();
  _timing_engine->reportTiming();
}

void TimingEval::updateEstimateDelay(const std::vector<TimingNet*>& timing_net_list)
{
  // set ieval netlist
  _timing_net_list = timing_net_list;

  // get sta_netlist
  auto netlist = _timing_engine->get_netlist();

  // reset rc info in timing graph
  _timing_engine->get_ista()->resetAllRcNet();

  for (auto& eval_net : _timing_net_list) {
    ista::Net* ista_net = netlist->findNet(eval_net->get_name().c_str());

    std::vector<std::pair<TimingPin*, TimingPin*>> pin_pair_list = eval_net->get_pin_pair_list();

    for (auto pin_pair : pin_pair_list) {
      TimingPin* first_pin = pin_pair.first;
      TimingPin* second_pin = pin_pair.second;

      ista::RctNode* first_node = nullptr;
      ista::RctNode* second_node = nullptr;

      if (first_pin->isRealPin()) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(first_pin->get_name().c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(first_pin->get_name().c_str());
        }
        first_node = _timing_engine->makeOrFindRCTreeNode(pin_port);
      } else {
        first_node = _timing_engine->makeOrFindRCTreeNode(ista_net, first_pin->get_id());
      }

      if (second_pin->isRealPin()) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(second_pin->get_name().c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(second_pin->get_name().c_str());
        }
        second_node = _timing_engine->makeOrFindRCTreeNode(pin_port);
      } else {
        second_node = _timing_engine->makeOrFindRCTreeNode(ista_net, second_pin->get_id());
      }

      int64_t wire_length = 0;
      wire_length = first_pin->get_coord().computeDist(second_pin->get_coord());

      std::optional<double> width = std::nullopt;

      if (_unit == -1) {
        _unit = 1000;
        std::cout << "Setting the default unit as 1000" << std::endl;
      }

      double cap
          = dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter())->getCapacitance(1, wire_length / 1.0 / _unit, width);
      double res
          = dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter())->getResistance(1, wire_length / 1.0 / _unit, width);

      _timing_engine->makeResistor(ista_net, first_node, second_node, res);
      _timing_engine->incrCap(first_node, cap / 2);
      _timing_engine->incrCap(second_node, cap / 2);
    }
    _timing_engine->updateRCTreeInfo(ista_net);
  }
  _timing_engine->updateTiming();
  _timing_engine->reportTiming();
}

void TimingEval::updateEstimateDelay(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                                     int propagation_level)
{
  // set ieval netlist
  _timing_net_list = timing_net_list;

  // get sta_netlist
  auto netlist = _timing_engine->get_netlist();

  for (auto& eval_net : _timing_net_list) {
    ista::Net* ista_net = netlist->findNet(eval_net->get_name().c_str());

    std::vector<std::pair<TimingPin*, TimingPin*>> pin_pair_list = eval_net->get_pin_pair_list();

    for (auto pin_pair : pin_pair_list) {
      TimingPin* first_pin = pin_pair.first;
      TimingPin* second_pin = pin_pair.second;

      ista::RctNode* first_node = nullptr;
      ista::RctNode* second_node = nullptr;

      if (first_pin->isRealPin()) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(first_pin->get_name().c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(first_pin->get_name().c_str());
        }
        first_node = _timing_engine->makeOrFindRCTreeNode(pin_port);
      } else {
        first_node = _timing_engine->makeOrFindRCTreeNode(ista_net, first_pin->get_id());
      }

      if (second_pin->isRealPin()) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(second_pin->get_name().c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(second_pin->get_name().c_str());
        }
        second_node = _timing_engine->makeOrFindRCTreeNode(pin_port);
      } else {
        second_node = _timing_engine->makeOrFindRCTreeNode(ista_net, second_pin->get_id());
      }

      int64_t wire_length = 0;
      wire_length = first_pin->get_coord().computeDist(second_pin->get_coord());

      std::optional<double> width = std::nullopt;

      double cap
          = dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter())->getCapacitance(1, wire_length / 1.0 / _unit, width);
      double res
          = dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter())->getResistance(1, wire_length / 1.0 / _unit, width);

      _timing_engine->makeResistor(ista_net, first_node, second_node, res);
      _timing_engine->incrCap(first_node, cap / 2);
      _timing_engine->incrCap(second_node, cap / 2);
    }
    _timing_engine->updateRCTreeInfo(ista_net);
  }

  for (auto& name : name_list) {
    _timing_engine->moveInstance(name.c_str(), propagation_level);
  }

  _timing_engine->incrUpdateTiming();
  _timing_engine->reportTiming();
}

void TimingEval::initTimingEngine(int32_t unit)
{
  _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  _unit = unit;
}

void TimingEval::initTimingEngine(idb::IdbBuilder* idb_builder, const char* sta_workspace_path, std::vector<const char*> lib_file_path_list,
                                  const char* sdc_file_path)
{
  _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  _timing_engine->set_num_threads(8);
  _timing_engine->set_design_work_space(sta_workspace_path);
  _timing_engine->readLiberty(lib_file_path_list);
  auto db_adapter = std::make_unique<ista::TimingIDBAdapter>(_timing_engine->get_ista());
  db_adapter->set_idb(idb_builder);
  db_adapter->convertDBToTimingNetlist();
  _timing_engine->set_db_adapter(std::move(db_adapter));
  _timing_engine->readSdc(sdc_file_path);
  _timing_engine->initRcTree();
  _timing_engine->buildGraph();
  _timing_engine->updateTiming();
  _unit = idb_builder->get_def_service()->get_design()->get_units()->get_micron_dbu();
}

double TimingEval::getEarlySlack(const std::string& pin_name) const
{
  double early_slack = 0;

  auto rise_value = _timing_engine->reportSlack(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
  auto fall_value = _timing_engine->reportSlack(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  early_slack = std::min(rise_value.value(), fall_value.value());

  return early_slack;
}

double TimingEval::getLateSlack(const std::string& pin_name) const
{
  double late_slack = 0;

  auto rise_value = _timing_engine->reportSlack(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  auto fall_value = _timing_engine->reportSlack(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  late_slack = std::min(rise_value.value(), fall_value.value());

  return late_slack;
}

double TimingEval::getArrivalEarlyTime(const std::string& pin_name) const
{
  double arrival_early_time = 0;

  auto rise_value = _timing_engine->reportAT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
  auto fall_value = _timing_engine->reportAT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MIN;
  }

  arrival_early_time = std::min(rise_value.value(), fall_value.value());

  return arrival_early_time;
}

double TimingEval::getArrivalLateTime(const std::string& pin_name) const
{
  double arrival_late_time = 0;

  auto rise_value = _timing_engine->reportAT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  auto fall_value = _timing_engine->reportAT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MIN;
  }

  arrival_late_time = std::max(rise_value.value(), fall_value.value());

  return arrival_late_time;
}

double TimingEval::getRequiredEarlyTime(const std::string& pin_name) const
{
  double required_early_time = 0;

  auto rise_value = _timing_engine->reportRT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
  auto fall_value = _timing_engine->reportRT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  required_early_time = std::max(rise_value.value(), fall_value.value());

  return required_early_time;
}

double TimingEval::getRequiredLateTime(const std::string& pin_name) const
{
  double required_late_time = 0;

  auto rise_value = _timing_engine->reportRT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  auto fall_value = _timing_engine->reportRT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  required_late_time = std::min(rise_value.value(), fall_value.value());

  return required_late_time;
}

double TimingEval::reportWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return _timing_engine->reportWNS(clock_name, mode);
}

double TimingEval::reportTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return _timing_engine->reportTNS(clock_name, mode);
}

// get net timing info ,called by iRT
double TimingEval::getNetDelay(const char* net_name, const char* load_pin_name, ista::AnalysisMode mode, ista::TransType trans_type)
{
  double net_delay = 0;
  net_delay = _timing_engine->reportNetDelay(net_name, load_pin_name, mode, trans_type);
  return net_delay;
}

bool TimingEval::checkClockName(const char* clock_name)
{
  const auto& sta_clock_list = _timing_engine->getClockList();
  bool isFindName = false;
  for (size_t i = 0; i < sta_clock_list.size(); ++i) {
    if (clock_name == sta_clock_list[i]->get_clock_name()) {
      isFindName = true;
    }
  }
  if (!isFindName) {
    std::cout << "Can not find " << clock_name << ". Pleace check the CLOCK_NAME is exist." << std::endl;
    std::cout << "clock_name FOR EXAMPLE: " << sta_clock_list[0]->get_clock_name() << std::endl;
  }
  return isFindName;
}

std::vector<const char*> TimingEval::getClockNameList()
{
  const auto& sta_clock_list = _timing_engine->getClockList();
  std::vector<const char*> clock_name_list;
  clock_name_list.reserve(sta_clock_list.size());
  for (size_t i = 0; i < sta_clock_list.size(); ++i) {
    clock_name_list.emplace_back(sta_clock_list[i]->get_clock_name());
  }
  return clock_name_list;
}

}  // namespace eval
