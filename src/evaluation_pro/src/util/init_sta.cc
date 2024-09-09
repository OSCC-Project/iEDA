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
 * @file init_sta.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 */

#include "init_sta.hh"

#include "RTInterface.hpp"
#include "api/PowerEngine.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "feature_irt.h"
#include "idm.h"
#include "salt/base/flute.h"
#include "salt/salt.h"
namespace ieval {
#define STA_INST (ista::TimingEngine::getOrCreateTimingEngine())
#define RT_INST (irt::RTInterface::getInst())
#define PW_INST (ipower::PowerEngine::getOrCreatePowerEngine())

InitSTA* InitSTA::_init_sta = nullptr;

InitSTA::~InitSTA()
{
  PW_INST->destroyPowerEngine();
  STA_INST->destroyTimingEngine();
}

InitSTA* InitSTA::getInst()
{
  if (_init_sta == nullptr) {
    _init_sta = new InitSTA();
  }
  return _init_sta;
}

void InitSTA::destroyInst()
{
  delete _init_sta;
  _init_sta = nullptr;
}

void InitSTA::runSTA()
{
  // auto routing_type_list = {"WLM", "HPWL", "FLUTE", "EGR", "DR"};
  initStaEngine();
  auto routing_type_list = {"HPWL", "FLUTE", "EGR", "DR"};
  std::ranges::for_each(routing_type_list, [&](const std::string& routing_type) {
    if (routing_type == "EGR" || routing_type == "DR") {
      callRT(routing_type);
    } else {
      buildRCTree(routing_type);
    }
    updateResult(routing_type);
  });
}

void InitSTA::initStaEngine()
{
  if (STA_INST->isBuildGraph()) {
    return;
  }
  STA_INST->readLiberty(dmInst->get_config().get_lib_paths());
  auto sta_db_adapter = std::make_unique<ista::TimingIDBAdapter>(STA_INST->get_ista());
  sta_db_adapter->set_idb(dmInst->get_idb_builder());
  sta_db_adapter->convertDBToTimingNetlist();
  STA_INST->set_db_adapter(std::move(sta_db_adapter));
  STA_INST->readSdc(dmInst->get_config().get_sdc_path().c_str());
  STA_INST->buildGraph();
  STA_INST->initRcTree();
}

void InitSTA::callRT(const std::string& routing_type)
{
  std::map<std::string, std::any> config_map;
  auto* idb_layout = dmInst->get_idb_lef_service()->get_layout();
  auto routing_layers = idb_layout->get_layers()->get_routing_layers();
  auto logic_layer_name = routing_layers.size() >= 2 ? routing_layers[1]->get_name() : routing_layers[0]->get_name();
  auto clock_layer_name = routing_layers.size() >= 3 ? routing_layers[routing_layers.size() - 3]->get_name() : logic_layer_name;
  // Hard Code, consider the clock layer is the last 3rd layer
  config_map.insert({"-bottom_routing_layer", logic_layer_name});
  config_map.insert({"-top_routing_layer", clock_layer_name});
  config_map.insert({"-enable_timing", 1});
  RT_INST.initRT(config_map);

  if (routing_type == "EGR") {
    RT_INST.runEGR();
  } else if (routing_type == "DR") {
    RT_INST.runRT();
  }
  RT_INST.destroyRT();
}

void InitSTA::buildRCTree(const std::string& routing_type)
{
  LOG_FATAL_IF(routing_type != "WLM" && routing_type != "HPWL" && routing_type != "FLUTE") << "Unsupported routing type";

  auto* idb_adapter = dynamic_cast<ista::TimingIDBAdapter*>(STA_INST->get_db_adapter());

  // init
  // 1. wirelength calculation
  auto* idb = dmInst->get_idb_builder();
  auto* idb_design = idb->get_def_service()->get_design();
  auto dbu = idb_design->get_units()->get_micron_dbu();

  auto calc_length = [&](const int64_t& x1, const int64_t& y1, const int64_t& x2, const int64_t& y2) {
    // Manhattan distance
    auto dist = std::abs(x1 - x2) + std::abs(y1 - y2);
    return 1.0 * dist / dbu;
  };

  // 2. cap and res calculation, if is clock net, return the last layer, otherwise return the first layer

  std::optional<double> width = std::nullopt;
  auto* idb_layout = dmInst->get_idb_lef_service()->get_layout();
  auto routing_layers = idb_layout->get_layers()->get_routing_layers();
  auto logic_layer = routing_layers.size() >= 2 ? 2 : 1;
  auto clock_layer
      = routing_layers.size() >= 3 ? routing_layers.size() - 3 : logic_layer;  // Hard Code, consider the clock layer is the last 3rd layer
  auto calc_res = [&](const bool& is_clock, const double& wirelength) {
    if (!is_clock) {
      return idb_adapter->getResistance(logic_layer, wirelength, width);
    }
    return idb_adapter->getResistance(clock_layer, wirelength, width);
  };
  auto calc_cap = [&](const bool& is_clock, const double& wirelength) {
    if (!is_clock) {
      return idb_adapter->getCapacitance(logic_layer, wirelength, width);
    }
    return idb_adapter->getCapacitance(clock_layer, wirelength, width);
  };

  // main flow
  auto* netlist = STA_INST->get_netlist();
  ista::Net* sta_net = nullptr;
  FOREACH_NET(netlist, sta_net)
  {
    STA_INST->resetRcTree(sta_net);
    // WLM
    if (routing_type == "WLM") {
      LOG_ERROR << "STA does not support WLM, TBD...";
      auto* driver = sta_net->getDriver();
      auto front_node = STA_INST->makeOrFindRCTreeNode(driver);

      double res = 0;  // rc TBD
      double cap = 0;  // rc TBD

      auto loads = sta_net->getLoads();
      for (auto load : loads) {
        auto back_node = STA_INST->makeOrFindRCTreeNode(load);
        STA_INST->makeResistor(sta_net, front_node, back_node, res);
        STA_INST->incrCap(front_node, cap / 2, true);
        STA_INST->incrCap(back_node, cap / 2, true);
      }
    }

    if (routing_type == "HPWL") {
      auto* driver = sta_net->getDriver();
      auto driver_loc = idb_adapter->idbLocation(driver);
      auto front_node = STA_INST->makeOrFindRCTreeNode(driver);

      auto loads = sta_net->getLoads();
      for (auto load : loads) {
        auto load_loc = idb_adapter->idbLocation(load);
        auto wirelength = calc_length(driver_loc->get_x(), driver_loc->get_y(), load_loc->get_x(), load_loc->get_y());
        double res = calc_res(sta_net->isClockNet(), wirelength);
        double cap = calc_cap(sta_net->isClockNet(), wirelength);
        auto back_node = STA_INST->makeOrFindRCTreeNode(load);
        STA_INST->makeResistor(sta_net, front_node, back_node, res);
        STA_INST->incrCap(front_node, cap / 2, true);
        STA_INST->incrCap(back_node, cap / 2, true);
      }
    }

    if (routing_type == "FLUTE") {
      // Flute
      std::vector<ista::DesignObject*> pin_ports = {sta_net->getDriver()};
      std::ranges::copy(sta_net->getLoads(), std::back_inserter(pin_ports));

      // makr rc node
      auto make_rc_node = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
        if (salt_node->pin) {
          return STA_INST->makeOrFindRCTreeNode(pin_ports[salt_node->id]);
        }
        // steiner node
        return STA_INST->makeOrFindRCTreeNode(sta_net, salt_node->id);
      };

      std::vector<std::shared_ptr<salt::Pin>> salt_pins;
      salt_pins.reserve(pin_ports.size());
      for (size_t i = 0; i < pin_ports.size(); ++i) {
        auto pin_port = pin_ports[i];
        auto* idb_loc = idb_adapter->idbLocation(pin_port);
        LOG_ERROR_IF(idb_loc == nullptr) << "The location of pin port: " << pin_port->getFullName() << " is not found.";
        LOG_ERROR_IF(idb_loc->is_negative()) << "The location of pin port: " << pin_port->getFullName() << " is negative.";
        auto pin = std::make_shared<salt::Pin>(idb_loc->get_x(), idb_loc->get_y(), i);
        salt_pins.push_back(pin);
      }
      salt::Net salt_net;
      salt_net.init(0, sta_net->get_name(), salt_pins);

      salt::Tree salt_tree;
      salt::FluteBuilder flute_builder;
      flute_builder.Run(salt_net, salt_tree);
      salt_tree.UpdateId();

      auto source = salt_tree.source;
      auto build_rc_tree = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
        if (salt_node->id == source->id) {
          return;
        }
        auto parent_salt_node = salt_node->parent;
        auto front_node = make_rc_node(parent_salt_node);
        auto back_node = make_rc_node(salt_node);
        auto wirelength = calc_length(parent_salt_node->loc.x, parent_salt_node->loc.y, salt_node->loc.x, salt_node->loc.y);
        auto res = calc_res(sta_net->isClockNet(), wirelength);
        auto cap = calc_cap(sta_net->isClockNet(), wirelength);

        STA_INST->makeResistor(sta_net, front_node, back_node, res);
        STA_INST->incrCap(front_node, cap / 2, true);
        STA_INST->incrCap(back_node, cap / 2, true);
      };
      salt::TreeNode::postOrder(source, build_rc_tree);
    }
    // update rc tree
    STA_INST->updateRCTreeInfo(sta_net);
  }
  STA_INST->updateTiming();
}

void InitSTA::initPowerEngine()
{
  if (!PW_INST->isBuildGraph()) {
    PW_INST->get_power()->initPowerGraphData();
    PW_INST->get_power()->initToggleSPData();
  }
  PW_INST->get_power()->updatePower();
}

void InitSTA::updateResult(const std::string& routing_type)
{
  // update timing
  _timing[routing_type] = std::map<std::string, std::map<std::string, double>>();
  auto clk_list = STA_INST->getClockList();
  std::ranges::for_each(clk_list, [&](ista::StaClock* clk) {
    auto clk_name = clk->get_clock_name();
    auto setup_tns = STA_INST->getTNS(clk_name, AnalysisMode::kMax);
    auto setup_wns = STA_INST->getWNS(clk_name, AnalysisMode::kMax);
    auto hold_tns = STA_INST->getTNS(clk_name, AnalysisMode::kMin);
    auto hold_wns = STA_INST->getWNS(clk_name, AnalysisMode::kMin);
    auto suggest_freq = 1000.0 / (clk->getPeriodNs() - setup_wns);
    _timing[routing_type][clk_name]["setup_tns"] = setup_tns;
    _timing[routing_type][clk_name]["setup_wns"] = setup_wns;
    _timing[routing_type][clk_name]["hold_tns"] = hold_tns;
    _timing[routing_type][clk_name]["hold_wns"] = hold_wns;
    _timing[routing_type][clk_name]["suggest_freq"] = suggest_freq;
  });

  // update power
  initPowerEngine();
  _power[routing_type] = std::map<std::string, double>();
  _net_power[routing_type] = std::unordered_map<std::string, double>();
  double static_power = 0;
  for (const auto& data : PW_INST->get_power()->get_leakage_powers()) {
    static_power += data->get_leakage_power();
  }
  double dynamic_power = 0;
  for (const auto& data : PW_INST->get_power()->get_internal_powers()) {
    dynamic_power += data->get_internal_power();
  }
  for (const auto& data : PW_INST->get_power()->get_switch_powers()) {
    dynamic_power += data->get_switch_power();
    auto* net = dynamic_cast<ista::Net*>(data->get_design_obj());
    _net_power[routing_type][net->get_name()] = data->get_switch_power();
  }
  _power[routing_type]["static_power"] = static_power;
  _power[routing_type]["dynamic_power"] = dynamic_power;
}

}  // namespace ieval