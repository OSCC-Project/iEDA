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
#include "idm.h"
#include "salt/base/flute.h"
#include "salt/salt.h"
#include "timing_db.hh"

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
  // auto routing_type_list = {"WLM", "HPWL", "FLUTE", "SALT", "EGR", "DR"}
  initStaEngine();
  auto routing_type_list = {"HPWL", "FLUTE", "SALT", "EGR", "DR"};
  std::ranges::for_each(routing_type_list, [&](const std::string& routing_type) {
    if (routing_type == "EGR" || routing_type == "DR") {
      callRT(routing_type);
    } else {
      buildRCTree(routing_type);
    }
    updateResult(routing_type);
  });
}

void InitSTA::evalTiming(const std::string& routing_type, const bool& rt_done)
{
  initStaEngine();
  if (routing_type == "EGR" || routing_type == "DR") {
    if (!rt_done) {
      callRT(routing_type);
    }
  } else {
    buildRCTree(routing_type);
  }

  updateResult(routing_type);
}

void InitSTA::leaglization(const std::vector<std::shared_ptr<salt::Pin>>& pins)
{
  if (pins.empty()) {
    return;
  }

  std::set<std::pair<double, double>> loc_set;
  bool is_legal = true;
  for (size_t i = 0; i < pins.size(); ++i) {
    if (loc_set.contains(std::make_pair(pins[i]->loc.x, pins[i]->loc.y))) {
      is_legal = false;
      break;
    }
    loc_set.insert(std::make_pair(pins[i]->loc.x, pins[i]->loc.y));
  }
  if (is_legal) {
    return;
  }

  // find all duplicated locations, and move them to a new location, objective: no duplicated locations and minimum total movement
  // x: pin->loc.x
  // y: pin->loc.y
  // Step 1: Group pins by their (x, y) locations
  std::map<std::pair<int, int>, std::vector<std::shared_ptr<salt::Pin>>> loc_map;
  for (const auto& pin : pins) {
    std::pair<int, int> coord = {pin->loc.x, pin->loc.y};
    loc_map[coord].push_back(pin);
  }

  // Step 2: Initialize a set to keep track of occupied locations
  std::unordered_set<long long> occupied;
  // Helper lambda to encode (x, y) into a unique key
  auto encode = [](int x, int y) -> long long {
    // Assuming x and y are within reasonable bounds to prevent overflow
    return static_cast<long long>(x) * 100000000 + y;
  };

  // Populate the occupied set with initial locations
  for (const auto& [coord, pin_list] : loc_map) {
    occupied.insert(encode(coord.first, coord.second));
  }

  // Step 3: Collect all pins that need to be moved
  std::vector<std::shared_ptr<salt::Pin>> pins_to_move;
  for (const auto& [coord, pin_list] : loc_map) {
    if (pin_list.size() > 1) {
      // Keep the first pin, move the rest
      for (size_t i = 1; i < pin_list.size(); ++i) {
        pins_to_move.push_back(pin_list[i]);
      }
    }
  }

  // Step 4: Define directions for BFS (8-connected grid)
  const std::vector<std::pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

  // Step 5: For each pin to move, find the nearest available location
  for (const auto& pin : pins_to_move) {
    int start_x = pin->loc.x;
    int start_y = pin->loc.y;

    // BFS initialization
    std::queue<std::pair<int, int>> q;
    std::unordered_set<long long> visited;
    q.push({start_x, start_y});
    visited.insert(encode(start_x, start_y));

    bool found = false;
    int new_x = start_x;
    int new_y = start_y;

    while (!q.empty() && !found) {
      int current_level_size = q.size();
      for (int i = 0; i < current_level_size; ++i) {
        auto [x, y] = q.front();
        q.pop();

        // Explore all directions
        for (const auto& [dx, dy] : directions) {
          int nx = x + dx;
          int ny = y + dy;
          long long key = encode(nx, ny);

          if (visited.find(key) == visited.end()) {
            // Check if the location is free
            if (occupied.find(key) == occupied.end()) {
              // Found a free location
              new_x = nx;
              new_y = ny;
              occupied.insert(key);
              found = true;
              break;
            }
            // Mark as visited and add to queue for further exploration
            visited.insert(key);
            q.push({nx, ny});
          }
        }
        if (found)
          break;
      }
    }

    if (!found) {
      // If no free location found in the immediate vicinity, expand the search
      // This can be optimized or have a maximum search radius
      // For simplicity, we'll assign a far away location
      LOG_FATAL << "No free location found for pin x=" << start_x << ", y=" << start_y;
    }

    // Update the pin's location
    pin->loc.x = new_x;
    pin->loc.y = new_y;
  }
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
  auto clock_layer_name = routing_layers.size() >= 4 ? routing_layers[routing_layers.size() - 4]->get_name() : logic_layer_name;
  // Hard Code, consider the clock layer is the last 4rd layer
  const std::string temp_path = dmInst->get_config().get_output_path() + "/rt/rt_temp_directory";
  config_map.insert({"-temp_directory_path", temp_path});
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
  LOG_FATAL_IF(routing_type != "WLM" && routing_type != "HPWL" && routing_type != "FLUTE" && routing_type != "SALT")
      << "The routing type: " << routing_type << " is not supported.";

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
      = routing_layers.size() >= 4 ? routing_layers.size() - 4 : logic_layer;  // Hard Code, consider the clock layer is the last 3rd layer
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
      auto loads = sta_net->getLoads();

      if (loads.empty()) {
        continue;
      }
      auto* driver = sta_net->getDriver();
      auto front_node = STA_INST->makeOrFindRCTreeNode(driver);

      double res = 0;  // rc TBD
      double cap = 0;  // rc TBD

      for (auto load : loads) {
        auto back_node = STA_INST->makeOrFindRCTreeNode(load);
        STA_INST->makeResistor(sta_net, front_node, back_node, res);
        STA_INST->incrCap(front_node, cap / 2, true);
        STA_INST->incrCap(back_node, cap / 2, true);
      }
    }

    if (routing_type == "HPWL") {
      auto loads = sta_net->getLoads();

      if (loads.empty()) {
        continue;
      }

      auto* driver = sta_net->getDriver();
      auto driver_loc = idb_adapter->idbLocation(driver);
      auto front_node = STA_INST->makeOrFindRCTreeNode(driver);

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

    if (routing_type == "FLUTE" || routing_type == "SALT") {
      std::vector<ista::DesignObject*> pin_ports = {sta_net->getDriver()};
      std::ranges::copy(sta_net->getLoads(), std::back_inserter(pin_ports));
      if (pin_ports.size() < 2) {
        continue;
      }
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
      leaglization(salt_pins);
      salt::Net salt_net;
      salt_net.init(0, sta_net->get_name(), salt_pins);

      salt::Tree salt_tree;
      if (routing_type == "FLUTE") {
        salt::FluteBuilder flute_builder;
        flute_builder.Run(salt_net, salt_tree);
      } else {
        salt::SaltBuilder salt_builder;
        salt_builder.Run(salt_net, salt_tree, 0);
      }
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

double InitSTA::getEarlySlack(const std::string& pin_name) const
{
  double early_slack = 0;

  auto rise_value = STA_INST->getSlack(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
  auto fall_value = STA_INST->getSlack(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  early_slack = std::min(rise_value.value(), fall_value.value());

  return early_slack;
}

double InitSTA::getLateSlack(const std::string& pin_name) const
{
  double late_slack = 0;

  auto rise_value = STA_INST->getSlack(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  auto fall_value = STA_INST->getSlack(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  late_slack = std::min(rise_value.value(), fall_value.value());

  return late_slack;
}

double InitSTA::getArrivalEarlyTime(const std::string& pin_name) const
{
  double arrival_early_time = 0;

  auto rise_value = STA_INST->getAT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
  auto fall_value = STA_INST->getAT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MIN;
  }

  arrival_early_time = std::min(rise_value.value(), fall_value.value());

  return arrival_early_time;
}

double InitSTA::getArrivalLateTime(const std::string& pin_name) const
{
  double arrival_late_time = 0;

  auto rise_value = STA_INST->getAT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  auto fall_value = STA_INST->getAT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MIN;
  }

  arrival_late_time = std::max(rise_value.value(), fall_value.value());

  return arrival_late_time;
}

double InitSTA::getRequiredEarlyTime(const std::string& pin_name) const
{
  double required_early_time = 0;

  auto rise_value = STA_INST->getRT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kRise);
  auto fall_value = STA_INST->getRT(pin_name.c_str(), ista::AnalysisMode::kMin, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  required_early_time = std::max(rise_value.value(), fall_value.value());

  return required_early_time;
}

double InitSTA::getRequiredLateTime(const std::string& pin_name) const
{
  double required_late_time = 0;

  auto rise_value = STA_INST->getRT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  auto fall_value = STA_INST->getRT(pin_name.c_str(), ista::AnalysisMode::kMax, ista::TransType::kFall);

  if (rise_value == std::nullopt || fall_value == std::nullopt) {
    return DBL_MAX;
  }

  required_late_time = std::min(rise_value.value(), fall_value.value());

  return required_late_time;
}

double InitSTA::reportWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return STA_INST->getWNS(clock_name, mode);
}

double InitSTA::reportTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return STA_INST->getTNS(clock_name, mode);
}

void InitSTA::updateTiming(const std::vector<TimingNet*>& timing_net_list, int32_t dbu_unit)
{
  // get sta_netlist
  auto netlist = STA_INST->get_netlist();

  // reset rc info in timing graph
  STA_INST->get_ista()->resetAllRcNet();

  for (auto& eval_net : timing_net_list) {
    ista::Net* ista_net = netlist->findNet(eval_net->net_name.c_str());

    std::vector<std::pair<TimingPin*, TimingPin*>> pin_pair_list = eval_net->pin_pair_list;

    for (auto pin_pair : pin_pair_list) {
      TimingPin* first_pin = pin_pair.first;
      TimingPin* second_pin = pin_pair.second;

      ista::RctNode* first_node = nullptr;
      ista::RctNode* second_node = nullptr;

      if (first_pin->is_real_pin) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(first_pin->pin_name.c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(first_pin->pin_name.c_str());
        }
        first_node = STA_INST->makeOrFindRCTreeNode(pin_port);
      } else {
        first_node = STA_INST->makeOrFindRCTreeNode(ista_net, first_pin->pin_id);
      }

      if (second_pin->is_real_pin) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(second_pin->pin_name.c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(second_pin->pin_name.c_str());
        }
        second_node = STA_INST->makeOrFindRCTreeNode(pin_port);
      } else {
        second_node = STA_INST->makeOrFindRCTreeNode(ista_net, second_pin->pin_id);
      }

      int64_t wire_length = 0;
      wire_length = std::abs(first_pin->x - second_pin->x) + std::abs(first_pin->y - second_pin->y);
      // wire_length = first_pin->get_coord().computeDist(second_pin->get_coord());

      std::optional<double> width = std::nullopt;

      // if (_unit == -1) {
      //   _unit = 1000;
      //   std::cout << "Setting the default unit as 1000" << std::endl;
      // }

      double cap
          = dynamic_cast<ista::TimingIDBAdapter*>(STA_INST->get_db_adapter())->getCapacitance(1, wire_length / 1.0 / dbu_unit, width);
      double res = dynamic_cast<ista::TimingIDBAdapter*>(STA_INST->get_db_adapter())->getResistance(1, wire_length / 1.0 / dbu_unit, width);

      // // tmp for test
      // double cap = (wire_length / 1.0 / _unit) * 1.6e-16;
      // double res = (wire_length / 1.0 / _unit) * 2.535;

      STA_INST->makeResistor(ista_net, first_node, second_node, res);
      STA_INST->incrCap(first_node, cap / 2);
      STA_INST->incrCap(second_node, cap / 2);
    }
    STA_INST->updateRCTreeInfo(ista_net);
  }
  STA_INST->updateTiming();
  STA_INST->reportTiming();
}

void InitSTA::updateTiming(const std::vector<TimingNet*>& timing_net_list, const std::vector<std::string>& name_list,
                           const int& propagation_level, int32_t dbu_unit)
{
  // get sta_netlist
  auto netlist = STA_INST->get_netlist();

  for (auto& eval_net : timing_net_list) {
    ista::Net* ista_net = netlist->findNet(eval_net->net_name.c_str());

    // reset rc info in timing graph
    STA_INST->get_ista()->resetRcNet(ista_net);

    std::vector<std::pair<TimingPin*, TimingPin*>> pin_pair_list = eval_net->pin_pair_list;

    for (auto pin_pair : pin_pair_list) {
      TimingPin* first_pin = pin_pair.first;
      TimingPin* second_pin = pin_pair.second;

      ista::RctNode* first_node = nullptr;
      ista::RctNode* second_node = nullptr;

      if (first_pin->is_real_pin) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(first_pin->pin_name.c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(first_pin->pin_name.c_str());
        }
        first_node = STA_INST->makeOrFindRCTreeNode(pin_port);
      } else {
        first_node = STA_INST->makeOrFindRCTreeNode(ista_net, first_pin->pin_id);
      }

      if (second_pin->is_real_pin) {
        ista::DesignObject* pin_port = nullptr;
        auto pin_port_list = netlist->findPin(second_pin->pin_name.c_str(), false, false);
        if (!pin_port_list.empty()) {
          pin_port = pin_port_list.front();
        } else {
          pin_port = netlist->findPort(second_pin->pin_name.c_str());
        }
        second_node = STA_INST->makeOrFindRCTreeNode(pin_port);
      } else {
        second_node = STA_INST->makeOrFindRCTreeNode(ista_net, second_pin->pin_id);
      }

      // int64_t wire_length = 0;
      // wire_length = first_pin->get_coord().computeDist(second_pin->get_coord());
      int64_t wire_length = 0;
      wire_length = std::abs(first_pin->x - second_pin->x) + std::abs(first_pin->y - second_pin->y);

      std::optional<double> width = std::nullopt;

      double cap
          = dynamic_cast<ista::TimingIDBAdapter*>(STA_INST->get_db_adapter())->getCapacitance(1, wire_length / 1.0 / dbu_unit, width);
      double res = dynamic_cast<ista::TimingIDBAdapter*>(STA_INST->get_db_adapter())->getResistance(1, wire_length / 1.0 / dbu_unit, width);

      STA_INST->makeResistor(ista_net, first_node, second_node, res);
      STA_INST->incrCap(first_node, cap / 2);
      STA_INST->incrCap(second_node, cap / 2);
    }
    STA_INST->updateRCTreeInfo(ista_net);
  }

  for (auto& name : name_list) {
    STA_INST->moveInstance(name.c_str(), propagation_level);
  }

  // STA_INST->incrUpdateTiming();
  STA_INST->updateTiming();
}

bool InitSTA::isClockNet(const std::string& net_name) const
{
  return STA_INST->isClockNet(net_name.c_str());
}

}  // namespace ieval