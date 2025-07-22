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
 * @file init_sta.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-25
 */

#include "init_sta.hh"

#include <yaml-cpp/yaml.h>

#include <algorithm>

#include "RTInterface.hpp"
#include "api/PowerEngine.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "idm.h"
#include "lm_layout.h"
#include "salt/base/flute.h"
#include "salt/salt.h"
#include "timing_db.hh"
#include "usage/usage.hh"

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

void InitSTA::runLmSTA(ilm::LmLayout* lm_layout, std::string work_dir)
{
  initStaEngine();

  buildLmRCTree(lm_layout, work_dir);

  updateResult("Large Model");
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

  // find all duplicated locations, and move them to a new location, objective:
  // no duplicated locations and minimum total movement x: pin->loc.x y:
  // pin->loc.y Step 1: Group pins by their (x, y) locations
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
  LOG_FATAL_IF(routing_type != "WLM" && routing_type != "HPWL" && routing_type != "FLUTE" && routing_type != "SALT"
               && routing_type != "WireGraph")
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

  // 2. cap and res calculation, if is clock net, return the last layer,
  // otherwise return the first layer

  std::optional<double> width = std::nullopt;
  auto* idb_layout = dmInst->get_idb_lef_service()->get_layout();
  auto routing_layers = idb_layout->get_layers()->get_routing_layers();
  auto logic_layer = routing_layers.size() >= 2 ? 2 : 1;
  auto clock_layer = routing_layers.size() >= 4 ? routing_layers.size() - 4 : logic_layer;  // Hard Code, consider the clock layer
                                                                                            // is the last 3rd layer
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
  auto idb_nets = idb_design->get_net_list()->get_net_list();
  auto* sta_netlist = STA_INST->get_netlist();
  ista::Net* sta_net = nullptr;
  for (size_t net_id = 0; net_id < idb_nets.size(); ++net_id) {
    auto* idb_net = idb_nets[net_id];
    sta_net = sta_netlist->findNet(idb_net->get_net_name().c_str());
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

void InitSTA::buildLmRCTree(ilm::LmLayout* lm_layout, std::string work_dir)
{
  // init
  auto* idb = dmInst->get_idb_builder();
  auto* idb_design = idb->get_def_service()->get_design();

  auto* idb_layout = dmInst->get_idb_layout();
  auto idb_layers = idb_layout->get_layers();
  auto layers = idb_layers->get_layers();
  auto idb_layer_1st = dmInst->get_config().get_routing_layer_1st();
  // find the first layer which get_name == "idb_layer_1st", erase the layers
  // before it
  auto lm_layers = layers | std::views::drop_while([&](auto layer) { return layer->get_name() != idb_layer_1st; });

  // main flow
  auto idb_nets = idb_design->get_net_list()->get_net_list();
  auto* sta_netlist = STA_INST->get_netlist();
  ista::Net* sta_net = nullptr;
  auto& wire_graph = lm_layout->get_graph().get_net_map();
  for (size_t net_id = 0; net_id < idb_nets.size(); ++net_id) {
    auto* idb_net = idb_nets[net_id];
    std::string the_idb_net_name = idb_net->get_net_name();
    the_idb_net_name = ieda::Str::replace(the_idb_net_name, R"(\\)", "");
    sta_net = sta_netlist->findNet(the_idb_net_name.c_str());

    if (!wire_graph.contains(net_id)) {
      continue;
    }

    auto lm_net = wire_graph.at(net_id);
    auto idb_inst_pins = idb_net->get_instance_pin_list()->get_pin_list();
    auto io_pins = idb_net->get_io_pins()->get_pin_list();
    auto& wires = lm_net.get_wires();
    // Check corner case
    if (wires.size() == 1) {
      auto& wire = wires[0];
      auto connected_nodes = wire.get_connected_nodes();
      auto* source = connected_nodes.first;
      auto* target = connected_nodes.second;
      if (source == target) {
        continue;
      }
    }
    auto sta_pin_ports = sta_net->get_pin_ports();
    std::unordered_map<std::string, ista::DesignObject*> sta_pin_port_map;
    std::ranges::for_each(sta_pin_ports, [&](ista::DesignObject* pin_port) { sta_pin_port_map[pin_port->getFullName()] = pin_port; });
    std::unordered_map<ilm::LmNode*, ista::RctNode*> lm_node_map;
    auto make_or_find_rc_node = [&](ilm::LmNode* lm_node) {
      if (lm_node_map.contains(lm_node)) {
        return lm_node_map[lm_node];
      }
      int pin_id = lm_node->get_node_data()->get_pin_id();
      ista::RctNode* rc_node = nullptr;
      if (pin_id >= 0) {
        auto pin_name_pair = lm_layout->findPinName(pin_id);
        auto [inst_name, pin_type_name] = pin_name_pair;
        auto pin_name = !inst_name.empty() ? (inst_name + ":" + pin_type_name) : pin_type_name;
        pin_name.erase(std::remove(pin_name.begin(), pin_name.end(), '\\'), pin_name.end());
        auto* sta_pin_port = sta_pin_port_map[pin_name];
        rc_node = STA_INST->makeOrFindRCTreeNode(sta_pin_port);
      } else {
        // steiner node
        rc_node = STA_INST->makeOrFindRCTreeNode(sta_net, lm_node->get_node_id());
      }
      lm_node_map[lm_node] = rc_node;
      return rc_node;
    };
    auto calc_res_cap = [&](ilm::LmNetWire& wire) {
      auto connected_nodes = wire.get_connected_nodes();
      auto* source = connected_nodes.first;
      auto* target = connected_nodes.second;
      auto source_layer = source->get_layer_id();
      auto target_layer = target->get_layer_id();
      if (source_layer != target_layer) {
        // is via
        return std::make_pair(0.0, 0.0);
      }
      auto& paths = wire.get_paths();
      int wirelength = 0;
      std::ranges::for_each(paths, [&](auto& path) {
        auto x1 = path.first->get_x();
        auto y1 = path.first->get_y();
        auto x2 = path.second->get_x();
        auto y2 = path.second->get_y();
        wirelength += std::abs(x1 - x2) + std::abs(y1 - y2);
      });

      auto* routing_layer = dynamic_cast<IdbLayerRouting*>(lm_layers[source_layer]);

      auto dbu = idb_layout->get_units()->get_micron_dbu();
      auto segment_width = ((double) routing_layer->get_width()) / dbu;

      double wirelength_um = ((double) wirelength) / dbu;

      auto lef_resistance = routing_layer->get_resistance();
      auto lef_capacitance = routing_layer->get_capacitance();
      auto lef_edge_capacitance = routing_layer->get_edge_capacitance();

      auto res = lef_resistance * wirelength_um / segment_width;
      auto cap = (lef_capacitance * wirelength_um * segment_width) + (lef_edge_capacitance * 2 * (wirelength_um + segment_width));

      return std::make_pair(res, cap);
    };
    std::ranges::for_each(wires, [&](ilm::LmNetWire& wire) {
      auto connected_nodes = wire.get_connected_nodes();
      auto* source = connected_nodes.first;
      auto* target = connected_nodes.second;

      auto* front_node = make_or_find_rc_node(source);
      auto* back_node = make_or_find_rc_node(target);

      auto [res, cap] = calc_res_cap(wire);
      STA_INST->makeResistor(sta_net, front_node, back_node, res);
      STA_INST->incrCap(front_node, cap / 2, true);
      STA_INST->incrCap(back_node, cap / 2, true);
    });

    std::vector<std::string> pin_names;
    std::ranges::for_each(wires, [&pin_names, lm_layout](ilm::LmNetWire& wire) {
      auto connected_nodes = wire.get_connected_nodes();
      auto* source = connected_nodes.first;
      auto* target = connected_nodes.second;

      for (auto* connected_node : {source, target}) {
        int pin_id = connected_node->get_node_data()->get_pin_id();

        if (pin_id != -1) {
          auto [inst_name, pin_type_name] = lm_layout->findPinName(pin_id);
          auto pin_name = !inst_name.empty() ? (inst_name + ":" + pin_type_name) : pin_type_name;

          pin_names.push_back(pin_name);
        }
      }
    });

    LOG_INFO << "Net " << idb_net->get_net_name() << " has " << pin_names.size() << " pins";

    // update rc tree
    STA_INST->updateRCTreeInfo(sta_net);
  }
  STA_INST->updateTiming();
  STA_INST->get_ista()->reportUsedLibs();

  std::string path_dir = work_dir;
  STA_INST->set_design_work_space(path_dir.c_str());
  STA_INST->reportWirePaths(10000);
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

double InitSTA::getNetResistance(const std::string& net_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  if (rc_net && ista_net->getDriver()) {
    double resistance = rc_net->getNetResistance();
    return resistance;
  }

  return 0.0;
}
double InitSTA::getNetCapacitance(const std::string& net_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  if (rc_net && ista_net->getDriver()) {
    double load = rc_net->load();
    return load;
  }

  return 0.0;
}

double InitSTA::getNetSlew(const std::string& net_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  if (!ista_net->getDriver()) {
    return 0.0;
  }

  if (!rc_net) {
    return 0.0;
  }

  double driver_slew = 0.0;
  auto* driver = rc_net->get_net()->getDriver();
  if (driver && driver->isPin()) {
    driver_slew = STA_INST->getSlew(driver->getFullName().c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  }
  // get driver slew for net slew.
  auto loads = ista_net->getLoads();

  double sum_load_slew = 0.0;
  for (auto* load : loads) {
    std::string load_name = load->getFullName();
    sum_load_slew += rc_net->slew(load_name.c_str(), driver_slew, ista::AnalysisMode::kMax, ista::TransType::kRise).value_or(0.0);
  }
  double net_avg_slew = (sum_load_slew / loads.size()) - driver_slew;
  return net_avg_slew;
}

std::map<std::string, double> InitSTA::getAllNodesSlew(const std::string& net_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  double driver_slew = 0.0;
  auto* driver = rc_net->get_net()->getDriver();
  if (driver && driver->isPin()) {
    driver_slew = STA_INST->getSlew(driver->getFullName().c_str(), ista::AnalysisMode::kMax, ista::TransType::kRise);
  }

  std::map<std::string, double> all_node_slews;

  if (rc_net->rct()) {
    all_node_slews = rc_net->getAllNodeSlew(driver_slew, ista::AnalysisMode::kMax, ista::TransType::kRise);
  }

  return all_node_slews;
}

double InitSTA::getNetDelay(const std::string& net_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  // get load average delay for net delay.
  auto loads = ista_net->getLoads();

  double sum_load_delay = 0.0;
  for (auto* load : loads) {
    std::string load_name = load->getFullName();
    sum_load_delay += rc_net->delay(load_name.c_str()).value_or(0.0);
  }

  double net_avg_delay = sum_load_delay / loads.size();
  return net_avg_delay;
}

std::pair<double, double> InitSTA::getNetToggleAndVoltage(const std::string& net_name) const
{
  return PW_INST->get_power()->getNetToggleAndVoltageData(net_name.c_str());
}

double InitSTA::getNetPower(const std::string& net_name) const
{
  // get net power from updated results.
  auto& nets_power = _net_power.begin()->second;
  if (nets_power.contains(net_name)) {
    double net_power = nets_power.at(net_name);
    return net_power;
  } else {
    return 0.0;
  }
}

double InitSTA::getWireResistance(const std::string& net_name, const std::string& wire_node_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  LOG_FATAL_IF(!rc_net) << "net " << net_name << " not found rc net.";

  double resistance = rc_net->getNodeResistance(wire_node_name.c_str());
  return resistance;
}

double InitSTA::getWireCapacitance(const std::string& net_name, const std::string& wire_node_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  LOG_FATAL_IF(!rc_net) << "net " << net_name << " not found rc net.";

  double load = rc_net->getNodeLoad(wire_node_name.c_str());
  return load;
}

double InitSTA::getWireDelay(const std::string& net_name, const std::string& wire_node_name) const
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  auto delay = rc_net->delay(wire_node_name.c_str());

  return delay.value_or(0.0);
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
      // wire_length =
      // first_pin->get_coord().computeDist(second_pin->get_coord());

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

TimingWireGraph InitSTA::getTimingWireGraph()
{
  LOG_INFO << "get wire timing graph start";
  ieda::Stats stats;

  TimingWireGraph timing_wire_graph;

  /// create node in wire graph
  auto create_node = [&timing_wire_graph](std::string& edge_node_name, bool is_pin, bool is_port) -> unsigned {
    auto index = timing_wire_graph.findNode(edge_node_name);
    if (!index) {
      TimingWireNode edge_node;
      edge_node._name = edge_node_name;
      edge_node._is_pin = is_pin;
      edge_node._is_port = is_port;

      index = timing_wire_graph.addNode(edge_node);
    }

    return index.value();
  };

  /// the node is StaNode
  auto create_inst_node = [&create_node](auto* the_node) -> unsigned {
    std::string node_name = the_node->getName();
    auto* design_obj = the_node->get_design_obj();
    bool is_pin = design_obj ? design_obj->isPin() : false;
    bool is_port = design_obj ? design_obj->isPort() : false;

    auto wire_node_index = create_node(node_name, is_pin, is_port);
    return wire_node_index;
  };

  /// the node is RC Node
  auto create_net_node = [&create_node](auto& the_node) -> unsigned {
    std::string node_name = the_node.get_name();
    bool is_pin = the_node.get_obj() ? the_node.get_obj()->isPin() : false;
    bool is_port = the_node.get_obj() ? the_node.get_obj()->isPort() : false;

    auto wire_node_index = create_node(node_name, is_pin, is_port);
    return wire_node_index;
  };

  auto* ista = STA_INST->get_ista();
  LOG_ERROR_IF(!ista->isBuildGraph()) << "timing graph is not build";

  auto* the_timing_graph = &(ista->get_graph());
  ista::StaArc* the_arc;

  timing_wire_graph._edges.reserve(the_timing_graph->get_arcs().size() * 100);
  timing_wire_graph._nodes.reserve(the_timing_graph->get_vertexes().size() * 10);
  FOREACH_ARC(the_timing_graph, the_arc)
  {
    if (the_arc->isNetArc()) {
      // for net arc, we need extract the wire topo.
      auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc);
      auto* the_net = the_net_arc->get_net();

      auto* rc_net = ista->getRcNet(the_net);

      if (rc_net) {
        auto* snk_node = the_arc->get_snk();
        auto snk_node_name = snk_node->get_design_obj()->getFullName();

        auto vertex_slew = the_arc->get_src()->getSlewNs(ista::AnalysisMode::kMax, TransType::kRise);
        if (!vertex_slew) {
          vertex_slew = the_arc->get_src()->getSlewNs(ista::AnalysisMode::kMax, TransType::kFall);
        }

        auto wire_topo = rc_net->getWireTopo(snk_node_name.c_str());
        for (auto* wire_edge : wire_topo | std::ranges::views::reverse) {
          ieda::Stats stats2;
          auto& from_node = wire_edge->get_from();
          auto& to_node = wire_edge->get_to();

          auto wire_from_node_index = create_net_node(from_node);
          auto wire_to_node_index = create_net_node(to_node);

          timing_wire_graph.addEdge(wire_from_node_index, wire_to_node_index);
        }
      } else {
        auto wire_from_node_index = create_inst_node(the_arc->get_src());
        auto wire_to_node_index = create_inst_node(the_arc->get_snk());

        auto& inst_wire_edge = timing_wire_graph.addEdge(wire_from_node_index, wire_to_node_index);
        inst_wire_edge._is_net_edge = true;
      }

    } else {
      auto wire_from_node_index = create_inst_node(the_arc->get_src());
      auto wire_to_node_index = create_inst_node(the_arc->get_snk());

      auto& inst_wire_edge = timing_wire_graph.addEdge(wire_from_node_index, wire_to_node_index);
      inst_wire_edge._is_net_edge = false;
    }
  }

  LOG_INFO << "wire timing graph nodes " << timing_wire_graph._nodes.size();
  LOG_INFO << "wire timing graph edges " << timing_wire_graph._edges.size();

  timing_wire_graph._nodes.shrink_to_fit();
  timing_wire_graph._edges.shrink_to_fit();

  LOG_INFO << "get wire timing graph end";

  LOG_INFO << "get wire timing graph memory usage " << stats.memoryDelta() << " MB";
  double total_time = stats.elapsedRunTime();
  LOG_INFO << "get wire timing graph elapsed time " << total_time << " s";

  // for debug
  // SaveTimingGraph(timing_wire_graph, "./timing_wire_graph.yaml");

  return timing_wire_graph;
}

bool InitSTA::getRcNet(const std::string& net_name)
{
  auto netlist = STA_INST->get_netlist();
  ista::Net* ista_net = netlist->findNet(net_name.c_str());
  auto* rc_net = STA_INST->get_ista()->getRcNet(ista_net);

  return rc_net ? true : false;
}

void SaveTimingGraph(const TimingWireGraph& timing_wire_graph, const std::string& yaml_file_name)
{
  LOG_INFO << "save wire timing graph start";

  std::ofstream file(yaml_file_name, std::ios::trunc);

  for (unsigned node_id = 0; auto& node : timing_wire_graph._nodes) {
    const char* node_name = Str::printf("node_%d", node_id++);
    LOG_INFO_EVERY_N(1000) << "write node " << node_id << " total " << timing_wire_graph._nodes.size();

    file << node_name << ":" << "\n";
    file << "  name: " << node._name << "\n";
    file << "  is_pin: " << node._is_pin << "\n";
    file << "  is_port: " << node._is_port << "\n";
  }

  for (unsigned edge_id = 0; auto& edge : timing_wire_graph._edges) {
    std::string edge_name = Str::printf("edge_%d", edge_id++);

    LOG_INFO_EVERY_N(1000) << "write edge " << edge_id << " total " << timing_wire_graph._edges.size();

    file << edge_name << ":" << "\n";
    file << "  from_node: " << edge._from_node << "\n";
    file << "  to_node: " << edge._to_node << "\n";
    file << "  is_net_edge: " << edge._is_net_edge << "\n";
  }

  // out << YAML::EndMap; // Close the YAML map
  file.close();

  LOG_INFO << "output wire graph yaml file path: " << yaml_file_name;
  LOG_INFO << "save wire timing graph end";
}
/// @brief Restore wire timing graph from yaml file.
/// @param yaml_file_name
/// @return
TimingWireGraph RestoreTimingGraph(const std::string& yaml_file_name)
{
  LOG_INFO << "restore wire timing graph start";
  TimingWireGraph timing_wire_graph;

  std::ifstream file(yaml_file_name);
  string line;

  bool is_node = true;
  TimingWireNode wire_node;
  TimingWireEdge wire_edge;

  while (getline(file, line)) {
    if (is_node && (line.rfind("edge_", 0) == 0)) {
      is_node = false;
    }

    if (is_node) {
      if (line.find("name:") != string::npos) {
        size_t pos = line.find(": ");
        wire_node._name = line.substr(pos + 2);
      } else if (line.find("is_pin:") != string::npos) {
        size_t pos = line.find(": ");
        wire_node._is_pin = stoi(line.substr(pos + 2));
      } else if (line.find("is_port:") != string::npos) {
        size_t pos = line.find(": ");
        wire_node._is_port = stoi(line.substr(pos + 2));
        timing_wire_graph._nodes.emplace_back(std::move(wire_node));
      }

    } else {
      if (line.find("from_node:") != string::npos) {
        size_t pos = line.find(": ");
        wire_edge._from_node = stoll(line.substr(pos + 2));
      } else if (line.find("to_node:") != string::npos) {
        size_t pos = line.find(": ");
        wire_edge._to_node = stoll(line.substr(pos + 2));
      } else if (line.find("is_net_edge:") != string::npos) {
        size_t pos = line.find(": ");
        wire_edge._is_net_edge = stoi(line.substr(pos + 2));
        timing_wire_graph._edges.emplace_back(std::move(wire_edge));
      }
    }
  }
  file.close();

  LOG_INFO << "restore wire timing graph end";

  LOG_INFO << "wire timing graph nodes " << timing_wire_graph._nodes.size();
  LOG_INFO << "wire timing graph edges " << timing_wire_graph._edges.size();

  return timing_wire_graph;
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
      // wire_length =
      // first_pin->get_coord().computeDist(second_pin->get_coord());
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

/**
 * @brief The timing map of the patch.
 *
 * @param patch
 * @return std::map<int, double>
 */
std::map<int, double> InitSTA::patchTimingMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch)
{
  std::map<int, double> patch_timing_map;
  auto inst_timing_map = STA_INST->get_ista()->displayTimingMap(ista::AnalysisMode::kMax);
  if (inst_timing_map.empty()) {
    LOG_ERROR << "No instance timing map found.";
    return patch_timing_map;
  }

  auto* idb_adapter = STA_INST->getIDBAdapter();
  auto dbu = idb_adapter->get_dbu();

  // 网格索引，减小搜索空间
  int64_t min_x = INT64_MAX;
  int64_t max_x = INT64_MIN;
  int64_t min_y = INT64_MAX;
  int64_t max_y = INT64_MIN;
  for (const auto& [coord, _] : inst_timing_map) {
    int64_t x = static_cast<int64_t>(coord.first) * dbu;
    int64_t y = static_cast<int64_t>(coord.second) * dbu;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }

  // 启发式确定网格大小
  int64_t grid_size_x = (max_x - min_x) / 100;
  int64_t grid_size_y = (max_y - min_y) / 100;

  // 创建网格: 二维网格，每个网格内存有对应的insts
  std::vector<std::vector<std::vector<std::pair<std::pair<int64_t, int64_t>, double>>>> grid;
  int64_t grid_width = (max_x - min_x) / grid_size_x + 1;
  int64_t grid_height = (max_y - min_y) / grid_size_y + 1;
  grid.resize(grid_width, std::vector<std::vector<std::pair<std::pair<int64_t, int64_t>, double>>>(grid_height));

  // 填充网格
  for (const auto& [coord, slack] : inst_timing_map) {
    int64_t x = static_cast<int64_t>(coord.first * dbu);
    int64_t y = static_cast<int64_t>(coord.second * dbu);
    int64_t grid_x = (x - min_x) / grid_size_x;
    int64_t grid_y = (y - min_y) / grid_size_y;
    grid[grid_x][grid_y].push_back({{x, y}, slack});
  }

  for (const auto& [patch_id, coord] : patch) {
    auto [l_range, u_range] = coord;
    const int64_t patch_lx = static_cast<int64_t>(l_range.first);
    const int64_t patch_ly = static_cast<int64_t>(l_range.second);
    const int64_t patch_ux = static_cast<int64_t>(u_range.first);
    const int64_t patch_uy = static_cast<int64_t>(u_range.second);

    // 计算覆盖的网格范围
    int64_t start_grid_x = std::max(static_cast<int64_t>(0), (patch_lx - min_x) / grid_size_x);
    int64_t end_grid_x = std::min(grid_width - 1, (patch_ux - min_x) / grid_size_x);
    int64_t start_grid_y = std::max(static_cast<int64_t>(0), (patch_ly - min_y) / grid_size_y);
    int64_t end_grid_y = std::min(grid_height - 1, (patch_uy - min_y) / grid_size_y);

    double min_slack = std::numeric_limits<double>::max();

    // 只检查覆盖的网格
    for (int64_t gx = start_grid_x; gx <= end_grid_x; ++gx) {
      for (int64_t gy = start_grid_y; gy <= end_grid_y; ++gy) {
        for (const auto& [inst_coord, inst_slack] : grid[gx][gy]) {
          int64_t inst_x = inst_coord.first;
          int64_t inst_y = inst_coord.second;
          if (patch_lx <= inst_x && inst_x <= patch_ux && patch_ly <= inst_y && inst_y <= patch_uy) {
            min_slack = std::min(min_slack, inst_slack);
          }
        }
      }
    }

    patch_timing_map[patch_id] = min_slack;
  }

  return patch_timing_map;
}
/**
 * @brief The power map of the patch.
 *
 * @param patch
 * @return std::map<int, double>
 */
std::map<int, double> InitSTA::patchPowerMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch)
{
  std::map<int, double> patch_power_map;
  auto inst_power_map = PW_INST->get_power()->displayInstancePowerMap();

  if (inst_power_map.empty()) {
    LOG_ERROR << "No instance power map found.";
    return patch_power_map;
  }

  auto* idb_adapter = STA_INST->getIDBAdapter();
  auto dbu = idb_adapter->get_dbu();

  // 网格索引，减小搜索空间
  int64_t min_x = INT64_MAX;
  int64_t max_x = INT64_MIN;
  int64_t min_y = INT64_MAX;
  int64_t max_y = INT64_MIN;
  for (const auto& [coord, _] : inst_power_map) {
    int64_t x = static_cast<int64_t>(coord.first * dbu);
    int64_t y = static_cast<int64_t>(coord.second * dbu);
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }

  // 启发式确定网格大小
  int64_t grid_size_x = (max_x - min_x) / 100;
  int64_t grid_size_y = (max_y - min_y) / 100;

  // 创建网格: 二维网格，每个网格内存有对应的insts
  std::vector<std::vector<std::vector<std::pair<std::pair<int64_t, int64_t>, double>>>> grid;
  int64_t grid_width = (max_x - min_x) / grid_size_x + 1;
  int64_t grid_height = (max_y - min_y) / grid_size_y + 1;
  grid.resize(grid_width, std::vector<std::vector<std::pair<std::pair<int64_t, int64_t>, double>>>(grid_height));

  // 填充网格
  for (const auto& [coord, power] : inst_power_map) {
    int64_t x = static_cast<int64_t>(coord.first) * dbu;
    int64_t y = static_cast<int64_t>(coord.second) * dbu;
    int64_t grid_x = (x - min_x) / grid_size_x;
    int64_t grid_y = (y - min_y) / grid_size_y;
    grid[grid_x][grid_y].push_back({{x, y}, power});
  }

  for (const auto& [patch_id, coord] : patch) {
    auto [l_range, u_range] = coord;
    const int64_t patch_lx = static_cast<int64_t>(l_range.first);
    const int64_t patch_ly = static_cast<int64_t>(l_range.second);
    const int64_t patch_ux = static_cast<int64_t>(u_range.first);
    const int64_t patch_uy = static_cast<int64_t>(u_range.second);

    // 计算覆盖的网格范围
    int64_t start_grid_x = std::max(static_cast<int64_t>(0), (patch_lx - min_x) / grid_size_x);
    int64_t end_grid_x = std::min(grid_width - 1, (patch_ux - min_x) / grid_size_x);
    int64_t start_grid_y = std::max(static_cast<int64_t>(0), (patch_ly - min_y) / grid_size_y);
    int64_t end_grid_y = std::min(grid_height - 1, (patch_uy - min_y) / grid_size_y);

    double total_power = 0.0;

    // 只检查覆盖的网格
    for (int64_t gx = start_grid_x; gx <= end_grid_x; ++gx) {
      for (int64_t gy = start_grid_y; gy <= end_grid_y; ++gy) {
        for (const auto& [inst_coord, inst_power] : grid[gx][gy]) {
          int64_t inst_x = inst_coord.first;
          int64_t inst_y = inst_coord.second;
          if (patch_lx <= inst_x && inst_x <= patch_ux && patch_ly <= inst_y && inst_y <= patch_uy) {
            total_power += inst_power;
          }
        }
      }
    }

    patch_power_map[patch_id] = total_power;
  }

  return patch_power_map;
}

/**
 * @brief The ir drop map of the patch.
 *
 * @param patch
 * @return std::map<int, double>
 */
std::map<int, double> InitSTA::patchIRDropMap(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& patch)
{
  std::map<int, double> patch_ir_drop_map;
  for (const auto& [patch_id, _] : patch) {
    patch_ir_drop_map[patch_id] = 0.0;
  }

  // hard code std cell power net is VDD
  std::string power_net_name = "VDD";
  PW_INST->runIRAnalysis(power_net_name);
  auto instance_to_ir_drop = PW_INST->getInstanceIRDrop();

  if (instance_to_ir_drop.empty()) {
    LOG_WARNING << "No IR drop data available, returning zero values for all patches";
    return patch_ir_drop_map;
  }

  auto* idb_adapter = STA_INST->getIDBAdapter();
  auto dbu = idb_adapter->get_dbu();

  // 网格索引，减小搜索空间
  int64_t min_x = INT64_MAX;
  int64_t max_x = INT64_MIN;
  int64_t min_y = INT64_MAX;
  int64_t max_y = INT64_MIN;
  std::vector<std::tuple<int64_t, int64_t, double>> instances;
  instances.reserve(instance_to_ir_drop.size());

  for (auto& [sta_inst, ir_drop] : instance_to_ir_drop) {
    auto coord = sta_inst->get_coordinate().value();
    int64_t x = static_cast<int64_t>(coord.first * dbu);
    int64_t y = static_cast<int64_t>(coord.second * dbu);
    instances.emplace_back(x, y, ir_drop);
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }
  // 启发式确定网格大小
  int64_t grid_size_x = (max_x - min_x) / 100;
  int64_t grid_size_y = (max_y - min_y) / 100;

  // 创建网格
  std::vector<std::vector<std::vector<std::pair<std::pair<int64_t, int64_t>, double>>>> grid;
  int64_t grid_width = (max_x - min_x) / grid_size_x + 1;
  int64_t grid_height = (max_y - min_y) / grid_size_y + 1;
  grid.resize(grid_width, std::vector<std::vector<std::pair<std::pair<int64_t, int64_t>, double>>>(grid_height));

  // 填充网格
  for (const auto& [x, y, ir_drop] : instances) {
    int64_t grid_x = (x - min_x) / grid_size_x;
    int64_t grid_y = (y - min_y) / grid_size_y;
    grid[grid_x][grid_y].push_back({{x, y}, ir_drop});
  }

  int processed_count = 0;
  for (const auto& [patch_id, coord] : patch) {
    auto [l_range, u_range] = coord;
    const int64_t patch_lx = static_cast<int64_t>(l_range.first);
    const int64_t patch_ly = static_cast<int64_t>(l_range.second);
    const int64_t patch_ux = static_cast<int64_t>(u_range.first);
    const int64_t patch_uy = static_cast<int64_t>(u_range.second);

    // 计算覆盖的网格范围
    int64_t start_grid_x = std::max(static_cast<int64_t>(0), (patch_lx - min_x) / grid_size_x);
    int64_t end_grid_x = std::min(grid_width - 1, (patch_ux - min_x) / grid_size_x);
    int64_t start_grid_y = std::max(static_cast<int64_t>(0), (patch_ly - min_y) / grid_size_y);
    int64_t end_grid_y = std::min(grid_height - 1, (patch_uy - min_y) / grid_size_y);

    double max_ir_drop = 0.0;

    // 只检查覆盖的网格
    for (int64_t gx = start_grid_x; gx <= end_grid_x; ++gx) {
      for (int64_t gy = start_grid_y; gy <= end_grid_y; ++gy) {
        for (const auto& [inst_coord, inst_ir_drop] : grid[gx][gy]) {
          int64_t inst_x = inst_coord.first;
          int64_t inst_y = inst_coord.second;
          if (patch_lx <= inst_x && inst_x <= patch_ux && patch_ly <= inst_y && inst_y <= patch_uy) {
            max_ir_drop = std::max(max_ir_drop, inst_ir_drop);
          }
        }
      }
    }

    patch_ir_drop_map[patch_id] = max_ir_drop;

    // 每处理5000个patch输出一次日志
    processed_count++;
    if (processed_count % 5000 == 0) {
      LOG_INFO << "Processed " << processed_count << " patches out of " << patch.size();
    }
  }

  return patch_ir_drop_map;
}

}  // namespace ieval