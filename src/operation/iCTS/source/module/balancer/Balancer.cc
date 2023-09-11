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
#include "Balancer.h"

#include "CTSAPI.hpp"
#include "Evaluator.h"
#include "SlewAware.h"
#include "Synthesis.h"
#include "log/Log.hh"

namespace icts {
void Balancer::init()
{
  LOG_INFO << "\033[1;31m";
  LOG_INFO << R"(  _           _                           )";
  LOG_INFO << R"( | |         | |                          )";
  LOG_INFO << R"( | |__   __ _| | __ _ _ __   ___ ___ _ __ )";
  LOG_INFO << R"( | '_ \ / _` | |/ _` | '_ \ / __/ _ \ '__|)";
  LOG_INFO << R"( | |_) | (_| | | (_| | | | | (_|  __/ |   )";
  LOG_INFO << R"( |_.__/ \__,_|_|\__,_|_| |_|\___\___|_|   )";
  LOG_INFO << "\033[0m";
  LOG_INFO << "Enter synthesis!";
}

void Balancer::balance()
{
  CTSAPIInst.saveToLog("\n\nBalance Log");
  // get clock arrival time
  auto same_clock_nets = getClkNetsGroupByMasterClocks();
  auto clock_name_map = getNetNameToClockNameMap();

  auto* design = CTSAPIInst.get_design();
  auto* timer = new TimingCalculator();

  for (auto [master_clocks_set, clk_nets] : same_clock_nets) {
    CTSAPIInst.saveToLog("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++");
    for (const auto& master_clock : master_clocks_set) {
      CTSAPIInst.saveToLog("Master Clock Group Member: ", master_clock);
    }
    // build data traits
    std::vector<BalanceTrait> balance_traits;
    for (auto* clk_net : clk_nets) {
      if (!clk_net->isClockRouted()) {
        continue;
      }
      CTSAPIInst.saveToLog("Balance Net: ", clk_net->get_net_name());
      CTSAPIInst.resetRCTree(clk_net->get_net_name());
      auto& clock_topo = design->findClockTopo(clk_net->get_net_name());
      auto* clock_inst = clk_net->get_driver_inst();

      auto* clock_node = CTSAPIInst.findTimingNode(clock_inst->get_name());
      LOG_FATAL_IF(clock_node == nullptr) << "can't find clock node";

      auto* root_node = clock_node->get_left();
      LOG_FATAL_IF(root_node == nullptr) << "can't find root node";

      clock_topo.disconnect_load(root_node->get_inst());

      auto* driver_pin = clk_net->get_driver_pin();

      auto belong_clock_name = clock_name_map[clk_net->get_net_name()];
      CTSAPIInst.saveToLog("Belong clock: ", belong_clock_name);
      double clock_at = 0.0;
      for (auto clock_name : master_clocks_set) {
        clock_at = std::max(
            clock_at, CTSAPIInst.getClockAT(driver_pin->is_io() ? driver_pin->get_pin_name() : driver_pin->get_full_name(), clock_name));
      }

      auto latency = clock_at + timer->calcElmoreDelay(clock_node, root_node) + root_node->get_delay_max();

      balance_traits.emplace_back(BalanceTrait{latency, clock_at, clock_node, root_node, clk_net});
      CTSAPIInst.saveToLog("Clock arrival time: ", clock_at);
      CTSAPIInst.saveToLog("Root node max delay: ", root_node->get_delay_max());
      CTSAPIInst.saveToLog("Total latency: ", latency, "\n");
    }
    // 1. find the max latency net
    auto cmp_by_latency = [](const BalanceTrait& lhs, const BalanceTrait& rhs) { return lhs.latency > rhs.latency; };
    std::sort(balance_traits.begin(), balance_traits.end(), cmp_by_latency);
    // 2. optimize max latency net
    auto max_trait = balance_traits.front();
    balance_traits.erase(balance_traits.begin());
    auto& max_clk_topo = design->findClockTopo(max_trait.clk_net->get_net_name());
    max_clk_topo.connect_load(max_trait.root_node->get_inst());

    auto max_latency = max_trait.latency;
    std::vector<ClockTopo> incre_topos;
    if (!balance_traits.empty()) {
      for (auto [latency, clock_at, clock_node, root_node, clk_net] : balance_traits) {
        auto incre_delay = max_latency - latency;
        CTSAPIInst.saveToLog("Balance Net: ", clk_net->get_net_name(), " incre delay: ", incre_delay);
        if (incre_delay < 0) {
          LOG_WARNING << "balance require delay less than 0";
          continue;
        }
        // insert buffer
        clock_node->set_type(TimingNodeType::kBuffer);
        timer->insertBuffer(clock_node, root_node, incre_delay);
        timer->updateCap(clock_node);
        timer->timingPropagate(clock_node);
        timer->updateTiming(clock_node);
        auto net_name = clock_node->get_name() + "_balance";
        auto* slew_aware = new SlewAware(net_name, {});
        // top down
        slew_aware->topDown(clock_node);
        // make topo
        slew_aware->buildClockTopo(clock_node->get_left(), net_name);
#ifdef TIMING_LOG
        slew_aware->timingLog();
#endif
        auto balance_topos = slew_aware->get_clk_topos();
        for (auto topo : balance_topos) {
          incre_topos.emplace_back(topo);
        }
        auto* final_root_buffer = clock_node->get_left()->get_inst();
        auto& clock_topo = design->findClockTopo(clk_net->get_net_name());
        clock_topo.connect_load(final_root_buffer);
        // add clock topo wire
        clock_topo.buildWires();
      }
    }
    for (auto topo : incre_topos) {
      design->addClockTopo(topo);
    }
  }
}

double Balancer::calcDumpCapOut(CtsNet* clock_net) const
{
  auto db_unit = CTSAPIInst.getDbUnit();
  auto unit_cap = CTSAPIInst.getClockUnitCap();
  double cap_out = 0;
  auto* driver_pin = clock_net->get_driver_pin();
  for (auto* load_pin : clock_net->get_load_pins()) {
    auto* load_inst = load_pin->get_instance();
    if (load_inst->get_type() != CtsInstanceType::kSink) {
      // auto length
      //     = 1.0 * pgl::manhattan_distance(driver_pin->get_instance()->get_location(), load_pin->get_instance()->get_location()) / db_unit;
      // auto pin_name = load_pin->is_io() ? load_pin->get_pin_name() : load_pin->get_full_name();
      // cap_out += length * unit_cap + CTSAPIInst.getCapOut(pin_name);
    }
  }
  return cap_out;
}

std::map<std::string, std::string> Balancer::getNetNameToClockNameMap() const
{
  std::map<std::string, std::string> clock_name_map;
  auto* design = CTSAPIInst.get_design();
  auto clocks = design->get_clocks();
  for (auto* clock : clocks) {
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clock_net : clock_nets) {
      clock_name_map[clock_net->get_net_name()] = clock->get_clock_name();
    }
  }
  return clock_name_map;
}

std::map<std::set<std::string>, std::vector<CtsNet*>> Balancer::getClkNetsGroupByMasterClocks() const
{
  std::map<std::set<std::string>, std::vector<CtsNet*>> same_clock_nets;
  auto* design = CTSAPIInst.get_design();
  auto clocks = design->get_clocks();
  for (auto* clock : clocks) {
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clock_net : clock_nets) {
      auto master_clocks = CTSAPIInst.getMasterClocks(clock_net);
      std::set<std::string> find_keys = {master_clocks.begin(), master_clocks.end()};
      std::vector<CtsNet*> find_values = {clock_net};
      for (auto itr = same_clock_nets.begin(); itr != same_clock_nets.end();) {
        bool is_find = false;
        auto keys = itr->first;
        auto values = itr->second;
        for (auto& cur_key : master_clocks) {
          if (std::find(keys.begin(), keys.end(), cur_key) != keys.end()) {
            is_find = true;
            // add keys and values to find_keys and find_values
            find_keys.insert(keys.begin(), keys.end());
            find_values.insert(find_values.end(), values.begin(), values.end());
            break;
          }
        }
        if (is_find) {
          // erase the key and value
          itr = same_clock_nets.erase(itr);
        } else {
          ++itr;
        }
      }
      same_clock_nets[find_keys] = find_values;
    }
  }
  return same_clock_nets;
}
}  // namespace icts