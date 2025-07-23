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
 * @file PowerEngine.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power engine for provide power and timing analysis api.
 * @version 0.1
 * @date 2024-02-26
 *
 */
#pragma once

#include <map>

#include "Power.hh"
#include "Type.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

namespace ipower {

/**
 * @brief cluster connection for iMP.
 *
 */
struct ClusterConnection {
  std::size_t _dst_cluster_id;
  std::vector<unsigned> _stages_each_hop;
  unsigned _hop;
};

/**
 * @brief macro connection for iMP.
 *
 */
struct MacroConnection {
  const char *_src_macro_name;
  const char *_dst_macro_name;
  std::vector<unsigned> _stages_each_hop;
  unsigned _hop;
};

/**
 * @brief The top class for power(include timing) engine.
 *
 */
class PowerEngine {
 public:
  static PowerEngine *getOrCreatePowerEngine();
  static void destroyPowerEngine();

  [[nodiscard]] Power *get_power() const { return _ipower; }
  [[nodiscard]] ista::TimingEngine *get_timing_engine() const {
    return _timing_engine;
  }

  bool isBuildGraph() { return _ipower->isBuildGraph(); }

  // api for dataflow, first is create seq graph, second is get cluster
  // connection for the max hop.
  unsigned creatDataflow();
  std::map<std::size_t, std::vector<ClusterConnection>> buildConnectionMap(
      std::vector<std::set<std::string>> clusters,
      std::set<std::string> src_instances, unsigned max_hop);

  // api for build only macro connection.
  std::vector<MacroConnection> buildMacroConnectionMap(unsigned max_hop);

  unsigned buildPGNetWireTopo();
  unsigned readPGSpef(const char *spef_file) {
    return _ipower->readPGSpef(spef_file);
  }

  void resetIRAnalysisData();
  auto *getRustPGRCData() { return _ipower->get_rust_pg_rc_data(); }
  unsigned runIRAnalysis(std::string power_net_name) {
    if (!getRustPGRCData()) {
      buildPGNetWireTopo();
      // set IR bump node locs.
      auto net_bump_node_locs = _pg_netlist_builder.get_net_bump_node_locs();
      _ipower->setBumpNodeLocs(net_bump_node_locs);
    }

    return _ipower->runIRAnalysis(power_net_name);
  }
  std::map<ista::Instance *, double> getInstanceIRDrop(
      std::string power_net_name = "VDD");

  std::map<ista::Instance::Coordinate, double> displayPowerMap() {
    return _ipower->displayInstancePowerMap();
  }
  std::map<ista::Instance::Coordinate, double> displayIRDropMap();

  unsigned reportIRAnalysis() { return _ipower->reportIRAnalysis(); }

#ifdef USE_GPU
  std::vector<MacroConnection> buildMacroConnectionMapWithGPU(unsigned max_hop);
#endif

 private:
  PowerEngine();
  ~PowerEngine();

  Power *_ipower = nullptr;
  ista::TimingEngine *_timing_engine = nullptr;

  IRPGNetlistBuilder _pg_netlist_builder;

  // Singleton power engine.
  static PowerEngine *_power_engine;
  FORBIDDEN_COPY(PowerEngine);
};

}  // namespace ipower