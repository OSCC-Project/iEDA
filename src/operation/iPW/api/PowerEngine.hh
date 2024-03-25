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

namespace ipower {

/**
 * @brief cluster connection for iMP.
 * 
 */
struct ClusterConnection {
    std::size_t _dst_cluster_id;
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

  // api for dataflow, first is create seq graph, second is get cluster
  // connection for the max hop.
  unsigned creatDataflow();
  std::map<std::size_t, std::vector<ClusterConnection>> buildConnectionMap(
      std::vector<std::set<std::string>> clusters, unsigned max_hop);

 private:
  PowerEngine();
  ~PowerEngine();

  Power *_ipower = nullptr;
  ista::TimingEngine *_timing_engine = nullptr;

  // Singleton power engine.
  static PowerEngine *_power_engine;
  FORBIDDEN_COPY(PowerEngine);
};

}  // namespace ipower