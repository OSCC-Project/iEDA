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
 * @file PowerEngine.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief  The power engine for provide power and timing analysis api.
 * @version 0.1
 * @date 2024-02-26
 *
 */
#include "PowerEngine.hh"

#include <tuple>
#include <vector>

namespace ipower {
PowerEngine* PowerEngine::_power_engine = nullptr;

PowerEngine::PowerEngine() {
  _timing_engine = TimingEngine::getOrCreateTimingEngine();
  _ipower = Power::getOrCreatePower(&(_timing_engine->get_ista()->get_graph()));
}

PowerEngine::~PowerEngine() {
  Power::destroyPower();
  TimingEngine::destroyTimingEngine();
}

/**
 * @brief Get the Or create power engine object.
 *
 * @return PowerEngine*
 */
PowerEngine* PowerEngine::getOrCreatePowerEngine() {
  static std::mutex mt;
  if (_power_engine == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_power_engine == nullptr) {
      _power_engine = new PowerEngine();
    }
  }
  return _power_engine;
}

/**
 * @brief Destory the power engine.
 *
 */
void PowerEngine::destroyPowerEngine() {
  delete _power_engine;
  _power_engine = nullptr;
}

/**
 * @brief create dataflow for macro placer.
 * To create dataflow, we build seq graph, the seq vertex is instance or
 * port.
 * @return unsigned
 */
unsigned PowerEngine::creatDataflow() {
  // build timing graph.
  if (!_timing_engine->isBuildGraph()) {
    _timing_engine->buildGraph();
    _timing_engine->updateTiming();
  }

  // build power graph & sequential graph.
  if (!_ipower->isBuildGraph()) {
    // build power graph.
    _ipower->buildGraph();

    // build seq graph
    _ipower->buildSeqGraph();
  }

  return 1;
}

/**
 * @brief build cluster connection map based on max hop.
 *
 * @param clusters cluster instances vector, cluster id from 0, so vector first
 * element is cluser id 0, second element is cluser id 1, and so on.
 * @param max_hop build connection not beyond max hop.
 * @return std::map<std::size_t, std::vector<PowerEngine::ClusterConnection>>
 * the map key is src cluster id, value is dst cluster id and hop.
 */
std::map<std::size_t, std::vector<ClusterConnection>>
PowerEngine::buildConnectionMap(
    std::vector<std::set<std::string>> clusters, unsigned max_hop) {
  auto& seq_graph = _ipower->get_power_seq_graph();
  auto* nl = _timing_engine->get_netlist();

  std::vector<std::tuple<std::size_t, std::string, unsigned>>
      cluster_connections;

  // dfs cluster connecton from one seq vertex.
  std::function<void(std::size_t, PwrSeqVertex*, unsigned)>
      dfs_from_src_seq_vertex = [&cluster_connections, &dfs_from_src_seq_vertex,
                                 max_hop](std::size_t src_cluster_id,
                                          PwrSeqVertex* current_seq_vertex,
                                          unsigned hop) {
        if (hop == 0) {
          return;
        }

        std::tuple<std::size_t, std::string, unsigned> one_connection;
        auto& src_arcs = current_seq_vertex->get_src_arcs();
        for (auto* src_arc : src_arcs) {
          auto* snk_seq_vertex = src_arc->get_snk();
          std::string snk_obj_name(snk_seq_vertex->get_obj_name());

          auto one_connection =
              std::make_tuple(src_cluster_id, snk_obj_name, max_hop - hop + 1);

          cluster_connections.push_back(one_connection);
          dfs_from_src_seq_vertex(src_cluster_id, snk_seq_vertex, hop - 1);
        }
      };

  // assume cluster id start from 0.
  for (std::size_t src_cluster_id = 0; auto& cluster_instances : clusters) {
    for (auto& obj_name : cluster_instances) {
      auto the_instance_or_port = nl->findObj(obj_name.data(), false, false);
      LOG_FATAL_IF(the_instance_or_port.size() != 1)
          << "the instance " << obj_name << " is not found";
      auto* seq_vertex = seq_graph.getSeqVertex(the_instance_or_port.front());

      // not seq instance, skip.
      if (!seq_vertex) {
        continue;
      }

      // dfs from src seq vertex.
      dfs_from_src_seq_vertex(src_cluster_id, seq_vertex, max_hop);
      ++src_cluster_id;
    }
  }

  // build connection map.
  std::map<std::size_t, std::vector<ClusterConnection>>
      connection_map;
  for (auto& one_connection : cluster_connections) {
    auto& src_cluster_id = std::get<0>(one_connection);
    auto& snk_obj_name = std::get<1>(one_connection);
    auto& hop = std::get<2>(one_connection);

    // find the snk cluster id.
    std::size_t snk_cluster_id = 0;
    bool is_found = false;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      if (clusters[i].contains(snk_obj_name)) {
        snk_cluster_id = i;
        is_found = true;
        break;
      }
    }

    LOG_FATAL_IF(!is_found)
        << "the snk cluster id " << snk_obj_name << " is not found";

    // add connection.
    connection_map[src_cluster_id].push_back(
        ClusterConnection(snk_cluster_id, hop));
  }

  return connection_map;
}

}  // namespace ipower