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
#include <tuple>
#include <vector>

#include "PowerEngine.hh"
#include "ThreadPool/ThreadPool.h"
#include "usage/usage.hh"

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
    _timing_engine->updateClockTiming();
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
PowerEngine::buildConnectionMap(std::vector<std::set<std::string>> clusters,
                                std::set<std::string> src_instances,
                                unsigned max_hop) {
  auto& seq_graph = _ipower->get_power_seq_graph();
  auto* nl = _timing_engine->get_netlist();

  std::vector<
      std::tuple<std::size_t, std::string, unsigned, std::vector<unsigned>>>
      cluster_connections;
  std::mutex connection_mutex;

  // dfs cluster connecton from one seq vertex.
  std::function<void(std::size_t, PwrSeqVertex*, std::vector<unsigned>&,
                     unsigned)>
      dfs_from_src_seq_vertex =
          [&cluster_connections, &connection_mutex, &dfs_from_src_seq_vertex,
           max_hop](std::size_t src_cluster_id,
                    PwrSeqVertex* current_seq_vertex,
                    std::vector<unsigned>& stages_each_hop, unsigned hop) {
            if (hop == 0) {
              return;
            }

            std::tuple<std::size_t, std::string, unsigned> one_connection;
            auto& src_arcs = current_seq_vertex->get_src_arcs();
            for (auto* src_arc : src_arcs) {
              auto* snk_seq_vertex = src_arc->get_snk();
              std::string snk_obj_name(snk_seq_vertex->get_obj_name());
              stages_each_hop[max_hop - hop] = src_arc->get_combine_depth();

              auto one_connection =
                  std::make_tuple(src_cluster_id, snk_obj_name,
                                  max_hop - hop + 1, stages_each_hop);
              {
                // add connection.
                std::lock_guard lk(connection_mutex);
                cluster_connections.push_back(one_connection);
              }

              dfs_from_src_seq_vertex(src_cluster_id, snk_seq_vertex,
                                      stages_each_hop, hop - 1);
            }
          };

  // assume cluster id start from 0.
  {
    ThreadPool thread_pool(48);
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
        if (src_instances.count(obj_name) == 0) {
          continue;
        }

// dfs from src seq vertex.
#if 1
        thread_pool.enqueue(
            [max_hop, &dfs_from_src_seq_vertex](auto src_cluster_id,
                                                auto* seq_vertex) {
              std::vector<unsigned> stages_each_hop(max_hop, 0);
              dfs_from_src_seq_vertex(src_cluster_id, seq_vertex,
                                      stages_each_hop, max_hop);
            },
            src_cluster_id, seq_vertex);

#else
        std::vector<unsigned> stages_each_hop(max_hop, 0);
        dfs_from_src_seq_vertex(src_cluster_id, seq_vertex, stages_each_hop,
                                max_hop);
#endif
      }
      ++src_cluster_id;
    }
  }

  // build connection map.
  std::map<std::size_t, std::vector<ClusterConnection>> connection_map;
  for (auto& one_connection : cluster_connections) {
    auto& src_cluster_id = std::get<0>(one_connection);
    auto& snk_obj_name = std::get<1>(one_connection);
    auto& hop = std::get<2>(one_connection);
    auto& stages_each_hop = std::get<3>(one_connection);

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

    // erase the extra not use stages.
    stages_each_hop.resize(hop);

    // add connection.
    connection_map[src_cluster_id].push_back(
        ClusterConnection(snk_cluster_id, stages_each_hop, hop));
  }

  return connection_map;
}

/**
 * @brief build connection for macro.
 *
 * @param max_hop
 * @return std::vector<MacroConnection>
 */
std::vector<MacroConnection> PowerEngine::buildMacroConnectionMap(
    unsigned max_hop) {
  ieda::Stats stats;
  LOG_INFO << "build macro connection map start";
  auto& seq_graph = _ipower->get_power_seq_graph();
  std::vector<MacroConnection> macro_connections;
  std::mutex connection_mutex;

  // dfs macro connecton from one seq vertex.
  std::function<void(PwrSeqVertex*, PwrSeqVertex*, std::vector<unsigned>&,
                     unsigned)>
      dfs_from_src_seq_vertex =
          [&macro_connections, &connection_mutex, &max_hop,
           &dfs_from_src_seq_vertex](PwrSeqVertex* src_marco_vertex,
                                     PwrSeqVertex* current_macro_vertex,
                                     std::vector<unsigned> stages_each_hop,
                                     unsigned hop) {
            if (hop == 0) {
              return;
            }

            std::tuple<std::size_t, std::string, unsigned> one_connection;
            auto& src_arcs = current_macro_vertex->get_src_arcs();
            for (auto* src_arc : src_arcs) {
              auto* snk_seq_vertex = src_arc->get_snk();
              std::string snk_obj_name(snk_seq_vertex->get_obj_name());
              stages_each_hop[max_hop - hop] = src_arc->get_combine_depth();
              if (snk_seq_vertex->isMacro()) {
                MacroConnection one_connection(
                    src_marco_vertex->get_obj_name().data(), snk_obj_name,
                    stages_each_hop, max_hop - hop + 1);
                {
                  // add connection.
                  std::lock_guard lk(connection_mutex);
                  macro_connections.push_back(one_connection);
                }
              }

              dfs_from_src_seq_vertex(src_marco_vertex, snk_seq_vertex,
                                      stages_each_hop, hop - 1);
            }
          };

  {
    ThreadPool thread_pool(48);
    PwrSeqVertex* seq_vertex;
    FOREACH_SEQ_VERTEX(&seq_graph, seq_vertex) {
      if (seq_vertex->isMacro()) {
        thread_pool.enqueue(
            [max_hop, &dfs_from_src_seq_vertex](auto* src_marco_vertex,
                                                auto* current_macro_vertex) {
              std::vector<unsigned> stages_each_hop(max_hop, 0);
              dfs_from_src_seq_vertex(src_marco_vertex, current_macro_vertex,
                                      stages_each_hop, max_hop);
            },
            seq_vertex, seq_vertex);
      }
    }
  }

  for (auto& macro_connection : macro_connections) {
    macro_connection._stages_each_hop.resize(macro_connection._hop);
  }

  LOG_INFO << "build macro connection map end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "build macro connection map memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "build macro connection map time elapsed " << time_delta << "s";

  return macro_connections;
}

}  // namespace ipower