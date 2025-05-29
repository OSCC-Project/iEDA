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
#ifdef USE_GPU
#include "gpu-kernel/power_kernel.cuh"
#endif
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
      dfs_from_src_seq_vertex = [&macro_connections, &connection_mutex,
                                 &max_hop, &dfs_from_src_seq_vertex](
                                    PwrSeqVertex* src_seq_vertex,
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
          stages_each_hop[max_hop - hop] = src_arc->get_combine_depth();
          if (snk_seq_vertex->isMacro() || snk_seq_vertex->isOutputPort()) {
            const char* src_vertex_name = src_seq_vertex->get_obj_name().data();
            const char* snk_vertex_name = snk_seq_vertex->get_obj_name().data();
            if (src_seq_vertex->isInputPort() ||
                snk_seq_vertex->isOutputPort()) {
              LOG_INFO_FIRST_N(100) << "port connection: " << src_vertex_name
                                    << " -> " << snk_vertex_name;
            }
            MacroConnection one_connection(src_vertex_name, snk_vertex_name,
                                           stages_each_hop, max_hop - hop + 1);
            {
              // add connection.
              std::lock_guard lk(connection_mutex);
              macro_connections.emplace_back(std::move(one_connection));

              LOG_INFO_EVERY_N(2000000)
                  << "build macro connection num: " << macro_connections.size();
            }
          }

          dfs_from_src_seq_vertex(src_seq_vertex, snk_seq_vertex,
                                  stages_each_hop, hop - 1);
        }
      };

  {
    ThreadPool thread_pool(48);
    PwrSeqVertex* seq_vertex;
    FOREACH_SEQ_VERTEX(&seq_graph, seq_vertex) {
      if (seq_vertex->isMacro() || seq_vertex->isInputPort()) {
        // dfs from src seq vertex.
        thread_pool.enqueue(
            [max_hop, &dfs_from_src_seq_vertex](auto* src_seq_vertex,
                                                auto* current_macro_vertex) {
              std::vector<unsigned> stages_each_hop(max_hop, 0);
              dfs_from_src_seq_vertex(src_seq_vertex, current_macro_vertex,
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
  LOG_INFO << "build macro connection num: " << macro_connections.size();
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "build macro connection map memory usage " << memory_delta
           << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "build macro connection map time elapsed " << time_delta << "s";

  return macro_connections;
}

#ifdef USE_GPU
/**
 * @brief build macro connection map with gpu kernel.
 *
 * @param max_hop
 * @return std::vector<MacroConnection>
 */
std::vector<MacroConnection> PowerEngine::buildMacroConnectionMapWithGPU(
    unsigned max_hop) {
  ieda::Stats stats;
  LOG_INFO << "gpu build macro connection map start";
  auto& seq_graph = _ipower->get_power_seq_graph();
  int num_macro = seq_graph.getMacroSeqVertexNum();
  // The first connection point is the src macro vertex.
  std::vector<GPU_Connection_Point> connection_points(num_macro);

  std::map<PwrSeqVertex*, unsigned> vertex_to_id;
  int num_seq_vertexes = seq_graph.get_vertexes().size();
  std::vector<unsigned> is_macros(num_seq_vertexes, 0);
  PwrSeqVertex* seq_vertex;
  int i = 0;  // for seq vertex index.
  int j = 0;  // for connection point index.
  FOREACH_SEQ_VERTEX(&seq_graph, seq_vertex) {
    if (seq_vertex->isMacro()) {
      connection_points[j]._src_id = i;
      connection_points[j++]._snk_id = i;
      is_macros[i] = 1;
    }
    vertex_to_id[seq_vertex] = i;
    ++i;
  }

  LOG_FATAL_IF(j != num_macro) << "Macro num is " << j;

  // build cpu data for gpu data communcation.
  std::vector<unsigned> seq_arcs;    // seq arc snk id vector.
  std::vector<unsigned> snk_arcs;    // snk arc [start, end] pair id vector.
  std::vector<unsigned> snk_depths;  // snk arc combine cell depth vector.
  FOREACH_SEQ_VERTEX(&seq_graph, seq_vertex) {
    PwrSeqArc* seq_arc;
    std::pair<int, int> snk_arc_id_pair{-1, -1};
    FOREACH_SRC_SEQ_ARC(seq_vertex, seq_arc) {
      auto* snk_seq_vertex = seq_arc->get_snk();
      unsigned snk_seq_vertex_id = vertex_to_id[snk_seq_vertex];
      seq_arcs.push_back(snk_seq_vertex_id);
      snk_depths.push_back(seq_arc->get_combine_depth());

      if (snk_arc_id_pair.first == -1) {
        snk_arc_id_pair.first = seq_arcs.size() - 1;
      } else {
        snk_arc_id_pair.second = seq_arcs.size() - 1;
      }
    }

    // for only one arc.
    snk_arcs.emplace_back(snk_arc_id_pair.first);
    if (snk_arc_id_pair.second == -1) {
      snk_arc_id_pair.second = snk_arc_id_pair.first;
    }
    snk_arcs.emplace_back(snk_arc_id_pair.second);
  }

  LOG_INFO_IF(seq_arcs.size() != seq_graph.getSeqArcNum())
      << "the seq arc size " << seq_arcs.size()
      << " is not equal to seq graph arc num " << seq_graph.getSeqArcNum();

  std::size_t output_connection_size = seq_arcs.size();
  std::vector<GPU_Connection_Point> out_connection_points(output_connection_size);
  int connection_point_num = connection_points.size();

  auto& seq_vertexes = seq_graph.get_vertexes();

  // use src arc to calc out connection num.
  auto calc_out_connection_num =
      [&seq_vertexes](auto& connection_points) -> int {
    int out_connection_point_num = 0;
    std::set<int> connection_ids;
    for (auto& connection_point : connection_points) {
      int connection_id = connection_point._snk_id;
      if (connection_ids.contains(connection_id)) {
        continue;
      }
      connection_ids.insert(connection_id);
      // get src arc num.
      auto& seq_vertex = seq_vertexes[connection_id];
      out_connection_point_num += seq_vertex->get_src_arcs().size();
    }

    return out_connection_point_num;
  };

  // the actual number of out connection points.
  int out_connection_point_num = calc_out_connection_num(connection_points);

  std::vector<MacroConnection> macro_connections;
  bool is_free_memory = false;
  // call gpu function to bfs seq arc.
  for (unsigned i = 0; i < max_hop; ++i) {
    is_free_memory = (i == max_hop - 1) ? true : false;
    build_macro_connection_map(connection_points.data(), is_macros.data(),
                               seq_arcs.data(), snk_depths.data(),
                               snk_arcs.data(), connection_point_num,
                               num_seq_vertexes, output_connection_size,
                               out_connection_points.data(), is_free_memory);

    // out connection give to connection and build connection map.
    connection_points.resize(out_connection_point_num);
    int connection_point_index = 0;
    for (std::size_t j = 0; j < output_connection_size; ++j) {
      int src_id = out_connection_points[j]._src_id;
      int snk_id = out_connection_points[j]._snk_id;
      // not tranversed, skip.
      if (snk_id == -1) {
        continue;
      }
      // set connection point
      connection_points[connection_point_index]._src_id = src_id;
      connection_points[connection_point_index++]._snk_id = snk_id;

      if (seq_vertexes.at(snk_id)->isMacro()) {
        // build macro connection.
        MacroConnection macro_connection;
        macro_connection._src_macro_name =
            seq_vertexes[src_id]->get_own_seq_inst()->get_name();
        macro_connection._dst_macro_name =
            seq_vertexes[snk_id]->get_own_seq_inst()->get_name();

        macro_connections.emplace_back(std::move(macro_connection));
      }

      // reset data.
      out_connection_points[j]._src_id = -1;
      out_connection_points[j]._snk_id = -1;
    }

    LOG_FATAL_IF(connection_point_index != out_connection_point_num)
        << "connection_point_index " << connection_point_index
        << " is not equal to out_connection_point_num "
        << out_connection_point_num;

    // update connection num and out connetion point num.
    connection_point_num = out_connection_point_num;
    out_connection_point_num = calc_out_connection_num(connection_points);
  }

  LOG_INFO << "gpu build macro connection map end";
  double memory_delta = stats.memoryDelta();
  LOG_INFO << "gpu build macro connection map memory usage " << memory_delta
           << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "gpu build macro connection map time elapsed " << time_delta
           << "s";

  return macro_connections;
}
#endif

/**
 * @brief build pg net wire topology.
 * 
 * @return unsigned 
 */
unsigned PowerEngine::buildPGNetWireTopo() {
  LOG_INFO << "build pg net wire topo start";
  
  // set the instance names for build wire topo skip some no power instance.
  std::vector<IRInstancePower> instance_power_data = _ipower->getInstancePowerData();
  std::set<std::string> instance_names;
  std::ranges::for_each(instance_power_data, [&instance_names](auto& instance_power) {
    std::string instance_name = instance_power._instance_name;
    instance_names.insert(std::move(instance_name));
  });
  _pg_netlist_builder.set_instance_names(std::move(instance_names));

  auto* idb_adapter =
      dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter());
  auto* idb_builder = idb_adapter->get_idb();

  // set layer name to id.
  IdbLayout* idb_layout = idb_builder->get_lef_service()->get_layout();
  vector<IdbLayer*>& routing_layers =
      idb_layout->get_layers()->get_routing_layers();
  for (int id = 1; auto* layer : routing_layers) {
    auto layer_name = layer->get_name();
    _pg_netlist_builder.setLayerNameToId(layer_name, id);
    ++id;
  }

  auto* special_net_list =
      idb_builder->get_def_service()->get_design()->get_special_net_list();
  auto* idb_design = idb_builder->get_def_service()->get_design();
  auto dbu = idb_design->get_units()->get_micron_dbu();
  _pg_netlist_builder.set_dbu(dbu);

  std::function<double(unsigned, unsigned, unsigned)> calc_resistance =
      [idb_adapter, dbu](unsigned layer_id, unsigned distance_dbu, unsigned width_dbu) -> double {
    double wire_length = double(distance_dbu) / dbu;
    double width = double(width_dbu) / dbu;
    double resistance = idb_adapter->getResistance(layer_id, wire_length, width);
    resistance *= c_resistance_coef;

    return resistance;
  };

  // buid pg netlist
  for (auto* power_net : special_net_list->get_net_list()) {
    auto* idb_design = idb_builder->get_def_service()->get_design();
    auto power_net_name = power_net->get_net_name();

    auto* io_pins = idb_design->get_io_pin_list();
    auto* power_io_pin = io_pins->find_pin(power_net_name);
    if (!power_io_pin) {
      continue;
    }

    _pg_netlist_builder.build(power_net, power_io_pin, calc_resistance);
  }

  bool is_empty = _pg_netlist_builder.get_pg_netlists().empty();
  if (is_empty) {
    LOG_INFO << "pg net netlist empty";
    return 0;
  }

  _pg_netlist_builder.createRustPGNetlist();
  _pg_netlist_builder.createRustRCData();

  auto* rc_data = _pg_netlist_builder.get_rust_rc_data();

  _ipower->set_rust_pg_rc_data(rc_data);

  LOG_INFO << "build pg net wire topo end";

  return 1;
}

/**
 * @brief reset ir data for rerun ir analysis.
 * 
 */
void PowerEngine::resetIRAnalysisData() {
  IRPGNetlistBuilder pg_netlist_builder;
  _pg_netlist_builder = std::move(pg_netlist_builder);

  _ipower->resetIRAnalysisData();
}

/**
 * @brief get instance ir drop map.
 * 
 * @return std::map<Instance*, double> 
 */
std::map<Instance*, double> PowerEngine::getInstanceIRDrop() {
  std::map<Instance*, double> instance_to_ir_drop;

  auto& instance_pin_to_ir_drop = _ipower->getInstanceIRDrop();
  auto sta_netlist = _timing_engine->get_netlist();

  for (auto& [instance_pin_name, inst_ir_drop] : instance_pin_to_ir_drop) {
    auto instance_name = Str::split(instance_pin_name.c_str(), ":").front();

    auto* sta_inst = sta_netlist->findInstance(instance_name.c_str());
    if (!sta_inst) {
      continue;
    }

    instance_to_ir_drop[sta_inst] = inst_ir_drop;
  }
  
  return instance_to_ir_drop;
}

/**
 * @brief function to display ir drop map.
 * 
 * @return std::map<Instance::Coordinate, double> 
 */
std::map<Instance::Coordinate, double> PowerEngine::displayIRDropMap() {
  LOG_INFO << "display IR Drop map start";

  std::map<Instance::Coordinate, double> coord_to_ir_drop_map;

  auto instance_to_ir_drop = getInstanceIRDrop();

  for (auto& [sta_inst, inst_ir_drop] : instance_to_ir_drop) {

    auto coord = sta_inst->get_coordinate().value();

    coord_to_ir_drop_map[coord] = inst_ir_drop;
  }

  LOG_INFO << "display IR Drop map end";

  return coord_to_ir_drop_map;
}

}  // namespace ipower