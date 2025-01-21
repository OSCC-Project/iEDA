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
 * @file StaDataPropagationBFS.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of data propagation use the BFS method.
 * @version 0.1
 * @date 2025-01-10
 *
 */

#include "StaDataPropagationBFS.hh"

#include <execution>

#include "Config.hh"
#include "StaDataSlewDelayPropagation.hh"
#include "StaDelayPropagation.hh"
#include "StaSlewPropagation.hh"
#include "ThreadPool/ThreadPool.h"
#include "propagation-cuda/fwd_propagation.cuh"

namespace ista {

/**
 * @brief propagate the arc to accumulated the path delay.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaFwdPropagationBFS::operator()(StaArc* the_arc) {
  std::lock_guard<std::mutex> lk(the_arc->get_snk()->get_fwd_mutex());

#if INTEGRATION_FWD
#if 0
  StaSlewPropagation slew_propagation;
  StaDelayPropagation delay_propagation;

  slew_propagation(the_arc);
  delay_propagation(the_arc);
#else
  StaDataSlewDelayPropagation slew_delay_propagation;
  slew_delay_propagation(the_arc);

#endif
#endif

  if (!the_arc->isCheckArc()) {
    // call parent operator.
    StaFwdPropagation::operator()(the_arc);
  }

  return 1;
}

/**
 * @brief The vertex propagate the vertex, and get the next bfs vertexes.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaFwdPropagationBFS::operator()(StaVertex* the_vertex) {
  if (the_vertex->is_const()) {
    return 1;
  }

  if (the_vertex->is_start() && !isIncremental()) {
    DLOG_INFO_FIRST_N(10) << "Thread " << std::this_thread::get_id()
                          << " date fwd propagate found start vertex."
                          << the_vertex->getName();

    createStartData(the_vertex);
  }

#if INTEGRATION_FWD
  // data propagation end at the clock vertex.
  if (the_vertex->is_end()) {
    // calc check arc at the end vertex.
    FOREACH_SNK_ARC(the_vertex, snk_arc) {
#if CUDA_PROPAGATION
      // collect different level arcs.
      addLevelArcs(the_vertex->get_level(), snk_arc);
#else
      snk_arc->exec(*this);
#endif
    }
  }
#endif

  if (isTracePath()) {
    addTracePathVertex(the_vertex);
  }

  FOREACH_SRC_ARC(the_vertex, src_arc) {
    if (!src_arc->isDelayArc()) {
      continue;
    }

    if (src_arc->is_loop_disable()) {
      continue;
    }

    auto* snk_vertex = src_arc->get_snk();
    if (!snk_vertex->get_prop_tag().is_prop()) {
      continue;
    }

#if CUDA_PROPAGATION
    // collect different level arcs.
    addLevelArcs(the_vertex->get_level(), src_arc);
#else
    if (!src_arc->exec(*this)) {
      LOG_FATAL << "data propagation error";
      break;
    }

#endif

    // get the next level bfs vertex and add it to the queue.
    if (snk_vertex->get_level() == (the_vertex->get_level() + 1)) {
      addNextBFSQueue(snk_vertex);
    }
  }

#if INTEGRATION_FWD
  the_vertex->set_is_slew_prop();
  the_vertex->set_is_delay_prop();
#endif

  the_vertex->set_is_fwd();

  return 1;
}

/**
 * @brief The data propagation using BFS.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaFwdPropagationBFS::operator()(StaGraph* the_graph) {
  ieda::Stats stats;
  LOG_INFO << "data fwd propagation bfs start";
  unsigned is_ok = 1;

  StaVertex* the_vertex;
  FOREACH_VERTEX(the_graph, the_vertex) {
    // start from the vertex which is level one and has slew prop.
    if ((the_vertex->get_level() == 1) && !the_vertex->is_fwd()) {
      // only propagate the vertex has slew.
      if (the_vertex->is_delay_prop()) {
        LOG_FATAL_IF(!the_vertex->is_delay_prop())
            << "the vertex should be delay propagated.";
        _bfs_queue.emplace_back(the_vertex);
      }
    }
  }

  // lambda for propagate the current queue.
  auto propagate_current_queue = [this](auto& current_queue) {
    LOG_INFO << "propagating current data queue vertexes number is "
             << current_queue.size();

#if 1
    {
      // create thread pool
      unsigned num_threads = getNumThreads();
      // unsigned num_threads = 1;
      ThreadPool pool(num_threads);

      for (auto* the_vertex : current_queue) {
        pool.enqueue(
            [this](StaVertex* the_vertex) { return the_vertex->exec(*this); },
            the_vertex);
      }
    }

#else
    for (auto* the_vertex : current_queue) {
      the_vertex->exec(*this);
    }

#endif
  };

  // do the bfs traverse for calc the clock slew/delay.
  do {
    propagate_current_queue(_bfs_queue);
    _bfs_queue.clear();

    // swap to the next bfs queue.
    std::swap(_bfs_queue, _next_bfs_queue);

  } while (!_bfs_queue.empty());

#if CUDA_PROPAGATION
  dispatchArcTask();
#endif

  LOG_INFO << "data fwd propagation bfs end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "data fwd propagation bfs memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "data fwd propagation bfs time elapsed " << time_delta << "s";

  return is_ok;
}

/**
 * @brief build gpu vertex slew data.
 * 
 * @param the_vertex 
 * @param gpu_vertex 
 * @param flatten_slew_data 
 */
void build_gpu_vertex_slew_data(StaVertex* the_vertex, GPU_Vertex& gpu_vertex,
                                std::vector<GPU_Fwd_Data>& flatten_slew_data) {
  // build slew data.
  the_vertex->initSlewData();
  gpu_vertex._slew_data._start_pos = flatten_slew_data.size();
  StaData* slew_data;
  FOREACH_SLEW_DATA(the_vertex, slew_data) {
    GPU_Fwd_Data gpu_slew_data;
    auto* the_slew_data = dynamic_cast<StaSlewData*>(slew_data);
    double slew_value = FS_TO_NS(the_slew_data->get_slew());

    gpu_slew_data._data_value = slew_value;
    gpu_slew_data._trans_type =
        the_slew_data->get_trans_type() == TransType::kRise
            ? GPU_Trans_Type::kRise
            : GPU_Trans_Type::kFall;
    gpu_slew_data._analysis_mode =
        the_slew_data->get_delay_type() == AnalysisMode::kMax
            ? GPU_Analysis_Mode::kMax
            : GPU_Analysis_Mode::kMin;
    flatten_slew_data.emplace_back(gpu_slew_data);
  }
  gpu_vertex._slew_data._num_fwd_data =
      flatten_slew_data.size() - gpu_vertex._slew_data._start_pos;
}

/**
 * @brief build gpu graph for gpu speed computation.
 *
 * @param the_sta_graph
 * @return GPU_Graph
 */
GPU_Graph build_gpu_graph(StaGraph* the_sta_graph) {
  GPU_Graph gpu_graph;
  unsigned num_vertex = the_sta_graph->numVertex();
  unsigned num_arc = the_sta_graph->numArc();

  std::vector<GPU_Vertex> gpu_vertices;
  gpu_vertices.reserve(num_vertex);
  std::vector<GPU_Arc> gpu_arcs;
  gpu_arcs.reserve(num_arc);

  std::vector<GPU_Fwd_Data> flatten_slew_data;
  flatten_slew_data.reserve(c_gpu_num_vertex_data * num_vertex);
  std::vector<GPU_Fwd_Data> flatten_at_data;
  flatten_at_data.reserve(c_gpu_num_vertex_data * num_vertex);

  // build gpu vertex
  StaVertex* the_vertex;
  FOREACH_VERTEX(the_sta_graph, the_vertex) { 
    GPU_Vertex gpu_vertex;
    build_gpu_vertex_slew_data(the_vertex, gpu_vertex, flatten_slew_data);
    gpu_vertices.emplace_back(std::move(gpu_vertex));
  }

  return gpu_graph;
}

/**
 * @brief dispatch arc propagation task on CPU or GPU.
 *
 */
void StaFwdPropagationBFS::dispatchArcTask() {
  if (_level_to_arcs.empty()) {
    return;
  }

  ieda::Stats stats;
  LOG_INFO << "dispatch arc task start";
  for (auto& [level, the_arcs] : _level_to_arcs) {
    std::for_each(std::execution::par, the_arcs.begin(), the_arcs.end(),
                  [this](auto* the_arc) { the_arc->exec(*this); });
  }
  _level_to_arcs.clear();

  LOG_INFO << "dispatch arc task end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "dispatch arc task memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "dispatch arc task time elapsed " << time_delta << "s";
}

}  // namespace ista