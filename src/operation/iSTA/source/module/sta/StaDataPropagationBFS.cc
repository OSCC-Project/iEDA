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

#include "Config.hh"
#include "StaDataSlewDelayPropagation.hh"
#include "StaDelayPropagation.hh"
#include "StaGPUFwdPropagation.hh"
#include "StaSlewPropagation.hh"
#include "ThreadPool/ThreadPool.h"

namespace ista {

/**
 * @brief propagate the arc to accumulated the path delay.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaFwdPropagationBFS::operator()(StaArc* the_arc) {
  std::lock_guard<std::mutex> lk(the_arc->get_snk()->get_fwd_mutex());

#if !CUDA_PROPAGATION || CPU_SIM
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
#endif

  if (!the_arc->isCheckArc()) {
    // call parent operator to arrive time propagation.
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
  // data propagation end at the end vertex.
  if (the_vertex->is_end()) {
    // calc check arc at the end vertex.
    FOREACH_SNK_ARC(the_vertex, snk_arc) {
      if (!snk_arc->isCheckArc()) {
        continue;
      }
      
#if CUDA_PROPAGATION
      // collect different level arcs.
      addLevelArcs(the_vertex->get_level(), snk_arc);

      snk_arc->exec(*this);
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

    if (src_arc->isMpwArc()) {
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
#endif

    if (!src_arc->exec(*this)) {
      LOG_FATAL << "data propagation error";
      break;
    }

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

#if CUDA_PROPAGATION
#if !CPU_SIM
  initFwdData(the_graph);
#endif
#endif

  the_graph->sortVertexByLevel();

  StaVertex* the_vertex;
  FOREACH_VERTEX(the_graph, the_vertex) {
    // start from the vertex which is level one and has slew prop.
    if ((the_vertex->get_level() == 1) && !the_vertex->is_fwd()) {
      // only propagate the vertex has clock slew or is input port.
      if (the_vertex->is_slew_prop() || the_vertex->is_port()) {
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
  dispatchArcTask(the_graph);
#endif

  LOG_INFO << "data fwd propagation bfs end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "data fwd propagation bfs memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "data fwd propagation bfs time elapsed " << time_delta << "s";

  return is_ok;
}

/**
 * @brief For use gpu propagation, we first init the sta fwd data.
 *
 * @param the_graph
 */
void StaFwdPropagationBFS::initFwdData(StaGraph* the_graph) {
  StaArc* the_arc;
  FOREACH_ARC(the_graph, the_arc) {
    the_arc->initArcDelayData();
  }
}

/**
 * @brief dispatch arc propagation task on CPU or GPU.
 *
 */
void StaFwdPropagationBFS::dispatchArcTask(StaGraph* the_graph) {
#if CUDA_PROPAGATION

  if (_level_to_arcs.empty()) {
    return;
  }

  StaGPUFwdPropagation gpu_fwd_propagation(std::move(_level_to_arcs));
  gpu_fwd_propagation.prepareGPUData(the_graph);

  ieda::Stats stats;
  gpu_fwd_propagation(the_graph);

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "dispatch arc task memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "dispatch arc task time elapsed " << time_delta << "s";

#endif
}

}  // namespace ista