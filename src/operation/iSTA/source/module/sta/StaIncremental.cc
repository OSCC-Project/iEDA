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
 * @file StaIncremental.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The incremental static timing analysis DAG graph.
 * @version 0.1
 * @date 2022-01-08
 */

#include "StaIncremental.hh"

#include "StaDataPropagation.hh"
#include "StaDelayPropagation.hh"
#include "StaSlewPropagation.hh"
#include "log/Log.hh"
#include "sta/StaVertex.hh"

namespace ista {

StaIncremental::StaIncremental()
    : _fwd_queue(min_heap_cmp), _bwd_queue(max_heap_cmp) {}

/**
 * @brief propagate the slew from the vertex to its fanout.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaIncremental::propagateSlew(StaVertex* the_vertex) {
  StaSlewPropagation slew_propagation;
  slew_propagation.set_is_incremental();

  return the_vertex->exec(slew_propagation);
}

/**
 * @brief propagate the delay from the vertex to its fanout.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaIncremental::propagateDelay(StaVertex* the_vertex) {
  StaDelayPropagation delay_propagation;
  delay_propagation.set_is_incremental();

  return the_vertex->exec(delay_propagation);
}

/**
 * @brief propagate the arrival time from the vertex to its fanout.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaIncremental::propagateAT(StaVertex* the_vertex) {
  StaFwdPropagation fwd_propagation;
  fwd_propagation.set_is_incremental();

  return the_vertex->exec(fwd_propagation);
}

/**
 * @brief propagate the required time from the vertex to its fanin.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaIncremental::propagateRT(StaVertex* the_vertex) {
  StaBwdPropagation bwd_propagation;
  bwd_propagation.set_is_incremental();

  return the_vertex->exec(bwd_propagation);
}

/**
 * @brief insert the vertex to fwd propagation queue.
 *
 * @param the_vertex
 */
void StaIncremental::insertFwdQueue(StaVertex* the_vertex) {
  _fwd_queue.push(the_vertex);
}

/**
 * @brief insert the vertex to bwd propagation queue.
 *
 * @param the_vertex
 */
void StaIncremental::insertBwdQueue(StaVertex* the_vertex) {
  _bwd_queue.push(the_vertex);
}

/**
 * @brief apply the vertex of fwd queue to fwd propagation.
 *
 * @return unsigned
 */
unsigned StaIncremental::applyFwdQueue() {
  unsigned is_ok = 1;

  while (!_fwd_queue.empty()) {
    auto* the_vertex = _fwd_queue.top();
    // DLOG_INFO << "fwd queue: " << the_vertex->getName();

    // need to parallel execute the follow task.
    is_ok &= propagateSlew(the_vertex);
    is_ok &= propagateDelay(the_vertex);
    is_ok &= propagateAT(the_vertex);

    if (!is_ok) {
      break;
    }

    _fwd_queue.pop();
  }

  return is_ok;
}

/**
 * @brief apply the vertex of bwd queue to bwd propagation.
 *
 * @return unsigned
 */
unsigned StaIncremental::applyBwdQueue() {
  unsigned is_ok = 1;

  while (!_fwd_queue.empty()) {
    auto* the_vertex = _fwd_queue.top();

    // need to parallel execute the follow task.
    is_ok &= propagateRT(the_vertex);

    if (!is_ok) {
      break;
    }

    _fwd_queue.pop();
  }

  return is_ok;
}

/**
 * @brief reset the vertex propgagation.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaResetPropagation::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_fwd_mutex());
  if (beyondLevel(the_vertex->get_level())) {
    return 1;
  }

  LOG_FATAL_IF(!_incr_func) << "incr_func is nullptr";
  if (_is_fwd) {
    if (the_vertex->is_fwd_reset()) {
      return 1;
    }

    the_vertex->reset_is_slew_prop();
    the_vertex->reset_is_delay_prop();
    the_vertex->reset_is_fwd();
    the_vertex->set_is_fwd_reset();

    _incr_func->insertFwdQueue(the_vertex);

    if (the_vertex->is_end()) {
      return 1;
    }

    FOREACH_SRC_ARC(the_vertex, src_arc) {
      if (!src_arc->isDelayArc()) {
        continue;
      }

      if (src_arc->is_loop_disable()) {
        continue;
      }

      src_arc->exec(*this);
    }
  } else {
    if (the_vertex->is_bwd_reset()) {
      return 1;
    }

    the_vertex->reset_is_bwd();
    the_vertex->set_is_bwd_reset();

    _incr_func->insertBwdQueue(the_vertex);

    if (the_vertex->is_start()) {
      return 1;
    }

    FOREACH_SNK_ARC(the_vertex, snk_arc) {
      if (!snk_arc->isDelayArc()) {
        continue;
      }

      if (snk_arc->is_loop_disable()) {
        continue;
      }
      snk_arc->exec(*this);
    }
  }

  return 1;
}

/**
 * @brief propagate the reset along with the arc.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaResetPropagation::operator()(StaArc* the_arc) {
  StaVertex* the_vertex;
  if (_is_fwd) {
    the_vertex = the_arc->get_snk();

  } else {
    the_vertex = the_arc->get_src();
  }
  return the_vertex->exec(*this);
}

}  // namespace ista
