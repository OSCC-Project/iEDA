/**
 * @file StaIncremental.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The incremental static timing analysis DAG graph.
 * @version 0.1
 * @date 2022-01-08
 */

#include "StaDataPropagation.hh"
#include "StaDelayPropagation.hh"
#include "StaIncremental.hh"
#include "StaSlewPropagation.hh"
#include "log/Log.hh"
#include "sta/StaVertex.hh"

namespace ista {

StaIncremental::StaIncremental() : _fwd_queue(cmp), _bwd_queue(cmp) {}

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
    the_vertex->reset_is_slew_prop();
    the_vertex->reset_is_delay_prop();
    the_vertex->reset_is_fwd();

    _incr_func->insertFwdQueue(the_vertex);

    FOREACH_SRC_ARC(the_vertex, src_arc) { src_arc->exec(*this); }
  } else {
    the_vertex->reset_is_bwd();
    _incr_func->insertBwdQueue(the_vertex);
    FOREACH_SNK_ARC(the_vertex, snk_arc) { snk_arc->exec(*this); }
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
