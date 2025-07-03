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
/**
 * @file PwrFunc.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The function base class of power analysis.
 * @version 0.1
 * @date 2023-02-14
 */
#pragma once

#include <iostream>
#include <stack>

#include "include/PwrConfig.hh"
#include "log/Log.hh"

namespace ipower {

class PwrGraph;
class PwrVertex;
class PwrArc;
class PwrCell;
class PwrSeqGraph;
class PwrSeqVertex;
class PwrSeqArc;

/**
 * @brief The functor of power analysis.
 *
 */
class PwrFunc {
 public:
  PwrFunc() = default;
  virtual ~PwrFunc() = default;

  void set_the_pwr_graph(PwrGraph* the_pwr_graph) {
    _the_pwr_graph = the_pwr_graph;
  }
  auto* get_the_pwr_graph() { return _the_pwr_graph; }

  void set_the_pwr_seq_graph(PwrSeqGraph* the_pwr_seq_graph) {
    _the_pwr_seq_graph = the_pwr_seq_graph;
  }
  auto* get_the_pwr_seq_graph() { return _the_pwr_seq_graph; }

  void set_num_threads(unsigned num_thread) { _num_threads = num_thread; }
  [[nodiscard]] unsigned get_num_threads() const { return _num_threads; }

  /*Power Graph functions.*/
  virtual unsigned operator()(PwrGraph* the_graph) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }
  virtual unsigned operator()(PwrVertex* the_vertex) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }
  virtual unsigned operator()(PwrArc* the_arc) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }
  virtual unsigned operator()(PwrCell* the_inst) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }

  /*Power sequential graph functions.*/
  virtual unsigned operator()(PwrSeqGraph* the_graph) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }
  virtual unsigned operator()(PwrSeqVertex* the_vertex) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }
  virtual unsigned operator()(PwrSeqArc* the_arc) {
    LOG_FATAL << "The func is not implemented";
    return 0;
  }

  virtual void printText(const char* file_name) {
    LOG_FATAL << "The func is not implemented";
  }

 protected:
  [[nodiscard]] unsigned isTrace() const { return _is_trace; }
  void set_is_trace(bool is_trace) { _is_trace = is_trace; }

  void addTraceSeqVertex(PwrSeqVertex* seq_vertex) {
    if (isTrace()) {
      _seq_vertex_track_stack.push(seq_vertex);
    }
  }
  void addTraceVertex(PwrVertex* pwr_vertex) {
    if (isTrace()) {
      _vertex_track_stack.push(pwr_vertex);
    }
  }

  void addTraceArc(PwrArc* pwr_arc) {
    if (isTrace()) {
      _arc_trace_stack.push(pwr_arc);
    }
  }

  void printSeqVertexTraceStack(const char* file_name, PwrFunc& dump_func);
  void printVertexTraceStack(const char* file_name, PwrFunc& dump_func);
  void printArcTraceStack(const char* file_name, PwrFunc& dump_func);

 private:
  PwrGraph* _the_pwr_graph = nullptr;  //!< The power functor need the power
                                       //!< graph, store here for useful.
  PwrSeqGraph* _the_pwr_seq_graph =
      nullptr;                            //!< The power functor need the seq
                                          //!< graph, store here for useful.
  unsigned _num_threads = c_num_threads;  //!< The num of threads.

  bool _is_trace = false;  // trace set.
  std::stack<PwrSeqVertex*>
      _seq_vertex_track_stack;  // seq trace stack for used record propagation.
  std::stack<PwrVertex*>
      _vertex_track_stack;  // trace stack for used record propagation vertex.
  std::stack<PwrArc*>
      _arc_trace_stack;  // trace stack for used record propagation arc.
};

}  // namespace ipower
