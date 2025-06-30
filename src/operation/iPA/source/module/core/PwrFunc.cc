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
 * @file PwrFunc.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of power analysis functor.
 * @version 0.1
 * @date 2023-02-14
 */
#include "PwrFunc.hh"

#include "PwrArc.hh"
#include "PwrSeqGraph.hh"

namespace ipower {

/**
 * @brief dump the seq trace stack information.
 *
 * @param out
 * @param the_power_func
 */
void PwrFunc::printSeqVertexTraceStack(const char* file_name,
                                       PwrFunc& dump_func) {
  while (!_seq_vertex_track_stack.empty()) {
    auto* seq_vertex = _seq_vertex_track_stack.top();
    seq_vertex->exec(dump_func);
    _seq_vertex_track_stack.pop();
  }

  dump_func.printText(file_name);
}

/**
 * @brief dump the vertex trace stack information.
 *
 * @param file_name
 * @param dump_func
 */
void PwrFunc::printVertexTraceStack(const char* file_name, PwrFunc& dump_func) {
  while (!_vertex_track_stack.empty()) {
    auto* pwr_vertex = _vertex_track_stack.top();
    pwr_vertex->exec(dump_func);
    _vertex_track_stack.pop();
  }

  dump_func.printText(file_name);
}

/**
 * @brief dump the arc trace stack information.
 * 
 * @param file_name 
 * @param dump_func 
 */
void PwrFunc::printArcTraceStack(const char* file_name, PwrFunc& dump_func) {
  while (!_arc_trace_stack.empty()) {
    auto* pwr_arc = _arc_trace_stack.top();
    pwr_arc->exec(dump_func);
    _arc_trace_stack.pop();
  }

  dump_func.printText(file_name);
}

}  // namespace ipower
