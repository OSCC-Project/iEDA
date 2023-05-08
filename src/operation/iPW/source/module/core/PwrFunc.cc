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
