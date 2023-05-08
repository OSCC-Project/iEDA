/**
 * @file StaFunc.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sta functor class.
 * @version 0.1
 * @date 2021-02-17
 */
#pragma once

#include <stack>

#include "Sta.hh"
#include "StaGraph.hh"
#include "Type.hh"
#include "log/Log.hh"
#include "sdc/SdcConstrain.hh"

namespace ista {

class StaVertx;
class StaGraph;
class StaClock;

/**
 * @brief The base functor of sta.
 *
 */
class StaFunc {
 public:
  StaFunc();
  virtual ~StaFunc();
  virtual unsigned operator()(StaGraph* the_graph);
  virtual unsigned operator()(StaVertex* the_vertex);
  virtual unsigned operator()(StaArc* the_arc);
  virtual unsigned operator()(StaClock* the_clock);

  virtual AnalysisMode get_analysis_mode();
  unsigned getNumThreads();
  Sta* getSta() { return _ista; }

  void set_is_trace_path() { _is_trace_path = true; }
  [[nodiscard]] bool isTracePath() const { return _is_trace_path; }
  void reset_is_trace_path() { _is_trace_path = false; }

  void set_is_incremental() { _is_incremental = true; }
  [[nodiscard]] bool isIncremental() const { return _is_incremental; }

  void PrintTraceRecord();

 protected:
  void addTracePathVertex(StaVertex* the_vertex) {
    _trace_path_record.push(the_vertex);
  }

 private:
  Sta* _ista;

  unsigned _is_trace_path : 1 = 0;
  unsigned _is_incremental : 1 = 0;
  unsigned _reserved : 30;

  std::stack<StaVertex*> _trace_path_record;
};

}  // namespace ista
