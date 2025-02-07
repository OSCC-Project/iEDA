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

/**
 * @brief The func for BFS processing.
 * 
 */
class StaBFSFunc {
  protected:

  void addNextBFSQueue(StaVertex* the_vertex) {
      static std::mutex g_mutex;
      std::lock_guard<std::mutex> lk(g_mutex);

      if (std::find(_next_bfs_queue.begin(), _next_bfs_queue.end(),
                    the_vertex) == _next_bfs_queue.end()) {        
        _next_bfs_queue.push_back(the_vertex);
      }
  }

  std::vector<StaVertex*> _bfs_queue; //!< The current bfs queue
  std::vector<StaVertex*> _next_bfs_queue; //!< For next bfs use.
};

}  // namespace ista
