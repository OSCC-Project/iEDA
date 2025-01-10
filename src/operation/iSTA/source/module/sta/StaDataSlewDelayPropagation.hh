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
 * @file StaDataSlewDelayPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The Data slew delay propagation using BFS method.
 * @version 0.1
 * @date 2024-12-26
 */
#pragma once

#include "StaFunc.hh"
#include <queue>
#include <mutex>

namespace ista {

/**
 * @brief The data slew and delay propagation with bfs.
 *
 */
class StaDataSlewDelayPropagation : public StaFunc {
 public:
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaGraph* the_graph) override;

  AnalysisMode get_analysis_mode() override { return AnalysisMode::kMaxMin; }

  private:

  void addNextBFSQueue(StaVertex* the_vertex) {
      static std::mutex g_mutex;

      if (std::find(_next_bfs_queue.begin(), _next_bfs_queue.end(),
                    the_vertex) == _next_bfs_queue.end()) {
        std::lock_guard<std::mutex> lk(g_mutex);
        _next_bfs_queue.push_back(the_vertex);
      }
  }

  std::vector<StaVertex*> _bfs_queue; //!< The current bfs queue
  std::vector<StaVertex*> _next_bfs_queue; //!< For next bfs use.
};

}