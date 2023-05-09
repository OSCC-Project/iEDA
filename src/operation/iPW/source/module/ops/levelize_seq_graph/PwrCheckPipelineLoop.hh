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
 * @file PwrCheckPipelineLoop.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Check the pipeline loop for the seq graph.
 * @version 0.1
 * @date 2023-03-14
 */

#pragma once

#include <vector>

#include "core/PwrFunc.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {

/**
 * @brief The class for record loop vertex.
 *
 */
class PwrSeqLoop {
 public:
  void insertVertex(PwrSeqVertex* the_vertex) {
    _pipeline_loop.emplace(the_vertex);
  }
  void print(std::ostream& out, unsigned serial_num) {
    out << "----loop " << serial_num << " begin ----\n";
    while (!_pipeline_loop.empty()) {
      out << _pipeline_loop.front()->get_obj_name() << " ---> ";
      _pipeline_loop.pop();
    }
    out << std::endl;
    out << "----loop " << serial_num << " end ----\n";
  }

 private:
  std::queue<PwrSeqVertex*> _pipeline_loop;
};

/**
 * @brief The class of check the pipeline loop for the seq graph.
 *
 */
class PwrCheckPipelineLoop : public PwrFunc {
 public:
  unsigned operator()(PwrSeqVertex* the_vertex) override;
  unsigned operator()(PwrSeqGraph* the_graph) override; 

  unsigned printPipelineLoop(std::ostream& out) {
    for (unsigned i = 0; auto& pipeline_loop : _pipeline_loops) {
      pipeline_loop.print(out, i);
      ++i;
    }

    return 1;
  };

 private:
  void pushStack(PwrSeqVertex* seq_vertex) {
    _dfs_stack.emplace_back(seq_vertex);
  }
  void popStack() { _dfs_stack.pop_back(); }
  auto getTop() { return _dfs_stack.front(); }

  /*record seq loop function.*/
  PwrSeqLoop getLoopFromStack(PwrSeqVertex* loop_vertex);
  void addLoop(PwrSeqLoop&& seq_loop) {
    _pipeline_loops.emplace_back(std::move(seq_loop));
  }
  auto& get_pipeline_loops() { return _pipeline_loops; }

  std::vector<PwrSeqVertex*>
      _dfs_stack;  //!< used for record dfs recursive vertex.
  std::vector<PwrSeqLoop>
      _pipeline_loops;  //!< The queue of a pipeline loop using for test.
  unsigned _record_num =
      1000;  //!< The pipeline record num that need to be printed.

  unsigned _loop_num = 0;  //!< The pipeline loop total num.
};
}  // namespace ipower
