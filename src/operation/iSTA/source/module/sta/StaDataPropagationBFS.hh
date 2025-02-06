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
 * @file StaDataPropagationBFS.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of data propagation use the BFS method.
 * @version 0.1
 * @date 2025-01-10
 *
 */
#pragma once

#include "StaDataPropagation.hh"
#include "StaFunc.hh"

namespace ista {

/**
 * @brief The forward propagation using BFS method.
 *
 */
class StaFwdPropagationBFS : public StaBFSFunc, public StaFwdPropagation {
 public:
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;

  unsigned operator()(StaGraph* the_graph) override;

  private:

#if CUDA_PROPAGATION
  void addLevelArcs(unsigned level, StaArc* the_arc) {
    static std::mutex the_mutex;
    std::lock_guard<std::mutex> lk(the_mutex);
    _level_to_arcs[level].emplace_back(the_arc);
  }
  auto& get_level_to_arcs() { return _level_to_arcs; }
#endif
  
  void initFwdData(StaGraph* the_graph);
  void dispatchArcTask(StaGraph* the_graph);

#if CUDA_PROPAGATION
  std::map<unsigned, std::vector<StaArc*>> _level_to_arcs;
#endif
};



}  // namespace ista