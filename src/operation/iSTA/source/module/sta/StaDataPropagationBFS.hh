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

#include "StaFunc.hh"
#include "StaDataPropagation.hh"

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

};

}  // namespace ista