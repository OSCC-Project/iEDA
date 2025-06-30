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
 * @file PwrLevelizeSeq.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Ranking of sequential logic units.
 * @version 0.1
 * @date 2023-02-27
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {

/**
 * @brief levelize the PwrLevelization by traveling power graph.
 *
 */
class PwrLevelizeSeq : public PwrFunc {
 public:
  unsigned operator()(PwrSeqVertex* the_vertex) override;
  unsigned operator()(PwrSeqGraph* the_graph) override;
};
}  // namespace ipower
