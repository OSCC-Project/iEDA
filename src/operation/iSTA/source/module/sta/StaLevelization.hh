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
 * @file StaLevelization.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The levelization from end vertex of the graph.
 * @version 0.1
 * @date 2021-09-16
 */
#pragma once

#include "StaFunc.hh"

namespace ista {

class StaLevelization : public StaFunc {
 public:
  StaLevelization() = default;
  ~StaLevelization() override = default;

  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;

  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista
