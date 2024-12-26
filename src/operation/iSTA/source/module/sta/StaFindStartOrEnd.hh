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
 * @file StaFindStartOrEnd.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The class for finding start or end points of the timing path.
 * @version 0.1
 * @date 2023-07-21
 */
#pragma once

#include "BTreeSet.hh"
#include "StaFunc.hh"

namespace ista {

/**
 * @brief The class for find the end pins in the given start pin in the
 * timing path.
 *
 */
class StaFindEnd : public StaFunc {
 public:
  unsigned operator()(StaVertex* start_vertex) override;
  unsigned operator()(StaGraph* the_graph) override;
};

/**
 * @brief The class for find the start pins in the given end pin in the
 * timing path.
 *
 */
class StaFindStart : public StaFunc {
 public:
  unsigned operator()(StaVertex* end_vertex) override;
  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista