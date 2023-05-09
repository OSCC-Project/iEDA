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
 * @file StaCheck.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The timing check for combination loop, liberty integrity, timing
 * constrain missing and else etc.
 * @version 0.1
 * @date 2021-03-01
 */

#pragma once

#include "StaFunc.hh"
#include "StaGraph.hh"
#include "sta/StaVertex.hh"

namespace ista {

/**
 * @brief The combination loop check functor.
 *
 */
class StaCombLoopCheck : public StaFunc {
 public:
  virtual unsigned operator()(StaGraph* the_graph);

 private:
  // loop record
  void printAndBreakLoop(bool is_fwd);
  std::queue<StaVertex*> _loop_record;
};

}  // namespace ista
