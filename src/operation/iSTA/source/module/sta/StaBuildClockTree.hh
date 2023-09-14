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
 * @file StaBuildClockTree.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief build clock tree for GUI and debug clock tree.
 * @version 0.1
 * @date 2023-09-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include "StaClockTree.hh"
#include "StaFunc.hh"

namespace ista {

/**
 * @brief function for build clock tree.
 *
 */
class StaBuildClockTree : public StaFunc {
 public:
  unsigned operator()(StaClock* the_clock) override;
  auto& takeClockTrees() { return _clock_trees; }

 private:
  void buildNextPin(
      StaClockTree* clock_tree, StaClockTreeNode* parent_node,
      StaVertex* parent_vertex,
      std::map<StaVertex*, std::vector<StaData*>>& vertex_to_datas);

  void addClockTree(StaClockTree* clock_tree) {
    _clock_trees.emplace_back(clock_tree);
  }

  std::vector<std::unique_ptr<StaClockTree>> _clock_trees;
};

}  // namespace ista
