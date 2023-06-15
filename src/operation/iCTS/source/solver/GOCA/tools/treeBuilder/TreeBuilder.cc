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
 * @file TreeBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "TreeBuilder.hh"

#include "CTSAPI.hpp"
#include "TimingPropagator.hh"

namespace icts {
/**
 * @brief shallow light tree
 *
 * @param loads
 * @param driver
 */
void TreeBuilder::shallowLightTree(const std::vector<Node*>& loads, Node* driver)
{
  CTSAPIInst.genShallowLightTree(loads, driver);
}
/**
 * @brief DME tree
 *
 * @param loads
 * @return std::vector<Node*> all the root nodes which are skew feasible
 */
std::vector<Node*> TreeBuilder::DMETree(const std::vector<Node*>& loads)
{
  return std::vector<Node*>();
}
/**
 * @brief recover tree
 *       remove trunk node and root node
 *       save the leaf node and disconnect the leaf node
 *
 * @param root
 */
void TreeBuilder::recoverTree(Node* root)
{
  std::vector<Node*> leafs;
  std::vector<Node*> to_be_removed;
  auto classify_node = [&leafs, &to_be_removed, &root](Node* node) {
    if (node->isSteiner() || node == root) {
      to_be_removed.push_back(node);
    } else {
      leafs.push_back(node);
    }
  };
  root->preOrderBy(classify_node);
  for (auto& node : leafs) {
    node->set_parent(nullptr);
    node->set_slew_in(0);
  }
  for (auto& node : to_be_removed) {
    delete node;
  }
}
}  // namespace icts