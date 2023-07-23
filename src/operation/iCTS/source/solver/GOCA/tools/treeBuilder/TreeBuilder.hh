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
 * @file TreeBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <string>
#include <vector>

#include "Node.hh"
namespace icts {
/**
 * @brief TreeBuilder for GOCA
 *       support:
 *          1. build tree:
 *                  1.1 shallow light tree
 *                  1.2 DME tree
 *          2. recover tree:
 *                  2.1 remove trunk node and root node
 */
class TreeBuilder
{
 public:
  TreeBuilder() = delete;
  ~TreeBuilder() = default;

  static void shallowLightTree(const std::vector<Node*>& loads, Node* driver);
  static std::vector<Node*> DMETree(const std::vector<Node*>& loads);
  static void recoverTree(Node* root);

 private:
};
}  // namespace icts