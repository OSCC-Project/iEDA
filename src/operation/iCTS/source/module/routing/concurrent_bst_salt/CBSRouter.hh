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
 * @file CBSRouter.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-16
 * @brief Standalone CBS router interface and custom SALT refinement builder.
 */
#pragma once

#include <memory>
#include <vector>

#include "RoutingTerminal.hh"
#include "SteinerTree.hh"

namespace salt {
class Net;
class Pin;
class Tree;
class TreeNode;
}  // namespace salt

namespace icts {

struct BSTRoutingConfig;

class CustomSaltBuilder
{
 public:
  CustomSaltBuilder() = default;
  ~CustomSaltBuilder();

  auto run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level = 3) -> void;

 private:
  auto relax(const std::shared_ptr<salt::TreeNode>& source_node, const std::shared_ptr<salt::TreeNode>& target_node) -> bool;
  auto dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps) -> void;
  auto init(const salt::Tree& min_tree, const std::shared_ptr<salt::Pin>& src_pin) -> void;
  auto finalize(const salt::Net& net, salt::Tree& tree) const -> void;

  std::vector<std::shared_ptr<salt::TreeNode>> _nodes;
  std::vector<double> _shortest_latency;
  std::vector<double> _cur_latency;
  std::shared_ptr<salt::TreeNode> _src = nullptr;
};

class CBSRouter
{
 public:
  using Terminal = ClockRoutingTerminal;
  using ClockSteinerTreeType = ClockSteinerTree<>;

  CBSRouter() = delete;
  ~CBSRouter() = default;

  static auto buildTree(const std::vector<Terminal>& load_terminals, const BSTRoutingConfig& parameters) -> ClockSteinerTreeType;
};

}  // namespace icts
