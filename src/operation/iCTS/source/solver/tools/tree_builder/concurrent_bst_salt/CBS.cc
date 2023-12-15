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
 * @file CBS.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CBS.hh"
#include "base/rsa.h"
#include "refine/refine.h"
namespace icts {
void CBSInterface::init(salt::Tree& min_tree, std::shared_ptr<salt::Pin> src_pin)
{
  min_tree.UpdateId();
  auto mt_nodes = min_tree.ObtainNodes();
  _nodes.resize(mt_nodes.size());
  _shortest_latency.resize(mt_nodes.size());
  _cur_latency.resize(mt_nodes.size());
  for (auto mt_node : mt_nodes) {
    _nodes[mt_node->id] = std::make_shared<salt::TreeNode>(mt_node->loc, mt_node->pin, mt_node->id);
    _shortest_latency[mt_node->id] = utils::Dist(src_pin->loc, mt_node->loc);
    _cur_latency[mt_node->id] = std::numeric_limits<double>::max();
  }
  _cur_latency[src_pin->id] = 0;
  _src = _nodes[src_pin->id];
}

void CBSInterface::finalize(const salt::Net& net, salt::Tree& tree)
{
  for (auto n : _nodes) {
    if (n->parent) {
      _nodes[n->parent->id]->children.push_back(n);
    }
  }
  tree.source = _src;
  tree.net = &net;
}

void TreeSaltBuilder::run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level)
{
  // input tree
  auto tree = input_tree;
  // Refine tree
  if (refine_level >= 1) {
    salt::Refine::flip(tree);
    salt::Refine::uShift(tree);
    salt::Refine::removeRedundantCoincident(tree);
  }

  // init
  init(tree, net.source());
  // dfs
  dfs(tree.source, _src, eps);
  finalize(net, input_tree);
  input_tree.RemoveTopoRedundantSteiner();

  // Connect breakpoints to source by RSA
  salt::RsaBuilder rsa_builder;
  rsa_builder.ReplaceRootChildren(input_tree);
  input_tree.UpdateId();
  // Refine SALT
  if (refine_level >= 1) {
    salt::Refine::cancelIntersect(input_tree);
    salt::Refine::flip(input_tree);
    salt::Refine::uShift(input_tree);
    if (refine_level >= 2) {
      salt::Refine::substitute(input_tree, eps, refine_level == 3);
    }
  }
}
bool TreeSaltBuilder::relax(const std::shared_ptr<salt::TreeNode>& u, const std::shared_ptr<salt::TreeNode>& v)
{
  auto new_latency = _cur_latency[u->id] + utils::Dist(u->loc, v->loc);
  if (_cur_latency[v->id] > new_latency) {
    _cur_latency[v->id] = new_latency;
    v->parent = u;
    return true;
  }
  if (_cur_latency[v->id] == new_latency && utils::Dist(u->loc, v->loc) < v->WireToParentChecked()) {
    v->parent = u;
    return true;
  }
  return false;
}
void TreeSaltBuilder::dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps)
{
  if (tree_node->pin && _cur_latency[cbs_node->id] > (1 + eps) * _shortest_latency[cbs_node->id]) {
    cbs_node->parent = _src;
    _cur_latency[cbs_node->id] = _shortest_latency[cbs_node->id];
  }
  for (auto c : tree_node->children) {
    relax(cbs_node, _nodes[c->id]);
    dfs(c, _nodes[c->id], eps);
    relax(_nodes[c->id], cbs_node);
  }
}

void TempBuilder::run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level)
{
  // tree
  auto tree = input_tree;

  // Refine tree
  if (refine_level >= 1) {
    salt::Refine::flip(tree);
    salt::Refine::uShift(tree);
  }

  // init
  init(tree, net.source());

  // dfs
  dfs(tree.source, _src, eps);
  finalize(net, input_tree);
  input_tree.RemoveTopoRedundantSteiner();

  // Connect breakpoints to source by RSA
  salt::RsaBuilder rsa_builder;
  rsa_builder.ReplaceRootChildren(input_tree);

  // Refine SALT
  if (refine_level >= 1) {
    salt::Refine::cancelIntersect(input_tree);
    salt::Refine::flip(input_tree);
    salt::Refine::uShift(input_tree);
    if (refine_level >= 2) {
      salt::Refine::substitute(input_tree, eps, refine_level == 3);
    }
  }
}

void TempBuilder::init(salt::Tree& min_tree, std::shared_ptr<salt::Pin> src_pin)
{
  min_tree.UpdateId();
  auto mt_nodes = min_tree.ObtainNodes();
  _nodes.resize(mt_nodes.size());
  _shortest_latency.resize(mt_nodes.size());
  _cur_latency.resize(mt_nodes.size());
  // update cap load
  _cap_loads.resize(mt_nodes.size());
  min_tree.postOrder([&](const std::shared_ptr<salt::TreeNode>& n) {
    if (n->pin) {
      _cap_loads[n->id] = n->pin->cap;
      return;
    }
    _cap_loads[n->id] = 0;
    std::ranges::for_each(n->children, [&](const std::shared_ptr<salt::TreeNode>& c) { _cap_loads[n->id] += _cap_loads[c->id]; });
  });

  for (auto mt_node : mt_nodes) {
    _nodes[mt_node->id] = std::make_shared<salt::TreeNode>(mt_node->loc, mt_node->pin, mt_node->id);
    _shortest_latency[mt_node->id] = utils::Dist(src_pin->loc, mt_node->loc);
    _cur_latency[mt_node->id] = std::numeric_limits<double>::max();
  }
  _cur_latency[src_pin->id] = 0;
  _src = _nodes[src_pin->id];
}

void TempBuilder::resetParent(const std::shared_ptr<salt::TreeNode>& child, const std::shared_ptr<salt::TreeNode>& parent)
{
  auto origin_parent = child->parent;
  auto cap_load = _cap_loads[child->id];
  if (origin_parent) {
    _cap_loads[origin_parent->id] -= cap_load;
  }
  child->parent = parent;
  _cap_loads[parent->id] += cap_load;
}

bool TempBuilder::relax(const std::shared_ptr<salt::TreeNode>& u, const std::shared_ptr<salt::TreeNode>& v)
{
  auto new_latency = _cur_latency[u->id] + delay(u, v);
  if (_cur_latency[v->id] > new_latency) {
    _cur_latency[v->id] = new_latency;
    resetParent(v, u);
    return true;
  }
  auto delay_from_parent = v->parent ? delay(v->parent, v) : 0;
  if (_cur_latency[v->id] == new_latency && delay(u, v) < delay_from_parent) {
    resetParent(v, u);
    return true;
  }
  return false;
}

void TempBuilder::dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps)
{
  if (tree_node->pin && _cur_latency[cbs_node->id] > (1 + eps) * _shortest_latency[cbs_node->id]) {
    resetParent(cbs_node, _src);
    _cur_latency[cbs_node->id] = _shortest_latency[cbs_node->id];
  }
  for (auto c : tree_node->children) {
    relax(cbs_node, _nodes[c->id]);
    dfs(c, _nodes[c->id], eps);
    relax(_nodes[c->id], cbs_node);
  }
}
}  // namespace icts