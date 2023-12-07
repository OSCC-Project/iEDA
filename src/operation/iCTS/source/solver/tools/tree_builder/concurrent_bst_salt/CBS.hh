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
 * @file CBS.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include "TimingPropagator.hh"
#include "salt.h"
namespace icts {
using namespace salt;
/**
 * @brief Balance skEw lAtency Tree
 *
 */
template <typename T>
concept DelayType = requires(T t)
{
  {
    t.id
  }
  ->std::convertible_to<int>;
  {
    t.loc
  }
  ->std::convertible_to<salt::Point>;
};

class CBSInterface
{
 protected:
  std::vector<std::shared_ptr<salt::TreeNode>> _nodes;  // nodes of the bound-skew tree
  std::vector<double> _shortest_latency;
  std::vector<double> _cur_latency;
  std::shared_ptr<salt::TreeNode> _src;  // source node of the bound-skew tree

  void init(salt::Tree& min_tree, std::shared_ptr<salt::Pin> src_pin);  // tree of minimum weight
  void finalize(const salt::Net& net, salt::Tree& tree);
  virtual bool relax(const std::shared_ptr<salt::TreeNode>& u, const std::shared_ptr<salt::TreeNode>& v) = 0;  // from u to v
  virtual void dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps) = 0;
};

class TreeSaltBuilder : public CBSInterface
{
 public:
  void run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level = 3);

 protected:
  bool relax(const std::shared_ptr<salt::TreeNode>& u,
             const std::shared_ptr<salt::TreeNode>& v);  // from u to v
  void dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps);
};

class TempBuilder : public CBSInterface
{
 public:
  void run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level = 3);

 protected:
  std::vector<double> _cap_loads;
  template <DelayType T1, DelayType T2>
  double delay(const T1& from, const T2& to) const
  {
    auto from_loc = from.loc;
    auto to_loc = to.loc;
    auto len = utils::Dist(from_loc, to_loc) / TimingPropagator::getDbUnit();
    auto cap_load = _cap_loads[to.id];
    return TimingPropagator::calcElmoreDelay(cap_load, len);
  }
  template <DelayType T1, DelayType T2>
  double delay(const std::shared_ptr<T1>& from, const std::shared_ptr<T2>& to) const
  {
    auto from_loc = from->loc;
    auto to_loc = to->loc;
    auto len = utils::Dist(from_loc, to_loc) / TimingPropagator::getDbUnit();
    auto cap_load = _cap_loads[to->id];
    return TimingPropagator::calcElmoreDelay(cap_load, len);
  }
  void init(salt::Tree& min_tree, std::shared_ptr<salt::Pin> src_pin);
  void resetParent(const std::shared_ptr<salt::TreeNode>& child, const std::shared_ptr<salt::TreeNode>& parent);
  bool relax(const std::shared_ptr<salt::TreeNode>& u,
             const std::shared_ptr<salt::TreeNode>& v);  // from u to v
  void dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps);
};

}  // namespace icts