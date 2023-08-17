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
 * @file BoundSkewTree.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <array>
#include <string>
#include <vector>

#include "Components.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"

namespace icts {
namespace bst {
/**
 * @brief bound skew tree
 *
 */
class BoundSkewTree
{
 public:
  BoundSkewTree(const std::string& net_name, const std::vector<Pin*>& pins, const std::optional<double>& skew_bound = std::nullopt)
      : _net_name(net_name)
  {
    _skew_bound = skew_bound.value_or(Timing::getSkewBound());
    std::ranges::for_each(pins, [&](Pin* pin) {
      LOG_FATAL_IF(!pin->isLoad()) << "pin " << pin->get_name() << " is not load pin";
      Timing::initLoadPinDelay(pin);
      Timing::updatePinCap(pin);
      auto* node = new BstNode(pin);
      _unmerged_nodes.push_back(node);
      _node_map.insert({pin->get_name(), pin});
    });
  }
  ~BoundSkewTree() = default;

  std::vector<Inst*> getInsertBufs() const { return _insert_bufs; }

  void run();

  void set_root_guide(const Point& root_guide)
  {
    auto x = root_guide.x();
    auto y = root_guide.y();
    _root_guide = Pt(1.0 * x / Timing::getDbUnit(), 1.0 * y / Timing::getDbUnit());
  }

 private:
  /**
   * @brief match
   *
   */
  using CostFunc = std::function<double(BstNode*, BstNode*)>;

  Match getBestMatch(CostFunc cost_func) const;

  double skewCost(BstNode* left, BstNode* right);

  double distanceCost(BstNode* left, BstNode* right) const;

  /**
   * @brief flow require
   *
   */
  void joinSegment(BstNode* parent, BstNode* left, BstNode* right);
  void merge(BstNode* parent, BstNode* left, BstNode* right);

  /**
   * @brief data
   *
   */
  std::string _net_name = "";

  std::vector<Inst*> _insert_bufs;
  std::set<Net*> _nets;
  std::vector<BstNode*> _unmerged_nodes;
  std::map<std::string, Node*> _node_map;
  std::optional<Pt> _root_guide;

  double _skew_bound = 0;

  constexpr static size_t kLeft = 0;
  constexpr static size_t kRight = 1;
  constexpr static size_t kMax = 0;
  constexpr static size_t kMin = 1;
  constexpr static LayerPattern kH = LayerPattern::kH;
  constexpr static LayerPattern kV = LayerPattern::kV;

  const int _db_unit = Timing::getDbUnit();
  const double _unit_h_cap = Timing::getUnitCap(kH);
  const double _unit_h_res = Timing::getUnitRes(kH);
  const double _unit_v_cap = Timing::getUnitCap(kV);
  const double _unit_v_res = Timing::getUnitRes(kV);

  Side<Pts> _join_region;
  Side<Pts> _join_segment;
  Side<Pts> _bal_points;  // balance points which skew equal to 0
  Side<Pt> _join_corner;
  Side<Pts> _fms_points;  // feasible merging section
};
}  // namespace bst
}  // namespace icts