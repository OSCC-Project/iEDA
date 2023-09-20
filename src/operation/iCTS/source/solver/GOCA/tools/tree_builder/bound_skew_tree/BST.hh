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
 * @file BST.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <string>
#include <vector>

#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"
#include "TimingPropagator.hh"
#include "pgl.h"
namespace icts {
struct MergeMatch
{
  Node* left;
  Node* right;
  double merge_cost;
};

/**
 * @brief bound skew tree
 *
 */
class BST
{
 public:
  BST(const std::string& net_name, const std::vector<Pin*>& pins, const std::optional<double>& skew_bound = std::nullopt)
      : _net_name(net_name)
  {
    if (!skew_bound.has_value()) {
      _skew_bound = TimingPropagator::getSkewBound();
    } else {
      _skew_bound = skew_bound.value();
    }
    std::ranges::for_each(pins, [&](Pin* pin) { _unmerged_nodes.push_back(pin); });
  }
  ~BST() = default;

  std::vector<Inst*> getInsertBufs() const { return _insert_bufs; }

  void run();

  void set_root_guide(const Point& root_guide) { _root_guide = root_guide; }

 private:
  /**
   * @brief match & merge
   *
   */
  using MergeCostFunc = std::function<double(Node*, Node*)>;

  MergeMatch getBestMatch(MergeCostFunc cost_func) const;

  double skewCost(Node* left, Node* right);

  double distanceCost(Node* left, Node* right) const;

  void updateTiming(Node* node) const;

  void merge(Node* left, Node* right);

  void fuse(Node* left, Node* right);

  void clearNode(Node* node);

  void clearGeom(Node* node);
  /**
   * @brief DME method
   *
   */
  void joinSegment(Node* left, Node* right);

  double endPointByZeroSkew(Node* left, Node* right, const std::optional<double>& init_delay_i = std::nullopt,
                            const std::optional<double>& init_delay_j = std::nullopt) const;

  std::pair<double, double> calcEndpointLoc(Node* left, Node* right) const;

  Polygon calcMergeRegion(Node* left, Node* right);

  bool isBoundMerge(Node* left, Node* right);
  /**
   * @brief flow require
   *
   */
  void timingInit();

  void preBuffering();

  Node* buffering(Node* node);

  bool amplifySubBufferSize(Node* node, const size_t& level = 1) const;

  bool bufferResizing(Node* node) const;

  void wireSnaking(Node* node) const;

  void insertBuffer(Node* parent, Node* child);

  double calcRequireSkew(Node* node) const;

  std::pair<double, double> calcGlobalDelayRange() const;

  bool skewFeasible(Node* left, Node* right);

  std::pair<Node*, Node*> timingOpt(Node* left, Node* right);

  void skewFix(Node* start);

  Node* getMinDelayChild(Node* node) const;

  Node* getMaxDelayChild(Node* node) const;

  Net* getNet(Node* node) const;

  void topdown(Node* root) const;

  void errorUpdate(Net* net) const;
  /**
   * @brief debug
   *
   */
  void reportSkew(Node* node) const;
  void reportMaxDelay(Node* node) const;
  /**
   * @brief data
   *
   */
  std::string _net_name = "";

  std::vector<Inst*> _insert_bufs;
  std::set<Net*> _nets;
  std::vector<Node*> _unmerged_nodes;
  std::map<Node*, Segment> _js_map;
  std::map<Node*, Polygon> _mr_map;
  std::optional<Point> _root_guide;
  double _skew_bound = 0;
  const int _db_unit = TimingPropagator::getDbUnit();
  const double _unit_cap = TimingPropagator::getUnitCap();
  const double _unit_res = TimingPropagator::getUnitRes();
};
}  // namespace icts