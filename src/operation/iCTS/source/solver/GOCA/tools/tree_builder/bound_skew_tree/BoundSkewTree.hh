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
#include "GeomCalc.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Pin.hh"

namespace icts {
namespace bst {
/**
 * @brief Tool namespace
 *
 */
using Geom = GeomCalc;
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
      auto* node = new Area(pin);
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
    _root_guide = Pt(1.0 * x / Timing::getDbUnit(), 1.0 * y / Timing::getDbUnit(), 0, 0, 0);
  }

  void set_pattern(const RCPattern& pattern) { _pattern = pattern; }

 private:
  /**
   * @brief match
   *
   */
  using CostFunc = std::function<double(Area*, Area*)>;
  Match getBestMatch(CostFunc cost_func) const;
  double skewCost(Area* left, Area* right);
  double distanceCost(Area* left, Area* right) const;

  /**
   * @brief flow require
   *
   */
  void merge(Area* parent, Area* left, Area* right);
  void constructMr(Area* parent, Area* left, Area* right);
  void jsProcess(Area* cur);
  void initSide();
  void updateJS(Area* cur, Line& left, Line& right, PtPair closest);

  void calcJr(Area* parent, Area* left, Area* right);
  LineType calcAreaLineType(Area* cur);
  void calcJrEndpoints(Area* cur);
  void calcNotManhattanJrEndpoints(Area* parent, Area* left, Area* right);
  void addJsPts(Area* parent, Area* left, Area* right);

  void calcJrCorner(Area* cur);
  void calcBalancePt(Area* cur);
  void calcFmsPt(Area* parent, Area* left, Area* right);

  bool existFmsOnJr();
  void constructFeasibleMr(Area* parent, Area* left, Area* right);
  void constructInfeasibleMr(Area* parent, Area* left, Area* right);
  void constructTrrMr(Area* cur);
  void calcConvexHull(Area* cur);
  void checkMr(Area* cur);

  double calcJrArea(const Line& l1, const Line& l2);
  void calcJS(Area* cur, Line& left, Line& right);
  void calcJS(Area* parent, Area* left, Area* right);
  void calcJsDelay(Area* left, Area* right);
  void calcBsLocated(Area* cur, Pt& pt, Line& line);
  void calcPtDelays(Area* cur, Pt& pt, Line& line);
  void updatePtDelaysBySide(Area* cur, const size_t& side, Pt& pt);
  void calcIrregularPtDelays(Area* cur, Pt& pt, Line& line);
  double ptDelayIncrease(Pt& p1, Pt& p2, const double& cap, const RCPattern& pattern = RCPattern::kHV);
  double calcDelayIncrease(const double& x, const double& y, const double& cap, const RCPattern& pattern = RCPattern::kHV);
  Line getJrLine(const size_t& side);
  Line getJsLine(const size_t& side);
  Line getJsLine(const size_t& side, const Side<Pts>& join_segment);
  void setJrLine(const size_t& side, const Line& line);
  void setJsLine(const size_t& side, const Line& line);
  double ptSkew(const Pt& pt);
  void checkPtDelay(Pt& pt);
  void checkJsMs();
  void checkUpdateJs(const Area* cur, Line& left, Line& right);
  void printPoint(const Pt& pt);
  void printArea(const Area* area);
  /**
   * @brief data
   *
   */
  std::string _net_name = "";

  std::vector<Inst*> _insert_bufs;
  std::set<Net*> _nets;
  std::vector<Area*> _unmerged_nodes;
  std::map<std::string, Node*> _node_map;
  std::optional<Pt> _root_guide;

  double _skew_bound = 0;
  RCPattern _pattern = RCPattern::kHV;

  const int _db_unit = Timing::getDbUnit();
  const double _unit_h_cap = Timing::getUnitCap(LayerPattern::kH);
  const double _unit_h_res = Timing::getUnitRes(LayerPattern::kH);
  const double _unit_v_cap = Timing::getUnitCap(LayerPattern::kV);
  const double _unit_v_res = Timing::getUnitRes(LayerPattern::kV);
  const Side<double> _K = {0.5 * _unit_h_res * _unit_h_cap, 0.5 * _unit_v_res* _unit_v_cap};

  Side<Pts> _join_region;
  Side<Pts> _join_segment;
  Side<Trr> _ms;
  Side<Pts> _bal_points;  // balance points which skew equal to 0
  Side<Pt> _join_corner;
  Side<Pts> _fms_points;  // feasible merging section
};
}  // namespace bst
}  // namespace icts