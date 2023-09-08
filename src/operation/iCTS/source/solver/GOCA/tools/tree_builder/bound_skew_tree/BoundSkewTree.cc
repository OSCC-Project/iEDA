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
 * @file BoundSkewTree.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "BoundSkewTree.hh"

#include "GeomOperator.hh"
#include "TreeBuilder.hh"
namespace icts {
namespace bst {
/**
 * @brief bst flow
 *
 */
BoundSkewTree::BoundSkewTree(const std::string& net_name, const std::vector<Pin*>& pins, const std::optional<double>& skew_bound)
{
  // Not input topology
  _net_name = net_name;
  _load_pins = pins;
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
BoundSkewTree::BoundSkewTree(const std::string& net_name, Pin* driver_pin, const std::optional<double>& skew_bound)
{
  _net_name = net_name;
  _skew_bound = skew_bound.value_or(Timing::getSkewBound());

  // Input topology
  std::unordered_map<Node*, Area*> node_area_map;
  driver_pin->postOrder([&](Node* node) {
    if (node->isPin() && node->isLoad()) {
      auto* pin = dynamic_cast<Pin*>(node);
      Timing::initLoadPinDelay(pin);
      Timing::updatePinCap(pin);
    }
    auto* area = new Area(node);
    // random select RCpattern
    auto pattern = static_cast<RCPattern>(1 + std::rand() % 2);
    area->set_pattern(pattern);
    _node_map.insert({node->get_name(), node});
    if (node->isPin() && node->isDriver()) {
      _root = area;
    }
    auto children = node->get_children();
    if (children.empty()) {
      auto* pin = dynamic_cast<Pin*>(node);
      _load_pins.push_back(pin);
      return;
    }
    LOG_FATAL_IF(children.size() != 2) << "node " << node->get_name() << " children size is not 2";
    auto left = children[kLeft];
    auto right = children[kRight];
    auto* left_area = node_area_map[left];
    auto* right_area = node_area_map[right];
    area->set_left(left_area);
    area->set_right(right_area);
    left_area->set_parent(area);
    right_area->set_parent(area);
  });
}
void BoundSkewTree::run()
{
  bottomUp();
  topDown();
}
void BoundSkewTree::convert()
{
  std::stack<Area*> stack;
  stack.push(_root);
  // pre-order build Node, leaf node will in _node_map
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();
    auto pt = cur->get_location();
    auto loc = Point(pt.x * _db_unit, pt.y * _db_unit);
    auto* parent = cur->get_parent();
    if (parent == nullptr) {
      // is root, make buffer
      auto* buf = TreeBuilder::genBufInst(_net_name, loc);
      cur->set_name(buf->get_name());
      _node_map.insert({buf->get_name(), buf->get_driver_pin()});
      _root_buf = buf;
    } else if (cur->get_left() == nullptr && cur->get_right() == nullptr) {
      // is load pin, find from _node_map
      auto* node = _node_map[cur->get_name()];
      LOG_FATAL_IF(node == nullptr) << "node " << cur->get_name() << " is not in _node_map";
      auto* parent_node = _node_map[parent->get_name()];
      LOG_FATAL_IF(parent_node == nullptr) << "node " << parent->get_name() << " is not in _node_map";
      parent_node->add_child(node);
      node->set_parent(parent_node);
    } else {
      // is steiner node
      auto* node = new Node(cur->get_name(), loc);
      auto* parent_node = _node_map[parent->get_name()];
      LOG_FATAL_IF(parent_node == nullptr) << "node " << parent->get_name() << " is not in _node_map";
      parent_node->add_child(node);
      node->set_parent(parent_node);
      _node_map.insert({node->get_name(), node});
    }

    if (cur->get_right()) {
      stack.push(cur->get_right());
    }
    if (cur->get_left()) {
      stack.push(cur->get_left());
    }
  }
  _net = Timing::genNet(_net_name, _root_buf->get_driver_pin(), _load_pins);
}
Match BoundSkewTree::getBestMatch(CostFunc cost_func) const
{
  auto min_cost = std::numeric_limits<double>::max();
  Match best_match;
  for (size_t i = 0; i < _unmerged_nodes.size(); ++i) {
    for (size_t j = i + 1; j < _unmerged_nodes.size(); ++j) {
      auto cost = cost_func(_unmerged_nodes[i], _unmerged_nodes[j]);
      if (cost < min_cost) {
        min_cost = cost;
        best_match = {_unmerged_nodes[i], _unmerged_nodes[j], cost};
      }
    }
  }
  return best_match;
}
double BoundSkewTree::distanceCost(Area* left, Area* right) const
{
  auto min_dist = std::numeric_limits<double>::max();
  auto left_mr = left->get_convex_hull();
  auto right_mr = right->get_convex_hull();
  for (auto left_pt : left_mr) {
    for (auto right_pt : right_mr) {
      min_dist = std::min(min_dist, Geom::distance(left_pt, right_pt));
    }
  }
  return min_dist;
}

void BoundSkewTree::bottomUp()
{
  // not input topo
  while (_unmerged_nodes.size() > 1) {
    auto cost_func = [&](Area* left, Area* right) { return distanceCost(left, right); };
    auto best_match = getBestMatch(cost_func);
    auto* left = best_match.left;
    auto* right = best_match.right;
    auto* parent = new Area();
    // random select RCpattern
    auto pattern = static_cast<RCPattern>(1 + std::rand() % 2);
    parent->set_pattern(pattern);
    merge(parent, left, right);
    // erase left and right
    _unmerged_nodes.erase(
        std::remove_if(_unmerged_nodes.begin(), _unmerged_nodes.end(), [&](Area* node) { return node == left || node == right; }),
        _unmerged_nodes.end());
    _unmerged_nodes.push_back(parent);
  }
  _root = _unmerged_nodes.front();
}
void BoundSkewTree::topDown()
{
  // set root location
  Pt root_loc;
  auto mr = _root->get_convex_hull();
  if (_root_guide.has_value()) {
    root_loc = Geom::closestPtOnRegion(_root_guide.value(), mr);
  } else {
    root_loc = Geom::centerPt(mr);
  }
  _root->set_location(root_loc);

  // recursive top down
  embedding(_root);
}
void BoundSkewTree::merge(Area* parent, Area* left, Area* right)
{
  parent->set_left(left);
  parent->set_right(right);
  left->set_parent(parent);
  right->set_parent(parent);
  calcJS(parent, left, right);
  parent->set_edge_len(kLeft, -1);
  parent->set_edge_len(kRight, -1);
  auto dist = parent->get_radius();
  jsProcess(parent);
  auto left_line = parent->get_line(kLeft);
  auto right_line = parent->get_line(kRight);
  constructMr(parent, left, right);
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan) {
    LOG_FATAL_IF(Geom::lineType(getJsLine(kRight)) != LineType::kManhattan) << "right js is not manhattan";
    if (Geom::isSegmentTrr(_ms[kLeft])) {
      Geom::msToLine(_ms[kLeft], left_line);
    }
    if (Geom::isSegmentTrr(_ms[kRight])) {
      Geom::msToLine(_ms[kRight], right_line);
    }
  }
  parent->set_line(kLeft, left_line);
  parent->set_line(kRight, right_line);

  parent->set_radius(dist);
  if (parent->get_edge_len(kLeft) + parent->get_edge_len(kRight) < 0) {
    parent->set_cap_load(left->get_cap_load() + right->get_cap_load() + parent->get_radius() * _unit_h_cap);
  } else {
    parent->set_cap_load(left->get_cap_load() + right->get_cap_load()
                         + (parent->get_edge_len(kLeft) + parent->get_edge_len(kRight)) * _unit_h_cap);
  }
}
void BoundSkewTree::calcJS(Area* parent, Area* left, Area* right)
{
  initSide();
  parent->set_radius(std::numeric_limits<double>::max());
  auto left_lines = left->getConvexHullLines();
  auto right_lines = right->getConvexHullLines();
  std::ranges::for_each(left_lines, [&](Line& left_line) {
    std::ranges::for_each(right_lines, [&](Line& right_line) {
      PtPair closest;
      Geom::lineDist(left_line, right_line, closest);
      calcJS(parent, left_line, right_line);
    });
  });
  calcJsDelay(left, right);
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan) {
    checkJsMs();
  }
}

void BoundSkewTree::jsProcess(Area* cur)
{
  auto swap = [](Pt& p1, Pt& p2) {
    auto temp = p1;
    p1 = p2;
    p2 = temp;
  };
  FOR_EACH_SIDE(side)
  {
    if (Equal(_join_segment[side][kHead].y, _join_segment[side][kTail].y)) {
      if (_join_segment[side][kHead].x < _join_segment[side][kTail].x) {
        swap(_join_segment[side][kHead], _join_segment[side][kTail]);
      }
    } else if (_join_segment[side][kHead].y < _join_segment[side][kTail].y) {
      swap(_join_segment[side][kHead], _join_segment[side][kTail]);
    }
  }
  FOR_EACH_SIDE(side)
  {
    setJrLine(side, getJsLine(side));
    cur->set_line(side, getJsLine(side));
  }
}

void BoundSkewTree::constructMr(Area* parent, Area* left, Area* right)
{
  calcJr(parent, left, right);
  calcJrCorner(parent);
  calcBalancePt(parent);
  calcFmsPt(parent);
  if (existFmsOnJr()) {
    constructFeasibleMr(parent, left, right);
  } else {
    constructInfeasibleMr(parent, left, right);
  }
  if (Geom::lineType(parent->get_line(kLeft)) == LineType::kManhattan && parent->get_edge_len(kLeft) >= 0) {
    LOG_FATAL_IF(parent->get_edge_len(kRight) < 0) << "right edge length is negative";
    constructTrrMr(parent);
  }
  auto mr = parent->get_mr();
  Geom::uniquePtsLoc(mr);
  parent->set_mr(mr);
  calcConvexHull(parent);
}
void BoundSkewTree::embedding(Area* cur) const
{
  auto* left = cur->get_left();
  auto* right = cur->get_right();
  if (!left || !right) {
    return;
  }
  embedding(cur, left, kLeft);
  embedding(left);

  embedding(cur, right, kRight);
  embedding(right);
  // update timing
  auto pt = cur->get_location();
  auto left_pt = left->get_location();
  auto right_pt = right->get_location();
  pt.min = std::numeric_limits<double>::max();
  pt.max = std::numeric_limits<double>::min();
  auto delay_left = ptDelayIncrease(pt, left_pt, cur->get_cap_load(), cur->get_pattern());
  auto delay_right = ptDelayIncrease(pt, right_pt, cur->get_cap_load(), cur->get_pattern());
  pt.min = std::min(left_pt.min + delay_left, right_pt.min + delay_right);
  pt.max = std::max(left_pt.max + delay_left, right_pt.max + delay_right);
  LOG_FATAL_IF(ptSkew(pt) > _skew_bound + kEpsilon) << "skew is larger than skew bound";
  cur->set_location(pt);
}
void BoundSkewTree::initSide()
{
  FOR_EACH_SIDE(side)
  {
    _join_region[side] = {Pt(), Pt()};
    _join_segment[side] = {Pt(), Pt()};
  }
}
void BoundSkewTree::calcJS(Area* cur, Line& left, Line& right)
{
  PtPair closest;
  auto line_dist = Geom::lineDist(left, right, closest);
  auto left_js_bak = getJsLine(kLeft);
  auto right_js_bak = getJsLine(kRight);
  auto left_ms_bak = _ms[kLeft];
  auto right_ms_bak = _ms[kRight];
  if (Equal(line_dist, cur->get_radius())) {
    cur->set_radius(line_dist);
    updateJS(cur, left, right, closest);
    auto origin_area = calcJrArea(left_js_bak, right_js_bak);
    auto new_area = calcJrArea(getJsLine(kLeft), getJsLine(kRight));
    if (origin_area >= new_area) {
      setJsLine(kLeft, left_js_bak);
      setJsLine(kRight, right_js_bak);
      if (Geom::lineType(left_js_bak) == LineType::kManhattan) {
        _ms[kLeft] = left_ms_bak;
        _ms[kRight] = right_ms_bak;
      }
    }
  } else if (line_dist < cur->get_radius()) {
    cur->set_radius(line_dist);
    updateJS(cur, left, right, closest);
  }
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan) {
    checkJsMs();
  }
}
void BoundSkewTree::calcJsDelay(Area* left, Area* right)
{
  FOR_EACH_SIDE(left_side)
  {
    Line line;
    calcBsLocated(left, _join_segment[kLeft][left_side], line);
    calcPtDelays(left, _join_segment[kLeft][left_side], line);
  }
  FOR_EACH_SIDE(right_side)
  {
    Line line;
    calcBsLocated(right, _join_segment[kRight][right_side], line);
    calcPtDelays(right, _join_segment[kRight][right_side], line);
  }
}
void BoundSkewTree::updateJS(Area* cur, Line& left, Line& right, PtPair closest)
{
  auto left_type = Geom::lineType(left);
  auto right_type = Geom::lineType(right);
  auto left_is_manhattan = left_type == LineType::kManhattan;
  auto right_is_manhattan = right_type == LineType::kManhattan;
  Trr left_ms, right_ms;
  if (left_is_manhattan) {
    Geom::lineToMs(left_ms, left);
  }
  if (right_is_manhattan) {
    Geom::lineToMs(right_ms, right);
  }
  if (!left_is_manhattan && right_is_manhattan) {
    left_ms.makeDiamond(closest[kLeft], 0);
  }
  if (left_is_manhattan && !right_is_manhattan) {
    right_ms.makeDiamond(closest[kRight], 0);
  }
  setJsLine(kLeft, {closest[kLeft], closest[kLeft]});
  setJsLine(kRight, {closest[kRight], closest[kRight]});
  if (left_is_manhattan || right_is_manhattan) {
    auto dist = Geom::msDistance(left_ms, right_ms);
    LOG_FATAL_IF(std::abs(dist - cur->get_radius()) > kEpsilon) << "ms distance is not equal to radius";
    cur->set_radius(dist);
    _ms[kLeft] = left_ms;
    _ms[kRight] = right_ms;
    Trr left_bound, right_bound, left_intersect, right_intersect;
    Geom::buildTrr(left_ms, dist, left_bound);
    Geom::buildTrr(right_ms, dist, right_bound);
    Geom::makeIntersect(right_bound, left_ms, left_intersect);
    Geom::makeIntersect(left_bound, right_ms, right_intersect);
    Geom::msToLine(left_intersect, _join_segment[kLeft][kHead], _join_segment[kLeft][kTail]);
    Geom::msToLine(right_intersect, _join_segment[kRight][kHead], _join_segment[kRight][kTail]);
  } else if (Geom::isParallel(left, right)) {
    auto min_x = std::max(std::min(left[kHead].x, left[kTail].x), std::min(right[kHead].x, right[kTail].x));
    auto max_x = std::min(std::max(left[kHead].x, left[kTail].x), std::max(right[kHead].x, right[kTail].x));
    auto min_y = std::max(std::min(left[kHead].y, left[kTail].y), std::min(right[kHead].y, right[kTail].y));
    auto max_y = std::min(std::max(left[kHead].y, left[kTail].y), std::max(right[kHead].y, right[kTail].y));
    if ((left_type == LineType::kVertical || left_type == LineType::kTilt) && max_y >= min_y) {
      Geom::calcCoord(_join_segment[kLeft][kHead], left, min_y);
      Geom::calcCoord(_join_segment[kLeft][kTail], left, max_y);
      Geom::calcCoord(_join_segment[kRight][kHead], right, min_y);
      Geom::calcCoord(_join_segment[kRight][kTail], right, max_y);
    } else if ((left_type == LineType::kHorizontal || left_type == LineType::kFlat) && max_x >= min_x) {
      Geom::calcCoord(_join_segment[kLeft][kHead], left, min_x);
      Geom::calcCoord(_join_segment[kLeft][kTail], left, max_x);
      Geom::calcCoord(_join_segment[kRight][kHead], right, min_x);
      Geom::calcCoord(_join_segment[kRight][kTail], right, max_x);
    }
  } else {
    // single point case
  }
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan && left_type != LineType::kManhattan) {
    _ms[kLeft].makeDiamond(closest[kLeft], 0);
    _ms[kRight].makeDiamond(closest[kRight], 0);
  }
  // checkUpdateJs(cur, left, right);
}

void BoundSkewTree::addJsPts(Area* parent, Area* left, Area* right)
{
  // add points on origin js lines
  FOR_EACH_SIDE(side)
  {
    LOG_FATAL_IF(Geom::isSame(_join_segment[side][kHead], _join_segment[side][kTail])) << "join segment is a point";
    auto mr = side == kLeft ? left->get_mr() : right->get_mr();
    for (auto pt : mr) {
      if (Geom::onLine(pt, getJsLine(side)) && !Geom::isSame(pt, _join_segment[side][kHead])
          && !Geom::isSame(pt, _join_segment[side][kTail])) {
        _join_segment[side].push_back(pt);
      }
    }
    Geom::sortPtsByFront(_join_segment[side]);
  }
  // add points on other side
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    auto other_mr = other_side == kLeft ? left->get_mr() : right->get_mr();
    auto relative_type = Geom::lineRelative(getJsLine(kLeft), getJsLine(kRight), other_side);
    for (auto pt : other_mr) {
      Geom::calcRelativeCoord(pt, relative_type, parent->get_radius());
      for (auto it = _join_segment[side].begin(); it != _join_segment[side].end() - 1; ++it) {
        Line line = {*it, *(it + 1)};
        // unique and sort
        if (Geom::onLine(pt, line) && !Geom::isSame(pt, *it) && !Geom::isSame(pt, *(it + 1))) {
          calcPtDelays(nullptr, pt, line);
          _join_segment[side].insert(it + 1, pt);
          break;
        }
      }
    }
  }
}

double BoundSkewTree::delayFromJs(const size_t& js_side, const size_t& side, const size_t& idx, const size_t& timing_type,
                                  const Side<double>& delay_from) const
{
  double delay = timing_type == kMin ? _join_segment[side][idx].min : _join_segment[side][idx].max;
  delay += js_side == side ? 0 : delay_from[side];
  return delay;
}
void BoundSkewTree::calcJr(Area* parent, Area* left, Area* right)
{
  if (calcAreaLineType(parent) == LineType::kManhattan) {
    calcJrEndpoints(parent);
  } else {
    calcNotManhattanJrEndpoints(parent, left, right);
  }
  addFmsToJr();
}

void BoundSkewTree::calcJrEndpoints(Area* cur)
{
  auto left_line = cur->get_line(kLeft);
  auto right_line = cur->get_line(kRight);
  LOG_FATAL_IF(!Geom::isSame(_join_segment[kLeft][kHead], left_line[kHead]) || !Geom::isSame(_join_segment[kLeft][kHead], left_line[kHead]))
      << "left join segment is not same as left line at head";
  LOG_FATAL_IF(!Geom::isSame(_join_segment[kLeft][kTail], left_line[kTail]) || !Geom::isSame(_join_segment[kLeft][kTail], left_line[kTail]))
      << "left join segment is not same as left line at tail";
  LOG_FATAL_IF(!Geom::isSame(_join_segment[kRight][kHead], right_line[kHead])
               || !Geom::isSame(_join_segment[kRight][kHead], right_line[kHead]))
      << "right join segment is not same as right line at head";
  LOG_FATAL_IF(!Geom::isSame(_join_segment[kRight][kTail], right_line[kTail])
               || !Geom::isSame(_join_segment[kRight][kTail], right_line[kTail]))
      << "right join segment is not same as right line at tail";
  _join_region[kLeft][kHead] = _join_segment[kLeft][kHead];
  _join_region[kLeft][kTail] = _join_segment[kLeft][kTail];
  _join_region[kRight][kHead] = _join_segment[kRight][kHead];
  _join_region[kRight][kTail] = _join_segment[kRight][kTail];
  updatePtDelaysByEndSide(cur, kHead, _join_region[kLeft][kHead]);
  updatePtDelaysByEndSide(cur, kHead, _join_region[kRight][kHead]);
  updatePtDelaysByEndSide(cur, kTail, _join_region[kLeft][kTail]);
  updatePtDelaysByEndSide(cur, kTail, _join_region[kRight][kTail]);
}
void BoundSkewTree::calcNotManhattanJrEndpoints(Area* parent, Area* left, Area* right)
{
  addJsPts(parent, left, right);
  Side<double> delay_from = {ptDelayIncrease(_join_segment[kLeft][kHead], _join_segment[kRight][kHead], left->get_cap_load()),
                             ptDelayIncrease(_join_segment[kLeft][kHead], _join_segment[kRight][kHead], right->get_cap_load())};
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    _join_region[side] = _join_segment[side];
    for (size_t i = 0; i < _join_segment[side].size(); ++i) {
      auto pt = _join_segment[side][i];
      pt.min = std::min(pt.min, _join_segment[other_side][i].min + delay_from[other_side]);
      pt.max = std::max(pt.max, _join_segment[other_side][i].max + delay_from[other_side]);
      _join_region[side][i] = pt;
    }
    // add JR turn points which delay slope is changed
    for (size_t i = 0; i < _join_region[side].size() - 1; ++i) {
      auto delta = (_join_segment[side][i].min - _join_segment[other_side][i].min - delay_from[other_side])
                   * (_join_segment[side][i + 1].min - _join_segment[other_side][i + 1].min - delay_from[other_side]);
      if (delta < -kEpsilon) {
        addTurnPt(side, i, kMin, delay_from);
      }
      delta = (_join_segment[side][i].max - _join_segment[other_side][i].max - delay_from[other_side])
              * (_join_segment[side][i + 1].max - _join_segment[other_side][i + 1].max - delay_from[other_side]);
      if (delta < -kEpsilon) {
        addTurnPt(side, i, kMax, delay_from);
      }
    }
    Geom::sortPtsByFront(_join_region[side]);
    // remove redundant turn points which have same slope
    for (size_t i = 0; i < _join_region[side].size() - 1; ++i) {
      auto pt1 = _join_region[side][i];
      auto pt2 = _join_region[side][i + 1];
      auto dist = Geom::distance(pt1, pt2);
      LOG_FATAL_IF(Equal(dist, 0)) << "distance is zero";
      _join_region[side][i].val = (ptSkew(pt2) - ptSkew(pt1)) / dist;
    }
    // remove redundant turn points which skew slope is not strictly monotone increasing
    Pts incr_pts = {_join_segment[side].front()};
    for (size_t j = 1; j < _join_segment[side].size() - 1; ++j) {
      auto cur_val = incr_pts.back().val;
      auto next_val = _join_region[side][j].val;
      LOG_FATAL_IF(cur_val > next_val) << "skew slope is not strictly monotone increasing";
      if (next_val > cur_val) {
        incr_pts.push_back(_join_segment[side][j]);
      }
    }
    incr_pts.push_back(_join_segment[side].back());
    _join_region[side] = incr_pts;
  }
}
void BoundSkewTree::addTurnPt(const size_t& side, const size_t& idx, const size_t& timing_type, const Side<double>& delay_from)
{
  auto p1 = _join_region[side][idx];
  auto p2 = _join_region[side][idx + 1];
  double alpha = 0;
  if (Equal(p1.x, p2.x)) {
    alpha = _K[kV];
  } else {
    alpha = _K[kH];
  }
  auto dist = Geom::distance(p1, p2);
  LOG_FATAL_IF(Equal(dist, 0)) << "distance is zero";
  Side<Side<double>> beta = Side<Side<double>>({Side<double>({0, 0}), Side<double>({0, 0})});
  FOR_EACH_SIDE(sub_side)  // left and right
  {
    FOR_EACH_SIDE(timing_side)  // min and max
    {
      auto t1 = delayFromJs(side, sub_side, idx, timing_side, delay_from);
      auto t2 = delayFromJs(side, sub_side, idx + 1, timing_side, delay_from);
      beta[sub_side][timing_side] = (t2 - t1) / dist - alpha * dist;
    }
  }
  auto turn_dist = (delayFromJs(side, kLeft, idx, timing_type, delay_from) - delayFromJs(side, kRight, idx, timing_type, delay_from))
                   / (beta[kRight][timing_type] - beta[kLeft][timing_type]);
  LOG_FATAL_IF(turn_dist <= 0 || turn_dist >= dist) << "turn dist is not in range";
  auto ref_dist = dist - turn_dist;
  Pt turn_pt((p1.x * ref_dist + p2.x * turn_dist) / dist, (p1.y * ref_dist + p2.y * turn_dist) / dist);
  Side<Side<double>> delay_bound = Side<Side<double>>({Side<double>({0, 0}), Side<double>({0, 0})});
  FOR_EACH_SIDE(sub_side)  // left and right
  {
    FOR_EACH_SIDE(timing_side)  // min and max
    {
      delay_bound[sub_side][timing_side] = delayFromJs(side, sub_side, idx, timing_side, delay_from) + alpha * turn_dist * turn_dist
                                           + beta[sub_side][timing_side] * turn_dist;
    }
  }
  turn_pt.min = std::min(delay_bound[kLeft][kMin], delay_bound[kRight][kMin]);
  turn_pt.max = std::max(delay_bound[kLeft][kMax], delay_bound[kRight][kMax]);
  _join_region[side].push_back(turn_pt);
}
void BoundSkewTree::addFmsToJr()
{
  FOR_EACH_SIDE(side)
  {
    for (size_t i = 0; i < _join_region[side].size() - 1; ++i) {
      auto pt_cur = _join_region[side][i];
      auto pt_next = _join_region[side][i + 1];
      auto delta_cur = ptSkew(pt_cur) - _skew_bound;
      auto delta_next = ptSkew(pt_next) - _skew_bound;
      if (delta_cur * delta_next < 0 && !Equal(delta_cur, 0) && !Equal(delta_next, 0)) {
        auto dist = Geom::distance(pt_cur, pt_next);
        auto turn_dist = (_skew_bound - ptSkew(pt_cur)) * dist / (ptSkew(pt_next) - ptSkew(pt_cur));
        auto ref_dist = dist - turn_dist;
        Pt turn_pt{(pt_cur.x * ref_dist + pt_next.x * turn_dist) / dist, (pt_cur.y * ref_dist + pt_next.y * turn_dist) / dist};
        Line line = {pt_cur, pt_next};
        calcPtDelays(nullptr, turn_pt, line);
        _join_region[side].insert(_join_region[side].begin() + i + 1, turn_pt);
      }
    }
  }
}
void BoundSkewTree::calcJrCorner(Area* cur)
{
  FOR_EACH_SIDE(side)
  {
    LOG_FATAL_IF(_join_segment[side].front().y < _join_segment[side].back().y) << "join segment direction is not correct";
  }
  if (calcAreaLineType(cur) == LineType::kManhattan && !Equal(cur->get_radius(), 0)) {
    FOR_EACH_SIDE(end_side)
    {
      if (jrCornerExist(end_side)) {
        auto p_left = _join_segment[kLeft][end_side];
        auto p_right = _join_segment[kRight][end_side];
        if ((p_left.x - p_right.x) * (p_left.y - p_right.y) < 0) {
          if (end_side == kHead) {
            _join_corner[end_side] = {std::max(p_left.x, p_right.x), std::max(p_left.y, p_right.y)};
          } else {
            _join_corner[end_side] = {std::min(p_left.x, p_right.x), std::min(p_left.y, p_right.y)};
          }
        } else {
          if (end_side == kHead) {
            _join_corner[end_side] = {std::min(p_left.x, p_right.x), std::max(p_left.y, p_right.y)};
          } else {
            _join_corner[end_side] = {std::max(p_left.x, p_right.x), std::min(p_left.y, p_right.y)};
          }
        }
        updatePtDelaysByEndSide(cur, end_side, _join_corner[end_side]);
      }
    }
  }
}
bool BoundSkewTree::jrCornerExist(const size_t& end_side) const
{
  auto pt1 = _join_segment[kLeft][end_side];
  auto pt2 = _join_segment[kRight][end_side];
  return !Equal(pt1.x, pt2.x) && !Equal(pt1.y, pt2.y);
}
void BoundSkewTree::calcBalancePt(Area* cur)
{
  if (Equal(cur->get_radius(), 0)) {
    return;
  }
  FOR_EACH_SIDE(end_side)
  {
    _bal_points[end_side].clear();
    auto left_line = cur->get_line(kLeft);
    auto right_line = cur->get_line(kRight);
    auto left_pt = left_line[end_side];
    auto right_pt = right_line[end_side];
    left_pt.val = cur->get_left()->get_cap_load();
    right_pt.val = cur->get_right()->get_cap_load();
    auto bal_ref_side = (left_pt.x - right_pt.x) * (left_pt.y - right_pt.y) < 0 ? 1 - end_side : end_side;
    FOR_EACH_SIDE(timing_type)
    {
      double dist_to_left = 0, dist_to_right = 0;
      Pt bal_pt;
      calcBalBetweenPts(left_pt, right_pt, timing_type, bal_ref_side, dist_to_left, dist_to_right, bal_pt);
      if (!Equal(dist_to_left, 0) && !Equal(dist_to_right, 0)) {
        updatePtDelaysByEndSide(cur, end_side, bal_pt);
        _bal_points[end_side].push_back(bal_pt);
      }
    }
  }
}
void BoundSkewTree::calcBalBetweenPts(Pt& p1, Pt& p2, const size_t& timing_type, const size_t& bal_ref_side, double& d1, double& d2,
                                      Pt& bal_pt) const
{
  auto h = std::abs(p1.x - p2.x);
  auto v = std::abs(p1.y - p2.y);
  if (Equal(h, 0) || Equal(v, 0)) {
    calcBalPtOnLine(p1, p2, timing_type, d1, d2, bal_pt);
  } else if (p1.x <= p2.x) {
    calcBalPtNotOnLine(p1, p2, timing_type, bal_ref_side, d1, d2, bal_pt);
  } else {
    calcBalPtNotOnLine(p2, p1, timing_type, bal_ref_side, d2, d1, bal_pt);
  }
}
void BoundSkewTree::calcBalPtOnLine(Pt& p1, Pt& p2, const size_t& timing_type, double& d1, double& d2, Pt& bal_pt) const
{
  auto h = std::abs(p1.x - p2.x);
  auto v = std::abs(p1.y - p2.y);
  LOG_FATAL_IF(!Equal(h, 0) && !Equal(v, 0)) << "h and v are not zero, which balance point is not on line";

  auto delay1 = timing_type == kMin ? p1.min : p1.max;
  auto delay2 = timing_type == kMin ? p2.min : p2.max;
  auto pattern = Equal(h, 0) ? LayerPattern::kV : LayerPattern::kH;
  auto r = pattern == LayerPattern::kH ? _unit_h_res : _unit_v_res;
  auto c = pattern == LayerPattern::kH ? _unit_h_cap : _unit_v_cap;
  calcMergeDist(r, c, p1.val, delay1, p2.val, delay2, h + v, d1, d2);
  calcPtCoordOnLine(p1, p2, d1, d2, bal_pt);
  double incr_delay1 = 0;
  double incr_delay2 = 0;
  if (Equal(h, 0)) {
    incr_delay1 = calcDelayIncrease(0, d1, p1.val);
    incr_delay2 = calcDelayIncrease(0, d2, p2.val);
  } else {
    incr_delay1 = calcDelayIncrease(d1, 0, p1.val);
    incr_delay2 = calcDelayIncrease(d2, 0, p2.val);
  }
  bal_pt.min = std::min(p1.min + incr_delay1, p2.min + incr_delay2);
  bal_pt.max = std::max(p1.max + incr_delay1, p2.max + incr_delay2);
}
void BoundSkewTree::calcBalPtNotOnLine(Pt& p1, Pt& p2, const size_t& timing_type, const size_t& bal_ref_side, double& d1, double& d2,
                                       Pt& bal_pt) const
{
  auto h = std::abs(p1.x - p2.x);
  auto v = std::abs(p1.y - p2.y);
  LOG_FATAL_IF(Equal(h, 0) || Equal(v, 0)) << "h or v is zero, which balance point is on line";
  LOG_FATAL_IF(p1.x > p2.x) << "p1 is not left of p2";

  auto delay1 = timing_type == kMin ? p1.min : p1.max;
  auto delay2 = timing_type == kMin ? p2.min : p2.max;
  auto x = calcXBalPosition(delay1, delay2, p1.val, p2.val, h, v, bal_ref_side);
  double y = 0;
  if (x < 0) {
    y = bal_ref_side == kX ? calcYBalPosition(delay1, delay2, p1.val, p2.val, h, v, bal_ref_side) : -1;
    x = y >= 0 ? 0 : x;
  } else if (x > h) {
    y = bal_ref_side == kX ? v + 1 : calcYBalPosition(delay1, delay2, p1.val, p2.val, h, v, bal_ref_side);
    x = y <= v ? h : x;
  } else {
    y = bal_ref_side == kX ? v : 0;
  }

  if (x < 0) {
    LOG_FATAL_IF(y >= 0) << "y is illegal";
    auto temp_pt = p1;
    auto incr_delay = calcDelayIncrease(h, v, p2.val);
    temp_pt.min = p2.min + incr_delay;
    temp_pt.max = p2.max + incr_delay;
    temp_pt.val = p2.val + _unit_h_cap * h + _unit_v_cap * v;
    calcBalPtOnLine(p1, temp_pt, timing_type, d1, d2, bal_pt);
    LOG_FATAL_IF(d1 != 0) << "dist to p1 should be zero";
    auto new_incr_delay = calcDelayIncrease(0, d2, temp_pt.val);
    LOG_FATAL_IF(!Equal(delay1, incr_delay + new_incr_delay + delay2)) << "delay is not equal";
    d2 += h + v;
  } else if (x > h) {
    LOG_FATAL_IF(y <= v) << "y is illegal";
    auto temp_pt = p2;
    auto incr_delay = calcDelayIncrease(h, v, p1.val);
    temp_pt.min = p1.min + incr_delay;
    temp_pt.max = p1.max + incr_delay;
    temp_pt.val = p1.val + _unit_h_cap * h + _unit_v_cap * v;
    calcBalPtOnLine(temp_pt, p2, timing_type, d1, d2, bal_pt);
    LOG_FATAL_IF(d2 != 0) << "dist to p2 should be zero";
    auto new_incr_delay = calcDelayIncrease(0, d1, temp_pt.val);
    LOG_FATAL_IF(!Equal(delay2, incr_delay + new_incr_delay + delay1)) << "delay is not equal";
    d1 += h + v;
  } else {
    LOG_FATAL_IF(y < 0 || y > v) << "y is not in range";
    bal_pt.x = p1.x + x;
    bal_pt.y = p1.y < p2.y ? p1.y + y : p1.y - y;
    auto incr_delay1 = calcDelayIncrease(x, y, p1.val);
    auto incr_delay2 = calcDelayIncrease(h - x, v - y, p2.val);
    bal_pt.min = std::min(p1.min + incr_delay1, p2.min + incr_delay2);
    bal_pt.max = std::max(p1.max + incr_delay1, p2.max + incr_delay2);
    d1 = x + y;
    d2 = h + v - d1;
    LOG_FATAL_IF(!Equal(incr_delay1 + delay1, incr_delay2 + delay2)) << "delay is not equal";
  }
  LOG_FATAL_IF(d1 + d2 < h + v - kEpsilon) << "dist out of range";
}
void BoundSkewTree::calcMergeDist(const double& r, const double& c, const double& cap1, const double& delay1, const double& cap2,
                                  const double& delay2, const double& dist, double& d1, double& d2) const
{
  auto target = (delay2 - delay1 + r * dist * (cap2 + c * dist / 2)) / (r * (cap1 + cap2 + c * dist));
  if (target < 0) {
    auto fact = cap2 / c;
    target = std::sqrt(fact * fact + 2 * (delay1 - delay2) / (r * c)) - fact;
    d1 = 0;
    d2 = target;
  } else if (target > dist) {
    auto fact = cap1 / c;
    target = std::sqrt(fact * fact + 2 * (delay2 - delay1) / (r * c)) - fact;
    d1 = target;
    d2 = 0;
  } else {
    d1 = target;
    d2 = dist - target;
  }
}
void BoundSkewTree::calcPtCoordOnLine(const Pt& p1, const Pt& p2, const double& d1, const double& d2, Pt& pt) const
{
  auto dist = d1 + d2;
  auto pt_dist = Geom::distance(p1, p2);
  LOG_FATAL_IF(!Equal(dist, pt_dist) && dist < pt_dist) << "dist is less than points dist";
  if (Equal(d1, 0)) {
    pt = p1;
  } else if (Equal(d2, 0)) {
    pt = p2;
  } else {
    pt = {(p1.x * d2 + p2.x * d1) / dist, (p1.y * d2 + p2.y * d1) / dist};
  }
}
double BoundSkewTree::calcXBalPosition(const double& delay1, const double& delay2, const double& cap1, const double& cap2, const double& h,
                                       const double& v, const size_t& bal_ref_side) const
{
  auto rc = _pattern == RCPattern::kHV ? _unit_v_res * _unit_h_cap : _unit_h_res * _unit_v_cap;
  double t = 0;
  if (bal_ref_side == kX) {
    // assume (x, v-y) and (h-x, y), then set y = 0
    t = delay2 - delay1 + _K[kH] * h * h - _K[kV] * v * v + _unit_h_res * h * cap2 - _unit_v_res * v * cap1;
  } else {
    // assume (x, y) and (h-x, v-y), then set y = 0
    t = delay2 - delay1 + _K[kH] * h * h + _K[kV] * v * v + cap2 * (_unit_h_res * h + _unit_v_res * v) + rc * h * v;
  }
  auto x = t / (_unit_h_res * (cap1 + cap2) + rc * v + 2 * h * _K[kH]);
  return x;
}
double BoundSkewTree::calcYBalPosition(const double& delay1, const double& delay2, const double& cap1, const double& cap2, const double& h,
                                       const double& v, const size_t& bal_ref_side) const
{
  auto rc = _pattern == RCPattern::kHV ? _unit_v_res * _unit_h_cap : _unit_h_res * _unit_v_cap;
  double t = 0;
  auto r = _unit_v_res * (cap1 + cap2) + 2 * v * _K[kV] + rc * h;
  double y = 0;
  if (bal_ref_side == kX) {
    // assume (x, y) and (h-x, v-y), then set x = 0
    t = delay2 - delay1 + _K[kH] * h * h + _K[kV] * v * v + cap2 * (_unit_h_res * h + _unit_v_res * v) + rc * h * v;
    y = t / r;
    LOG_FATAL_IF(y > v) << "y is larger than v";
  } else {
    // assume (h-x, y) and (x, v-y), then set x = 0
    t = delay2 - delay1 + _K[kV] * v * v - _K[kH] * h * h + _unit_v_res * v * cap2 - _unit_h_res * h * cap1;
    y = t / r;
    LOG_FATAL_IF(y < 0) << "y is less than zero";
  }
  return y;
}
void BoundSkewTree::calcFmsPt(Area* cur)
{
  FOR_EACH_SIDE(end_side)
  {
    _fms_points[end_side].clear();
    Side<Pt> candidate;
    if (end_side == kHead) {
      candidate[kLeft] = _join_region[kLeft].front();
      candidate[kRight] = _join_region[kRight].front();
    } else {
      candidate[kLeft] = _join_region[kLeft].back();
      candidate[kRight] = _join_region[kRight].back();
    }
    bool exist = false;
    if (jrCornerExist(end_side)) {
      exist = calcFmsOnLine(cur, candidate[kLeft], _join_corner[end_side], end_side);
      if (exist) {
        exist = calcFmsOnLine(cur, candidate[kRight], _join_corner[end_side], end_side);
        if (!exist) {
          exist = calcFmsOnLine(cur, _join_corner[end_side], candidate[kLeft], end_side);
          LOG_FATAL_IF(!exist) << "can't find feasible merge section on line";
        }
      } else {
        exist = calcFmsOnLine(cur, _join_corner[end_side], candidate[kRight], end_side);
        if (exist) {
          exist = calcFmsOnLine(cur, candidate[kRight], _join_corner[end_side], end_side);
          LOG_FATAL_IF(!exist) << "can't find feasible merge section on line";
        }
      }
    } else {
      exist = calcFmsOnLine(cur, candidate[kLeft], candidate[kRight], end_side);
      if (exist) {
        calcFmsOnLine(cur, candidate[kRight], candidate[kLeft], end_side);
      }
    }
    Geom::uniquePtsLoc(_fms_points[end_side]);
  }
}
bool BoundSkewTree::calcFmsOnLine(Area* cur, Pt& pt, const Pt& q, const size_t& end_side)
{
  auto skew = ptSkew(pt);
  if (Equal(skew, _skew_bound) || skew < _skew_bound) {
    _fms_points[end_side].push_back(pt);
    return true;
  }
  auto min_dist_pt = q;
  std::ranges::for_each(_bal_points[end_side], [&min_dist_pt, &pt](const Pt& bal_pt) {
    if (Geom::distance(pt, bal_pt) < Geom::distance(pt, min_dist_pt)) {
      min_dist_pt = bal_pt;
    }
  });
  skew = ptSkew(min_dist_pt);
  if (Equal(skew, _skew_bound)) {
    _fms_points[end_side].push_back(min_dist_pt);
    return true;
  }
  if (skew < _skew_bound) {
    Pt fms_pt;
    calcFmsBetweenPts(pt, min_dist_pt, fms_pt);
    updatePtDelaysByEndSide(cur, end_side, fms_pt);
    if (!Equal(ptSkew(fms_pt), _skew_bound)) {
      auto p1 = pt;
      auto p2 = min_dist_pt;
      updatePtDelaysByEndSide(cur, end_side, p1);
      updatePtDelaysByEndSide(cur, end_side, p2);
      LOG_FATAL << "feasible merge section point should in skew bound";
    }
    _fms_points[end_side].push_back(fms_pt);
    return true;
  }
  return false;
}
void BoundSkewTree::calcFmsBetweenPts(const Pt& high_skew_pt, const Pt& low_skew_pt, Pt& fms_pt) const
{
  auto high_skew = ptSkew(high_skew_pt);
  auto low_skew = ptSkew(low_skew_pt);
  LOG_FATAL_IF(low_skew > _skew_bound) << "low skew is larger than skew bound";
  LOG_FATAL_IF(high_skew < low_skew + kEpsilon) << "high skew is less than low skew";
  auto dist = Geom::distance(high_skew_pt, low_skew_pt);
  LOG_FATAL_IF(dist <= kEpsilon) << "distance is less than epsilon";
  auto dist_to_low = dist * (_skew_bound - low_skew) / (high_skew - low_skew);
  calcPtCoordOnLine(high_skew_pt, low_skew_pt, dist - dist_to_low, dist_to_low, fms_pt);
}
bool BoundSkewTree::existFmsOnJr() const
{
  if (!_fms_points[kHead].empty() || !_fms_points[kTail].empty()) {
    return true;
  }
  FOR_EACH_SIDE(side)
  {
    for (auto pt : _join_region[side]) {
      if (ptSkew(pt) <= _skew_bound) {
        return true;
      }
    }
  }
  return false;
}
void BoundSkewTree::constructFeasibleMr(Area* parent, Area* left, Area* right) const
{
  if (calcAreaLineType(parent) == LineType::kManhattan) {
    // parallel manhattan arc
    mrBetweenJs(parent, kHead);
    if (!parent->get_mr().empty() && !jRisLine()) {
      mrBetweenJs(parent, kTail);
    }
  } else {
    // parallel horizontal or vertical arc
    mrBetweenJs(parent, kHead);
    mrOnJs(parent, kLeft);
    mrBetweenJs(parent, kTail);
    if (parent->get_radius() > kEpsilon) {
      mrOnJs(parent, kRight);
    }
  }
}
bool BoundSkewTree::jRisLine() const
{
  auto left_head = _join_segment[kLeft][kHead];
  auto left_tail = _join_segment[kLeft][kTail];
  auto right_head = _join_segment[kRight][kHead];
  auto right_tail = _join_segment[kRight][kTail];
  auto min_x = std::min({left_head.x, left_tail.x, right_head.x, right_tail.x});
  auto min_y = std::min({left_head.y, left_tail.y, right_head.y, right_tail.y});
  auto max_x = std::max({left_head.x, left_tail.x, right_head.x, right_tail.x});
  auto max_y = std::max({left_head.y, left_tail.y, right_head.y, right_tail.y});
  if (Equal(min_x, max_x) || Equal(min_y, max_y)) {
    return true;
  }
  return false;
}
void BoundSkewTree::mrBetweenJs(Area* cur, const size_t& end_side) const
{
  Pts mr_pts;
  std::ranges::for_each(_bal_points[end_side], [&mr_pts](const Pt& p) { mr_pts.push_back(p); });
  std::ranges::for_each(_fms_points[end_side], [&mr_pts](const Pt& p) { mr_pts.push_back(p); });
  if (jrCornerExist(end_side) && ptSkew(_join_corner[end_side]) < _skew_bound + kEpsilon) {
    mr_pts.push_back(_join_corner[end_side]);
  }
  if (mr_pts.empty()) {
    return;
  }
  auto left_line = cur->get_line(kLeft);
  auto right_line = cur->get_line(kRight);
  Pt ref_js_pt = end_side == kHead ? left_line[end_side] : right_line[end_side];
  std::ranges::for_each(mr_pts, [&](Pt& pt) { pt.val = Geom::distance(pt, ref_js_pt); });
  Geom::uniquePtsVal(mr_pts);
  Geom::sortPtsByVal(mr_pts);
  std::ranges::for_each(mr_pts, [&cur](const Pt& p) { cur->add_mr_point(p); });
}
void BoundSkewTree::mrOnJs(Area* cur, const size_t& side) const
{
  LOG_FATAL_IF(_join_region[side].size() < 2) << "join region size is less than 2";
  auto other_side = side == kLeft ? kRight : kLeft;
  auto p = _join_region[side].front();
  auto q = _join_region[other_side].front();
  size_t jr_left_id = 1;
  if (_fms_points[kHead].empty() && ptSkew(p) < ptSkew(q)) {
    for (; jr_left_id < _join_region[side].size() - 1; ++jr_left_id) {
      if (Equal(ptSkew(_join_region[side][jr_left_id]), _skew_bound)) {
        break;
      }
    }
  }
  p = _join_region[side].back();
  q = _join_region[other_side].back();
  size_t jr_right_id = _join_region[other_side].size() - 2;
  if (_fms_points[kTail].empty() && ptSkew(p) < ptSkew(q)) {
    for (; jr_right_id > 0; --jr_right_id) {
      if (Equal(ptSkew(_join_region[side][jr_right_id]), _skew_bound)) {
        break;
      }
    }
  }
  if (side == kLeft) {
    for (size_t i = jr_left_id; i <= jr_right_id; ++i) {
      fmsOfLineExist(cur, side, i);
    }
  } else {
    for (size_t i = jr_right_id; i >= jr_left_id; --i) {
      fmsOfLineExist(cur, side, i);
    }
  }
}
void BoundSkewTree::fmsOfLineExist(Area* cur, const size_t& side, const size_t& idx) const
{
  auto pt = _join_region[side][idx];
  auto slope = calcSkewSlope(cur);
  auto dist = (ptSkew(pt) - _skew_bound) / slope;
  if (dist <= 0) {
    cur->add_mr_point(pt);
  } else if (dist <= cur->get_radius()) {
    auto other_side = side == kLeft ? kRight : kLeft;
    auto relative_type = Geom::lineRelative(getJsLine(side), getJsLine(other_side), other_side);
    Geom::calcRelativeCoord(pt, relative_type, dist);
    auto x = std::abs(pt.x - _join_segment[side][idx].x);
    auto y = std::abs(pt.y - _join_segment[side][idx].y);
    LOG_FATAL_IF(!Equal(x, 0) && !Equal(y, 0)) << "not horizontal or vertical";
    auto incr_delay = side == kLeft ? calcDelayIncrease(x, y, cur->get_left()->get_cap_load())
                                    : calcDelayIncrease(x, y, cur->get_right()->get_cap_load());
    pt.min += incr_delay;
    pt.max = _skew_bound + pt.min;
    cur->add_mr_point(pt);
  }
}
double BoundSkewTree::calcSkewSlope(Area* cur) const
{
  auto left_x = cur->get_line(kLeft)[kHead].x;
  auto left_y = cur->get_line(kLeft)[kHead].y;
  auto right_x = cur->get_line(kRight)[kHead].x;
  auto right_y = cur->get_line(kRight)[kHead].y;
  auto left_cap = cur->get_left()->get_cap_load();
  auto right_cap = cur->get_right()->get_cap_load();
  if (Equal(left_x, right_x)) {
    return _unit_v_res * (left_cap + right_cap + cur->get_radius() * _unit_v_cap);
  } else if (Equal(left_y, right_y)) {
    return _unit_h_res * (left_cap + right_cap + cur->get_radius() * _unit_h_cap);
  }
  LOG_FATAL << "line is not horizontal or vertical";
}
void BoundSkewTree::constructInfeasibleMr(Area* parent, Area* left, Area* right) const
{
  calcMinSkewSection(parent);
  calcDetourEdgeLen(parent);
  refineMrDelay(parent);
}
void BoundSkewTree::calcMinSkewSection(Area* cur) const
{
  auto min_skew = std::numeric_limits<double>::max();
  auto min_skew_side = kLeft;
  FOR_EACH_SIDE(side)
  {
    auto min_side_skew = std::numeric_limits<double>::max();
    std::ranges::for_each(_join_region[side], [&](const Pt& pt) { min_side_skew = std::min(min_side_skew, ptSkew(pt)); });
    if (min_side_skew < min_skew) {
      min_skew = min_side_skew;
      min_skew_side = side;
    }
  }
  std::ranges::for_each(_join_region[min_skew_side], [&](const Pt& pt) {
    if (Equal(ptSkew(pt), min_skew)) {
      cur->add_mr_point(pt);
    }
  });
}
void BoundSkewTree::calcDetourEdgeLen(Area* cur) const
{
  auto left_pt = cur->get_line(kLeft)[kHead];
  auto right_pt = cur->get_line(kRight)[kHead];
  left_pt.val = cur->get_left()->get_cap_load();
  right_pt.val = cur->get_right()->get_cap_load();
  auto delta = ptSkew(cur->get_mr().front()) - _skew_bound;
  LOG_FATAL_IF(delta <= 0) << "remain skew less than 0";
  auto h = std::abs(left_pt.x - right_pt.x);
  auto v = std::abs(left_pt.y - right_pt.y);
  if (left_pt.max > right_pt.max) {
    right_pt.max = left_pt.max - delta - calcDelayIncrease(h, v, right_pt.val);
    double d1 = 0, d2 = 0;
    Pt bal_pt;
    calcBalBetweenPts(left_pt, right_pt, kMin, kX, d1, d2, bal_pt);
    LOG_FATAL_IF(d1 != 0) << "dist to left_pt should be zero";
    cur->set_edge_len(kLeft, 0);
    cur->set_edge_len(kRight, d2);
  } else {
    left_pt.max = right_pt.max - delta - calcDelayIncrease(h, v, left_pt.val);
    double d1 = 0, d2 = 0;
    Pt bal_pt;
    calcBalBetweenPts(right_pt, left_pt, kMin, kX, d1, d2, bal_pt);
    LOG_FATAL_IF(d2 != 0) << "dist to right_pt should be zero";
    cur->set_edge_len(kLeft, d1);
    cur->set_edge_len(kRight, 0);
  }
}
void BoundSkewTree::refineMrDelay(Area* cur) const
{
  auto mr = cur->get_mr();
  std::ranges::for_each(mr, [&](Pt& pt) { pt.min = pt.max - _skew_bound; });
  cur->set_mr(mr);
}
void BoundSkewTree::constructTrrMr(Area* cur) const
{
  Trr trr_left;
  Geom::buildTrr(_ms[kLeft], cur->get_edge_len(kLeft), trr_left);
  Trr trr_right;
  Geom::buildTrr(_ms[kRight], cur->get_edge_len(kRight), trr_right);

  Trr intersect;
  Geom::makeIntersect(trr_left, trr_right, intersect);
  Geom::trrCore(intersect, intersect);
  Region mr;
  Geom::trrToRegion(intersect, mr);
  auto old_pt = cur->get_mr().front();
  std::ranges::for_each(mr, [&](Pt& p) {
    p.min = old_pt.min;
    p.max = old_pt.max;
  });
  cur->set_mr(mr);
}
void BoundSkewTree::embedding(Area* parent, Area* child, const size_t& side) const
{
  Pt child_loc;
  auto parent_loc = parent->get_location();
  auto mr = child->get_mr();
  if (mr.size() == 4 && isTrrArea(parent)) {
    Trr trr;
    mrToTrr(mr, trr);
    auto dist = Geom::ptToTrrDist(parent_loc, trr);
    Trr parent_trr(parent_loc, dist);
    Trr ms;
    Geom::makeIntersect(parent_trr, trr, ms);
    Geom::coreMidPoint(ms, child_loc);
  } else {
    auto js_line = parent->get_line(side);
    auto head = js_line[kHead];
    auto tail = js_line[kTail];
    Line temp;
    calcBsLocated(child, head, temp);
    calcBsLocated(child, tail, temp);
    auto x = std::abs(head.x - tail.x);
    auto y = std::abs(head.y - tail.y);
    if (Equal(x, 0) && Equal(y, 0)) {
      // parent loc is same as child loc
      child_loc = head;
    } else if (Equal(x, 0)) {
      // vertical
      child_loc.x = head.x;
      child_loc.y = parent_loc.y;
    } else if (Equal(y, 0)) {
      // horizontal
      child_loc.x = parent_loc.x;
      child_loc.y = head.y;
    } else {
      // others
      Geom::ptToLineDist(parent_loc, js_line, child_loc);
    }
    LOG_FATAL_IF(!Geom::onLine(child_loc, js_line)) << "child loc is not on js line";
  }
  child->set_location(child_loc);
  if (parent->get_edge_len(side) >= 0) {
    LOG_FATAL_IF(parent->get_edge_len(side) < Geom::distance(parent_loc, child_loc) - kEpsilon) << "edge len is less than distance";
  }
}
bool BoundSkewTree::isTrrArea(Area* cur) const
{
  if (isManhattanArea(cur)) {
    return true;
  }
  auto mr = cur->get_mr();
  if (mr.size() != 4) {
    return false;
  }
  int count = 0;
  for (auto line : cur->getMrLines()) {
    if (Geom::lineType(line) == LineType::kManhattan) {
      count++;
    }
    auto t_min = std::abs(line[kHead].min - line[kTail].min);
    auto t_max = std::abs(line[kHead].max - line[kTail].max);
    if (t_min > kEpsilon || t_max > kEpsilon) {
      return false;
    }
  }
  return count == 2;
}
bool BoundSkewTree::isManhattanArea(Area* cur) const
{
  auto mr = cur->get_mr();
  if (mr.size() == 1) {
    return true;
  }
  if (mr.size() == 2 && Geom::lineType(mr[kHead], mr[kTail]) == LineType::kManhattan) {
    return true;
  }
  return false;
}

void BoundSkewTree::mrToTrr(const Region& mr, Trr& trr) const
{
  if (mr.size() == 1) {
    trr.makeDiamond(mr.front(), 0);
    return;
  }
  if (mr.size() == 2) {
    Geom::lineToMs(trr, mr[kHead], mr[kTail]);
    return;
  }
  if (mr.size() == 4) {
    Trr trr_left;
    Geom::lineToMs(trr_left, mr[kLeft + kHead], mr[kLeft + kTail]);
    Trr trr_right;
    Geom::lineToMs(trr_right, mr[kRight + kHead], mr[kRight + kTail]);
    trr = trr_left;
    trr.enclose(trr_right);
    return;
  }
  LOG_FATAL << "mr size is not 1, 2 or 4";
}

LineType BoundSkewTree::calcAreaLineType(Area* cur) const
{
  auto line = cur->get_line(kLeft);
  return Geom::lineType(line);
}
void BoundSkewTree::calcConvexHull(Area* cur) const
{
  auto mr = cur->get_mr();
  Geom::convexHull(mr);
  cur->set_convex_hull(mr);
}

double BoundSkewTree::calcJrArea(const Line& l1, const Line& l2) const
{
  auto min_x = std::min({l1[kHead].x, l1[kTail].x, l2[kHead].x, l2[kTail].x});
  auto max_x = std::max({l1[kHead].x, l1[kTail].x, l2[kHead].x, l2[kTail].x});
  auto min_y = std::min({l1[kHead].y, l1[kTail].y, l2[kHead].y, l2[kTail].y});
  auto max_y = std::max({l1[kHead].y, l1[kTail].y, l2[kHead].y, l2[kTail].y});
  auto bound_area = (max_x - min_x) * (max_y - min_y);
  auto tri_area_1 = 0.5 * std::abs(l1[kHead].x - l1[kTail].x) * std::abs(l1[kHead].y - l1[kTail].y);
  auto tri_area_2 = 0.5 * std::abs(l2[kHead].x - l2[kTail].x) * std::abs(l2[kHead].y - l2[kTail].y);
  auto jr_area = bound_area - tri_area_1 - tri_area_2;
  LOG_FATAL_IF(jr_area < 0) << "jr area is negative";
  return jr_area;
}

void BoundSkewTree::calcBsLocated(Area* cur, Pt& pt, Line& line) const
{
  for (auto mr_line : cur->getMrLines()) {
    line = mr_line;
    if (Geom::onLine(pt, line)) {
      return;
    }
  }
  printPoint(pt);
  printArea(cur);
  LOG_FATAL << "point is not located in area";
}
void BoundSkewTree::calcPtDelays(Area* cur, Pt& pt, Line& line) const
{
  LOG_FATAL_IF(!Geom::onLine(pt, line)) << "point is not located in line";
  auto dist = Geom::distance(pt, line[kHead]);
  auto x = std::abs(line[kHead].x - line[kTail].x);
  auto y = std::abs(line[kHead].y - line[kTail].y);
  auto length = x + y;
  if (Equal(dist, 0)) {
    pt.min = line[kHead].min;
    pt.max = line[kHead].max;
  } else if (Geom::isSame(pt, line[kTail])) {
    pt.min = line[kTail].min;
    pt.max = line[kTail].max;
  } else if (Equal(x, y)) {
    // line is manhattan arc
    LOG_FATAL_IF(!Equal(line[kHead].min, line[kTail].min) || !Equal(line[kHead].max, line[kTail].max))
        << "manhattan arc endpoint's delay is not same";
    pt.min = line[kHead].min = line[kTail].min;
    pt.max = line[kHead].max = line[kTail].max;
  } else if (Equal(x, 0) || Equal(y, 0)) {
    // line is vertical or horizontal
    auto alpha = Equal(x, 0) ? _K[kV] : _K[kH];
    auto beta = (line[kTail].min - line[kHead].min) / length - alpha * length;
    pt.min = line[kHead].min + alpha * dist * dist + beta * dist;
    beta = (line[kTail].max - line[kHead].max) / length - alpha * length;
    pt.max = line[kHead].max + alpha * dist * dist + beta * dist;
  } else {
    LOG_FATAL_IF(!cur) << "cur is nullptr";
    LOG_FATAL_IF(!Equal(ptSkew(line[kHead]), _skew_bound) || !Equal(ptSkew(line[kTail]), _skew_bound))
        << "thera are skew reservation in line";
    calcIrregularPtDelays(cur, pt, line);
  }
  checkPtDelay(pt);
}
void BoundSkewTree::updatePtDelaysByEndSide(Area* cur, const size_t& end_side, Pt& pt) const
{
  auto left_line = cur->get_line(kLeft);
  auto right_line = cur->get_line(kRight);
  auto delay_left = ptDelayIncrease(pt, left_line[end_side], cur->get_left()->get_cap_load());
  auto delay_right = ptDelayIncrease(pt, right_line[end_side], cur->get_right()->get_cap_load());
  pt.min = std::min(left_line[end_side].min + delay_left, right_line[end_side].min + delay_right);
  pt.max = std::max(left_line[end_side].max + delay_left, right_line[end_side].max + delay_right);
}
void BoundSkewTree::calcIrregularPtDelays(Area* cur, Pt& pt, Line& line) const
{
  auto x = std::abs(line[kHead].x - line[kTail].x);
  auto y = std::abs(line[kHead].y - line[kTail].y);
  auto left_line = cur->get_line(kLeft);
  auto right_line = cur->get_line(kRight);
  auto js_type = Geom::lineType(cur->get_line(kLeft));
  if (js_type == LineType::kManhattan) {
    LOG_FATAL_IF(!Geom::isSame(left_line[kHead], left_line[kTail]) || !Geom::isSame(right_line[kHead], right_line[kTail]))
        << "endpoint should be same";
    auto delay_left = ptDelayIncrease(left_line[kHead], pt, cur->get_left()->get_cap_load());
    auto delay_right = ptDelayIncrease(right_line[kHead], pt, cur->get_right()->get_cap_load());
    pt.min = std::min(left_line[kHead].min + delay_left, right_line[kHead].min + delay_right);
    pt.max = std::max(left_line[kHead].max + delay_left, right_line[kHead].max + delay_right);
    LOG_FATAL_IF(ptSkew(pt) >= _skew_bound + kEpsilon) << "skew is larger than skew bound";
  } else {
    LOG_FATAL_IF(js_type != LineType::kVertical && js_type != LineType::kHorizontal) << "js type is not vertical or horizontal";
    auto dist = Geom::distance(pt, line[kHead]);
    auto length = x + y;
    double alpha = 0;
    if (x > y) {
      auto m = y / x;
      auto ratio = std::pow(1 + std::abs(m), 2);
      alpha = (_K[kH] + m * m * _K[kV]) / ratio;
    } else {
      auto m = x / y;
      auto ratio = std::pow(1 + std::abs(m), 2);
      alpha = (_K[kV] + m * m * _K[kH]) / ratio;
    }
    auto beta = (line[kTail].max - line[kHead].max) / length - alpha * length;
    pt.max = line[kHead].max + alpha * dist * dist + beta * dist;
    beta = (line[kTail].min - line[kHead].min) / length - alpha * length;
    pt.min = line[kHead].min + alpha * dist * dist + beta * dist;
  }
}
double BoundSkewTree::ptDelayIncrease(Pt& p1, Pt& p2, const double& cap, const RCPattern& pattern) const
{
  auto delay = calcDelayIncrease(std::abs(p1.x - p2.x), std::abs(p1.y - p2.y), cap, pattern);
  LOG_FATAL_IF(delay < 0) << "point increase delay is negative";
  return delay;
}
double BoundSkewTree::calcDelayIncrease(const double& x, const double& y, const double& cap, const RCPattern& pattern) const
{
  double delay = 0;
  switch (pattern) {
    case RCPattern::kHV:
      delay = _unit_h_res * x * (_unit_h_cap * x / 2 + cap) + _unit_v_res * y * (_unit_v_cap * y / 2 + cap + x * _unit_h_cap);
      break;
    case RCPattern::kVH:
      delay = _unit_v_res * y * (_unit_v_cap * y / 2 + cap) + _unit_h_res * x * (_unit_h_cap * x / 2 + cap + y * _unit_v_cap);
      break;
    default:
      LOG_FATAL << "unknown pattern";
      break;
  }
  return delay;
}

double BoundSkewTree::ptSkew(const Pt& pt) const
{
  return pt.max - pt.min;
}
Line BoundSkewTree::getJrLine(const size_t& side) const
{
  auto jr = _join_region[side];
  return {jr[kHead], jr[kTail]};
}
Line BoundSkewTree::getJsLine(const size_t& side) const
{
  auto js = _join_segment[side];
  return {js[kHead], js[kTail]};
}
Line BoundSkewTree::getJsLine(const size_t& side, const Side<Pts>& join_segment) const
{
  auto js = join_segment[side];
  return {js[kHead], js[kTail]};
}
void BoundSkewTree::setJrLine(const size_t& side, const Line& line)
{
  _join_region[side][kHead] = line[kHead];
  _join_region[side][kTail] = line[kTail];
}
void BoundSkewTree::setJsLine(const size_t& side, const Line& line)
{
  _join_segment[side][kHead] = line[kHead];
  _join_segment[side][kTail] = line[kTail];
}
void BoundSkewTree::checkPtDelay(Pt& pt) const
{
  LOG_FATAL_IF(pt.min <= -kEpsilon) << "pt min delay is negative";
  LOG_FATAL_IF(pt.max - pt.min <= -kEpsilon) << "pt skew is negative";
  if (pt.min < 0) {
    pt.min = 0;
  }
  if (pt.max < pt.min) {
    pt.max = pt.min;
  }
}
void BoundSkewTree::checkJsMs() const
{
  Trr left, right;
  auto left_js = getJsLine(kLeft);
  auto right_js = getJsLine(kRight);
  Geom::lineToMs(left, left_js);
  Geom::lineToMs(right, right_js);
  LOG_FATAL_IF(!Geom::isTrrContain(left, _ms[kLeft])) << "left js is not contain in left ms";
  LOG_FATAL_IF(!Geom::isTrrContain(right, _ms[kRight])) << "right js is not contain in right ms";
}
void BoundSkewTree::checkUpdateJs(const Area* cur, Line& left, Line& right) const
{
  auto is_parallel = Geom::isParallel(left, right);
  auto line_type = Geom::lineType(left);
  if (is_parallel) {
    LOG_FATAL_IF(line_type == LineType::kFlat || line_type == LineType::kTilt) << "not consider case";
  }
  auto left_js = getJsLine(kLeft);
  auto right_js = getJsLine(kRight);
  PtPair temp;
  auto dist = Geom::lineDist(left_js, right_js, temp);
  LOG_FATAL_IF(!Geom::isSame(left_js[kHead], left_js[kTail]) && !Geom::isSame(right_js[kHead], right_js[kTail])
               && !Geom::isParallel(left_js, right_js))
      << "js line error";
  LOG_FATAL_IF(!Equal(dist, cur->get_radius())) << "distance between joinsegments not equal to radius";
  LOG_FATAL_IF(!Geom::onLine(left_js[kHead], left) || !Geom::onLine(left_js[kTail], left)) << "left_js not in left section";
  LOG_FATAL_IF(!Geom::onLine(right_js[kHead], right) || !Geom::onLine(right_js[kTail], right)) << "left_js not in left section";
}
void BoundSkewTree::printPoint(const Pt& pt) const
{
  LOG_INFO << "x: " << pt.x << " y: " << pt.y << " max: " << pt.max << " min: " << pt.min << " val: " << pt.val;
}
void BoundSkewTree::printArea(const Area* area) const
{
  LOG_INFO << "area: " << area->get_name();
  std::ranges::for_each(area->getMrLines(), [&](Line& line) {
    printPoint(line[kHead]);
    printPoint(line[kTail]);
  });
}
}  // namespace bst
}  // namespace icts