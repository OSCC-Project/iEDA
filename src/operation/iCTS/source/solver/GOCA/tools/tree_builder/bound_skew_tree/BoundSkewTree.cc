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
void BoundSkewTree::run()
{
}
void BoundSkewTree::merge(Area* parent, Area* left, Area* right)
{
  parent->set_left(left);
  parent->set_right(right);
  auto dist = parent->get_radius();
  jsProcess(parent);
  auto left_line = parent->get_line(kLeft);
  auto right_line = parent->get_line(kRight);
  // TBD checkMergeArea(parent);
  constructMr(parent, left, right);
  // TBD checkMergeArea(parent,left,right);
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan) {
    LOG_FATAL_IF(Geom::lineType(getJsLine(kRight)) != LineType::kManhattan) << "right js is not manhattan";
    // TBD jsProcess_sub(area)
  } else {
    parent->set_line(kLeft, left_line);
    parent->set_line(kRight, right_line);
  }
  parent->set_radius(dist);
  switch (_pattern) {
    case RCPattern::kHV:
      parent->set_cap_load(left->get_cap_load() + right->get_cap_load()
                           + (parent->get_edge_len(kLeft) + parent->get_edge_len(kRight)) * _unit_h_cap);
      break;
    case RCPattern::kVH:
      parent->set_cap_load(left->get_cap_load() + right->get_cap_load()
                           + (parent->get_edge_len(kLeft) + parent->get_edge_len(kRight)) * _unit_v_cap);
      break;
    default:
      LOG_FATAL << "unknown pattern";
      break;
  }
}
void BoundSkewTree::constructMr(Area* parent, Area* left, Area* right)
{
  // calcJr(parent, left, right);
  // calcJrCorner(parent);
  // calcBalancePt(parent);
  // calcFmsPt(parent, left, right);
  // if (existFmsOnJr()) {
  //   constructFeasibleMr(parent, left, right);
  // } else {
  //   constructInfeasibleMr(parent, left, right);
  // }
  // if (Geom::lineType(parent->get_line(kLeft)) == LineType::kManhattan && parent->get_edge_len(kLeft) >= 0) {
  //   LOG_FATAL_IF(parent->get_edge_len(kRight) < 0) << "right edge length is negative";
  //   constructTrrMr(parent);
  // }
  // checkMr(parent);
  // auto new_mr = uniqueSortPts(parent->get_mr());
  // parent->set_mr(new_mr);
  // calcConvexHull(parent);
  // checkMr(parent);
  // checkMr2(parent);
  // checkMr3(parent);
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
void BoundSkewTree::initSide()
{
  FOR_EACH_SIDE(side)
  {
    _join_region[side] = {Pt(), Pt()};
    _join_segment[side] = {Pt(), Pt()};
  }
}
void BoundSkewTree::updateJS(Area* cur, Line& left, Line& right, PtPair closest)
{
  initSide();
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
    LOG_WARNING << "unknow case";
  }
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan && left_type != LineType::kManhattan) {
    _ms[kLeft].makeDiamond(closest[kLeft], 0);
    _ms[kRight].makeDiamond(closest[kRight], 0);
  }
  checkUpdateJs(cur, left, right);
}
double BoundSkewTree::calcJrArea(const Line& l1, const Line& l2)
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
void BoundSkewTree::calcJS(Area* cur, Line& left, Line& right)
{
  PtPair closest;
  auto line_dist = Geom::lineDist(left, right, closest);
  auto left_js_bak = getJsLine(kLeft);
  auto right_js_bak = getJsLine(kRight);
  auto left_ms_bak = _ms[kLeft];
  auto right_ms_bak = _ms[kRight];
  if (Equal(line_dist, cur->get_radius())) {
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
void BoundSkewTree::calcJS(Area* cur, Area* left, Area* right)
{
  auto left_lines = left->getConvexHullLines();
  auto right_lines = right->getConvexHullLines();
  std::ranges::for_each(left_lines, [&](Line& left_line) {
    std::ranges::for_each(right_lines, [&](Line& right_line) {
      PtPair closest;
      Geom::lineDist(left_line, right_line, closest);
      calcJS(cur, left_line, right_line);
    });
  });
  calcJsDelay(cur, left, right);
  auto left_js = getJsLine(kLeft);
  if (Geom::lineType(left_js) == LineType::kManhattan) {
    checkJsMs();
  }
}
void BoundSkewTree::calcJsDelay(Area* cur, Area* left, Area* right)
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
void BoundSkewTree::calcBsLocated(Area* cur, Pt& pt, Line& line)
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
void BoundSkewTree::calcPtDelays(Area* cur, Pt& pt, Line& line)
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
void BoundSkewTree::calcIrregularPtDelays(Area* cur, Pt& pt, Line& line)
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
double BoundSkewTree::ptDelayIncrease(Pt& p1, Pt& p2, const double& cap, const RCPattern& pattern)
{
  auto delay = calcDelayIncrease(std::abs(p1.x - p2.x), std::abs(p1.y - p2.y), cap, pattern);
  LOG_FATAL_IF(delay < 0) << "point increase delay is negative";
  return delay;
}
double BoundSkewTree::calcDelayIncrease(const double& x, const double& y, const double& cap, const RCPattern& pattern)
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
Line BoundSkewTree::getJrLine(const size_t& side)
{
  auto jr = _join_region[side];
  return {jr[kHead], jr[kTail]};
}
Line BoundSkewTree::getJsLine(const size_t& side)
{
  auto js = _join_segment[side];
  return {js[kHead], js[kTail]};
}
Line BoundSkewTree::getJsLine(const size_t& side, const Side<Pts>& join_segment)
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
double BoundSkewTree::ptSkew(const Pt& pt)
{
  return pt.max - pt.min;
}
void BoundSkewTree::checkPtDelay(Pt& pt)
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
void BoundSkewTree::checkJsMs()
{
  Trr left, right;
  auto left_js = getJsLine(kLeft);
  auto right_js = getJsLine(kRight);
  Geom::lineToMs(left, left_js);
  Geom::lineToMs(right, right_js);
  LOG_FATAL_IF(!Geom::isContain(left, _ms[kLeft])) << "left js is not contain in left ms";
  LOG_FATAL_IF(!Geom::isContain(right, _ms[kRight])) << "right js is not contain in right ms";
}
void BoundSkewTree::checkUpdateJs(const Area* cur, Line& left, Line& right)
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
void BoundSkewTree::printPoint(const Pt& pt)
{
  LOG_INFO << "x: " << pt.x << " y: " << pt.y << " max: " << pt.max << " min: " << pt.min << " cap: " << pt.cap;
}
void BoundSkewTree::printArea(const Area* area)
{
  LOG_INFO << "area: " << area->get_name();
  std::ranges::for_each(area->getMrLines(), [&](Line& line) {
    printPoint(line[kHead]);
    printPoint(line[kTail]);
  });
}
}  // namespace bst
}  // namespace icts