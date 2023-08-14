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
 * @file TimingCalculator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "TimingCalculator.h"

#include <fstream>

#include "CTSAPI.hpp"
#include "Operator.h"
#include "log/Log.hh"

namespace icts {
TimingCalculator::TimingCalculator()
{
  auto* config = CTSAPIInst.get_config();
  _unit_res = CTSAPIInst.getClockUnitRes() / 1000;
  _unit_cap = CTSAPIInst.getClockUnitCap();
  // init timing model from api
  _delay_libs = CTSAPIInst.getAllBufferLibs();
  _skew_bound = config->get_skew_bound();
  _db_unit = CTSAPIInst.getDbUnit();
  _max_buf_tran = config->get_max_buf_tran();
  _max_sink_tran = config->get_max_sink_tran();
  _max_cap = config->get_max_cap();
  _max_fanout = config->get_max_fanout();
  _max_length = config->get_max_length();
  _min_insert_delay = _delay_libs.front()->getDelayIntercept();
}
// basic calc
double TimingCalculator::calcShortestLength(TimingNode* i, TimingNode* j) const
{
  auto length = 1.0 * pgl::manhattan_distance(i->get_join_segment(), j->get_join_segment()) / _db_unit;
  return length;
}

double TimingCalculator::calcFarthestLength(TimingNode* i, TimingNode* j) const
{
  auto farthest_length = 0.0;
  auto edges_i = i->get_merge_region().get_edges();
  auto edges_j = j->get_merge_region().get_edges();
  for (auto& edge_i : edges_i) {
    for (auto& edge_j : edges_j) {
      auto length = pgl::manhattan_distance(edge_i, edge_j);
      if (length > farthest_length) {
        farthest_length = length;
      }
    }
  }
  auto length = 1.0 * farthest_length / _db_unit;
  return length;
}

double TimingCalculator::calcCapLoad(TimingNode* k) const
{
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (k->is_sink() || !(left || right)) {
    return k->get_cap_out();
  }
  auto cap_out = 0.0;
  if (left) {
    auto* left_lib = findLib(left);
    cap_out += _unit_cap * (calcShortestLength(k, left) + left->get_need_snake())
               + (left->is_buffer() ? left_lib->get_init_cap() : left->get_cap_out());
  }
  if (right) {
    auto* right_lib = findLib(right);
    cap_out += _unit_cap * (calcShortestLength(k, right) + right->get_need_snake())
               + (right->is_buffer() ? right_lib->get_init_cap() : right->get_cap_out());
  }
  return cap_out;
}

double TimingCalculator::minSubSlewConstraint(TimingNode* k) const
{
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (!left && !right) {
    return k->get_slew_constraint();
  }
  auto min_slew = std::numeric_limits<double>::max();
  if (left) {
    min_slew = std::min(min_slew, left->get_slew_constraint());
  }
  if (right) {
    min_slew = std::min(min_slew, right->get_slew_constraint());
  }
  return min_slew;
}

double TimingCalculator::calcElmoreDelay(TimingNode* s, TimingNode* t) const
{
  auto length = calcShortestLength(s, t) + t->get_need_snake();
  auto delay = _unit_res * length * (_unit_cap * length / 2 + t->get_cap_out());
  return delay;
}

double TimingCalculator::calcTempElmoreDelay(TimingNode* s, TimingNode* t) const
{
  auto center = pgl::center(s->get_merge_region());
  auto length = 1.0 * pgl ::manhattan_distance(center, t->get_join_segment()) / _db_unit + t->get_need_snake();
  auto delay = _unit_res * length * (_unit_cap * length / 2 + t->get_cap_out());
  return delay;
}

double TimingCalculator::calcIdealSlew(TimingNode* s, TimingNode* t) const
{
  return std::log(9) * calcElmoreDelay(s, t);
}

double TimingCalculator::calcMaxIdealSlew(TimingNode* i, TimingNode* j) const
{
  auto* temp_k = new TimingNode();
  // Polygon region;
  // calcMergeRegion(region, i, j); // calcBestSlewMergeRegion
  // TBD use the merge region
  auto region
      = Polygon({i->get_join_segment().low(), i->get_join_segment().high(), j->get_join_segment().low(), j->get_join_segment().high()});
  auto mid_js_point = pgl::center(region);
  temp_k->set_join_segment(Segment({mid_js_point, mid_js_point}));
  temp_k->set_merge_region(Polygon({mid_js_point, mid_js_point}));
  auto ideal_i = calcIdealSlew(temp_k, i);
  auto ideal_j = calcIdealSlew(temp_k, j);
  delete temp_k;
  return std::max(ideal_i, ideal_j);
}

double TimingCalculator::endPointByZeroSkew(TimingNode* i, TimingNode* j, const std::optional<double>& init_delay_i,
                                            const std::optional<double>& init_delay_j) const
{
  auto delay_i = init_delay_i ? init_delay_i.value() : i->get_delay_max();
  auto delay_j = init_delay_j ? init_delay_j.value() : j->get_delay_max();
  auto length = calcShortestLength(i, j);
  auto factor = (i->get_cap_out() + j->get_cap_out() + _unit_cap * length);
  auto length_to_i
      = (delay_j - delay_i) / _unit_res / factor + (j->get_cap_out() * length + 0.5 * std::pow(length, 2) * _unit_cap) / factor;
  return length_to_i;
}

std::tuple<int, double> TimingCalculator::calcEvenlyInsertNum(CtsCellLib* buf_lib, const double& length,
                                                              const double& target_low_delay) const
{
  // target_delay means the final delay of the path
  int num = 1;
  while (true) {
    auto avg_length = length / (num + 1);
    auto avg_cap_out = _unit_cap * avg_length + buf_lib->get_init_cap();
    auto avg_slew_out = buf_lib->calcSlew(avg_cap_out);
    auto avg_elmore_delay = _unit_res * avg_length * (_unit_cap * avg_length / 2 + avg_cap_out);
    auto avg_ideal_slew = std::log(9) * avg_elmore_delay;
    auto avg_slew_wire = std::sqrt(std::pow(avg_slew_out, 2) + std::pow(avg_ideal_slew, 2));
    auto avg_insert_delay = buf_lib->calcDelay(avg_slew_wire, avg_cap_out);
    auto total_elmore_delay = (num + 1) * avg_elmore_delay;
    auto total_delay = num * avg_insert_delay + total_elmore_delay;
    if (total_delay >= target_low_delay) {
      return std::make_tuple(num, total_delay);
    }
    ++num;
  }
  return std::make_tuple(std::numeric_limits<int>::max(), std::numeric_limits<double>::max());
}

std::vector<TimingNode*> TimingCalculator::screenNodes(const std::vector<TimingNode*>& nodes) const
{
  double max_delay = -std::numeric_limits<double>::max();
  double min_delay = std::numeric_limits<double>::max();
  double all_delay = 0;
  for (auto* node : nodes) {
    max_delay = std::max(max_delay, node->get_delay_max());
    min_delay = std::min(min_delay, node->get_delay_max());
    all_delay += node->get_delay_max();
  }
  double avg_delay = all_delay / nodes.size();
  std::vector<TimingNode*> lower_delay_nodes;
  auto max_delay_limit = avg_delay + (max_delay - avg_delay) * 0.5;
  for (auto* node : nodes) {
    if (node->get_delay_max() <= max_delay_limit) {
      lower_delay_nodes.emplace_back(node);
    }
  }
  return lower_delay_nodes;
}

Point TimingCalculator::guideCenter(const std::vector<TimingNode*>& nodes) const
{
  int64_t x = 0.0;
  int64_t y = 0.0;
  for (auto* node : nodes) {
    auto point = pgl::center(node->get_merge_region());
    x += point.x();
    y += point.y();
  }
  x /= nodes.size();
  y /= nodes.size();
  return Point(x, y);
}

int TimingCalculator::guideDist(const std::vector<TimingNode*>& nodes) const
{
  auto avg_delay = 0.0;
  auto max_delay = 0.0;
  for (auto* node : nodes) {
    avg_delay += node->get_delay_max();
    max_delay = std::max(max_delay, node->get_delay_max());
  }
  avg_delay /= nodes.size();
  auto delta_delay = max_delay - avg_delay;
  int bound_dist = std::sqrt(2 * delta_delay / (_unit_cap * _unit_res)) * _db_unit;
  int limit_dist = _max_length * _db_unit;
  // DLOG_INFO << "Bound dist: " << bound_dist;
  // DLOG_INFO << "Limit dist: " << limit_dist;
  return std::min(bound_dist, limit_dist);
}

std::pair<double, double> TimingCalculator::calcEndpointLoc(TimingNode* i, TimingNode* j, const double& skew_bound) const
{
  // check skew whether in bound (el_l <= el_r)
  joinSegment(i, j);
  auto ep_l = endPointByZeroSkew(i, j, i->get_delay_min() + skew_bound, j->get_delay_max());
  auto ep_r = endPointByZeroSkew(i, j, i->get_delay_max(), j->get_delay_min() + skew_bound);
  if (ep_l > ep_r) {
    if (std::fabs(ep_l - ep_r) < 1e-6) {
      ep_r = ep_l;
    }
  }
  return std::make_pair(ep_l, ep_r);
}

std::pair<double, double> TimingCalculator::calcTargetSkewRange(TimingNode* k) const
{
  LOG_FATAL_IF(!k->get_left() || !k->get_right()) << "current node does not have both left and right nodes";
  auto* left = k->get_left();
  auto* right = k->get_right();
  auto left_max = left->get_delay_max() + calcElmoreDelay(k, left);
  auto left_min = left->get_delay_min() + calcElmoreDelay(k, left);
  auto right_max = right->get_delay_max() + calcElmoreDelay(k, right);
  auto right_min = right->get_delay_min() + calcElmoreDelay(k, right);
  if (left_max > right_max) {
    auto up_bound = left_min + _skew_bound;
    auto low_bound = left_max - _skew_bound;
    return std::make_pair(low_bound - right_min, up_bound - right_max);
  } else {
    auto up_bound = right_min + _skew_bound;
    auto low_bound = right_max - _skew_bound;
    return std::make_pair(low_bound - left_min, up_bound - left_max);
  }
}

// merge calc
double TimingCalculator::cardanoMaxRealRoot(const double& a, const double& b, const double& c, const double& d) const
{
  double p = (3 * a * c - b * b) / (3 * a * a);
  double q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);
  double delta = q * q / 4 + p * p * p / 27;
  if (delta < 0) {
    return -std::numeric_limits<double>::infinity();
  }
  double u = -q / 2 + std::sqrt(delta);
  double v = -q / 2 - std::sqrt(delta);
  double y = std::pow(std::fabs(u), 1.0 / 3.0);
  double z = std::pow(std::fabs(v), 1.0 / 3.0);
  double x1 = (u < 0) ? -y - z : y + z;
  double x2 = -x1 / 2 + (b / (3 * a));
  double x3 = -x1 / 2 - (b / (3 * a));
  auto root = std::max({x1, x2, x3});
  return root;
}

double TimingCalculator::calcMaxRealRoot(const std::vector<double>& coeffs) const
{
  auto real_roots = CTSAPIInst.solvePolynomialRealRoots(coeffs);
  if (real_roots.empty()) {
    return -std::numeric_limits<double>::infinity();
  }
  return *std::max_element(real_roots.begin(), real_roots.end());
}

double TimingCalculator::calcBestSlewEP(TimingNode* i, TimingNode* j) const
{
  auto length = calcShortestLength(i, j);
  double factor_1 = j->get_cap_out() + _unit_cap * length;
  double factor_2 = 0.5 * _unit_cap * std::pow(length, 2) + j->get_cap_out() * length;
  double a = _unit_cap * (i->get_cap_out() + factor_1);
  double b = std::pow(i->get_cap_out(), 2) - std::pow(factor_1, 2) - _unit_cap * factor_2;
  double c = 2 * factor_1 * factor_2;
  double d = -std::pow(factor_2, 2)
             - (std::pow(i->get_slew_constraint(), 2) - std::pow(j->get_slew_constraint(), 2)) / std::pow(std::log(9) * _unit_res, 2);
  auto coeffs = std::vector<double>{a, b, c, d};
  auto ep_m = calcMaxRealRoot(coeffs);
  if (ep_m == -std::numeric_limits<double>::infinity()) {
    auto delta = std::fabs(a * length * length * length + b * length * length + c * length + d);
    if (delta < std::fabs(d)) {
      ep_m = length;
    } else {
      ep_m = 0;
    }
  }
  return ep_m;
}

double TimingCalculator::calcFeasibleWirelengthBySlew(TimingNode* k, CtsCellLib* lib) const
{
  auto cap_out = calcCapLoad(k);
  auto slew_const = minSubSlewConstraint(k);
  auto slew_coef = lib->get_slew_coef();
  auto s1 = slew_coef[0];
  auto s2 = slew_coef[1];

  double a = std::pow(std::log(9) * _unit_res * _unit_cap / 2, 2);
  double b = std::pow(std::log(9) * _unit_res, 2) * _unit_cap * cap_out;
  double c = std::pow(std::log(9) * _unit_res * cap_out, 2) + std::pow(s2 * _unit_cap, 2);
  double d = 2 * _unit_cap * s2 * (s2 * cap_out + s1);
  double e = std::pow(s2 * cap_out, 2) + 2 * s1 * s2 * cap_out + std::pow(s1, 2) - slew_const;
  auto coeffs = std::vector<double>{a, b, c, d, e};
  auto max_root = calcMaxRealRoot(coeffs);
  if (max_root == -std::numeric_limits<double>::infinity()) {
    return std::numeric_limits<double>::max();
  }
  return max_root;
}

int TimingCalculator::calcSlewDivenDist(TimingNode* k, const int& dist_limit) const
{
  for (auto* lib : _delay_libs) {
    auto wire_length = calcFeasibleWirelengthBySlew(k, lib);
    if (wire_length != std::numeric_limits<double>::max()) {
      k->set_cell_master(lib->get_cell_master());
      int slew_guide_dist = wire_length * _db_unit;
      return std::min(slew_guide_dist, dist_limit);
    }
  }
  LOG_FATAL << "No feasible guide liberty and wirelength found, please check "
               "the config parameter: \"max_buf_tran\" and \"max_sink_tran\"";
  return dist_limit;
}

double TimingCalculator::calcMergeWireSlew(TimingNode* i, TimingNode* j) const
{
  auto length = calcShortestLength(i, j) + i->get_need_snake() + j->get_need_snake();
  // if (length > 2 * _max_length) {
  //   return length;
  // }
  double cap_out = i->get_cap_out() + j->get_cap_out() + _unit_cap * length;
  auto* slew_lib = findLib();
  double slew_out = slew_lib->calcSlew(cap_out);
  auto max_ideal_slew = calcMaxIdealSlew(i, j);
  double slew_wire = std::sqrt(std::pow(max_ideal_slew, 2) + std::pow(slew_out, 2));
  return slew_wire;
}

double TimingCalculator::calcMergeCost(TimingNode* i, TimingNode* j) const
{
  joinSegment(i, j);
  return calcShortestLength(i, j);
  // auto length = calcShortestLength(i, j);
  // auto length_i = endPointByZeroSkew(i, j);
  // if (length_i < 0) {
  //   return i->get_delay_max() + (i->get_delay_max() - j->get_delay_max()) + 0.5 * _unit_res * _unit_cap * length_i * length_i
  //          - _unit_res * length_i * (i->get_cap_out() + j->get_cap_out()) / 2;
  // }
  // if (length_i > length) {
  //   return j->get_delay_max() + (j->get_delay_max() - i->get_delay_max())
  //          + 0.5 * _unit_res * _unit_cap * (length_i - length) * (length_i - length)
  //          + _unit_res * (length_i - length) * (i->get_cap_out() + j->get_cap_out()) / 2;
  // }
  // return i->get_delay_max() + 0.5 * _unit_res * _unit_cap * length_i * length_i + _unit_res * length_i * i->get_cap_out();
}

TimingNode* TimingCalculator::calcMergeNode(TimingNode* i, TimingNode* j) const
{
  joinSegment(i, j);
  i->set_merge_region(i->get_join_segment());
  j->set_merge_region(j->get_join_segment());
  updateTiming(i);
  updateTiming(j);
  auto* k = new TimingNode(i, j);
  // check fanout
  if (i->get_fanout() + j->get_fanout() > _max_fanout) {
    if (i->get_delay_max() - j->get_delay_max() > _min_insert_delay && i->get_fanout() > j->get_fanout()) {
      insertBuffer(i);
    } else if (j->get_delay_max() - i->get_delay_max() > _min_insert_delay && j->get_fanout() > i->get_fanout()) {
      insertBuffer(j);
    } else {
      insertBuffer(i);
      insertBuffer(j);
    }
    // updateTiming(k);
  }
  // left and rigth maybe changed
  auto* left = k->get_left();
  auto* right = k->get_right();

  if (balanceTiming(k)) {
    Polygon merge_region;
    calcBestSlewMergeRegion(merge_region, left, right);
    k->set_join_segment(Segment(merge_region.get_points().front(), merge_region.get_points().back()));
    k->set_merge_region(merge_region);
    // calcMergeRegion(merge_region, left, right);
    // k->set_join_segment(Segment(merge_region.get_points().front(), merge_region.get_points().back()));
    // k->set_merge_region(merge_region);
  }
  updateTiming(k);
  k->set_merged();
  k->set_is_origin();
  // if (k->get_fanout() == _max_fanout) {
  //   insertBuffer(k);
  //   updateTiming(k);
  // }
  return k;
}

void TimingCalculator::mergeNode(TimingNode* k) const
{
  auto* i = k->get_left();
  auto* j = k->get_right();
  joinSegment(i, j);
  i->set_merge_region(i->get_join_segment());
  j->set_merge_region(j->get_join_segment());
  updateTiming(i);
  updateTiming(j);
  auto length = calcShortestLength(i, j);
  auto total_cap = i->get_cap_out() + j->get_cap_out() + _unit_cap * length;
  while (length > 2 * _max_length || total_cap > _max_cap) {
    // fix length & cap
    auto guide_dist = _max_length * _db_unit;
    Point guide_center;
    if (i->get_delay_max() > j->get_delay_max()) {
      guide_center = pgl::mid(i->get_join_segment().low(), i->get_join_segment().high());
      insertBuffer(j, guide_center, guide_dist);
    } else {
      guide_center = pgl::mid(j->get_join_segment().low(), j->get_join_segment().high());
      insertBuffer(i, guide_center, guide_dist);
    }
    length = calcShortestLength(i, j);
    total_cap = i->get_cap_out() + j->get_cap_out() + _unit_cap * length;
  }
  // left and rigth maybe changed
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (k->get_fanout() == _max_fanout) {
    insertBuffer(k);
    updateTiming(k);
  } else if (left->get_fanout() + right->get_fanout() > _max_fanout) {
    if (left->get_delay_max() - right->get_delay_max() > _min_insert_delay && left->get_fanout() > right->get_fanout()) {
      insertBuffer(left);
    } else if (right->get_delay_max() - left->get_delay_max() > _min_insert_delay && right->get_fanout() > left->get_fanout()) {
      insertBuffer(right);
    } else {
      insertBuffer(left);
      insertBuffer(right);
    }
  }
  // left and rigth maybe changed
  left = k->get_left();
  right = k->get_right();

  if (balanceTiming(k)) {
    Polygon merge_region;
    calcBestSlewMergeRegion(merge_region, left, right);
    k->set_join_segment(Segment(merge_region.get_points().front(), merge_region.get_points().back()));
    k->set_merge_region(merge_region);
  }
  updateTiming(k);
  k->set_merged();
  k->set_is_origin();
}

void TimingCalculator::calcMergeRegion(Polygon& merge_region, TimingNode* i, TimingNode* j) const
{
  joinSegment(i, j);
  updateDelay(i);
  updateDelay(j);
  auto ep_pair = calcEndpointLoc(i, j, _skew_bound);
  auto ep_l = ep_pair.first;
  auto ep_r = ep_pair.second;
  auto length = calcShortestLength(i, j);

  auto js_i = i->get_join_segment();
  auto js_j = j->get_join_segment();

  if (length == 0 || ep_r <= 0 || ep_l >= length) {
    merge_region
        = ep_l >= 0 && ep_r >= 0 ? merge_region = Polygon({js_j.low(), js_j.high()}) : merge_region = Polygon({js_i.low(), js_i.high()});
    return;
  }
  ep_l = ep_l < 0 ? 0 : ep_l;
  ep_r = ep_r > length ? length : ep_r;

  Polygon sdr;
  calcSDR(sdr, js_i, js_j);

  // Type 1: rectilinear line
  int left_radius = std::floor(ep_r * _db_unit);
  int right_radius = std::floor((length - ep_l) * _db_unit);
  if (left_radius + right_radius < length * _db_unit) {
    ++right_radius;
  }
  if (sdr.size() == 2) {
    auto pair_point = pgl::cutSegment(Segment(sdr.get_points()[0], sdr.get_points()[1]), left_radius, right_radius);
    merge_region = Polygon({pair_point.first, pair_point.second});
    return;
  }
  // Type 2: polygon (maybe merge a line type polygon)
  Polygon left_rect;
  pgl::tilted_rect_region(left_rect, js_i, left_radius);
  Polygon right_rect;
  pgl::tilted_rect_region(right_rect, js_j, right_radius);
  auto bound = intersectionByBg(left_rect, right_rect);
  // Type 2.1: bound is a line
  if (bound.size() == 2) {
    auto seg = Segment(bound.get_points()[0], bound.get_points()[1]);
    merge_region = intersectionByBg(sdr, seg);
    return;
  }
  // Type 2.2: bound is a polygon
  merge_region = intersectionByBg(bound, sdr);
  return;
}

void TimingCalculator::calcBestSlewMergeRegion(Polygon& merge_region, TimingNode* i, TimingNode* j) const
{
  joinSegment(i, j);
  updateDelay(i);
  updateDelay(j);
  auto ep_pair = calcEndpointLoc(i, j, _skew_bound);
  auto ep_l = ep_pair.first;
  auto ep_r = ep_pair.second;
  auto length = calcShortestLength(i, j);

  auto js_i = i->get_join_segment();
  auto js_j = j->get_join_segment();

  if (length == 0 || ep_r <= 0 || ep_l >= length) {
    merge_region
        = ep_l >= 0 && ep_r >= 0 ? merge_region = Polygon({js_j.low(), js_j.high()}) : merge_region = Polygon({js_i.low(), js_i.high()});
    return;
  }
  ep_l = ep_l < 0 ? 0 : ep_l;
  ep_r = ep_r > length ? length : ep_r;
  // choose minimum slew join segment
  auto ep_m = calcBestSlewEP(i, j);
  ep_m = ep_m < ep_l ? ep_l : ep_m > ep_r ? ep_r : ep_m;
  int radius = std::ceil(ep_m * _db_unit);

  Polygon sdr;
  calcSDR(sdr, js_i, js_j);
  if (ep_m == 0) {
    merge_region = Polygon({js_i.low(), js_i.high()});
    return;
  }
  if (ep_m == length) {
    merge_region = Polygon({js_j.low(), js_j.high()});
    return;
  }
  if (sdr.size() == 2) {
    auto js_point = pgl::cutSegment(Segment(sdr.get_points()[0], sdr.get_points()[1]), radius);
    merge_region = Polygon({js_point, js_point});
    return;
  }
  Polygon rect;
  pgl::tilted_rect_region(rect, js_i, radius);

  auto points = intersectionPointByBg(sdr, rect);
  merge_region = Polygon({points[0], points[1]});
}

bool TimingCalculator::balanceTiming(TimingNode* k) const
{
  // maybe update left and right
  auto* left = k->get_left();
  auto* right = k->get_right();
  auto ep_pair = calcEndpointLoc(left, right, _skew_bound);
  auto ep_l = ep_pair.first;
  auto ep_r = ep_pair.second;
  auto length = calcShortestLength(left, right);

  if (ep_r <= 0) {
    auto js = intersectJS(right->get_join_segment(), left->get_join_segment(), std::floor(1 / 3 * length * _db_unit));
    k->set_join_segment(js);
    k->set_merge_region(js);
    updateDelay(k);
    auto skew_range = calcTargetSkewRange(k);
    auto low_skew = skew_range.first;
    if (low_skew >= _min_insert_delay) {
      insertBuffer(k, right);
    } else {
      wireSnaking(k, right, low_skew);
    }
    return false;
  }
  if (ep_l >= length) {
    auto js = intersectJS(left->get_join_segment(), right->get_join_segment(), std::floor(1 / 3 * length * _db_unit));
    k->set_join_segment(js);
    k->set_merge_region(js);
    updateDelay(k);
    auto skew_range = calcTargetSkewRange(k);
    auto low_skew = skew_range.first;
    if (low_skew >= _min_insert_delay) {
      insertBuffer(k, left);
    } else {
      wireSnaking(k, left, low_skew);
    }
    return false;
  }
  return true;
}

void TimingCalculator::fixTiming(TimingNode* k) const
{
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (left && right) {
    if (!skewFeasible(k)) {
      auto skew = k->get_delay_max() - k->get_delay_min();
      auto left_max = left->get_delay_max() + calcElmoreDelay(k, left);
      auto right_max = right->get_delay_max() + calcElmoreDelay(k, right);
      if (skew >= _min_insert_delay) {
        if (left_max > right_max) {
          insertBuffer(k, right);
        } else {
          insertBuffer(k, left);
        }
      } else {
        auto skew_range = calcTargetSkewRange(k);
        auto low_skew = skew_range.first;
        if (left_max > right_max) {
          wireSnaking(k, right, low_skew);
        } else {
          wireSnaking(k, left, low_skew);
        }
        // TBD propagate k ?
      }
    }
  }
}

void TimingCalculator::connect(TimingNode* parent, TimingNode* left, TimingNode* right) const
{
  parent->set_left(left);
  parent->set_right(right);
  left->set_parent(parent);
  right->set_parent(parent);
}

void TimingCalculator::insertConnect(TimingNode* i, TimingNode* insert, TimingNode* j) const
{
  i->get_left() == j ? i->set_left(insert) : i->set_right(insert);
  insert->set_parent(i);
  insert->set_left(j);
  j->set_parent(insert);
}

TimingNode* TimingCalculator::genSteinerNode() const
{
  auto* inst = new CtsInstance("", "", CtsInstanceType::kSteinerPoint, Point(-1, -1));
  auto* node = new TimingNode(inst);
  node->set_type(TimingNodeType::kSteiner);
  return node;
}

TimingNode* TimingCalculator::genBufferNode(const std::string& cell_master) const
{
  auto* buffer = new CtsInstance("", cell_master, CtsInstanceType::kBuffer, Point(-1, -1));
  auto* node = new TimingNode(buffer);
  node->set_slew_constraint(_max_buf_tran);
  node->set_type(TimingNodeType::kBuffer);
  return node;
}

// timing update
void TimingCalculator::updateTiming(TimingNode* k, const bool& update_cap, const bool& update_delay,
                                    const bool& update_slew_constraint) const
{
  auto* left = k->get_left();
  auto* right = k->get_right();
  auto cap_out = 0.0;
  auto left_max = -std::numeric_limits<double>::max();
  auto left_min = std::numeric_limits<double>::max();
  auto right_max = -std::numeric_limits<double>::max();
  auto right_min = std::numeric_limits<double>::max();
  auto sub_slew_constraint = std::numeric_limits<double>::max();
  auto sub_ideal_slew = -std::numeric_limits<double>::max();
  auto fanout = 0;
  auto level = 1;
  auto net_length = 0.0;
  auto* lib = findLib(k);
  if (k->is_sink() || !(left || right)) {
    return;
  }
  if (left) {
    if (update_cap) {
      if (k->is_buffer()) {
        cap_out = lib->get_init_cap();
      } else {
        cap_out += _unit_cap * (calcShortestLength(k, left) + left->get_need_snake()) + left->get_cap_out();
      }
    }
    if (update_delay) {
      auto left_delay = calcElmoreDelay(k, left);
      left_min = left->get_delay_min() + left_delay;
      left_max = left->get_delay_max() + left_delay;
    }
    if (update_slew_constraint && !k->is_buffer()) {
      sub_slew_constraint = std::min(left->get_slew_constraint(), sub_slew_constraint);
      sub_ideal_slew = std::max(calcIdealSlew(k, left), sub_ideal_slew);
    }
    fanout += left->is_steiner() ? left->get_fanout() : 1;
    level = std::max(level, left->get_level() + 1);
    net_length += left->is_steiner() ? (calcShortestLength(k, left) + left->get_need_snake() + left->get_net_length())
                                     : (calcShortestLength(k, left) + left->get_need_snake());
  }
  if (right) {
    if (update_cap) {
      if (k->is_buffer()) {
        cap_out = lib->get_init_cap();
      } else {
        cap_out += _unit_cap * (calcShortestLength(k, right) + right->get_need_snake()) + right->get_cap_out();
      }
    }
    if (update_delay) {
      auto right_delay = calcElmoreDelay(k, right);
      right_min = right->get_delay_min() + right_delay;
      right_max = right->get_delay_max() + right_delay;
    }
    if (update_slew_constraint && !k->is_buffer()) {
      sub_slew_constraint = std::min(right->get_slew_constraint(), sub_slew_constraint);
      sub_ideal_slew = std::max(calcIdealSlew(k, right), sub_ideal_slew);
    }
    fanout += right->is_steiner() ? right->get_fanout() : 1;
    level = std::max(level, right->get_level() + 1);
    net_length += right->is_steiner() ? (calcShortestLength(k, right) + right->get_need_snake() + right->get_net_length())
                                      : (calcShortestLength(k, right) + right->get_need_snake());
  }
  if (update_cap) {
    k->set_cap_out(cap_out);
  }
  if (update_delay) {
    k->set_delay_max(std::max(left_max, right_max) + k->get_insertion_delay());
    k->set_delay_min(std::min(left_min, right_min) + k->get_insertion_delay());
  }
  if (update_slew_constraint) {
    if (k->is_buffer()) {
      k->set_slew_constraint(_max_buf_tran);
    } else {
      auto factor = std::pow(sub_slew_constraint, 2) - std::pow(sub_ideal_slew, 2) - std::pow(lib->calcSlew(cap_out), 2);
      // debug compute way diff
      if (factor < 0) {
        k->set_slew_constraint(0);
      } else {
        auto slew_constraint = std::sqrt(factor);
        k->set_slew_constraint(slew_constraint);
      }
    }
  }
  k->set_fanout(fanout);
  k->set_level(level);
  k->set_net_length(net_length);
  // remain delay
  if (!skewFeasible(k) && k->get_delay_max() - k->get_delay_min() < _skew_bound + 1e-6) {
    k->set_delay_max(k->get_delay_min() + _skew_bound);
  }
}

void TimingCalculator::updateCap(TimingNode* k) const
{
  auto cap_out = calcCapLoad(k);
  k->set_cap_out(cap_out);
}

void TimingCalculator::updateDelay(TimingNode* k) const
{
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (k->is_sink() || !(left || right)) {
    return;
  }
  auto left_max = -std::numeric_limits<double>::max();
  auto left_min = std::numeric_limits<double>::max();
  auto right_max = -std::numeric_limits<double>::max();
  auto right_min = std::numeric_limits<double>::max();
  if (left) {
    auto left_delay = calcElmoreDelay(k, left);
    left_min = left->get_delay_min() + left_delay;
    left_max = left->get_delay_max() + left_delay;
  }
  if (right) {
    auto right_delay = calcElmoreDelay(k, right);
    right_min = right->get_delay_min() + right_delay;
    right_max = right->get_delay_max() + right_delay;
  }
  k->set_delay_max(std::max(left_max, right_max) + k->get_insertion_delay());
  k->set_delay_min(std::min(left_min, right_min) + k->get_insertion_delay());
}

void TimingCalculator::timingPropagate(TimingNode* k, const bool& propagate_head) const
{
  k->set_merge_region(k->get_join_segment());
  // k is a buffer node
  auto* left = k->get_left();
  auto* right = k->get_right();
  auto* lib = findLib(k);
  if (propagate_head) {
    // is head
    auto slew_out = lib->calcSlew(k->get_cap_out());
    if (left) {
      auto ideal_slew_left = calcIdealSlew(k, left);
      auto wire_slew_left = std::sqrt(std::pow(slew_out, 2) + std::pow(ideal_slew_left, 2));
      left->set_slew_in(wire_slew_left);
    }
    if (right) {
      auto ideal_slew_right = calcIdealSlew(k, right);
      auto wire_slew_right = std::sqrt(std::pow(slew_out, 2) + std::pow(ideal_slew_right, 2));
      right->set_slew_in(wire_slew_right);
    }
  } else if (k->is_steiner()) {
    // is steiner
    // wire slew calc
    if (left) {
      auto left_slew_in = std::sqrt(std::pow(k->get_slew_in(), 2) + std::pow(calcIdealSlew(k, left), 2));
      left->set_slew_in(left_slew_in);
    }
    if (right) {
      auto right_slew_in = std::sqrt(std::pow(k->get_slew_in(), 2) + std::pow(calcIdealSlew(k, right), 2));
      right->set_slew_in(right_slew_in);
    }
  } else {
    // is buffer
    if (k->is_buffer()) {
      // modify insertion delay
      auto slew_in = k->get_slew_in();
      auto cap_out = calcCapLoad(k);
      auto insert_delay = lib->calcDelay(slew_in, cap_out);
      k->set_insertion_delay(insert_delay);
      updateDelay(k);
    }
    return;
  }
  if (left) {
    timingPropagate(left, false);
  }
  if (right) {
    timingPropagate(right, false);
  }
  // just update insert delay, because cap, slew, slew constraint won't change
  updateDelay(k);
  fixTiming(k);
}

void TimingCalculator::simplePropagate(TimingNode* k, const bool& propagate_head) const
{
  k->set_merge_region(k->get_join_segment());
  // k is a buffer node
  auto* left = k->get_left();
  auto* right = k->get_right();
  auto* lib = findLib(k);
  if (propagate_head) {
    // is head
    auto slew_out = lib->calcSlew(k->get_cap_out());
    if (left) {
      auto ideal_slew_left = calcIdealSlew(k, left);
      auto wire_slew_left = std::sqrt(std::pow(slew_out, 2) + std::pow(ideal_slew_left, 2));
      left->set_slew_in(wire_slew_left);
    }
    if (right) {
      auto ideal_slew_right = calcIdealSlew(k, right);
      auto wire_slew_right = std::sqrt(std::pow(slew_out, 2) + std::pow(ideal_slew_right, 2));
      right->set_slew_in(wire_slew_right);
    }
  } else if (k->is_steiner()) {
    // is steiner
    // wire slew calc
    if (left) {
      auto left_slew_in = std::sqrt(std::pow(k->get_slew_in(), 2) + std::pow(calcIdealSlew(k, left), 2));
      left->set_slew_in(left_slew_in);
    }
    if (right) {
      auto right_slew_in = std::sqrt(std::pow(k->get_slew_in(), 2) + std::pow(calcIdealSlew(k, right), 2));
      right->set_slew_in(right_slew_in);
    }
  } else {
    // is buffer
    if (k->is_buffer()) {
      // modify insertion delay
      auto slew_in = k->get_slew_in();
      auto cap_out = calcCapLoad(k);
      auto insert_delay = lib->calcDelay(slew_in, cap_out);
      k->set_insertion_delay(insert_delay);
      updateDelay(k);
    }
    return;
  }
  if (left) {
    timingPropagate(left, false);
  }
  if (right) {
    timingPropagate(right, false);
  }
  // just update insert delay, because cap, slew, slew constraint won't change
  updateDelay(k);
}

void TimingCalculator::wireSnaking(TimingNode* s, TimingNode* t, const double& incre_delay) const
{
  LOG_FATAL_IF(incre_delay < 0) << "incre_delay: " << incre_delay << " less than 0";
  // incre_delay means the delay increase of the path
  auto length = calcShortestLength(s, t) + t->get_need_snake();
  auto factor = length + t->get_cap_out() / _unit_cap;
  auto snake_length = std::sqrt(std::pow(factor, 2) + 2 * incre_delay / (_unit_res * _unit_cap)) - factor;
  t->set_need_snake(snake_length + t->get_need_snake());
  // update timing
  updateTiming(s);
}

void TimingCalculator::makeBuffer(TimingNode* k) const
{
  LOG_WARNING_IF(!k->is_steiner()) << "make buffer on a non-steiner node";
  k->set_type(TimingNodeType::kBuffer);
  k->set_insert_type(InsertType::kMake);
  setMinCostCellLib(k);
  k->set_slew_constraint(_max_buf_tran);
  timingPropagate(k);
  // virtual insertion delay
  k->set_insertion_delay(predictInsertDelay(k, findLib(k)));
  updateTiming(k);
}

void TimingCalculator::insertBuffer(TimingNode* k, std::optional<Point> guide_point, const int& guide_dist) const
{
  // TBD multiple method (delay, slew, cap driver)
  // remain cap insertion
  if (!k->is_steiner() && guide_point == std::nullopt) {
    // LOG_WARNING << "current node is steiner, and not guide point is given";
    return;
  }
  if (k->is_steiner()) {
    makeBuffer(k);
    return;
  }
  if (guide_point != std::nullopt) {
    auto* temp = k->copy(true);
    k->set_type(TimingNodeType::kBuffer);
    k->set_insert_type(InsertType::kTopInsert);
    k->set_slew_constraint(_max_buf_tran);
    k->set_left(temp);
    k->set_right(nullptr);
    temp->set_parent(k);
    if (temp->get_left()) {
      temp->get_left()->set_parent(temp);
    }
    if (temp->get_right()) {
      temp->get_right()->set_parent(temp);
    }
    auto js_seg = temp->get_join_segment();
    auto guide = guide_point == std::nullopt ? pgl::mid(js_seg.low(), js_seg.high()) : guide_point.value();
    auto center_js = Segment(guide, guide);

    auto closest_point = pgl::closest_point(guide, js_seg);
    js_seg = Segment(closest_point, closest_point);
    temp->set_join_segment(js_seg);
    temp->set_merge_region(js_seg);
    updateTiming(temp);
    auto remain_cap = _max_cap - temp->get_cap_out();
    int remain_cap_dist = std::abs(remain_cap / _unit_cap) * _db_unit;
    auto feasible_dist = remain_cap > 0 ? std::min(pgl::manhattan_distance(guide, js_seg) / 2, remain_cap_dist) : 0;
    feasible_dist = std::min(guide_dist, feasible_dist);
    feasible_dist = calcSlewDivenDist(k, feasible_dist);
    int remain_net_dist = (_max_length - k->get_net_length()) * _db_unit;
    remain_net_dist = std::max(0, remain_net_dist);
    feasible_dist = std::min(feasible_dist, remain_net_dist);
    auto js = intersectJS(center_js, js_seg, feasible_dist);
    k->set_join_segment(js);
    k->set_merge_region(js);
    // timing update
    timingPropagate(k);
    // virtual insertion delay
    k->set_insertion_delay(predictInsertDelay(k, findLib(k)));
    updateTiming(k);
    fixTiming(k);
    return;
  }
}

void TimingCalculator::insertBuffer(TimingNode* s, TimingNode* t) const
{
  if (t->is_steiner()) {
    makeBuffer(t);
    if (s->is_buffer()) {
      updateCap(s);
      timingPropagate(s);
      s->set_insertion_delay(predictInsertDelay(s, findLib(s)));
    }
    updateTiming(s);
  }
  if (skewFeasible(s)) {
    return;
  } else {
    auto skew_range = calcTargetSkewRange(s);
    auto low_skew = skew_range.first;
    if (0 < low_skew && low_skew <= _min_insert_delay) {
      auto ref = s->get_left() == t ? s->get_right() : s->get_left();
      if (ref->get_delay_max() + calcElmoreDelay(s, ref) > t->get_delay_max() + calcElmoreDelay(s, t)) {
        wireSnaking(s, t, low_skew);
      } else {
        wireSnaking(s, ref, low_skew);
      }
      return;
    }
  }
  joinSegment(s, t);
  s->set_merge_region(s->get_join_segment());
  t->set_merge_region(t->get_join_segment());
  updateTiming(t);
  auto skew_range = calcTargetSkewRange(s);
  auto low_skew = skew_range.first;
  auto high_skew = skew_range.second;
  auto length = calcShortestLength(s, t);

  // calc insert buffer evenly
  CtsCellLib* lib;
  int insert_num;
  std::tie(lib, insert_num) = findMinCostEvenlyCellLib(length, low_skew + calcElmoreDelay(s, t), high_skew + calcElmoreDelay(s, t));
  // DLOG_INFO << "Insert Buffer num: " << insert_num;
  auto avg_length = length / (insert_num + 1);
  auto* current = t;
  for (int i = 0; i < insert_num; ++i) {
    auto* insert_buf_node = genBufferNode(lib->get_cell_master());
    insert_buf_node->set_insert_type(InsertType::kMultiple);
    auto buf_js = intersectJS(s->get_join_segment(), current->get_join_segment(), std::floor(avg_length * _db_unit));
    insert_buf_node->set_merge_region(buf_js);
    insert_buf_node->set_join_segment(buf_js);
    // update connection
    insertConnect(s, insert_buf_node, current);
    // timing propagate
    timingPropagate(insert_buf_node);
    // virtual insertion delay
    insert_buf_node->set_insertion_delay(i > 0 ? current->get_insertion_delay() : predictInsertDelay(insert_buf_node, lib));
    updateTiming(insert_buf_node);
    updateTiming(s);
    if (s->is_buffer()) {
      simplePropagate(s);
    }
    auto temp_skew_range = calcTargetSkewRange(s);
    auto temp_low_skew = temp_skew_range.first;
    auto temp_high_skew = temp_skew_range.second;
    if (temp_low_skew < 0) {
      return;
    }
    if (temp_high_skew <= _min_insert_delay) {
      fixTiming(s);
      return;
    }
    current = insert_buf_node;
  }

  updateTiming(s);
  if (s->is_buffer()) {
    timingPropagate(s);
  }
  fixTiming(s);
  // add init head insertion delay (TBD with level change, cap, slew, sub delay)
}

void TimingCalculator::insertBuffer(TimingNode* s, TimingNode* t, const double& incre_delay) const
{
  auto origin_delay = t->get_delay_max() + calcElmoreDelay(s, t);
  joinSegment(s, t);
  s->set_merge_region(s->get_join_segment());
  t->set_merge_region(t->get_join_segment());
  updateTiming(t);
  if (t->is_steiner()) {
    makeBuffer(t);
    updateTiming(s);
  }
  auto incre_require_delay = incre_delay - (t->get_delay_max() + calcElmoreDelay(s, t) - origin_delay);
  auto length = calcShortestLength(s, t);
  CtsCellLib* lib;
  int insert_num;
  std::tie(lib, insert_num) = findMinCostEvenlyCellLib(length, incre_require_delay, std::numeric_limits<double>::max());
  // TBD safety num (increse default insert delay or plus snake strategy)
  // DLOG_INFO << "Insert Buffer(L2 type) num: " << insert_num;
  auto avg_length = length / (insert_num + 1);
  auto* current = t;
  for (int i = 0; i < insert_num; ++i) {
    auto* insert_buf_node = genBufferNode(lib->get_cell_master());
    insert_buf_node->set_insert_type(InsertType::kMultiple);
    auto buf_js = intersectJS(s->get_join_segment(), current->get_join_segment(), std::floor(avg_length * _db_unit));
    insert_buf_node->set_merge_region(buf_js);
    insert_buf_node->set_join_segment(buf_js);
    // update connection
    insertConnect(s, insert_buf_node, current);
    // timing propagate
    timingPropagate(insert_buf_node);
    // virtual insertion delay
    insert_buf_node->set_insertion_delay(i > 0 ? current->get_insertion_delay() : predictInsertDelay(insert_buf_node, lib));
    updateTiming(insert_buf_node);
    current = insert_buf_node;
  }
  updateTiming(s);
}

CtsCellLib* TimingCalculator::findLib(TimingNode* k) const
{
  std::string cell_master = "";
  if (k && k->is_buffer()) {
    cell_master = k->get_inst()->get_cell_master();
  }
  if (cell_master.empty()) {
    cell_master = _delay_libs.front()->get_cell_master();
  }
  return CTSAPIInst.getCellLib(cell_master);
}

void TimingCalculator::setMinCostCellLib(TimingNode* k) const
{
  if (!k->get_cell_master().empty()) {
    LOG_FATAL << "Cell master is not empty";
  }
  // TBD multiple cost
  // CtsCellLib *min_cost_lib = nullptr;
  // double cost = std::numeric_limits<double>::max();
  for (auto* lib : _delay_libs) {
    if (checkSlew(k, lib)) {
      k->set_cell_master(lib->get_cell_master());
      return;
    }
  }
  // slew violation
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (left && left->is_steiner()) {
    makeBuffer(left);
  }
  if (right && right->is_steiner()) {
    makeBuffer(right);
  }
  for (auto* lib : _delay_libs) {
    if (checkSlew(k, lib)) {
      k->set_cell_master(lib->get_cell_master());
      return;
    }
  }
  LOG_ERROR << "No feasible lib, please check "
               "the config parameter: \"max_buf_tran\", \"max_sink_tran\" and "
               "\"max_cap\"";
  k->set_cell_master(_delay_libs.back()->get_cell_master());
}

CtsCellLib* TimingCalculator::findFeasibleLib(TimingNode* k) const
{
  for (auto* lib : _delay_libs) {
    if (checkSlew(k, lib)) {
      return lib;
    }
  }
  // slew violation
  auto* left = k->get_left();
  auto* right = k->get_right();
  if (left && left->is_steiner()) {
    makeBuffer(left);
  }
  if (right && right->is_steiner()) {
    makeBuffer(right);
  }
  for (auto* lib : _delay_libs) {
    if (checkSlew(k, lib)) {
      return lib;
    }
  }
  LOG_ERROR << "No feasible lib, please check "
               "the config parameter: \"max_buf_tran\", \"max_sink_tran\" and "
               "\"max_cap\"";
  return _delay_libs.back();
}

bool TimingCalculator::checkSlew(TimingNode* k, CtsCellLib* lib) const
{
  double cap_out = calcCapLoad(k);

  auto* left = k->get_left();
  auto* right = k->get_right();

  auto slew_out = lib->calcSlew(cap_out);

  auto chekc_slew = [&](TimingNode* t) {
    double slew_wire = calcIdealSlew(k, t);
    double slew = std::sqrt(slew_out * slew_out + slew_wire * slew_wire);
    return slew < t->get_slew_constraint();
  };

  if (left && !chekc_slew(left)) {
    return false;
  }
  if (right && !chekc_slew(right)) {
    return false;
  }
  return true;
}

std::tuple<CtsCellLib*, int> TimingCalculator::findMinCostEvenlyCellLib(const double& length, const double& target_low_skew,
                                                                        const double& target_high_skew) const
{
  std::tuple<CtsCellLib*, int> min_base_cost_tuple;
  std::tuple<CtsCellLib*, int> min_feasible_cost_tuple;
  double base_cost = std::numeric_limits<double>::max();
  double feasible_cost = std::numeric_limits<double>::max();
  for (auto* lib : _delay_libs) {
    int num;
    double total_delay;
    std::tie(num, total_delay) = calcEvenlyInsertNum(lib, length, target_low_skew);
    auto unit_area = CTSAPIInst.getCellArea(lib->get_cell_master());
    // TBD leak power cost
    auto temp_cost = 1.0 * num * unit_area;
    if (total_delay < target_high_skew && temp_cost < feasible_cost) {
      min_feasible_cost_tuple = std::make_tuple(lib, num);
      feasible_cost = temp_cost;
    }
    if (temp_cost < base_cost) {
      min_base_cost_tuple = std::make_tuple(lib, num);
      base_cost = temp_cost;
    }
  }
  LOG_FATAL_IF(base_cost == std::numeric_limits<double>::max())
      << "No feasible lib in find minimal cost cell lib, which skew range:[" << target_low_skew << ", " << target_high_skew
      << "], please check "
         "the config parameter: \"skew_bound\", \"max_buf_tran\", \"max_sink_tran\" and "
         "\"max_cap\"";
  if (feasible_cost != std::numeric_limits<double>::max()) {
    return min_feasible_cost_tuple;
  }
  return min_base_cost_tuple;
}

void TimingCalculator::breakLongWire(TimingNode* s, TimingNode* t) const
{
  auto remain_length = calcShortestLength(s, t);
  while (remain_length > _max_length) {
    auto remain_cap = _max_cap - t->get_cap_out();
    auto length = calcShortestLength(s, t);
    auto l_1 = remain_cap / _unit_cap;
    l_1 = l_1 >= length ? length : l_1;
    l_1 = l_1 > _max_length ? _max_length : l_1;
    // make buffer node
    auto l_1_buf_js = intersectJS(s->get_join_segment(), t->get_join_segment(), std::floor(l_1 * _db_unit));
    // DLOG_INFO << "Insert Buffer(L1 type)";
    auto* buf_node = genBufferNode();
    buf_node->set_insert_type(InsertType::kBreakWire);
    buf_node->set_merge_region(l_1_buf_js);
    buf_node->set_join_segment(l_1_buf_js);
    // update connection
    insertConnect(s, buf_node, t);
    // timing propagate
    auto* lib = findFeasibleLib(buf_node);
    buf_node->set_cell_master(lib->get_cell_master());
    timingPropagate(buf_node);
    // virtual insertion delay
    buf_node->set_insertion_delay(predictInsertDelay(buf_node, lib));
    updateTiming(buf_node);
    updateTiming(s);
    remain_length -= calcShortestLength(buf_node, t);
    t = buf_node;
  }
}

double TimingCalculator::predictSlewIn(TimingNode* k) const
{
#if (defined PY_MODEL) && (defined USE_EXTERNAL_MODEL)
  // predict slew in and find lib to set insertion delay
  if (_external_model) {
    std::vector<double> x_type;
    switch (k->get_insert_type()) {
      case InsertType::kMake:
        x_type = {1, 0, 0, 0};
        break;
      case InsertType::kMultiple:
        x_type = {0, 1, 0, 0};
        break;
      case InsertType::kTopInsert:
        x_type = {0, 0, 1, 0};
        break;
      case InsertType::kBreakWire:
        x_type = {0, 0, 0, 1};
        break;
      default:
        x_type = {0, 0, 0, 0};
        break;
    }
    updateTiming(k);
    double level = k->get_level();
    double cap_out = calcCapLoad(k);
    double fanout = k->get_fanout();
    double min_delay = k->get_delay_min();
    double max_delay = k->get_delay_max();
    std::vector<double> x_feat = {level, cap_out, fanout, min_delay, max_delay};
    std::vector<double> x;
    x.insert(x.end(), x_type.begin(), x_type.end());
    x.insert(x.end(), x_feat.begin(), x_feat.end());
    auto slew_in = _external_model->predict(x);
    return slew_in;
  }
#endif
  return 0;
}

double TimingCalculator::predictInsertDelay(TimingNode* k, CtsCellLib* lib) const
{
#if (defined PY_MODEL) && (defined USE_EXTERNAL_MODEL)
  // predict slew in and find lib to set insertion delay
  if (_external_model && k->get_level() < 10) {
    auto slew_in = predictSlewIn(k);
    auto cap_out = calcCapLoad(k);
    return lib->calcDelay(slew_in, cap_out);
  }
#endif
  auto cap_out = calcCapLoad(k);
  return lib->calcDelay(0, cap_out);
}

// polygon calc
Point TimingCalculator::bgToPglPoint(const bg_Point& p) const
{
  return Point(p.x(), p.y());
}

Segment TimingCalculator::bgToPglSegment(const bg_Segment& s) const
{
  return Segment(bgToPglPoint(s.front()), bgToPglPoint(s.back()));
}

Polygon TimingCalculator::bgToPglPolygon(const bg_Polygon& p) const
{
  Polygon poly;
  for (auto& point : p.outer()) {
    poly.add_point(bgToPglPoint(point));
  }
  return poly;
}

bg_Point TimingCalculator::pglToBgPoint(const Point& p) const
{
  return bg_Point(p.x(), p.y());
}

bg_Segment TimingCalculator::pglToBgSegment(const Segment& s) const
{
  return bg_Segment({pglToBgPoint(s.low()), pglToBgPoint(s.high())});
}

bg_Polygon TimingCalculator::pglToBgPolygon(const Polygon& p) const
{
  bg_Polygon bg_poly;
  for (auto& point : p.get_points()) {
    bg_poly.outer().push_back(pglToBgPoint(point));
  }
  if (!bg::is_valid(bg_poly)) {
    bg::correct(bg_poly);
  }
  return bg_poly;
}

std::vector<Point> TimingCalculator::intersectionPointByBg(const Polygon& poly_a, const Polygon& poly_b) const
{
  auto bg_poly_a = pglToBgPolygon(poly_a);
  auto bg_poly_b = pglToBgPolygon(poly_b);

  std::vector<bg_Point> intersection;
  bg::intersection(bg_poly_a, bg_poly_b, intersection);
  std::vector<Point> points;
  for (auto point : intersection) {
    points.push_back(Point(point.x(), point.y()));
  }
  return points;
}

std::vector<Point> TimingCalculator::intersectionPointByBg(const Polygon& poly, const Segment& seg) const
{
  auto bg_poly = pglToBgPolygon(poly);
  auto bg_seg = pglToBgSegment(seg);

  std::vector<bg_Point> intersection;
  bg::intersection(bg_poly, bg_seg, intersection);
  std::vector<Point> points;
  for (auto point : intersection) {
    points.push_back(Point(point.x(), point.y()));
  }
  return points;
}

Point TimingCalculator::intersectionPointByBg(const Segment& seg_a, const Segment& seg_b) const
{
  auto bg_seg_a = pglToBgSegment(seg_a);
  auto bg_seg_b = pglToBgSegment(seg_b);
  std::vector<bg_Point> intersection;
  bg::intersection(bg_seg_a, bg_seg_b, intersection);
  if (intersection.empty()) {
    return Point(0, 0);
  }
  return Point(intersection.front().x(), intersection.front().y());
}

Polygon TimingCalculator::intersectionByBg(const Polygon& poly_a, const Polygon& poly_b) const
{
  auto bg_poly_a = pglToBgPolygon(poly_a);
  auto bg_poly_b = pglToBgPolygon(poly_b);
  std::vector<bg_Polygon> intersection;
  bg::intersection(bg_poly_a, bg_poly_b, intersection);
  if (intersection.empty()) {
    auto is_points = intersectionPointByBg(poly_a, poly_b);
    return Polygon(is_points);
  }
  Polygon poly;
  for (auto point : intersection.front().outer()) {
    poly.add_point(Point(point.x(), point.y()));
  }
  return poly;
}

Polygon TimingCalculator::intersectionByBg(const Polygon& poly, const Segment& seg) const
{
  auto bg_poly = pglToBgPolygon(poly);
  auto bg_seg = pglToBgSegment(seg);
  std::vector<bg_Point> intersection;
  bg::intersection(bg_poly, bg_seg, intersection);
  Polygon is_poly;
  for (auto& point : intersection) {
    is_poly.add_point(Point(point.x(), point.y()));
  }
  return is_poly;
}

void TimingCalculator::lineJoinSegment(TimingNode* i, TimingNode* j) const
{
  auto mr_i = i->get_merge_region();
  auto mr_j = j->get_merge_region();
  auto point_pair = pgl::closest_point_pair(mr_i.get_edges().front(), mr_j.get_edges().front());
  auto join_seg_i = Segment(point_pair.first, point_pair.first);
  auto join_seg_j = Segment(point_pair.second, point_pair.second);
  i->set_join_segment(join_seg_i);
  j->set_join_segment(join_seg_j);
}

void TimingCalculator::joinSegment(TimingNode* i, TimingNode* j) const
{
  auto mr_i = i->get_merge_region();
  auto mr_j = j->get_merge_region();
  auto intersection = intersectionByBg(mr_i, mr_j);
  if (intersection.size() > 0) {
    Segment seg;
    if (intersection.size() == 1) {
      seg = Segment(intersection.get_points().front(), intersection.get_points().front());
    } else {
      pgl::longest_segment(seg, intersection.get_edges());
    }
    i->set_join_segment(seg);
    j->set_join_segment(seg);
    return;
  }
  // not intersect
  auto edge_pair = pgl::closestEdge(mr_i, mr_j);
  auto join_seg_i = edge_pair.first;
  auto join_seg_j = edge_pair.second;
  // two closest edge are manhattan arc
  if (!pgl::manhattan_arc(join_seg_i) || !pgl::manhattan_arc(join_seg_j)) {
    auto point_pair = pgl::closest_point_pair(join_seg_i, join_seg_j);
    join_seg_i = Segment(point_pair.first, point_pair.first);
    join_seg_j = Segment(point_pair.second, point_pair.second);
  }
  Polygon trr;
  auto radius = pgl::manhattan_distance(join_seg_i, join_seg_j);
  if (radius == 0) {
    auto point = intersectionPointByBg(join_seg_i, join_seg_j);
    auto seg = Segment(point, point);
    i->set_join_segment(seg);
    j->set_join_segment(seg);
    return;
  }
  auto seg_i = join_seg_i;
  auto seg_j = join_seg_j;
  if (seg_i.low() != seg_i.high()) {
    pgl::tilted_rect_region(trr, seg_j, radius);
    auto js_i_points = intersectionPointByBg(trr, seg_i);
    join_seg_i
        = js_i_points.size() == 2 ? Segment(js_i_points.front(), js_i_points.back()) : Segment(js_i_points.front(), js_i_points.front());
  }
  if (seg_j.low() != seg_j.high()) {
    pgl::tilted_rect_region(trr, seg_i, radius);
    auto js_j_points = intersectionPointByBg(trr, seg_j);
    join_seg_j
        = js_j_points.size() == 2 ? Segment(js_j_points.front(), js_j_points.back()) : Segment(js_j_points.front(), js_j_points.front());
  }
  i->set_join_segment(join_seg_i);
  j->set_join_segment(join_seg_j);
}

Segment TimingCalculator::intersectJS(const Segment& js_i, const Segment& js_j, const int& radius_by_j) const
{
  if (radius_by_j == 0) {
    return js_j;
  }
  if (radius_by_j == pgl::manhattan_distance(js_i, js_j)) {
    return js_i;
  }
  if (js_i.low() == js_i.high() && js_j.low() == js_j.high() && pgl::rectilinear(js_i.low(), js_j.low())) {
    Point point;
    if (js_i.low().x() == js_j.low().x()) {
      point = Point(js_i.low().x(),
                    js_j.low().y() + radius_by_j * ((js_i.low().y() - js_j.low().y()) / std::abs((js_i.low().y() - js_j.low().y()))));
      return Segment(point, point);
    }
    point = Point(js_j.low().x() + radius_by_j * ((js_i.low().x() - js_j.low().x()) / std::abs((js_i.low().x() - js_j.low().x()))),
                  js_j.low().y());
    return Segment(point, point);
  }

  Polygon poly_j;
  pgl::tilted_rect_region(poly_j, js_j, radius_by_j);
  Polygon sdr;
  calcSDR(sdr, js_i, js_j);
  auto is_points = intersectionPointByBg(sdr, poly_j);
  if (is_points.size() > 2) {
    // it means sdr radius plus 1, should remove some points
    auto dist = pgl::manhattan_distance(js_i, js_j);
    for (auto itr = is_points.begin(); itr != is_points.end();) {
      if (pgl::manhattan_distance(*itr, js_i) > dist) {
        is_points.erase(itr);
      } else {
        ++itr;
      }
    }
  }
  auto js = Segment(is_points[0], is_points[1]);
  return pgl::fixJoinSegment(js);
}

void TimingCalculator::calcSDR(Polygon& sdr, const Segment& seg_i, const Segment& seg_j) const
{
  if (seg_i == seg_j) {
    sdr = seg_i.low() == seg_i.high() ? Polygon({seg_i.low()}) : Polygon({seg_i.low(), seg_i.high()});
    return;
  }
  if (seg_i.low() == seg_i.high() && seg_j.low() == seg_j.high() && pgl::rectilinear(seg_i.low(), seg_j.low())) {
    sdr = Polygon({seg_i.low(), seg_j.low()});
    return;
  }
  Rectangle bbox;
  std::vector<Point> points{seg_i.low(), seg_i.high(), seg_j.low(), seg_j.high()};
  pgl::convex_hull(points);
  pgl::extents(bbox, points);

  auto box = Polygon({bbox.low(), Point(bbox.low().x(), bbox.high().y()), bbox.high(), Point(bbox.high().x(), bbox.low().y()), bbox.low()});
  if (seg_i.low() == seg_i.high() && seg_j.low() == seg_j.high()) {
    sdr = box;
    return;
  }
  CtsPolygon<int64_t> trr_a, trr_b;
  auto radius = pgl::manhattan_distance(seg_i, seg_j);
  if (radius == 0) {
    auto point = intersectionPointByBg(seg_i, seg_j);
    sdr = Polygon({point});
    return;
  }

  pgl::tilted_rect_region(trr_a, seg_i, radius);
  pgl::tilted_rect_region(trr_b, seg_j, radius);

  auto first_region = intersectionByBg(box, trr_a);
  sdr = intersectionByBg(first_region, trr_b);
  LOG_FATAL_IF(sdr.empty()) << "sdr is empty";
}
}  // namespace icts