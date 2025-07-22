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

#include <filesystem>
#include <numbers>
#include <random>
#include <stack>

#include "TreeBuilder.hh"
namespace icts {
namespace bst {
/**
 * @brief bst flow
 *
 */
BoundSkewTree::BoundSkewTree(const std::string& net_name, const std::vector<Pin*>& pins, const std::optional<double>& skew_bound,
                             const TopoType& topo_type, const bool& estimation)
{
  LOG_FATAL_IF(topo_type == TopoType::kInputTopo) << "error topo type";
  // Not input topology
  _net_name = net_name;
  _load_pins = pins;
  _skew_bound = skew_bound.value_or(Timing::getSkewBound());
  _topo_type = topo_type;
  std::ranges::for_each(pins, [&](Pin* pin) {
    LOG_FATAL_IF(!pin->isLoad()) << "pin " << pin->get_name() << " is not load pin";
    if (estimation) {
      Timing::initLoadPinDelay(pin, true);
    } else {
      auto* inst = pin->get_inst();
      if (!inst->isSink()) {
        auto* driver_pin = inst->get_driver_pin();
        pin->set_min_delay(driver_pin->get_min_delay() + inst->get_insert_delay());
        pin->set_max_delay(driver_pin->get_max_delay() + inst->get_insert_delay());
      }
    }
    Timing::updatePinCap(pin);
    if (!Timing::skewFeasible(pin, _skew_bound)) {
#ifdef DEBUG_ICTS_BST
      LOG_ERROR << "pin " << pin->get_name() << " skew is not feasible with error: " << Timing::calcSkew(pin) - _skew_bound;
#endif
      pin->set_min_delay(pin->get_max_delay() - _skew_bound);
    }
    auto* node = new Area(pin);
    _unmerged_nodes.push_back(node);
    _node_map.insert({pin->get_name(), pin});
  });
}
BoundSkewTree::BoundSkewTree(const std::string& net_name, Pin* driver_pin, const std::optional<double>& skew_bound, const bool& estimation)
{
  _net_name = net_name;
  _root_buf = driver_pin->get_inst();
  _skew_bound = skew_bound.value_or(Timing::getSkewBound());
  TreeBuilder::convertToBinaryTree(driver_pin);
  // Copy topology
  std::unordered_map<Node*, Area*> node_area_map;
  driver_pin->postOrder([&](Node* node) {
    if (node->isPin() && node->isLoad()) {
      auto* pin = dynamic_cast<Pin*>(node);
      if (estimation) {
        Timing::initLoadPinDelay(pin, true);
      } else {
        auto* inst = pin->get_inst();
        if (!inst->isSink()) {
          auto* driver_pin = inst->get_driver_pin();
          pin->set_min_delay(driver_pin->get_min_delay() + inst->get_insert_delay());
          pin->set_max_delay(driver_pin->get_max_delay() + inst->get_insert_delay());
        }
      }
      Timing::updatePinCap(pin);
      if (!Timing::skewFeasible(pin, _skew_bound)) {
        // LOG_ERROR << "pin " << pin->get_name() << " skew is not feasible with error: " << Timing::calcSkew(pin) - _skew_bound;
#ifdef DEBUG_ICTS_BST
        LOG_ERROR << "pin " << pin->get_name() << " skew is not feasible with error: " << Timing::calcSkew(pin) - _skew_bound;
#endif
        pin->set_min_delay(pin->get_max_delay() - _skew_bound);
      }
    }
    auto* area = new Area(node);
    node_area_map[node] = area;
    area->set_pattern(node->get_pattern());
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
  convert();
  auto pins = _load_pins;
  pins.push_back(_root_buf->get_driver_pin());
  TreeBuilder::localPlace(pins);
}
void BoundSkewTree::convert()
{
  if (_topo_type == TopoType::kInputTopo) {
    inputTopologyConvert();
  } else {
    noneInputTopologyConvert();
  }
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
double BoundSkewTree::mergeCost(Area* left, Area* right) const
{
  auto min_dist = std::numeric_limits<double>::max();
  auto left_mr = left->get_mr();
  auto right_mr = right->get_mr();
  Pt l_pt, r_pt;
  for (auto left_pt : left_mr) {
    for (auto right_pt : right_mr) {
      if (Geom::distance(left_pt, right_pt) >= min_dist) {
        continue;
      }
      min_dist = Geom::distance(left_pt, right_pt);
      l_pt = left_pt;
      r_pt = right_pt;
    }
  }
  auto left_max = l_pt.max;
  auto right_max = r_pt.max;
  auto factor = left->get_cap_load() + right->get_cap_load() + _unit_h_cap * min_dist;
  auto len_to_left
      = ((right_max - left_max) / _unit_h_res + 0.5 * _unit_h_cap * min_dist * min_dist + min_dist * right->get_cap_load()) / factor;
  if (len_to_left < 0) {
    len_to_left = -len_to_left;
  } else if (len_to_left > min_dist) {
    len_to_left -= min_dist;
  }
  auto latency = left_max + 0.5 * _unit_h_res * _unit_h_cap * len_to_left * len_to_left + _unit_h_res * len_to_left * left->get_cap_load();
  return latency;
}
double BoundSkewTree::distanceCost(Area* left, Area* right) const
{
  auto min_dist = std::numeric_limits<double>::max();
  auto left_mr = left->get_mr();
  auto right_mr = right->get_mr();
  for (auto left_pt : left_mr) {
    for (auto right_pt : right_mr) {
      min_dist = std::min(min_dist, Geom::distance(left_pt, right_pt));
    }
  }
  return min_dist;
}

Area* BoundSkewTree::merge(Area* left, Area* right)
{
  auto* parent = new Area(++_id);
  parent->set_pattern(_pattern);
  parent->set_left(left);
  parent->set_right(right);
  left->set_parent(parent);
  right->set_parent(parent);
  return parent;
}

void BoundSkewTree::areaReset()
{
  _unmerged_nodes.clear();
  _unmerged_nodes.push_back(_root);
  ptReset(_root);
}

void BoundSkewTree::ptReset(Area* cur)
{
  auto pt = cur->get_location();
  pt.val = 0;
  cur->set_location(pt);
  if (cur->get_left()) {
    ptReset(cur->get_left());
  }
  if (cur->get_right()) {
    ptReset(cur->get_right());
  }
}
/**
 * @brief BiPartition method
 *
 */
void BoundSkewTree::biPartition()
{
  LOG_FATAL_IF(_unmerged_nodes.size() < 2) << "unmerged nodes size is less than 2";
  _root = biPartition(_unmerged_nodes);
  areaReset();
}
Area* BoundSkewTree::biPartition(std::vector<Area*>& areas)
{
  LOG_FATAL_IF(areas.empty()) << "areas is empty";

  if (areas.size() == 1) {
    return areas.front();
  }
  Area* parent = nullptr;
  if (areas.size() == 2) {
    parent = merge(areas.front(), areas.back());
  } else {
    auto [left_areas, right_areas] = octagonDivide(areas);
    auto* left = biPartition(left_areas);
    auto* right = biPartition(right_areas);
    parent = merge(left, right);
  }
  std::vector<Pt> pts;
  std::ranges::for_each(areas, [&](Area* area) { pts.push_back(area->get_location()); });
  auto loc = Geom::centerPt(pts);
  parent->set_location(loc);
  return parent;
}
std::pair<std::vector<Area*>, std::vector<Area*>> BoundSkewTree::octagonDivide(std::vector<Area*>& areas) const
{
  auto cap_sum = std::accumulate(areas.begin(), areas.end(), 0.0, [](double sum, Area* area) { return sum + area->get_cap_load(); });
  auto half_cap = 1.0 * cap_sum / 2;

  auto octagon = calcOctagon(areas);
  auto bound_areas = areaOnOctagonBound(areas, octagon);
  auto num = bound_areas.size();
  auto half_num = num / 2;

  bound_areas.insert(bound_areas.end(), bound_areas.begin(), bound_areas.begin() + half_num);

  auto calc_diameter = [](Area* area, const std::vector<Area*>& refs) {
    auto min_dist = std::numeric_limits<double>::max();
    auto max_dist = std::numeric_limits<double>::min();
    std::ranges::for_each(refs, [&area, &min_dist, &max_dist](const Area* ref) {
      auto dist = Geom::distance(area->get_location(), ref->get_location());
      min_dist = std::min(min_dist, dist);
      max_dist = std::max(max_dist, dist);
    });
    return max_dist + min_dist;
  };

  auto bound_diameter = [&](const std::vector<Area*>& ref) {
    auto oct = calcOctagon(ref);
    auto bound = areaOnOctagonBound(ref, oct);
    double max_dist = std::numeric_limits<double>::min();
    for (size_t i = 0; i < bound.size(); ++i) {
      for (size_t j = i + 1; j < bound.size(); ++j) {
        max_dist = std::max(max_dist, Geom::distance(bound[i]->get_location(), bound[j]->get_location()));
      }
    }
    return max_dist;
  };

  std::vector<Area*> left_set;
  std::vector<Area*> right_set;
  auto min_cost = std::numeric_limits<double>::max();

  for (size_t i = 0; i < num; ++i) {
    auto ref_set = std::vector<Area*>(bound_areas.begin() + i, bound_areas.begin() + i + half_num);
    std::ranges::for_each(areas, [&ref_set, &calc_diameter](Area* area) {
      auto pt = area->get_location();
      auto diameter = calc_diameter(area, ref_set);
      pt.val = diameter;
      area->set_location(pt);
    });
    std::ranges::sort(areas, [](Area* left, Area* right) { return left->get_location().val < right->get_location().val; });
    // find half cap idx, which diff of half_cap with left's cap is minimum
    int left_num = 0;
    double cap_count = 0;
    double diff = std::numeric_limits<double>::max();
    for (size_t j = 0; j < areas.size() - 1; ++j) {
      cap_count += areas[j]->get_cap_load();
      auto cur_diff = std::abs(cap_count - half_cap);
      if (cur_diff < diff) {
        diff = cur_diff;
        left_num = j + 1;
      }
    }
    auto left = std::vector<Area*>(areas.begin(), areas.begin() + left_num);
    auto right = std::vector<Area*>(areas.begin() + left_num, areas.end());
    auto cost = bound_diameter(left) + bound_diameter(right);
    if (cost < min_cost) {
      min_cost = cost;
      left_set = left;
      right_set = right;
    }
  }
  return {left_set, right_set};
}
std::vector<Pt> BoundSkewTree::calcOctagon(const std::vector<Area*>& areas) const
{
  auto x_p = std::numeric_limits<double>::min(), y_p = std::numeric_limits<double>::min(), ymx_p = std::numeric_limits<double>::min(),
       ypx_p = std::numeric_limits<double>::min();
  auto x_m = std::numeric_limits<double>::max(), y_m = std::numeric_limits<double>::max(), ymx_m = std::numeric_limits<double>::max(),
       ypx_m = std::numeric_limits<double>::max();
  std::ranges::for_each(areas, [&](const Area* area) {
    auto loc = area->get_location();
    auto x = loc.x;
    auto y = loc.y;
    x_p = std::max(x, x_p);
    x_m = std::min(x, x_m);
    y_p = std::max(y, y_p);
    y_m = std::min(y, y_m);
    ymx_p = std::max(y - x, ymx_p);
    ymx_m = std::min(y - x, ymx_m);
    ypx_p = std::max(y + x, ypx_p);
    ypx_m = std::min(y + x, ypx_m);
  });

  std::vector<Pt> octagon{Pt(y_p - ymx_p, y_p), Pt(ypx_p - y_p, y_p), Pt(x_p, ypx_p - x_p), Pt(x_p, x_p + ymx_m),
                          Pt(y_m - ymx_m, y_m), Pt(ypx_m - y_m, y_m), Pt(x_m, ypx_m - x_m), Pt(x_m, x_m + ymx_p)};
  Geom::convexHull(octagon);
  return octagon;
}
std::vector<Area*> BoundSkewTree::areaOnOctagonBound(const std::vector<Area*> areas, const std::vector<Pt>& octagon) const
{
  std::vector<Area*> result;
  std::ranges::for_each(areas, [&result, &octagon](Area* area) {
    for (size_t i = 0; i < octagon.size(); ++i) {
      auto line = Side<Pt>{octagon[i], octagon[(i + 1) % octagon.size()]};
      auto pt = area->get_location();
      if (Geom::onLine(pt, line)) {
        result.push_back(area);
        break;
      }
    }
  });
  auto center = Geom::centerPt(octagon);
  std::ranges::for_each(areas, [&center](Area* area) {
    auto pt = area->get_location();
    auto arc_tan2 = std::atan2(pt.y - center.y, pt.x - center.x);
    if (arc_tan2 < 0) {
      arc_tan2 += 2 * std::numbers::pi;
    }
    pt.val = arc_tan2;
    area->set_location(pt);
  });
  std::ranges::sort(result, [](Area* left, Area* right) { return left->get_location().val < right->get_location().val; });
  return result;
}
/**
 * @brief BiCluster method
 *
 */
void BoundSkewTree::biCluster()
{
  LOG_FATAL_IF(_unmerged_nodes.size() < 2) << "unmerged nodes size is less than 2";
  _root = biCluster(_unmerged_nodes);
  areaReset();
}
Area* BoundSkewTree::biCluster(const std::vector<Area*>& areas)
{
  LOG_FATAL_IF(areas.empty()) << "areas is empty";

  if (areas.size() == 1) {
    return areas.front();
  }
  Area* parent = nullptr;
  if (areas.size() == 2) {
    parent = merge(areas.front(), areas.back());
  } else {
    auto clusters = kMeansPlus(areas, 2);
    auto* left = biCluster(clusters.front());
    auto* right = biCluster(clusters.back());
    parent = merge(left, right);
  }
  std::vector<Pt> pts;
  std::ranges::for_each(areas, [&](Area* area) { pts.push_back(area->get_location()); });
  auto loc = Geom::centerPt(pts);
  parent->set_location(loc);
  return parent;
}
std::vector<std::vector<Area*>> BoundSkewTree::kMeansPlus(const std::vector<Area*>& areas, const size_t& k, const int& seed,
                                                          const size_t& max_iter) const
{
  std::vector<std::vector<Area*>> best_clusters(k);

  std::vector<Pt> centers;
  size_t num_instances = areas.size();
  std::vector<int> assignments(num_instances);

  // Randomly choose first center from instances
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<> dis(0, num_instances - 1);
  auto loc = areas[dis(gen)]->get_location();
  centers.emplace_back(loc);
  // Choose k-1 remaining centers using kmeans++ algorithm
  while (centers.size() < k) {
    std::vector<double> distances(num_instances, std::numeric_limits<double>::max());
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = Geom::distance(areas[i]->get_location(), centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }
    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    auto select_loc = areas[selected_index]->get_location();
    centers.emplace_back(select_loc);
  }

  size_t num_iterations = 0;
  double mss = std::numeric_limits<double>::max();
  while (num_iterations++ < max_iter) {
    // Assignment step
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = Geom::distance(areas[i]->get_location(), centers[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_center_index = j;
        }
      }
      assignments[i] = min_center_index;
    }
    // Update step
    std::vector<Pt> new_centers(k, Pt(0, 0));
    std::vector<int> center_counts(k, 0);
    for (size_t i = 0; i < num_instances; i++) {
      int center_index = assignments[i];
      new_centers[center_index] += areas[i]->get_location();
      center_counts[center_index]++;
    }
    for (size_t i = 0; i < k; i++) {
      if (center_counts[i] > 0) {
        new_centers[i] /= center_counts[i];
      }
    }
    centers = new_centers;
    // Check mss
    double new_mss = 0;
    for (size_t i = 0; i < num_instances; i++) {
      int center_index = assignments[i];
      new_mss += Geom::distance(areas[i]->get_location(), centers[center_index]);
    }
    // update clustering
    if (new_mss < mss) {
      best_clusters.clear();
      best_clusters.resize(k);
      mss = new_mss;
      for (size_t i = 0; i < num_instances; i++) {
        int center_index = assignments[i];
        best_clusters[center_index].push_back(areas[i]);
      }
    }
  }
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Area*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}

void BoundSkewTree::bottomUp()
{
  switch (_topo_type) {
    case TopoType::kBiCluster:
      bottomUpTopoBased();
      break;
    case TopoType::kBiPartition:
      bottomUpTopoBased();
      break;
    case TopoType::kInputTopo:
      bottomUpTopoBased();
      break;
    case TopoType::kGreedyDist:
      bottomUpAllPairBased();
      break;
    case TopoType::kGreedyMerge:
      bottomUpAllPairBased();
      break;
    default:
      LOG_FATAL << "topo type is not supported";
      break;
  }
}
void BoundSkewTree::bottomUpAllPairBased()
{
  // none input topo
  while (_unmerged_nodes.size() > 1) {
    // switch cost_func by topo_type
    CostFunc cost_func;
    switch (_topo_type) {
      case TopoType::kGreedyDist:
        cost_func = [&](Area* left, Area* right) { return distanceCost(left, right); };
        break;
      case TopoType::kGreedyMerge:
        cost_func = [&](Area* left, Area* right) { return mergeCost(left, right); };
        break;
      default:
        LOG_FATAL << "topo type is not supported";
        break;
    }
    auto best_match = getBestMatch(cost_func);
    auto* left = best_match.left;
    auto* right = best_match.right;
    auto* parent = new Area(++_id);
    // random select RCpattern
    parent->set_pattern(_pattern);
    merge(parent, left, right);
    // erase left and right
    _unmerged_nodes.erase(
        std::remove_if(_unmerged_nodes.begin(), _unmerged_nodes.end(), [&](Area* node) { return node == left || node == right; }),
        _unmerged_nodes.end());
    _unmerged_nodes.push_back(parent);
  }
  _root = _unmerged_nodes.front();
}
void BoundSkewTree::bottomUpTopoBased()
{
  switch (_topo_type) {
    case TopoType::kBiCluster:
      biCluster();
      break;
    case TopoType::kBiPartition:
      biPartition();
      break;
    case TopoType::kInputTopo:
      break;
    default:
      LOG_FATAL << "topo type is not supported";
      break;
  }
  recursiveBottomUp(_root);
}
void BoundSkewTree::recursiveBottomUp(Area* cur)
{
  auto* left = cur->get_left();
  auto* right = cur->get_right();
  if (left && right) {
    recursiveBottomUp(left);
    recursiveBottomUp(right);
    merge(cur, left, right);
  }
}
void BoundSkewTree::topDown()
{
  // set root location
  Pt root_loc;
  auto mr = _root->get_mr();
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
  auto delay_left = ptDelayIncrease(pt, left_pt, cur->get_edge_len(kLeft), left->get_cap_load(), _pattern);
  auto delay_right = ptDelayIncrease(pt, right_pt, cur->get_edge_len(kRight), right->get_cap_load(), _pattern);
  pt.min = std::min(left_pt.min + delay_left, right_pt.min + delay_right);
  pt.max = std::max(left_pt.max + delay_left, right_pt.max + delay_right);
  LOG_FATAL_IF(ptSkew(pt) > _skew_bound + 100 * kEpsilon) << "skew is so larger than skew bound, skew: " << ptSkew(pt);
  if (ptSkew(pt) > _skew_bound + kEpsilon) {
    LOG_WARNING << cur->get_name() << " max delay: " << pt.max << " min delay: " << pt.min;
    LOG_WARNING << "skew is larger than skew bound with error: " << ptSkew(pt) - _skew_bound;
    pt.min = pt.max - _skew_bound + kEpsilon;
  }
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
  if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan && left_type != LineType::kManhattan && right_type != LineType::kManhattan) {
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
  auto new_js = _join_segment;
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    auto other_mr = other_side == kLeft ? left->get_mr() : right->get_mr();
    auto relative_type = Geom::lineRelative(getJsLine(kLeft), getJsLine(kRight), other_side);
    for (auto pt : other_mr) {
      Geom::calcRelativeCoord(pt, relative_type, parent->get_radius());
      for (size_t i = 0; i < _join_segment[side].size() - 1; ++i) {
        Line line = {_join_segment[side][i], _join_segment[side][i + 1]};
        if (Geom::onLine(pt, line) && !Geom::isSame(pt, _join_segment[side][i]) && !Geom::isSame(pt, _join_segment[side][i + 1])) {
          calcPtDelays(nullptr, pt, line);
          new_js[side].push_back(pt);
          break;
        }
      }
    }
    Geom::sortPtsByFront(new_js[side]);
  }
  FOR_EACH_SIDE(side) { _join_segment[side] = new_js[side]; }
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
  Side<double> delay_from = {ptDelayIncrease(_join_segment[kLeft][kHead], _join_segment[kRight][kHead], left->get_cap_load(), _pattern),
                             ptDelayIncrease(_join_segment[kLeft][kHead], _join_segment[kRight][kHead], right->get_cap_load(), _pattern)};
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
    Geom::uniquePtsLoc(_join_region[side]);
  }
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    // add JR turn points which delay slope is changed
    auto n = _join_region[side].size() - 1;
    for (size_t i = 0; i < n; ++i) {
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
    Geom::uniquePtsLoc(_join_region[side]);
  }
  FOR_EACH_SIDE(side)
  {
    // remove redundant turn points which have same slope
    for (size_t i = 0; i < _join_region[side].size() - 1; ++i) {
      auto pt1 = _join_region[side][i];
      auto pt2 = _join_region[side][i + 1];
      auto dist = Geom::distance(pt1, pt2);
      LOG_FATAL_IF(Equal(dist, 0)) << "distance is zero";
      _join_region[side][i].val = (ptSkew(pt2) - ptSkew(pt1)) / dist;
    }
    // remove redundant turn points which skew slope is not strictly monotone increasing
    Pts incr_pts = {_join_region[side].front()};
    for (size_t j = 1; j < _join_region[side].size() - 1; ++j) {
      auto cur_val = incr_pts.back().val;
      auto next_val = _join_region[side][j].val;
      LOG_FATAL_IF(cur_val > next_val + 100 * kEpsilon)
          << "cur_val: " << cur_val << "> next_val: " << next_val << ", skew slope is not strictly monotone increasing";
      if (next_val > cur_val) {
        incr_pts.push_back(_join_region[side][j]);
      }
    }
    incr_pts.push_back(_join_region[side].back());
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
    LOG_FATAL_IF(_join_segment[side].front().y + kEpsilon < _join_segment[side].back().y) << "join segment direction is not correct";
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
  FOR_EACH_SIDE(end_side) { _bal_points[end_side].clear(); }
  if (Equal(cur->get_radius(), 0)) {
    return;
  }
  FOR_EACH_SIDE(end_side)
  {
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
      calcBalBetweenPts(left_pt, right_pt, timing_type, bal_ref_side, dist_to_left, dist_to_right, bal_pt, _pattern);
      if (!Equal(dist_to_left, 0) && !Equal(dist_to_right, 0)) {
        updatePtDelaysByEndSide(cur, end_side, bal_pt);
        _bal_points[end_side].push_back(bal_pt);
      }
    }
  }
}
void BoundSkewTree::calcBalBetweenPts(Pt& p1, Pt& p2, const size_t& timing_type, const size_t& bal_ref_side, double& d1, double& d2,
                                      Pt& bal_pt, const RCPattern& pattern) const
{
  auto h = std::abs(p1.x - p2.x);
  auto v = std::abs(p1.y - p2.y);
  if (Equal(h, 0) || Equal(v, 0)) {
    calcBalPtOnLine(p1, p2, timing_type, d1, d2, bal_pt, pattern);
  } else if (p1.x <= p2.x) {
    calcBalPtNotOnLine(p1, p2, timing_type, bal_ref_side, d1, d2, bal_pt, pattern);
  } else {
    calcBalPtNotOnLine(p2, p1, timing_type, bal_ref_side, d2, d1, bal_pt, pattern);
  }
}
void BoundSkewTree::calcBalPtOnLine(Pt& p1, Pt& p2, const size_t& timing_type, double& d1, double& d2, Pt& bal_pt,
                                    const RCPattern& pattern) const
{
  auto h = std::abs(p1.x - p2.x);
  auto v = std::abs(p1.y - p2.y);
  LOG_FATAL_IF(!Equal(h, 0) && !Equal(v, 0)) << "h and v are not zero, which balance point is not on line";

  auto delay1 = timing_type == kMin ? p1.min : p1.max;
  auto delay2 = timing_type == kMin ? p2.min : p2.max;
  auto r = Equal(h, 0) ? _unit_v_res : _unit_h_res;
  auto c = Equal(h, 0) ? _unit_v_cap : _unit_h_cap;
  calcMergeDist(r, c, p1.val, delay1, p2.val, delay2, h + v, d1, d2);
  calcPtCoordOnLine(p1, p2, d1, d2, bal_pt);
  double incr_delay1 = 0;
  double incr_delay2 = 0;
  if (Equal(h, 0)) {
    incr_delay1 = calcDelayIncrease(0, d1, p1.val, pattern);
    incr_delay2 = calcDelayIncrease(0, d2, p2.val, pattern);
  } else {
    incr_delay1 = calcDelayIncrease(d1, 0, p1.val, pattern);
    incr_delay2 = calcDelayIncrease(d2, 0, p2.val, pattern);
  }
  bal_pt.min = std::min(p1.min + incr_delay1, p2.min + incr_delay2);
  bal_pt.max = std::max(p1.max + incr_delay1, p2.max + incr_delay2);
}
void BoundSkewTree::calcBalPtNotOnLine(Pt& p1, Pt& p2, const size_t& timing_type, const size_t& bal_ref_side, double& d1, double& d2,
                                       Pt& bal_pt, const RCPattern& pattern) const
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
    auto incr_delay = calcDelayIncrease(h, v, p2.val, pattern);
    temp_pt.min = p2.min + incr_delay;
    temp_pt.max = p2.max + incr_delay;
    temp_pt.val = p2.val + _unit_h_cap * h + _unit_v_cap * v;
    calcBalPtOnLine(p1, temp_pt, timing_type, d1, d2, bal_pt, pattern);
    LOG_FATAL_IF(d1 > kEpsilon) << "dist to p1 should be zero";
    auto new_incr_delay = calcDelayIncrease(0, d2, temp_pt.val, pattern);
    LOG_FATAL_IF(!Equal(delay1, incr_delay + new_incr_delay + delay2)) << "delay is not equal";
    d2 += h + v;
  } else if (x > h) {
    LOG_FATAL_IF(y <= v) << "y: " << y << " is not greater than v: " << v;
    auto temp_pt = p2;
    auto incr_delay = calcDelayIncrease(h, v, p1.val, pattern);
    temp_pt.min = p1.min + incr_delay;
    temp_pt.max = p1.max + incr_delay;
    temp_pt.val = p1.val + _unit_h_cap * h + _unit_v_cap * v;
    calcBalPtOnLine(temp_pt, p2, timing_type, d1, d2, bal_pt, pattern);
    LOG_FATAL_IF(d2 > kEpsilon) << "dist to p2 should be zero";
    auto new_incr_delay = calcDelayIncrease(0, d1, temp_pt.val, pattern);
    LOG_FATAL_IF(!Equal(delay2, incr_delay + new_incr_delay + delay1)) << "delay is not equal";
    d1 += h + v;
  } else {
    LOG_FATAL_IF(y < -kEpsilon || y > v + kEpsilon) << "y: " << y << " is not in range [0, " << v << "]";
    bal_pt.x = p1.x + x;
    bal_pt.y = p1.y < p2.y ? p1.y + y : p1.y - y;
    auto incr_delay1 = calcDelayIncrease(x, y, p1.val, pattern);
    auto incr_delay2 = calcDelayIncrease(h - x, v - y, p2.val, pattern);
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
    LOG_FATAL_IF(y > v + kEpsilon) << "y: " << y << " is larger than v: " << v;
  } else {
    // assume (h-x, y) and (x, v-y), then set x = 0
    t = delay2 - delay1 + _K[kV] * v * v - _K[kH] * h * h + _unit_v_res * v * cap2 - _unit_h_res * h * cap1;
    y = t / r;
    LOG_FATAL_IF(y < -kEpsilon) << "y: " << y << " is less than 0";
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
  Geom::sortPtsByValDec(mr_pts);
  Geom::uniquePtsLoc(mr_pts);
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
  size_t jr_right_id = _join_region[side].size() - 2;
  if (_fms_points[kTail].empty() && ptSkew(p) < ptSkew(q)) {
    for (; jr_right_id >= jr_left_id; --jr_right_id) {
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
    auto relative_type = Geom::lineRelative(getJsLine(kLeft), getJsLine(kRight), side);
    Geom::calcRelativeCoord(pt, relative_type, dist);
    auto x = std::abs(pt.x - _join_region[side][idx].x);
    auto y = std::abs(pt.y - _join_region[side][idx].y);
    LOG_FATAL_IF(!Equal(x, 0) && !Equal(y, 0)) << "not horizontal or vertical";
    auto incr_delay = side == kLeft ? calcDelayIncrease(x, y, cur->get_left()->get_cap_load(), _pattern)
                                    : calcDelayIncrease(x, y, cur->get_right()->get_cap_load(), _pattern);
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
    right_pt.max = left_pt.max - delta - calcDelayIncrease(h, v, right_pt.val, _pattern);
    double d1 = 0, d2 = 0;
    Pt bal_pt;
    calcBalBetweenPts(left_pt, right_pt, kMax, kX, d1, d2, bal_pt, _pattern);
    LOG_FATAL_IF(d1 > kEpsilon) << "dist to left_pt should be zero";
    cur->set_edge_len(kLeft, 0);
    cur->set_edge_len(kRight, d2);
  } else {
    left_pt.max = right_pt.max - delta - calcDelayIncrease(h, v, left_pt.val, _pattern);
    double d1 = 0, d2 = 0;
    Pt bal_pt;
    calcBalBetweenPts(left_pt, right_pt, kMax, kX, d1, d2, bal_pt, _pattern);
    LOG_FATAL_IF(d2 > kEpsilon) << "dist to right_pt should be zero";
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
  if (mr.size() == 4 && isTrrArea(child)) {
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
      // kHead loc is same as kTail loc
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
  } else {
    parent->set_edge_len(side, Geom::distance(parent_loc, child_loc));
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
    if (Geom::lineType(mr[0], mr[1]) == LineType::kManhattan) {
      Geom::lineToMs(trr_left, mr[0], mr[1]);
    } else {
      LOG_FATAL_IF(Geom::lineType(mr[2], mr[1]) != LineType::kManhattan) << "mr is not manhattan";
      Geom::lineToMs(trr_left, mr[1], mr[2]);
    }
    Trr trr_right;
    if (Geom::lineType(mr[2], mr[3]) == LineType::kManhattan) {
      Geom::lineToMs(trr_right, mr[2], mr[3]);
    } else {
      LOG_FATAL_IF(Geom::lineType(mr[0], mr[3]) != LineType::kManhattan) << "mr is not manhattan";
      Geom::lineToMs(trr_right, mr[3], mr[0]);
    }
    trr = trr_left;
    trr.enclose(trr_right);
    return;
  }
  LOG_FATAL << "mr size is not 1, 2 or 4";
}

void BoundSkewTree::inputTopologyConvert()
{
  std::stack<Area*> stack;
  stack.push(_root);
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();

    if (cur->get_right()) {
      stack.push(cur->get_right());
    }
    if (cur->get_left()) {
      stack.push(cur->get_left());
    }

    auto pt = cur->get_location();
    auto loc = Point(pt.x * _db_unit, pt.y * _db_unit);
    auto* node = _node_map[cur->get_name()];
    if (node == nullptr) {
      node = new Node(cur->get_name(), loc);
      _node_map.insert({node->get_name(), node});
    }

    if (node->isPin() && node->isDriver()) {
      _root_buf->set_location(loc);
    } else {
      node->set_location(loc);
    }
    auto* parent = cur->get_parent();
    if (parent) {
      auto direction = parent->get_left() == cur ? kLeft : kRight;
      auto edge_len = parent->get_edge_len(direction);
      auto snake = edge_len - Geom::distance(parent->get_location(), cur->get_location());
      LOG_FATAL_IF(snake < -kEpsilon) << "snake is less than 0";
      node->set_required_snake(snake);
    }
  }
}

void BoundSkewTree::noneInputTopologyConvert()
{
  std::stack<Area*> stack;
  stack.push(_root);
  // pre-order build Node, leaf node will in _node_map
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();

    if (cur->get_right()) {
      stack.push(cur->get_right());
    }
    if (cur->get_left()) {
      stack.push(cur->get_left());
    }

    auto pt = cur->get_location();
    auto loc = Point(pt.x * _db_unit, pt.y * _db_unit);
    auto* parent = cur->get_parent();
    if (parent == nullptr) {
      // is root, make buffer
      _root_buf = TreeBuilder::genBufInst(_net_name, loc);
      cur->set_name(_root_buf->get_name());
      _node_map.insert({_root_buf->get_name(), _root_buf->get_driver_pin()});
      continue;
    }

    Node* node = nullptr;
    Node* parent_node = _node_map[parent->get_name()];
    LOG_FATAL_IF(parent_node == nullptr) << "node " << parent->get_name() << " is not in _node_map";
    if (cur->get_left() == nullptr && cur->get_right() == nullptr) {
      // is load pin, find from _node_map
      node = _node_map[cur->get_name()];
      LOG_FATAL_IF(node == nullptr) << "node " << cur->get_name() << " is not in _node_map";
    } else {
      // is steiner node
      node = new Node(cur->get_name(), loc);
      _node_map.insert({node->get_name(), node});
    }
    parent_node->add_child(node);
    node->set_parent(parent_node);
    auto direction = parent->get_left() == cur ? kLeft : kRight;
    auto edge_len = parent->get_edge_len(direction);
    auto snake = edge_len - Geom::distance(parent->get_location(), cur->get_location());
    LOG_FATAL_IF(snake < -kEpsilon) << "snake is less than 0";
    node->set_required_snake(snake);
  }
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
  auto delay_left = ptDelayIncrease(pt, left_line[end_side], cur->get_left()->get_cap_load(), _pattern);
  auto delay_right = ptDelayIncrease(pt, right_line[end_side], cur->get_right()->get_cap_load(), _pattern);
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
        << "endpoint should be same, left head: [" << left_line[kHead].x << ", " << left_line[kHead].y << "], left tail: ["
        << left_line[kTail].x << ", " << left_line[kTail].y << "], right head: [" << right_line[kHead].x << ", " << right_line[kHead].y
        << "], right tail: [" << right_line[kTail].x << ", " << right_line[kTail].y << "]";

    auto delay_left = ptDelayIncrease(left_line[kHead], pt, cur->get_left()->get_cap_load(), _pattern);
    auto delay_right = ptDelayIncrease(right_line[kHead], pt, cur->get_right()->get_cap_load(), _pattern);
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
double BoundSkewTree::ptDelayIncrease(Pt& p1, Pt& p2, const double& len, const double& cap, const RCPattern& pattern) const
{
  auto h = std::abs(p1.x - p2.x);
  auto v = std::abs(p1.y - p2.y);
  LOG_FATAL_IF(!Equal(len, h + v) && len < h + v) << "len is less than h + v";
  double delay = 0;
  if (Equal(h, 0)) {
    delay = calcDelayIncrease(0, len, cap, pattern);
  } else if (Equal(v, 0)) {
    delay = calcDelayIncrease(len, 0, cap, pattern);
  } else {
    delay = calcDelayIncrease(h, v, cap, pattern);
    if (len > h + v) {
      delay += calcDelayIncrease(0, len - h - v, cap + _unit_h_cap * h + _unit_v_cap * v, pattern);
    }
  }
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
    case RCPattern::kSingle:
      delay = _unit_h_res * (x + y) * (_unit_h_cap * (x + y) / 2 + cap);
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
  return Line{jr[kHead], jr[kTail]};
}
Line BoundSkewTree::getJsLine(const size_t& side) const
{
  auto js = _join_segment[side];
  return Line{js[kHead], js[kTail]};
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
  // LOG_ERROR_IF(pt.min <= -kEpsilon) << "pt min delay is negative";
  LOG_FATAL_IF(pt.max - pt.min <= -kEpsilon) << "pt skew is negative";
  if (pt.min < -kEpsilon) {
    pt.min = 0;
  }
  if (pt.max < pt.min + kEpsilon) {
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
void BoundSkewTree::writePy(const std::vector<Pt>& pts, const std::string& file) const
{
  auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  std::ofstream ofs(dir + "/" + file + ".py");
  ofs.setf(std::ios::fixed, std::ios::floatfield);
  ofs.precision(4);
  ofs << "import matplotlib.pyplot as plt\n";
  ofs << "import numpy as np\n";
  ofs << "x = [";
  for (auto pt : pts) {
    ofs << pt.x << ", ";
  }
  ofs << pts.front().x << "]\n";
  ofs << "y = [";
  for (auto pt : pts) {
    ofs << pt.y << ", ";
  }
  ofs << pts.front().y << "]\n";
  ofs << "plt.plot(x, y)\n";
  ofs << "plt.show()\n";
  ofs << "plt.savefig('" + file + ".png')\n";
  ofs.close();
}

void BoundSkewTree::writePy(Area* area, const std::string& file) const
{
  auto dir = CTSAPIInst.get_config()->get_work_dir() + "/file";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  std::ofstream ofs(dir + "/" + file + ".py");
  ofs.setf(std::ios::fixed, std::ios::floatfield);
  ofs.precision(4);
  ofs << "import matplotlib.pyplot as plt\n";
  ofs << "import numpy as np\n";
  std::stack<Area*> stack;
  stack.push(area);
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();
    if (cur->get_right()) {
      stack.push(cur->get_right());
    }
    if (cur->get_left()) {
      stack.push(cur->get_left());
    }
    if (cur->get_mr().empty()) {
      continue;
    }

    ofs << "x = [";
    for (auto pt : cur->get_convex_hull()) {
      ofs << pt.x << ", ";
    }
    ofs << cur->get_convex_hull().front().x << "]\n";
    ofs << "y = [";
    for (auto pt : cur->get_convex_hull()) {
      ofs << pt.y << ", ";
    }
    ofs << cur->get_convex_hull().front().y << "]\n";

    ofs << "plt.plot(x, y, \"--b\", linewidth=1)\n";

    ofs << "x = [";
    for (auto pt : cur->get_mr()) {
      ofs << pt.x << ", ";
    }
    ofs << cur->get_mr().front().x << "]\n";
    ofs << "y = [";
    for (auto pt : cur->get_mr()) {
      ofs << pt.y << ", ";
    }
    ofs << cur->get_mr().front().y << "]\n";
    ofs << "plt.plot(x, y, \"o\", color='red', markersize=1)\n";

    auto center = Geom::centerPt(cur->get_convex_hull());
    ofs << "plt.text(" << center.x << ", " << center.y << ", '" << cur->get_name() << "', fontsize=4)\n\n";
  }
  ofs << "plt.savefig('" + file + ".png', dpi=900)\n";
  ofs << "plt.show()\n";
  ofs.close();
}
}  // namespace bst
}  // namespace icts