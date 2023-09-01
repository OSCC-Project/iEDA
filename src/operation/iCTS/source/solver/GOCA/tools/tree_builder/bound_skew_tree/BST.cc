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
 * @file BST.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "BST.hh"

#include "GeomOperator.hh"
#include "TreeBuilder.hh"
namespace icts {
/**
 * @brief bst flow
 *
 */
void BST::run()
{
  timingInit();
  // preBuffering();
  while (_unmerged_nodes.size() > 1) {
    auto skew_cost = [&](Node* left, Node* right) { return distanceCost(left, right); };
    auto best_match = getBestMatch(skew_cost);
    auto left = best_match.left;
    auto right = best_match.right;
    merge(left, right);
    auto merged_node = _unmerged_nodes.back();
    if (merged_node->get_sub_len() > 0.35 * TimingPropagator::getMaxLength()) {
      _unmerged_nodes.erase(_unmerged_nodes.end() - 1);
      auto* buf_node = buffering(merged_node);
      _unmerged_nodes.push_back(buf_node);
    }
  }
  // buffering root
  auto root = _unmerged_nodes[0];
  updateTiming(root);
  if (_root_guide != std::nullopt) {
    auto mr = _mr_map[root];
    auto guide_loc = *(_root_guide);
    if (pgl::contains(mr, guide_loc)) {
      root->set_location(guide_loc);
    } else {
      auto min_loc_it = std::min_element(mr.begin(), mr.end(), [&](const Point& p1, const Point& p2) {
        return TimingPropagator::calcDist(p1, guide_loc) < TimingPropagator::calcDist(p2, guide_loc);
      });
      root->set_location(*min_loc_it);
    }
  }
  auto* final_root = root->isBufferPin() ? root : buffering(root);
  // topdown
  topdown(final_root);
}

MergeMatch BST::getBestMatch(MergeCostFunc cost_func) const
{
  // for all pair combinations of _unmerged_nodes, find the best match
  double best_cost = std::numeric_limits<double>::max();
  MergeMatch best_match(nullptr, nullptr, best_cost);
  for (size_t i = 0; i < _unmerged_nodes.size(); ++i) {
    for (size_t j = i + 1; j < _unmerged_nodes.size(); ++j) {
      double cost = cost_func(_unmerged_nodes[i], _unmerged_nodes[j]);
      if (cost < best_cost) {
        best_cost = cost;
        best_match = MergeMatch(_unmerged_nodes[i], _unmerged_nodes[j], best_cost);
      }
    }
  }
  return best_match;
}

double BST::skewCost(Node* left, Node* right)
{
  joinSegment(left, right);
  auto length = TimingPropagator::calcLen(left, right);
  auto length_i = endPointByZeroSkew(left, right);
  if (length_i < 0) {
    return left->get_max_delay() + (left->get_max_delay() - right->get_max_delay()) + 0.5 * _unit_res * _unit_cap * length_i * length_i
           - _unit_res * length_i * (left->get_cap_load() + right->get_cap_load()) / 2;
  }
  if (length_i > length) {
    return right->get_max_delay() + (right->get_max_delay() - left->get_max_delay())
           + 0.5 * _unit_res * _unit_cap * (length_i - length) * (length_i - length)
           + _unit_res * (length_i - length) * (left->get_cap_load() + right->get_cap_load()) / 2;
  }
  return left->get_max_delay() + 0.5 * _unit_res * _unit_cap * length_i * length_i + _unit_res * length_i * left->get_cap_load();
}

double BST::distanceCost(Node* left, Node* right) const
{
  auto len = TimingPropagator::calcLen(left, right);
  return len;
}

void BST::updateTiming(Node* node) const
{
  // update timing
  std::vector<Inst*> loads;
  node->preOrder([&](Node* cur) {
    if (cur->isPin() && cur->isLoad()) {
      auto* pin = dynamic_cast<Pin*>(cur);
      auto* inst = pin->get_inst();
      loads.push_back(inst);
    }
  });
  std::ranges::for_each(loads, [&](Inst* inst) {
    if (inst->isSink()) {
      return;
    }
    auto* driver_pin = inst->get_driver_pin();
    auto* net = driver_pin->get_net();
    if (net) {
      TimingPropagator::update(net);
    }
    auto* load_pin = inst->get_load_pin();
    if (load_pin->get_slew_in() == 0) {
      TimingPropagator::initLoadPinDelay(load_pin);
    } else {
      TimingPropagator::updateCellDelay(inst);
    }
  });
  TimingPropagator::updateNetLen(node);
  TimingPropagator::updateCapLoad(node);
  TimingPropagator::updateWireDelay(node);
  if (!TimingPropagator::skewFeasible(node, _skew_bound)) {
    if (node->get_max_delay() - node->get_min_delay() - _skew_bound < 1e-6) {
      node->set_max_delay(node->get_min_delay() + _skew_bound);
    }
  }
}

void BST::merge(Node* left, Node* right)
{
  LOG_FATAL_IF(!TimingPropagator::skewFeasible(left, _skew_bound) || !TimingPropagator::skewFeasible(right, _skew_bound))
      << "Input node has skew violation";
  clearNode(left);
  clearNode(right);
  joinSegment(left, right);
  auto left_js = _js_map[left];
  auto right_js = _js_map[right];
  _mr_map[left] = Polygon({left_js.low(), left_js.high()});
  _mr_map[right] = Polygon({right_js.low(), right_js.high()});
  // if (_js_map.at(left) == _js_map.at(right)) {
  //   // fuse left and right
  //   fuse(left, right);
  //   return;
  // }
  // normal merge
  auto mr = calcMergeRegion(left, right);
  auto loc = Point(-1, -1);
  if (mr.empty()) {
    if (left->get_max_delay() > right->get_max_delay()) {
      loc = left->get_location();
    } else {
      loc = right->get_location();
    }
  } else {
    loc = pgl::center(mr);
  }
  auto* parent = new Node(loc);
  TreeBuilder::connect(parent, left);
  TreeBuilder::connect(parent, right);
  updateTiming(parent);
  if (!TimingPropagator::skewFeasible(parent, _skew_bound)) {
    parent = buffering(parent);  // will fix skew
    loc = parent->get_location();
    // mr = Polygon({loc});
  }
  _js_map[parent] = Segment(loc, loc);
  _mr_map[parent] = Polygon({loc});
  _unmerged_nodes.push_back(parent);
}

void BST::fuse(Node* left, Node* right)
{
  LOG_FATAL_IF(left->isPin() && right->isPin()) << "can't fuse two pin in clock tree";
  if (left->isPin()) {
    std::ranges::for_each(right->get_children(), [&](Node* right_child) { TreeBuilder::connect(left, right_child); });
    delete right;
    auto js = _js_map.at(left);
    auto mr = Polygon({js.low(), js.high()});
    _unmerged_nodes.push_back(left);
    _mr_map[left] = mr;
  } else {
    // right node is pin, or both left node and right node are steiner node
    std::ranges::for_each(left->get_children(), [&](Node* left_child) { TreeBuilder::connect(right, left_child); });
    delete left;
    auto js = _js_map.at(right);
    auto mr = Polygon({js.low(), js.high()});
    _unmerged_nodes.push_back(right);
    _mr_map[right] = mr;
  }
}

void BST::clearNode(Node* node)
{
  _unmerged_nodes.erase(std::remove_if(_unmerged_nodes.begin(), _unmerged_nodes.end(), [&](const Node* cur) { return cur == node; }),
                        _unmerged_nodes.end());
}

void BST::clearGeom(Node* node)
{
  _js_map.erase(node);
  _mr_map.erase(node);
}

void BST::joinSegment(Node* left, Node* right)
{
  auto mr_i = left->isPin() ? Polygon({left->get_location()}) : _mr_map.at(left);
  auto mr_j = right->isPin() ? Polygon({right->get_location()}) : _mr_map.at(right);
  auto intersection = GeomOperator::intersectionByBg(mr_i, mr_j);
  if (intersection.size() > 0) {
    Segment seg;
    if (intersection.size() == 1) {
      seg = Segment(intersection.get_points().front(), intersection.get_points().front());
    } else {
      pgl::longest_segment(seg, intersection.get_edges());
    }
    _js_map[left] = seg;
    _js_map[right] = seg;
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
    auto point = GeomOperator::intersectionPointByBg(join_seg_i, join_seg_j);
    auto seg = Segment(point, point);
    _js_map[left] = seg;
    _js_map[right] = seg;
    return;
  }
  auto seg_i = join_seg_i;
  auto seg_j = join_seg_j;
  if (seg_i.low() != seg_i.high()) {
    pgl::tilted_rect_region(trr, seg_j, radius);
    auto js_i_points = GeomOperator::intersectionPointByBg(trr, seg_i);
    join_seg_i
        = js_i_points.size() == 2 ? Segment(js_i_points.front(), js_i_points.back()) : Segment(js_i_points.front(), js_i_points.front());
  }
  if (seg_j.low() != seg_j.high()) {
    pgl::tilted_rect_region(trr, seg_i, radius);
    auto js_j_points = GeomOperator::intersectionPointByBg(trr, seg_j);
    join_seg_j
        = js_j_points.size() == 2 ? Segment(js_j_points.front(), js_j_points.back()) : Segment(js_j_points.front(), js_j_points.front());
  }
  _js_map[left] = join_seg_i;
  _js_map[right] = join_seg_j;
}

double BST::endPointByZeroSkew(Node* left, Node* right, const std::optional<double>& init_delay_i,
                               const std::optional<double>& init_delay_j) const
{
  auto delay_i = init_delay_i.value_or(left->get_max_delay());
  auto delay_j = init_delay_j.value_or(right->get_max_delay());
  auto length = TimingPropagator::calcLen(left, right);
  auto factor = (left->get_cap_load() + right->get_cap_load() + _unit_cap * length);
  auto length_to_i
      = (delay_j - delay_i) / _unit_res / factor + (right->get_cap_load() * length + 0.5 * std::pow(length, 2) * _unit_cap) / factor;
  return length_to_i;
}

std::pair<double, double> BST::calcEndpointLoc(Node* left, Node* right) const
{
  auto left_min = left->get_min_delay();
  auto left_max = left->get_max_delay();
  auto right_min = right->get_min_delay();
  auto right_max = right->get_max_delay();
  auto ep_l = endPointByZeroSkew(left, right, left_min + _skew_bound, right_max);
  auto ep_r = endPointByZeroSkew(left, right, left_max, right_min + _skew_bound);
  if (ep_l > ep_r) {
    if (std::fabs(ep_l - ep_r) < 1e-6) {
      ep_r = ep_l;
    }
  }
  return std::make_pair(ep_l, ep_r);
}

Polygon BST::calcMergeRegion(Node* left, Node* right)
{
  joinSegment(left, right);
  updateTiming(left);
  updateTiming(right);
  auto ep_pair = calcEndpointLoc(left, right);
  auto ep_l = ep_pair.first;
  auto ep_r = ep_pair.second;
  auto length = TimingPropagator::calcLen(left, right);
  auto left_js = _js_map.at(left);
  auto right_js = _js_map.at(right);
  if (ep_l > length || ep_r < 0) {
    return Polygon({});
  }
  if (length == 0 && (ep_r >= 0)) {
    return Polygon({right_js.low(), right_js.high()});
  }
  ep_l = ep_l < 0 ? 0 : ep_l;
  ep_r = ep_r > length ? length : ep_r;

  Polygon sdr;
  GeomOperator::calcSDR(sdr, left_js, right_js);
  // Type 1: single point
  if (sdr.size() == 1) {
    return sdr;
  }
  // Type 2: rectilinear line
  int left_radius = std::floor(ep_r * _db_unit);
  int right_radius = std::floor((length - ep_l) * _db_unit);
  if (left_radius + right_radius < length * _db_unit) {
    ++right_radius;
  }
  if (sdr.size() == 2) {
    auto pair_point = pgl::cutSegment(Segment(sdr.get_points()[0], sdr.get_points()[1]), left_radius, right_radius);
    return Polygon({pair_point.first, pair_point.second});
  }
  // Type 3: polygon (maybe merge a line type polygon)
  Polygon left_rect;
  pgl::tilted_rect_region(left_rect, left_js, left_radius);
  Polygon right_rect;
  pgl::tilted_rect_region(right_rect, right_js, right_radius);
  auto bound = GeomOperator::intersectionByBg(left_rect, right_rect);
  // Type 2.1: bound is a line
  if (bound.size() == 2) {
    auto seg = Segment(bound.get_points()[0], bound.get_points()[1]);
    return GeomOperator::intersectionByBg(sdr, seg);
  }
  // Type 2.2: bound is a polygon
  return GeomOperator::intersectionByBg(bound, sdr);
}

bool BST::isBoundMerge(Node* left, Node* right)
{
  joinSegment(left, right);
  updateTiming(left);
  updateTiming(right);
  auto ep_pair = calcEndpointLoc(left, right);
  auto ep_l = ep_pair.first;
  auto ep_r = ep_pair.second;
  auto length = TimingPropagator::calcLen(left, right);
  if (length == 0 || ep_r <= 0 || ep_l >= length) {
    return true;
  }
  return false;
}

void BST::timingInit()
{
  std::ranges::for_each(_unmerged_nodes, [&](Node* node) {
    LOG_FATAL_IF(!node->isPin()) << "Node: " << node->get_name() << " is not Pin";
    auto* pin = dynamic_cast<Pin*>(node);
    if (pin->isBufferPin()) {
      TimingPropagator::initLoadPinDelay(pin);
    }
    auto loc = node->get_location();
    _js_map[node] = Segment(loc, loc);
    _mr_map[node] = Polygon({loc});
  });
}

void BST::preBuffering()
{
  if (_unmerged_nodes.size() < 8) {
    return;
  }
  auto delay_range = calcGlobalDelayRange();
  auto min_delay = delay_range.first;
  auto max_delay = delay_range.second;
  if (max_delay - min_delay > _skew_bound) {
    std::vector<Node*> lower_nodes;
    std::vector<Node*> higher_nodes;
    auto lower_bound = min_delay + _skew_bound * 0.25;
    auto higher_bound = max_delay - _skew_bound * 0.25;
    std::ranges::for_each(_unmerged_nodes, [&](Node* node) {
      auto node_min_delay = node->get_min_delay();
      auto node_max_delay = node->get_max_delay();
      if (node_min_delay < lower_bound && node_max_delay > higher_bound) {
        auto min_delta = lower_bound - node_min_delay;
        auto max_delta = node_max_delay - higher_bound;
        if (min_delta < max_delta) {
          lower_nodes.push_back(node);
        } else {
          higher_nodes.push_back(node);
        }
      } else if (node_min_delay < lower_bound) {
        lower_nodes.push_back(node);
      } else if (node_max_delay > higher_bound) {
        higher_nodes.push_back(node);
      }
    });
    // lower delay insert buffer
    std::ranges::for_each(lower_nodes, [&](Node* node) {
      clearNode(node);
      auto* new_node = buffering(node);
      _unmerged_nodes.push_back(new_node);
    });
    // higher delay amplify buffer
    std::ranges::for_each(higher_nodes, [&](Node* node) {
      auto* pin = dynamic_cast<Pin*>(node);
      auto* inst = pin->get_inst();
      TreeBuilder::amplifyBufferSize(inst, 2);
    });
    LOG_INFO << "Pre-Buffering info: lower delay insert " << lower_nodes.size() << " buffer(s), amplify " << higher_nodes.size()
             << " buffer(s).";
  }
}

Node* BST::buffering(Node* node)
{
  auto loc = node->get_location();
  auto net_name = CTSAPIInst.toString(_net_name, "_", CTSAPIInst.genId());

  Inst* buffer = nullptr;
  if (node->isSteiner()) {
    // case 1: steiner node
    // convert to buffer
    buffer = TreeBuilder::toBufInst(net_name, node);
  } else {
    // case 2: sink node or buffer node
    // generate a buffer and connect them
    buffer = TreeBuilder::genBufInst(net_name, loc);
    TreeBuilder::connect(buffer->get_driver_pin(), node);
  }

  buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  auto* driver_pin = buffer->get_driver_pin();

  TreeBuilder::place(buffer);
  auto* net = TimingPropagator::genNet(net_name, driver_pin);
  TimingPropagator::update(net);
  // skew violation by buffer resize
  while (!TimingPropagator::skewFeasible(driver_pin, _skew_bound)) {
    skewFix(driver_pin);
    TimingPropagator::updateLoads(net);
    TimingPropagator::update(net);
  }
  auto* load_pin = buffer->get_load_pin();
  TimingPropagator::initLoadPinDelay(load_pin);
  auto final_loc = buffer->get_location();
  _js_map[load_pin] = Segment(final_loc, final_loc);
  _mr_map[load_pin] = Polygon({final_loc});
  _insert_bufs.push_back(buffer);
  _nets.insert(net);
  return load_pin;
}

bool BST::amplifySubBufferSize(Node* node, const size_t& level) const
{
  // skew check
  bool is_ok = false;
  if (node->isBufferPin()) {
    auto* pin = dynamic_cast<Pin*>(node);
    auto* inst = pin->get_inst();
    auto* max_lib = TimingPropagator::getMaxSizeLib();
    if (inst->get_cell_master() == max_lib->get_cell_master()) {
      return false;
    }
    TreeBuilder::amplifyBufferSize(inst, level);
    is_ok = true;
  } else {
    auto find_buf = [&](Node* cur) {
      if (cur->isBufferPin()) {
        auto* pin = dynamic_cast<Pin*>(cur);
        auto* inst = pin->get_inst();
        auto* max_lib = TimingPropagator::getMaxSizeLib();
        if (inst->get_cell_master() != max_lib->get_cell_master()) {
          is_ok = true;
          TreeBuilder::amplifyBufferSize(inst, level);
        }
      }
    };
    node->preOrder(find_buf);
    updateTiming(node);
  }
  return is_ok;
}

bool BST::bufferResizing(Node* node) const
{
  bool can_fix = true;
  if (TimingPropagator::skewFeasible(node, _skew_bound)) {
    LOG_WARNING << "skew is feasible, no need to fix skew";
    return true;
  }

  auto max_delay_child = getMaxDelayChild(node);
  if (!TimingPropagator::skewFeasible(max_delay_child, _skew_bound)) {
    can_fix = bufferResizing(max_delay_child);
  } else {
    auto* max_lib = TimingPropagator::getMaxSizeLib();
    auto max_cell = max_lib->get_cell_master();
    std::vector<Inst*> to_be_amplify;
    auto find_sub_insts = [&](Node* cur) {
      if (cur->isPin()) {
        auto* pin = dynamic_cast<Pin*>(cur);
        auto* inst = pin->get_inst();
        if (inst->isSink() || inst->get_cell_master() == max_cell) {
          can_fix = false;
        }
        to_be_amplify.push_back(inst);
      }
    };
    max_delay_child->preOrder(find_sub_insts);
    LOG_FATAL_IF(to_be_amplify.empty()) << "error: no inst to be amplified";
    if (can_fix) {
      // try to amplify buffer size, if skew greater than before, then rollback
      auto all_skew_feasible = [&](const std::vector<Inst*>& insts) {
        bool all_feasible = true;
        std::ranges::for_each(insts, [&](Inst* inst) {
          auto* driver_pin = inst->get_driver_pin();
          auto new_skew = TimingPropagator::calcSkew(driver_pin);
          if (new_skew > _skew_bound) {
            all_feasible = false;
          }
        });
        return all_feasible;
      };
      std::ranges::for_each(to_be_amplify, [&](Inst* inst) { TreeBuilder::amplifyBufferSize(inst); });
      bool skew_feasible = all_skew_feasible(to_be_amplify);
      if (!skew_feasible) {
        std::ranges::for_each(to_be_amplify, [&](Inst* inst) { TreeBuilder::reduceBufferSize(inst); });
        return false;
      }
    } else {
      return can_fix;
    }
  }
  if (node->isPin() && node->isDriver()) {
    auto* driver_pin = dynamic_cast<Pin*>(node);
    auto* net = driver_pin->get_net();
    TimingPropagator::update(net);
  } else {
    updateTiming(node);
  }

  return can_fix;
}

void BST::wireSnaking(Node* node) const
{
  auto low_skew = calcRequireSkew(node);
  LOG_FATAL_IF(low_skew < 0) << "snaking required skew is less than 0";
  LOG_FATAL_IF(low_skew > TimingPropagator::getMinInsertDelay()) << "snaking required skew is less than 0";
  auto snaking = [&](Node* parent, Node* child, const double& required_delay) {
    auto length = TimingPropagator::calcLen(parent, child);
    auto factor = length + child->get_cap_load() / _unit_cap;
    auto snake_length = std::sqrt(std::pow(factor, 2) + 2 * required_delay / (_unit_res * _unit_cap)) - factor;
    child->set_required_snake(child->get_required_snake() + snake_length);
  };
  auto* min_delay_node = getMinDelayChild(node);
  snaking(node, min_delay_node, low_skew);

  if (node->isPin()) {
    LOG_FATAL_IF(!node->isDriver()) << "wire snaking can only be applied to driver pin";
    auto* pin = dynamic_cast<Pin*>(node);
    auto* net = pin->get_net();
    TimingPropagator::update(net);
  } else {
    updateTiming(node);
  }
}

void BST::insertBuffer(Node* parent, Node* child)
{
  auto* origin_net = getNet(parent);
  auto loc = (parent->get_location() + child->get_location()) / 2;
  auto net_name = CTSAPIInst.toString(_net_name, "_", CTSAPIInst.genId());
  auto* buffer = TreeBuilder::genBufInst(net_name, loc);
  buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  TreeBuilder::place(buffer);
  auto* driver_pin = buffer->get_driver_pin();
  auto* load_pin = buffer->get_load_pin();
  // get origin net, and add load_pin to net; update origin net after insert buffer
  TreeBuilder::disconnect(parent, child);
  TreeBuilder::connect(parent, load_pin);
  TreeBuilder::connect(driver_pin, child);
  auto required_snake = child->get_required_snake();
  child->set_required_snake(0.5 * required_snake);
  load_pin->set_required_snake(0.5 * required_snake);
  auto* new_net = TimingPropagator::genNet(net_name, driver_pin);
  TimingPropagator::update(new_net);
  if (load_pin->get_slew_in() == 0) {
    TimingPropagator::initLoadPinDelay(load_pin);
  } else {
    TimingPropagator::updateCellDelay(buffer);
  }

  if (!TimingPropagator::skewFeasible(driver_pin, _skew_bound)) {
    skewFix(driver_pin);
  }

  if (origin_net) {
    TimingPropagator::updateLoads(origin_net);
    TimingPropagator::update(origin_net);
  }
  _insert_bufs.push_back(buffer);
  _nets.insert(new_net);
}

double BST::calcRequireSkew(Node* node) const
{
  auto children = node->get_children();
  LOG_FATAL_IF(children.size() < 2) << "can't calculate the skew range while child node less than 2";
  double max_delay = std::numeric_limits<double>::min();
  double min_delay = std::numeric_limits<double>::max();
  std::ranges::for_each(children, [&](Node* child) {
    max_delay = std::max(max_delay, child->get_max_delay() + TimingPropagator::calcElmoreDelay(node, child));
    min_delay = std::min(min_delay, child->get_min_delay() + TimingPropagator::calcElmoreDelay(node, child));
  });
  return max_delay - min_delay - _skew_bound;
}

std::pair<double, double> BST::calcGlobalDelayRange() const
{
  double min_delay = std::numeric_limits<double>::max();
  double max_delay = std::numeric_limits<double>::min();
  std::ranges::for_each(_unmerged_nodes, [&min_delay, &max_delay](const Node* node) {
    auto max_val = node->get_max_delay();
    auto min_val = node->get_min_delay();
    min_delay = std::min(min_delay, min_val);
    max_delay = std::max(max_delay, max_val);
  });
  return std::make_pair(min_delay, max_delay);
}

bool BST::skewFeasible(Node* left, Node* right)
{
  if (!TimingPropagator::skewFeasible(left, _skew_bound) || !TimingPropagator::skewFeasible(right, _skew_bound)) {
    return false;
  }
  joinSegment(left, right);
  updateTiming(left);
  updateTiming(right);
  auto ep_pair = calcEndpointLoc(left, right);
  auto ep_l = ep_pair.first;
  auto ep_r = ep_pair.second;
  auto length = TimingPropagator::calcLen(left, right);
  if (ep_r < 0 || ep_l > length) {
    return false;
  }
  return true;
}

std::pair<Node*, Node*> BST::timingOpt(Node* left, Node* right)
{
  auto* new_left = left;
  auto* new_right = right;
  auto ep_pair = calcEndpointLoc(left, right);
  auto ep_r = ep_pair.second;
  if (ep_r < 0) {
    // right delay too low, buffering right or amplify left buffer size
    size_t num = 0;
    while (!skewFeasible(new_left, new_right)) {
      auto left_max = left->get_max_delay();
      auto right_max = right->get_max_delay();
      // case 1: buffer sizing can fix
      if (std::fabs(left_max - right_max) < TimingPropagator::getMinInsertDelay()) {
        auto is_ok = bufferResizing(left);
        // TBD add wire snaking
        LOG_FATAL_IF(!is_ok) << "can't fix skew by buffer resizing";
      } else {
        new_right = buffering(new_right);
      }
      ++num;
      LOG_FATAL_IF(num > 100) << "can't fix skew, left delay: " << new_left->get_max_delay()
                              << ", right delay: " << new_right->get_max_delay();
    }
  } else {
    // left delay too low, buffering left or amplify right buffer size
    size_t num = 0;
    while (!skewFeasible(new_left, new_right)) {
      auto left_max = left->get_max_delay();
      auto right_max = right->get_max_delay();
      // case 1: buffer sizing can fix
      if (std::fabs(left_max - right_max) < TimingPropagator::getMinInsertDelay()) {
        auto is_ok = bufferResizing(right);
        LOG_FATAL_IF(!is_ok) << "can't fix skew by buffer resizing";
      } else {
        new_left = buffering(new_left);
      }
      ++num;
      LOG_FATAL_IF(num > 100) << "can't fix skew, left delay: " << new_left->get_max_delay()
                              << ", right delay: " << new_right->get_max_delay();
    }
  }
  return std::make_pair(new_left, new_right);
}

void BST::skewFix(Node* start)
{
  LOG_FATAL_IF(TimingPropagator::skewFeasible(start, _skew_bound)) << "not need to fix skew";

  auto children = start->get_children();
  if (children.size() == 1) {
    skewFix(children[0]);
    updateTiming(start);
    return;
  }
  auto min_insert_delay = TimingPropagator::getMinInsertDelay();
  size_t resize_limit = TimingPropagator::getDelayLibs().size();

  auto delta = start->get_max_delay() - start->get_min_delay() - _skew_bound;
  bool can_fix = true;
  if (start->isBufferPin() && delta < resize_limit * min_insert_delay) {
    // case 1 try to buffer resizing
    while (can_fix && !TimingPropagator::skewFeasible(start, _skew_bound)) {
      can_fix = bufferResizing(start);

      LOG_FATAL_IF(children.size() != 2) << "this part should be fixed, avoid [fuse]";
      if (!TimingPropagator::skewFeasible(start, _skew_bound) && skewFeasible(children[0], children[1])) {
        // relocation
        auto mr = calcMergeRegion(children[0], children[1]);
        LOG_FATAL_IF(mr.empty()) << "relocation after buffering, but location leagalization failed";
        auto new_loc = pgl::center(mr);

        auto* pin = dynamic_cast<Pin*>(start);
        auto* buffer = pin->get_inst();
        TreeBuilder::cancelPlace(buffer);
        buffer->set_location(new_loc);
        TreeBuilder::place(buffer);
        if (TimingPropagator::skewFeasible(start, _skew_bound)) {
          can_fix = true;
        } else {
          can_fix = false;
        }
      }
    }
    if (can_fix) {
      return;
    }
  }
  start->postOrder([&](Node* node) {
    updateTiming(node);
    if (TimingPropagator::skewFeasible(node, _skew_bound)) {
      return;
    }
    size_t opt_step = 0;
    while (!TimingPropagator::skewFeasible(node, _skew_bound)) {
      auto low_skew = calcRequireSkew(node);
      LOG_FATAL_IF(low_skew < 0) << "illegal skew range which lower skew: " << low_skew << " while node skew fesible";
      min_insert_delay = TimingPropagator::getMinInsertDelay();
      if (low_skew > min_insert_delay) {
        // case 2 try to insert buffer
        auto* min_child = getMinDelayChild(node);
        // auto* max_child = getMaxDelayChild(node);
        // auto min_len = TimingPropagator::calcLen(node, min_child);
        // auto max_len = TimingPropagator::calcLen(node, max_child);
        // calc range
        // if (max_len > 0.5 * TimingPropagator::getMaxLength()) {
        // insertBuffer(node, max_child);
        // } else {
        insertBuffer(node, min_child);
        // }
      } else {
        // case 3 try to wire snaking on pin node
        wireSnaking(node);
        if (start->isPin()) {
          auto* driver_pin = dynamic_cast<Pin*>(start);
          auto* net = driver_pin->get_net();
          TimingPropagator::update(net);
          if (!TimingPropagator::skewFeasible(node, _skew_bound)) {
            skewFix(node);
          }
        }
      }
      LOG_FATAL_IF(opt_step > 100) << "optimization step seems too long";
      ++opt_step;
    }
  });
}

Node* BST::getMinDelayChild(Node* node) const
{
  auto children = node->get_children();
  LOG_FATAL_IF(children.empty());
  Node* min_delay_node = nullptr;
  std::ranges::for_each(children, [&](Node* child) {
    if (!min_delay_node) {
      min_delay_node = child;
    }
    if (min_delay_node->get_max_delay() + TimingPropagator::calcElmoreDelay(node, min_delay_node)
        > child->get_max_delay() + TimingPropagator::calcElmoreDelay(node, child)) {
      min_delay_node = child;
    }
  });
  return min_delay_node;
}

Node* BST::getMaxDelayChild(Node* node) const
{
  auto children = node->get_children();
  LOG_FATAL_IF(children.empty());
  Node* max_delay_node = nullptr;
  std::ranges::for_each(children, [&](Node* child) {
    if (!max_delay_node) {
      max_delay_node = child;
    }
    if (max_delay_node->get_max_delay() + TimingPropagator::calcElmoreDelay(node, max_delay_node)
        < child->get_max_delay() + TimingPropagator::calcElmoreDelay(node, child)) {
      max_delay_node = child;
    }
  });
  return max_delay_node;
}

Net* BST::getNet(Node* node) const
{
  if (node->isPin()) {
    auto* pin = dynamic_cast<Pin*>(node);
    return pin->get_net();
  }
  auto* parent = node->get_parent();
  while (parent) {
    if (parent->isPin()) {
      auto* pin = dynamic_cast<Pin*>(parent);
      return pin->get_net();
    }
    parent = parent->get_parent();
  }
  return nullptr;
}

void BST::topdown(Node* root) const
{
  auto* root_load_pin = dynamic_cast<Pin*>(root);
  auto* root_inst = root_load_pin->get_inst();
  auto* root_driver_pin = root_inst->get_driver_pin();
  auto* root_net = root_driver_pin->get_net();
  std::queue<Net*> q;
  q.push(root_net);
  while (!q.empty()) {
    auto* net = q.front();
    q.pop();
    auto* driver_pin = net->get_driver_pin();
    driver_pin->preOrder([&](Node* node) {
      if (node->isPin() && (node->isSinkPin() || node->isDriver())) {
        return;
      }
      if (node->isSteiner()) {
        // set loc
        auto* parent = node->get_parent();
        auto parent_loc = parent->get_location();
        auto js = _js_map.at(node);
        auto loc = pgl::closest_point(parent_loc, js);
        node->set_location(loc);
        return;
      } else {
        auto* load_pin = dynamic_cast<Pin*>(node);
        auto* inst = load_pin->get_inst();
        // add sub net
        auto* driver_pin = inst->get_driver_pin();
        auto* net = driver_pin->get_net();
        if (_nets.find(net) != _nets.end()) {
          q.push(net);
        }
      }
    });
  }
}

void BST::reportSkew(Node* node) const
{
  LOG_INFO << "Node skew: " << TimingPropagator::calcSkew(node);
}

void BST::reportMaxDelay(Node* node) const
{
  LOG_INFO << "Node max delay: " << node->get_max_delay();
}

}  // namespace icts