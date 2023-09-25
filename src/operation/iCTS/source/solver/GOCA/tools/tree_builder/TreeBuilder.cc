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
 * @file TreeBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "TreeBuilder.hh"

#include <filesystem>
#include <stack>
#include <unordered_set>

#include "BEAT.hh"
#include "CTSAPI.hh"
#include "CtsConfig.hh"
#include "GeomCalc.hh"
#include "LocalLegalization.hh"
#include "Node.hh"
#include "TimingPropagator.hh"
#include "salt/base/flute.h"
#include "salt/salt.h"
namespace icts {

const std::unordered_map<SteinerTreeFunc, std::string> TreeBuilder::kSteinterTreeName = {
    {shallowLightTree, "ShallowLightTree"},
    {fluteTree, "FluteTree"},
};
const std::unordered_map<SkewTreeFunc, std::string> TreeBuilder::kSkewTreeName = {
    {boundSkewTree, "BoundSkewTree"},
    {bstSaltTree, "BstSaltTree"},
    {beatTree, "BeatTree"},
};
/**
 * @brief get the sub insts of inst
 *
 * @param inst
 * @return std::vector<Inst*>
 */
std::vector<Inst*> TreeBuilder::getSubInsts(Inst* inst)
{
  if (inst->isSink()) {
    return {};
  }
  auto* driver_pin = inst->get_driver_pin();
  auto* net = driver_pin->get_net();
  auto load_pins = net->get_load_pins();
  std::vector<Inst*> sub_insts;
  std::ranges::for_each(load_pins, [&sub_insts](Pin* pin) { sub_insts.push_back(pin->get_inst()); });
  return sub_insts;
}
/**
 * @brief generate buffer inst
 *       name: prefix + "_buf"
 *       cell: none
 *
 * @param prefix
 * @param location
 * @return Inst*
 */
Inst* TreeBuilder::genBufInst(const std::string& prefix, const Point& location)
{
  auto buf_name = prefix + "_buf";
  auto buf_inst = new Inst(buf_name, location, InstType::kBuffer);
  return buf_inst;
}
/**
 * @brief convert node to buffer inst
 *
 * @param prefix
 * @param driver_node
 * @return Inst*
 */
Inst* TreeBuilder::toBufInst(const std::string& prefix, Node* driver_node)
{
  auto buf_name = prefix + "_buf";
  auto buf_inst = new Inst(buf_name, driver_node->get_location(), driver_node);
  return buf_inst;
}
/**
 * @brief amplify the buffer size of inst
 *
 * @param inst
 * @param level
 */
void TreeBuilder::amplifyBufferSize(Inst* inst, const size_t& level)
{
  if (level == 0 || inst->isSink()) {
    return;
  }
  if (level > 1) {
    auto sub_insts = getSubInsts(inst);
    bool is_all_buf = true;
    for (auto inst : sub_insts) {
      if (inst->isSink()) {
        is_all_buf = false;
        break;
      }
    }
    if (is_all_buf) {
      std::ranges::for_each(sub_insts, [&](Inst* sub_inst) { amplifyBufferSize(sub_inst, level - 1); });
    }
  }
  auto cell_master = inst->get_cell_master();
  if (cell_master == TimingPropagator::getMaxSizeLib()->get_cell_master()) {
    LOG_WARNING << inst->get_name() << " can't be amplified";
    return;
  }
  auto libs = TimingPropagator::getDelayLibs();
  // find next cell master
  for (size_t i = 0; i < libs.size() - 1; ++i) {
    if (cell_master == libs[i]->get_cell_master()) {
      auto next_lib = libs[i + 1];
      inst->set_cell_master(next_lib->get_cell_master());
      auto* driver_pin = inst->get_driver_pin();
      auto* net = driver_pin->get_net();
      TimingPropagator::update(net);
      auto* load_pin = inst->get_load_pin();
      if (load_pin->get_slew_in() == 0) {
        TimingPropagator::initLoadPinDelay(load_pin);
      } else {
        TimingPropagator::updateCellDelay(inst);
      }
      return;
    }
  }
}
/**
 * @brief reduce the buffer size of inst
 *
 * @param inst
 * @param level
 */
void TreeBuilder::reduceBufferSize(Inst* inst, const size_t& level)
{
  if (level == 0 || inst->isSink()) {
    return;
  }
  if (level > 1) {
    auto sub_insts = getSubInsts(inst);
    bool is_all_buf = true;
    for (auto inst : sub_insts) {
      if (inst->isSink()) {
        is_all_buf = false;
        break;
      }
    }
    if (is_all_buf) {
      std::ranges::for_each(sub_insts, [&](Inst* sub_inst) { reduceBufferSize(sub_inst, level - 1); });
    }
  }
  auto cell_master = inst->get_cell_master();
  if (cell_master == TimingPropagator::getMinSizeLib()->get_cell_master()) {
    LOG_WARNING << inst->get_name() << " can't be reduced";
    return;
  }
  auto libs = TimingPropagator::getDelayLibs();
  // find next cell master
  for (size_t i = 1; i < libs.size(); ++i) {
    if (cell_master == libs[i]->get_cell_master()) {
      auto before_lib = libs[i - 1];
      inst->set_cell_master(before_lib->get_cell_master());
      auto* driver_pin = inst->get_driver_pin();
      auto* net = driver_pin->get_net();
      TimingPropagator::update(net);
      auto* load_pin = inst->get_load_pin();
      if (load_pin->get_slew_in() == 0) {
        TimingPropagator::initLoadPinDelay(load_pin);
      } else {
        TimingPropagator::updateCellDelay(inst);
      }
      return;
    }
  }
}
/**
 * @brief find all feasible cell of inst
 *
 * @param inst
 * @param skew_bound
 * @return std::vector<std::string>
 */
std::vector<std::string> TreeBuilder::feasibleCell(Inst* inst, const double& skew_bound)
{
  LOG_FATAL_IF(!inst->isBuffer()) << inst->get_name() << " is not buffer";
  auto origin_cell = inst->get_cell_master();
  auto libs = TimingPropagator::getDelayLibs();

  auto* driver_pin = inst->get_driver_pin();
  auto* net = driver_pin->get_net();
  std::vector<std::string> feasible_cells;
  std::ranges::for_each(libs, [&](CtsCellLib* lib) {
    auto cell = lib->get_cell_master();
    inst->set_cell_master(cell);
    TimingPropagator::update(net);
    if (TimingPropagator::skewFeasible(driver_pin, skew_bound)) {
      feasible_cells.push_back(cell);
    }
  });
  inst->set_cell_master(origin_cell);
  TimingPropagator::update(net);
  return feasible_cells;
}
/**
 * @brief connect parent node and child node
 *
 * @param parent
 * @param child
 */
void TreeBuilder::connect(Node* parent, Node* child)
{
  parent->add_child(child);
  child->set_parent(parent);
}
/**
 * @brief disconnect parent node and child node
 *
 * @param parent
 * @param child
 */
void TreeBuilder::disconnect(Node* parent, Node* child)
{
  parent->remove_child(child);
  child->set_parent(nullptr);
}
/**
 * @brief directly connected tree
 *
 * @param driver
 * @param load
 */
void TreeBuilder::directConnectTree(Pin* driver, Pin* load)
{
  driver->add_child(load);
  load->set_parent(driver);
}
/**
 * @brief Flute Tree
 *
 * @param net_name
 * @param driver
 * @param loads
 */
void TreeBuilder::fluteTree(const std::string& net_name, Pin* driver, const std::vector<Pin*>& loads)
{
  std::vector<icts::Pin*> pins{driver};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;

  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net net;
  net.init(0, net_name, salt_pins);

  salt::Tree tree;
  salt::FluteBuilder flute_builder;
  flute_builder.Run(net, tree);
  tree.UpdateId();
  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  auto source = tree.source;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = CTSAPIInst.toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
}
/**
 * @brief shallow light tree
 *
 * @param net_name
 * @param driver
 * @param loads
 */
void TreeBuilder::shallowLightTree(const std::string& net_name, Pin* driver, const std::vector<Pin*>& loads)
{
  std::vector<icts::Pin*> pins{driver};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;

  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net net;
  net.init(0, net_name, salt_pins);

  salt::Tree tree;
  salt::SaltBuilder salt_builder;
  salt_builder.Run(net, tree, 0);

  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  auto source = tree.source;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = CTSAPIInst.toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
}
/**
 * @brief bound skew tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */
Inst* TreeBuilder::boundSkewTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                                 const std::optional<Point>& guide_loc, const TopoType& topo_type)
{
  auto solver = bst::BoundSkewTree(net_name, loads, skew_bound, topo_type);
  if (guide_loc.has_value()) {
    solver.set_root_guide(*guide_loc);
  }
  solver.run();
  return solver.get_root_buf();
}
/**
 * @brief BEAT Salt Tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */
Inst* TreeBuilder::bstSaltTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                               const std::optional<Point>& guide_loc, const TopoType& topo_type)
{
  // build BST
  auto* buf = icts::TreeBuilder::boundSkewTree(net_name, loads, skew_bound, guide_loc, topo_type);
  auto* driver_pin = buf->get_driver_pin();
  buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  auto* bst_net = TimingPropagator::genNet("BoundSkewTree", driver_pin, loads);
  TimingPropagator::update(bst_net);

  std::vector<Pin*> pins{driver_pin};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::unordered_map<Pin*, std::shared_ptr<salt::Pin>> salt_pin_map;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  // init
  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pin_map[pin] = salt_pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net salt_net;
  salt_net.init(0, net_name, salt_pins);
  // convert bound skew tree to salt data structure
  std::unordered_map<Node*, std::shared_ptr<salt::TreeNode>> salt_node_map;
  int id = pins.size();
  driver_pin->preOrder([&](Node* node) {
    auto loc = salt::Point(node->get_location().x(), node->get_location().y());
    std::shared_ptr<salt::TreeNode> salt_node;
    if (node->isPin()) {
      auto salt_pin = salt_pin_map[dynamic_cast<Pin*>(node)];
      salt_node = std::make_shared<salt::TreeNode>(loc, salt_pin, salt_pin->id);
    } else {
      salt_node = std::make_shared<salt::TreeNode>(loc, nullptr, id++);
    }
    salt_node_map[node] = salt_node;
    if (node->get_parent()) {
      auto salt_parent = salt_node_map[node->get_parent()];
      salt_node->parent = salt_parent;
      salt_parent->children.push_back(salt_node);
    }
  });
  salt::Tree bound_skew_tree(salt_node_map[driver_pin], &salt_net);
  // BEAT Salt
  TreeSaltBuilder builder;
  builder.run(salt_net, bound_skew_tree, 0);

  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  icts::TimingPropagator::resetNet(bst_net);
  auto source = bound_skew_tree.source;
  buf = icts::TreeBuilder::genBufInst(net_name, icts::Point(source->loc.x, source->loc.y));
  driver_pin = buf->get_driver_pin();
  id_to_node[0] = driver_pin;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = CTSAPIInst.toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
  return buf;
}
/**
 * @brief BEAT Tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */
Inst* TreeBuilder::beatTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                            const std::optional<Point>& guide_loc, const TopoType& topo_type)
{
  auto* inst = bstSaltTree(net_name, loads, skew_bound, guide_loc, topo_type);
  auto solver = bst::BoundSkewTree(net_name, inst->get_driver_pin(), skew_bound);
  solver.run();
  return inst;
}
/**
 * @brief temp tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */
Inst* TreeBuilder::tempTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                            const std::optional<Point>& guide_loc, const TopoType& topo_type)
{
  // build BST
  auto* buf = icts::TreeBuilder::boundSkewTree(net_name, loads, skew_bound, guide_loc, topo_type);
  auto* driver_pin = buf->get_driver_pin();
  buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  auto* bst_net = TimingPropagator::genNet("BoundSkewTree", driver_pin, loads);
  TimingPropagator::update(bst_net);

  std::vector<Pin*> pins{driver_pin};
  std::ranges::copy(loads, std::back_inserter(pins));

  std::unordered_map<int, Node*> id_to_node;
  std::unordered_map<Pin*, std::shared_ptr<salt::Pin>> salt_pin_map;
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  // init
  for (size_t i = 0; i < pins.size(); ++i) {
    auto pin = pins[i];
    auto salt_pin = std::make_shared<salt::Pin>(pin->get_location().x(), pin->get_location().y(), i, pin->get_cap_load());
    id_to_node[i] = pin;
    salt_pin_map[pin] = salt_pin;
    salt_pins.push_back(salt_pin);
  }
  salt::Net salt_net;
  salt_net.init(0, net_name, salt_pins);
  // convert bound skew tree to salt data structure
  std::unordered_map<Node*, std::shared_ptr<salt::TreeNode>> salt_node_map;
  int id = pins.size();
  driver_pin->preOrder([&](Node* node) {
    auto loc = salt::Point(node->get_location().x(), node->get_location().y());
    std::shared_ptr<salt::TreeNode> salt_node;
    if (node->isPin()) {
      auto salt_pin = salt_pin_map[dynamic_cast<Pin*>(node)];
      salt_node = std::make_shared<salt::TreeNode>(loc, salt_pin, salt_pin->id);
    } else {
      salt_node = std::make_shared<salt::TreeNode>(loc, nullptr, id++);
    }
    salt_node_map[node] = salt_node;
    if (node->get_parent()) {
      auto salt_parent = salt_node_map[node->get_parent()];
      salt_node->parent = salt_parent;
      salt_parent->children.push_back(salt_node);
    }
  });
  salt::Tree bound_skew_tree(salt_node_map[driver_pin], &salt_net);
  // BEAT Salt
  icts::TempBuilder beat_builder;
  beat_builder.run(salt_net, bound_skew_tree, 0);

  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  icts::TimingPropagator::resetNet(bst_net);
  auto source = bound_skew_tree.source;
  buf = icts::TreeBuilder::genBufInst(net_name, icts::Point(source->loc.x, source->loc.y));
  driver_pin = buf->get_driver_pin();
  id_to_node[0] = driver_pin;
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    if (salt_node->id == source->id) {
      return;
    }
    // steiner point, need to create a new node
    if (salt_node->id > static_cast<int>(loads.size())) {
      auto name = CTSAPIInst.toString(net_name, "_", salt_node->id);
      auto node = new icts::Node(name, icts::Point(salt_node->loc.x, salt_node->loc.y));
      id_to_node[salt_node->id] = node;
    }
    // connect to parent
    auto* current_node = id_to_node[salt_node->id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = id_to_node[parent_id];
    current_node->set_parent(parent_node);
    parent_node->add_child(current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
  return buf;
}
/**
 * @brief convert to binary tree by add steiner node
 *
 * @param root
 */
void TreeBuilder::convertToBinaryTree(Node* root)
{
  // convert to binary tree for bound skew tree
  auto one_child_refine = [&](Node* node) {
    auto children = node->get_children();
    LOG_FATAL_IF(children.size() != 1) << "node " << node->get_name() << " children size is not 1";

    // upstream refine
    auto* parent = node->get_parent();
    auto* copy_node = new Node(CTSAPIInst.toString("steiner_", CTSAPIInst.genId()), node->get_location());
    if (parent) {
      disconnect(parent, node);
      connect(parent, copy_node);
    }
    // case 1: size is 1
    // downstream refine
    LOG_FATAL_IF(!node->isPin()) << "node " << node->get_name() << " is not pin";
    auto* child = children.front();
    disconnect(node, child);
    connect(copy_node, child);
    connect(copy_node, node);

    return copy_node;
  };

  auto two_child_refine = [&](Node* node) {
    auto children = node->get_children();
    LOG_FATAL_IF(!(node->isPin() && node->isLoad()) || children.size() != 2)
        << "node " << node->get_name() << " is not Pin or children size is not 2";

    // upstream refine
    auto* parent = node->get_parent();
    auto* copy_node = new Node(CTSAPIInst.toString("steiner_", CTSAPIInst.genId()), node->get_location());
    if (parent) {
      disconnect(parent, node);
      connect(parent, copy_node);
    }
    // case 2: size is 2 and node is pin
    // downstream refine
    connect(copy_node, node);
    auto* trunk = new Node(CTSAPIInst.toString("steiner_", CTSAPIInst.genId()), node->get_location());
    connect(copy_node, trunk);
    std::ranges::for_each(children, [&](Node* child) {
      disconnect(node, child);
      connect(trunk, child);
    });

    return copy_node;
  };

  auto three_child_refine = [&](Node* node) {
    auto children = node->get_children();
    LOG_FATAL_IF(children.size() != 3) << "node " << node->get_name() << " children size is not 3";

    // case 3: size is 3

    // find closest 2 node in children TBD use heuristic
    std::ranges::sort(children, [&](const Node* lhs, const Node* rhs) {
      return Point::manhattanDistance(lhs->get_location(), node->get_location())
             < Point::manhattanDistance(rhs->get_location(), node->get_location());
    });
    // downstream refine
    auto* left_child = children[0];
    auto* right_child = children[1];
    auto* trunk = new Node(CTSAPIInst.toString("steiner_", CTSAPIInst.genId()), node->get_location());
    disconnect(node, left_child);
    disconnect(node, right_child);
    connect(node, trunk);
    connect(trunk, left_child);
    connect(trunk, right_child);
    return node;
  };

  auto four_child_refine = [&](Node* node) {
    auto children = node->get_children();
    LOG_FATAL_IF(children.size() != 4) << "node " << node->get_name() << " children size is not 4";

    // upstream check
    LOG_FATAL_IF(node->get_parent()) << "node " << node->get_name() << "have 4 children, so parent should be nullptr";
    // case 4: size is 4
    auto* copy_left_node = new Node(CTSAPIInst.toString("steiner_", CTSAPIInst.genId()), node->get_location());
    auto* copy_right_node = new Node(CTSAPIInst.toString("steiner_", CTSAPIInst.genId()), node->get_location());
    std::ranges::for_each(children, [&](Node* child) { disconnect(node, child); });
    connect(node, copy_left_node);
    connect(node, copy_right_node);
    // find closest 2 node in children TBD use heuristic
    std::ranges::sort(children, [&](const Node* lhs, const Node* rhs) {
      return Point::manhattanDistance(lhs->get_location(), node->get_location())
             < Point::manhattanDistance(rhs->get_location(), node->get_location());
    });
    // downstream refine
    auto left_children = std::vector<Node*>(children.begin(), children.begin() + 2);
    std::ranges::for_each(left_children, [&](Node* child) { connect(copy_left_node, child); });
    auto right_children = std::vector<Node*>(children.begin() + 2, children.end());
    std::ranges::for_each(right_children, [&](Node* child) { connect(copy_right_node, child); });
    return node;
  };

  auto to_binary = [&](Node* node) {
    auto children = node->get_children();
    if (children.empty()) {
      return node;
    }
    if (children.size() == 1) {
      return one_child_refine(node);
    }
    if (children.size() == 2) {
      return (node->isPin() && node->isLoad()) ? two_child_refine(node) : node;
    }
    if (children.size() == 3) {
      return three_child_refine(node);
    }
    LOG_FATAL_IF(children.size() != 4) << "node " << node->get_name() << " children size is " << children.size() << " not 4";
    return four_child_refine(node);
  };
  std::stack<Node*> stack;
  stack.push(root);
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();

    cur = to_binary(cur);
    auto children = cur->get_children();
    if (children.empty()) {
      continue;
    }

    LOG_FATAL_IF(children.size() != 2) << "It's not a binary tree";
    stack.push(children.front());
    stack.push(children.back());
  }
}
/**
 * @brief find steiner tree function name
 *
 * @param func
 * @return std::string
 */
std::string TreeBuilder::funcName(const SteinerTreeFunc& func)
{
  if (kSteinterTreeName.find(func) == kSteinterTreeName.end()) {
    LOG_FATAL << "Unsupported function";
  }
  return kSteinterTreeName.at(func);
}
/**
 * @brief find skew tree function name
 *
 * @param func
 * @return std::string
 */
std::string TreeBuilder::funcName(const SkewTreeFunc& func)
{
  if (kSkewTreeName.find(func) == kSkewTreeName.end()) {
    LOG_FATAL << "Unsupported function";
  }
  return kSkewTreeName.at(func);
}
/**
 * @brief get all steiner tree functions
 *
 * @return std::vector<SteinerTreeFunc>
 */
std::vector<SteinerTreeFunc> TreeBuilder::getSteinerTreeFuncs()
{
  return {fluteTree, shallowLightTree};
}
/**
 * @brief get all skew tree functions
 *
 * @return std::vector<SkewTreeFunc>
 */
std::vector<SkewTreeFunc> TreeBuilder::getSkewTreeFuncs()
{
  return {boundSkewTree, bstSaltTree, beatTree};
}
/**
 * @brief local place, if location is repeated, then move the inst to the feasible location
 *
 * @param inst
 * @param load_pins
 */
void TreeBuilder::localPlace(Inst* inst, const std::vector<Pin*>& load_pins)
{
  LocalLegalization(inst, load_pins);
}
void TreeBuilder::localPlace(std::vector<Point>& variable_locs, const std::vector<Point>& fixed_locs)
{
  LocalLegalization(variable_locs, fixed_locs);
}
/**
 * @brief print tree to graphviz
 *
 * @param root
 * @param prefix
 */
void TreeBuilder::printGraphviz(Node* root, const std::string& name)
{
  auto* config = CTSAPIInst.get_config();
  auto dir = config->get_sta_workspace();
  auto file_name = dir + "/" + name + ".dot";
  std::ofstream out(file_name);
  if (!out.is_open()) {
    LOG_ERROR << "Can't open file: " << file_name;
    return;
  }
  out << "digraph G {\n";
  std::function<void(Node*)> print_node = [&out](Node* node) {
    // include node info

    if (node->isSteiner()) {
      out << node->get_name() << " [shape=point];\n";
    } else {
      out << node->get_name() << " [shape=box];\n";
    }
  };
  std::function<void(Node*)> print_edge = [&out](Node* node) {
    if (node->isSteiner()) {
      for (auto child : node->get_children()) {
        out << node->get_name() << " -> " << child->get_name() << ";\n";
      }
    }
  };
  root->preOrder(print_node);
  root->preOrder(print_edge);
  out << "}\n";
  out.close();
}
/**
 * @brief write tree to python
 *
 * @param root
 * @param name
 */
void TreeBuilder::writePy(Node* root, const std::string& name)
{
  auto* config = CTSAPIInst.get_config();
  auto dir = config->get_sta_workspace() + "/file";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directory(dir);
  }
  auto file_name = dir + "/" + name + ".py";
  std::ofstream out(file_name);
  if (!out.is_open()) {
    LOG_ERROR << "Can't open file: " << file_name;
    return;
  }
  out << "import matplotlib.pyplot as plt\n";
  out << "fig = plt.figure(figsize=(8,6), dpi=300)\n";
  root->preOrder([&out](Node* node) {
    if (node->isPin() && node->isDriver()) {
      // more big
      out << "plt.plot(" << node->get_location().x() << "," << node->get_location().y() << ",'*r')\n";
    } else if (node->isSteiner()) {
      out << "plt.plot(" << node->get_location().x() << "," << node->get_location().y() << ",'.k')\n";
    } else {
      out << "plt.plot(" << node->get_location().x() << "," << node->get_location().y() << ",'ob')\n";
    }
    // out << "plt.text(" << node->get_location().x() << "," << node->get_location().y() << ",'[" << std::fixed << std::setprecision(4)
    //     << node->get_name() << "]')\n";
    auto* parent = node->get_parent();
    if (parent) {
      out << "plt.plot([" << parent->get_location().x() << "," << node->get_location().x() << "],"
          << "[" << parent->get_location().y() << "," << node->get_location().y() << "],'-k')\n";
    }
  });
  out << "plt.show()\n";
  out << "plt.savefig('./" + name + ".png')\n";
  out.close();
}
}  // namespace icts