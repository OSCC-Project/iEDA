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

#include "CBS.hh"
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
    {cbsTree, "CBSTree"},
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
  if (cell_master == TimingPropagator::getMaxSizeCell()) {
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
  if (cell_master == TimingPropagator::getMinSizeCell()) {
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
      auto name = CTSAPIInst.toString("steiner_", salt_node->id);
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
      auto name = CTSAPIInst.toString("steiner_", salt_node->id);
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
  if (loads.size() == 1) {
    auto loc = guide_loc.value_or(loads.front()->get_location());
    auto* buf = genBufInst(net_name, loc);
    directConnectTree(buf->get_driver_pin(), loads.front());
    return buf;
  }
  auto solver = bst::BoundSkewTree(net_name, loads, skew_bound, topo_type);
  if (guide_loc.has_value()) {
    solver.set_root_guide(*guide_loc);
  }
  solver.run();
  return solver.get_root_buf();
}
/**
 * @brief bound skew tree without estimate
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */
Inst* TreeBuilder::noneEstBoundSkewTree(const std::string& net_name, const std::vector<Pin*>& loads,
                                        const std::optional<double>& skew_bound, const std::optional<Point>& guide_loc,
                                        const TopoType& topo_type)
{
  if (loads.size() == 1) {
    auto loc = guide_loc.value_or(loads.front()->get_location());
    auto* buf = genBufInst(net_name, loc);
    directConnectTree(buf->get_driver_pin(), loads.front());
    return buf;
  }
  auto solver = bst::BoundSkewTree(net_name, loads, skew_bound, topo_type, false);
  if (guide_loc.has_value()) {
    solver.set_root_guide(*guide_loc);
  }
  solver.run();
  return solver.get_root_buf();
}
/**
 * @brief Flute-BST Salt Tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @return Inst*
 */
Inst* TreeBuilder::fluteBstSaltTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                                    const std::optional<Point>& guide_loc)
{
  // build flute-BST
  auto* buf = genBufInst(net_name, guide_loc.value_or(loads.front()->get_location()));
  auto* driver_pin = buf->get_driver_pin();
  std::vector<Pin*> pins{driver_pin};
  std::ranges::copy(loads, std::back_inserter(pins));
  localPlace(pins);
  // flute
  fluteTree(net_name, driver_pin, loads);
  // flute-BST
  auto solver = bst::BoundSkewTree(net_name, driver_pin, skew_bound);
  solver.run();

  buf->set_cell_master(TimingPropagator::getMinSizeCell());
  auto* bst_net = TimingPropagator::genNet("BoundSkewTree", driver_pin, loads);
  removeRedundant(driver_pin);
  TimingPropagator::update(bst_net);

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

  // BST Salt
  salt::Tree bound_skew_tree(salt_node_map[driver_pin], &salt_net);
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
      auto name = CTSAPIInst.toString("steiner_", salt_node->id);
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
 * @brief BST Salt Tree
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
  removeRedundant(driver_pin);
  TimingPropagator::update(bst_net);

  TreeBuilder::updateId(driver_pin);
  int num = 0;
  driver_pin->preOrder([&](Node* node) { ++num; });
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  std::vector<Node*> cts_nodes(num);
  std::vector<std::shared_ptr<salt::TreeNode>> salt_nodes(num);
  driver_pin->preOrder([&](Node* node) {
    std::shared_ptr<salt::TreeNode> salt_node;
    auto id = node->get_id();
    auto loc = salt::Point(node->get_location().x(), node->get_location().y());
    if (node->isPin()) {
      auto* pin = dynamic_cast<Pin*>(node);
      auto salt_pin = std::make_shared<salt::Pin>(loc, id, pin->get_cap_load());
      salt_node = std::make_shared<salt::TreeNode>(loc, salt_pin, id);
      salt_pins.push_back(salt_pin);
    } else {
      salt_node = std::make_shared<salt::TreeNode>(loc, nullptr, id);
    }
    salt_nodes[id] = salt_node;
    cts_nodes[id] = node;
  });
  salt::Net salt_net;
  salt_net.init(0, net_name, salt_pins);
  // convert bound skew tree to salt data structure
  driver_pin->preOrder([&](Node* node) {
    if (!node->get_parent()) {
      return;
    }
    auto cur_id = node->get_id();
    auto parent_id = node->get_parent()->get_id();
    auto salt_node = salt_nodes[cur_id];
    auto salt_parent = salt_nodes[parent_id];
    salt::TreeNode::setParent(salt_node, salt_parent);
  });

  // BST Salt
  salt::Tree bound_skew_tree(salt_nodes[0], &salt_net);
  TreeSaltBuilder builder;
  builder.run(salt_net, bound_skew_tree, 0);
  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  icts::TimingPropagator::resetNet(bst_net);
  auto source = bound_skew_tree.source;
  buf = icts::TreeBuilder::genBufInst(net_name, icts::Point(source->loc.x, source->loc.y));
  driver_pin = buf->get_driver_pin();
  cts_nodes[source->id] = driver_pin;
  num = 0;
  auto count_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) { ++num; };
  salt::TreeNode::preOrder(source, count_func);
  cts_nodes.resize(num);
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    // steiner point, need to create a new node
    auto id = salt_node->id;
    if (id == source->id) {
      return;
    }
    if (!salt_node->pin) {
      auto* node = new icts::Node(id, icts::Point(salt_node->loc.x, salt_node->loc.y));
      cts_nodes[id] = node;
    }
    // connect to parent
    auto* current_node = cts_nodes[id];
    auto parent_id = salt_node->parent->id;
    auto* parent_node = cts_nodes[parent_id];
    connect(parent_node, current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
  return buf;
}
/**
 * @brief CBS Tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */
Inst* TreeBuilder::cbsTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                           const std::optional<Point>& guide_loc, const TopoType& topo_type)
{
  if (loads.size() == 1) {
    auto loc = guide_loc.value_or(loads.front()->get_location());
    auto* buf = genBufInst(net_name, loc);
    directConnectTree(buf->get_driver_pin(), loads.front());
    return buf;
  }
  auto* inst = bstSaltTree(net_name, loads, skew_bound, guide_loc, topo_type);
  auto solver = bst::BoundSkewTree(net_name, inst->get_driver_pin(), skew_bound);
  solver.run();
  return inst;
}
/**
 * @brief CBS tree with shift
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @param shift
 * @param max_len
 * @return Inst*
 */
Inst* TreeBuilder::shiftCBSTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                                const std::optional<Point>& guide_loc, const TopoType& topo_type, const bool& shift,
                                const std::optional<double>& max_len)
{
  {
    auto* buf = defaultTree(net_name, loads, skew_bound, guide_loc, topo_type);
    auto* driver_pin = buf->get_driver_pin();
    auto driver_loc = driver_pin->get_location();
    driver_pin->postOrder(TimingPropagator::updateNetLen<Node>);
    if (shift && guide_loc.has_value() && driver_loc != guide_loc
        && driver_pin->get_sub_len() < max_len.value_or(TimingPropagator::getMaxLength())) {
      auto id = driver_pin->getMaxId();
      auto remain_dist = (max_len.value_or(TimingPropagator::getMaxLength()) - driver_pin->get_sub_len()) * TimingPropagator::getDbUnit();
      auto guide_dist = TimingPropagator::calcDist(driver_loc, guide_loc.value());
      auto feasible_loc
          = remain_dist > guide_dist ? guide_loc.value() : driver_loc + (guide_loc.value() - driver_loc) * remain_dist / guide_dist;
      auto* steiner = new Node(++id, driver_loc);
      auto children = driver_pin->get_children();
      std::ranges::for_each(children, [&](Node* child) {
        disconnect(driver_pin, child);
        connect(steiner, child);
      });
      connect(driver_pin, steiner);
      buf->set_location(feasible_loc);
    }
    return buf;
  }
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
  buf->set_cell_master(TimingPropagator::getMinSizeCell());
  auto* bst_net = TimingPropagator::genNet("BoundSkewTree", driver_pin, loads);
  removeRedundant(driver_pin);
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
  // bst Salt
  icts::TempBuilder temp_builder;
  temp_builder.run(salt_net, bound_skew_tree, 0);

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
      auto name = CTSAPIInst.toString("steiner_", salt_node->id);
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
 * @brief For testing
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @param topo_type
 * @return Inst*
 */

Inst* TreeBuilder::defaultTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                               const std::optional<Point>& guide_loc, const TopoType& topo_type)
{
  auto* config = CTSAPIInst.get_config();
  auto use_skew_tree_alg = config->get_use_skew_tree_alg();
  if (use_skew_tree_alg) {
    return cbsTree(net_name, loads, skew_bound, guide_loc, topo_type);
  }
  if (loads.size() == 1) {
    auto loc = guide_loc.value_or(loads.front()->get_location());
    auto* buf = genBufInst(net_name, loc);
    directConnectTree(buf->get_driver_pin(), loads.front());
    return buf;
  }
  auto* buf = genBufInst(net_name, guide_loc.value_or(BalanceClustering::calcCentroid(loads)));
  auto* driver_pin = buf->get_driver_pin();
  std::vector<Pin*> pins{driver_pin};
  std::ranges::copy(loads, std::back_inserter(pins));
  localPlace(pins);
  shallowLightTree(net_name, driver_pin, loads);
  return buf;
}

/**
 * @brief iterative fix skew
 *
 * @param net
 * @param skew_bound
 * @param guide_loc
 */
void TreeBuilder::iterativeFixSkew(Net* net, const std::optional<double>& skew_bound, const std::optional<Point>& guide_loc)
{
  TimingPropagator::update(net);
  auto* driver_pin = net->get_driver_pin();
  auto loads = net->get_load_pins();
  auto solver = bst::BoundSkewTree("iterFix", driver_pin, skew_bound, false);
  solver.set_root_guide(guide_loc.value_or(driver_pin->get_location()));
  solver.run();
  TimingPropagator::update(net);
}
/**
 * @brief convert to binary tree by add steiner node
 *
 * @param root
 */
void TreeBuilder::convertToBinaryTree(Node* root)
{
  auto id = root->getMaxId();

  auto pin_node_refine = [&](Node* node) {
    if (!node->isPin() || !node->isLoad()) {
      return node;
    }
    auto children = node->get_children();
    if (children.empty()) {
      return node;
    }
    auto* parent = node->get_parent();
    auto* copy_node = new Node(++id, node->get_location());
    LOG_FATAL_IF(!parent) << "node is load pin but not have parent node";
    disconnect(parent, node);
    connect(parent, copy_node);
    connect(copy_node, node);

    // downstream refine
    std::ranges::for_each(children, [&](Node* child) {
      disconnect(node, child);
      connect(copy_node, child);
    });
    return copy_node;
  };

  // convert to binary tree for bound skew tree
  auto one_child_refine = [&](Node* node) {
    auto children = node->get_children();
    LOG_FATAL_IF(children.size() != 1) << "node " << node->get_name() << " children size is not 1";
    // case 1: size is 1
    auto* child = children.front();
    if ((node->isPin() && node->isLoad())) {
      // upstream refine
      auto* parent = node->get_parent();
      auto* copy_node = new Node(++id, node->get_location());
      if (parent) {
        disconnect(parent, node);
        connect(parent, copy_node);
      }
      // downstream refine
      LOG_FATAL_IF(!node->isPin()) << "node " << node->get_name() << " is not pin";
      disconnect(node, child);
      connect(copy_node, child);
      connect(copy_node, node);
      return copy_node;
    }
    auto grand_children = child->get_children();
    LOG_FATAL_IF(grand_children.empty()) << "node " << child->get_name() << " children size is 0";
    std::ranges::for_each(grand_children, [&](Node* grand_child) {
      disconnect(child, grand_child);
      connect(node, grand_child);
    });
    if (!child->isPin()) {
      disconnect(node, child);
      delete child;
    }
    return node;
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
    auto* trunk = new Node(++id, node->get_location());
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
    // case 4: size is 4
    auto* copy_left_node = new Node(++id, node->get_location());
    auto* copy_right_node = new Node(++id, node->get_location());
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
    node = pin_node_refine(node);
    auto children = node->get_children();
    LOG_FATAL_IF(node->isSteiner() && children.empty()) << "steiner node but not children";
    if (children.empty()) {
      return node;
    }
    if (children.size() == 1) {
      return one_child_refine(node);
    }
    if (children.size() == 2) {
      return node;
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
    if (cur == root) {
      // driver_pin should be checked
      cur = to_binary(cur);
    }
    auto children = cur->get_children();
    if (children.empty()) {
      continue;
    }

    LOG_FATAL_IF(children.size() != 2) << "It's not a binary tree";
    stack.push(children.front());
    stack.push(children.back());
  }
}
void TreeBuilder::removeRedundant(Node* root)
{
  std::vector<Node*> to_be_removed;
  std::stack<Node*> stack;
  stack.push(root);
  while (!stack.empty()) {
    auto* node = stack.top();
    stack.pop();
    auto children = node->get_children();
    bool exist_opt = false;
    for (auto* child : children) {
      if (node->get_location() == child->get_location()) {
        exist_opt = true;
        break;
      }
    }
    if (!exist_opt) {
      std::ranges::for_each(children, [&](Node* child) { stack.push(child); });
      continue;
    }
    // move target child's children to node, and remove target child
    if (node->isPin()) {
      std::ranges::for_each(children, [&](Node* child) {
        if (node->get_location() != child->get_location()) {
          return;
        }
        LOG_FATAL_IF(node->isPin() && child->isPin()) << "Both of Node and child are pins, but have the same location";
        // move child's children to node, and add child to to_be_removed
        disconnect(node, child);
        auto sub_children = child->get_children();
        std::ranges::for_each(sub_children, [&](Node* sub_child) {
          disconnect(child, sub_child);
          connect(node, sub_child);
        });
        to_be_removed.push_back(child);
      });
      stack.push(node);  // recheck new children whether have same location
      continue;
    }
    // let target child be new node
    for (auto* child : children) {
      if (node->get_location() != child->get_location()) {
        continue;
      }
      auto* parent = node->get_parent();
      disconnect(node, child);
      disconnect(parent, node);
      connect(parent, child);
      std::ranges::for_each(children, [&](Node* sub_child) {
        if (sub_child == child) {
          return;
        }
        disconnect(node, sub_child);
        connect(child, sub_child);
      });
      stack.push(child);
      node->set_children({});
      to_be_removed.push_back(node);
      break;
    }
  }
  std::ranges::for_each(to_be_removed, [&](Node* node) { delete node; });
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
  return {boundSkewTree, bstSaltTree, cbsTree};
}
/**
 * @brief local place, if location is repeated, then move the driver_pin (and inst) to the feasible location
 *
 * @param driver_pin
 * @param load_pins
 */
void TreeBuilder::localPlace(Pin* driver_pin, const std::vector<Pin*>& load_pins)
{
  LocalLegalization(driver_pin, load_pins);
}
void TreeBuilder::localPlace(std::vector<Pin*>& pins)
{
  auto sovler = LocalLegalization(pins);
}
void TreeBuilder::localPlace(std::vector<Point>& variable_locs, const std::vector<Point>& fixed_locs)
{
  LocalLegalization(variable_locs, fixed_locs);
}
/**
 * @brief update tree id
 *
 * @param root
 */
void TreeBuilder::updateId(Node* root)
{
  std::vector<Node*> pin_nodes;
  std::vector<Node*> steiner_nodes;
  root->preOrder([&](Node* node) {
    if (node->isPin()) {
      pin_nodes.push_back(node);
    } else {
      steiner_nodes.push_back(node);
    }
  });
  int id = 0;
  std::ranges::for_each(pin_nodes, [&](Node* node) { node->set_id(id++); });
  std::ranges::for_each(steiner_nodes, [&](Node* node) {
    node->set_id(id++);
    node->set_name(CTSAPIInst.toString("steiner_", node->get_id()));
  });
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
  auto dir = config->get_work_dir() + "/file";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
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
      out << "\"" << node->get_name() << "\" [shape=point];\n";
    } else {
      out << "\"" << node->get_name() << "\" [shape=box];\n";
    }
  };
  std::function<void(Node*)> print_edge = [&out](Node* node) {
    for (auto child : node->get_children()) {
      out << "\"" << node->get_name() << "\" -> \"" << child->get_name() << "\";\n";
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
  auto dir = config->get_work_dir() + "/file";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  auto file_name = dir + "/" + name + ".py";
  std::ofstream out(file_name);
  if (!out.is_open()) {
    LOG_ERROR << "Can't open file: " << file_name;
    return;
  }
  out << "import matplotlib.pyplot as plt\n";
  out << "fig = plt.figure(figsize=(6,6), dpi=300)\n";
  root->preOrder([&out](Node* node) {
    auto* parent = node->get_parent();
    if (parent) {
      out << "plt.plot([" << parent->get_location().x() << "," << node->get_location().x() << "],"
          << "[" << parent->get_location().y() << "," << node->get_location().y() << "],'-k')\n";
    }
  });
  root->preOrder([&out](Node* node) {
    if (node->isPin() && node->isDriver()) {
      // more big
      out << "plt.plot(" << node->get_location().x() << "," << node->get_location().y() << ", c='r', marker='s', mew=0, ms=10)\n";
    } else if (node->isSteiner()) {
      out << "plt.plot(" << node->get_location().x() << "," << node->get_location().y() << ", c='k', marker='o', mew=0, ms=6)\n";
    } else {
      out << "plt.plot(" << node->get_location().x() << "," << node->get_location().y() << ", c='k', marker='s', mew=0, ms=8)\n";
    }
    if (node->isPin()) {
      out << "plt.text(" << node->get_location().x() << "," << node->get_location().y() << ",'[" << std::fixed << std::setprecision(4)
          << node->get_min_delay() << "," << node->get_max_delay() << "]', fontsize=6)\n";
    }
  });
  out << "plt.axis('square')\n";
  out << "plt.axis('off')\n";
  out << "plt.savefig('./" + name + ".pdf', dpi=300, bbox_inches='tight')\n";
  out.close();
}
/**
 * @brief write instance info to file
 *
 * @param root
 * @param name
 */
void TreeBuilder::writeInstInfo(Node* root, const std::string& name)
{
  auto* config = CTSAPIInst.get_config();
  auto dir = config->get_work_dir() + "/file";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  auto file_name = dir + "/" + name + ".inst";
  std::ofstream out(file_name);
  if (!out.is_open()) {
    LOG_ERROR << "Can't open file: " << file_name;
    return;
  }
  root->preOrder([&](Node* node) {
    if (node->isPin() && node->isLoad()) {
      auto* pin = dynamic_cast<Pin*>(node);
      auto* inst = pin->get_inst();
      auto type = inst->get_type() == InstType::kSink ? "sink" : "buf";
      out << node->get_name() << " " << inst->get_location().x() << " " << inst->get_location().y() << " " << node->get_cap_load() << " "
          << type << "\n";
    }
  });
  out.close();
}
}  // namespace icts