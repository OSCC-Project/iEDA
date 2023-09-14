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

#include <unordered_set>

#include "CTSAPI.hpp"
#include "CtsConfig.h"
#include "CtsDBWrapper.h"
#include "LocalLegalization.hh"
#include "Node.hh"
#include "TimingPropagator.hh"
#include "bound_skew_tree/BST.hh"
#include "bound_skew_tree/BoundSkewTree.hh"
namespace icts {

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
  auto cts_buf_inst = new CtsInstance(buf_name, TimingPropagator::getMinSizeLib()->get_cell_master(), CtsInstanceType::kBuffer, location);
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  db_wrapper->linkIdb(cts_buf_inst);
  auto buf_inst = new Inst(cts_buf_inst, InstType::kBuffer);
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
  auto cts_buf_inst = new CtsInstance(buf_name, TimingPropagator::getMinSizeLib()->get_cell_master(), CtsInstanceType::kBuffer,
                                      driver_node->get_location());
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  db_wrapper->linkIdb(cts_buf_inst);
  auto buf_inst = new Inst(cts_buf_inst, driver_node);
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
 * @brief shallow light tree
 *
 * @param loads
 * @param driver
 */
void TreeBuilder::shallowLightTree(Pin* driver, const std::vector<Pin*>& loads)
{
  CTSAPIInst.genShallowLightTree(driver, loads);
}
/**
 * @brief DME tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @return std::vector<Inst*>
 */
std::vector<Inst*> TreeBuilder::dmeTree(const std::string& net_name, const std::vector<Pin*>& loads,
                                        const std::optional<double>& skew_bound, const std::optional<Point>& guide_loc)
{
  auto solver = BST(net_name, loads, skew_bound);
  if (guide_loc.has_value()) {
    solver.set_root_guide(*guide_loc);
  }
  solver.run();
  return solver.getInsertBufs();
}
/**
 * @brief bound skew tree
 *
 * @param net_name
 * @param loads
 * @param skew_bound
 * @param guide_loc
 * @return Inst*
 */
Inst* TreeBuilder::boundSkewTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
                                 const std::optional<Point>& guide_loc)
{
  auto solver = bst::BoundSkewTree(net_name, loads, skew_bound);
  if (guide_loc.has_value()) {
    solver.set_root_guide(*guide_loc);
  }
  solver.run();
  solver.convert();
  return solver.get_root_buf();
}
/**
 * @brief recover net
 *       remove root node
 *       save the leaf node and disconnect the leaf node
 *
 * @param net
 */
void TreeBuilder::recoverNet(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  auto load_pins = net->get_load_pins();
  std::vector<Node*> to_be_removed;
  auto find_steiner = [&to_be_removed](Node* node) {
    if (node->isSteiner()) {
      to_be_removed.push_back(node);
    }
  };
  driver_pin->preOrder(find_steiner);
  // recover load pins' timing
  std::ranges::for_each(load_pins, [](Pin* pin) {
    pin->set_parent(nullptr);
    pin->set_children({});
    pin->set_slew_in(0);
    pin->set_net(nullptr);
    TimingPropagator::initLoadPinDelay(pin);
  });
  // release buffer and its pins
  auto* buffer = driver_pin->get_inst();
  delete buffer;
  // release steiner node
  std::ranges::for_each(to_be_removed, [](Node* node) { delete node; });
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
  auto dir = config->get_sta_workspace();
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
    // out << "plt.text(" << node->get_location().x() << "," << node->get_location().y() << ",'[" << node->get_min_delay() << ","
    //     << node->get_max_delay() << "]')\n";
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