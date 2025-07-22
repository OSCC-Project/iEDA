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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-06 14:45:35
 * @LastEditTime: 2023-03-09 10:04:58
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/module/global_placer/electrostatic_placer/NesterovPlace.cc
 * Contact : https://github.com/sjchanson
 */
#include "NesterovPlace.hh"

#include <limits.h>

#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cfloat>
#include <cmath>
#include <random>

#include "Log.hh"
#include "PLAPI.hh"
#include "ipl_io.h"
#include "json/json.hpp"
#include "omp.h"
#include "tool_manager.h"
#include "usage/usage.hh"

#ifdef BUILD_QT
#include "utility/Image.hh"
#endif
namespace gtl = boost::polygon;
using namespace gtl::operators;
typedef gtl::polygon_90_set_data<int> PolygonSet;

namespace ipl {

#define PRINT_LONG_NET 0
#define PRINT_COORDI 0
#define PLOT_IMAGE 1
#define RECORD_ITER_INFO 0
#define PRINT_DENSITY_MAP 0

#define SQRT2 1.414213562373095048801L

void NesterovPlace::initNesConfig(Config* config)
{
  _nes_config = config->get_nes_config();

  if (_nes_config.isOptMaxWirelength() || _nes_config.isOptTiming()) {
    _nes_config.add_opt_target_overflow(0.15);
    _nes_config.add_opt_target_overflow(0.20);
    _nes_config.add_opt_target_overflow(0.25);
    _nes_config.add_opt_target_overflow(0.30);
  }
}

void NesterovPlace::initNesDatabase(PlacerDB* placer_db)
{
  _nes_database = new NesterovDatabase();
  _nes_database->_placer_db = placer_db;

  wrapNesInstanceList();
  wrapNesNetList();
  wrapNesPinList();
  completeConnection();

  if (_nes_config.isAdaptiveBin()) {
    this->calculateAdaptiveBinCnt();
    LOG_INFO << "Change to Adaptive Bin: (" << _nes_config.get_bin_cnt_x() << "," << _nes_config.get_bin_cnt_y() << ")" << std::endl;
  }

  initGridManager();
  initTopologyManager();
  notifyPLBinSize();
  initHPWLEvaluator();
  initWAWLGradientEvaluator();
  if (_nes_config.isOptTiming()) {
    initTimingAnnotation();
  }
}

void NesterovPlace::calculateAdaptiveBinCnt()
{
  int32_t inst_cnt = _nes_database->_nInstances_range;
  int32_t side_cnt = 2;
  while (side_cnt * side_cnt < inst_cnt) {
    side_cnt *= 2;
  }

  int32_t bin_cnt_x = side_cnt;
  int32_t bin_cnt_y = side_cnt;

  const Layout* layout = _nes_database->_placer_db->get_layout();
  int32_t core_width = layout->get_core_shape().get_width();
  int32_t core_height = layout->get_core_shape().get_height();

  if (core_width > core_height) {
    bin_cnt_x *= std::pow(2, static_cast<int32_t>(round(static_cast<float>(core_width) / core_height)) - 1);
  } else {
    bin_cnt_y *= std::pow(2, static_cast<int32_t>(round(static_cast<float>(core_height) / core_width)) - 1);
  }

  _nes_config.set_bin_cnt_x(bin_cnt_x);
  _nes_config.set_bin_cnt_y(bin_cnt_y);
}

void NesterovPlace::notifyPLBinSize()
{
  int32_t bin_size_x = _nes_database->_grid_manager->get_grid_size_x();
  int32_t bin_size_y = _nes_database->_grid_manager->get_grid_size_y();

  PlacerDBInst.bin_size_x = bin_size_x;
  PlacerDBInst.bin_size_y = bin_size_y;
}

void NesterovPlace::wrapNesInstanceList()
{
  auto inst_list = _nes_database->_placer_db->get_design()->get_instance_list();

  for (auto* inst : inst_list) {
    // skip the outside inst
    if (inst->isOutsideInstance()) {
      continue;
    }

    NesInstance* n_inst = new NesInstance(inst->get_name());

    wrapNesInstance(inst, n_inst);

    _nes_database->_nInstance_list.push_back(n_inst);
    _nes_database->_nInstance_map.emplace(inst, n_inst);
    _nes_database->_instance_map.emplace(n_inst, inst);
    n_inst->set_inst_id(_nes_database->_nInstances_range);
    _nes_database->_nInstances_range += 1;
  }
}

void NesterovPlace::wrapNesInstance(Instance* inst, NesInstance* nesInst)
{
  nesInst->set_origin_shape(std::move(inst->get_shape()));

  if (inst->isFixed()) {
    nesInst->set_fixed();
  }

  if (inst->get_cell_master() && inst->get_cell_master()->isMacro()) {
    nesInst->set_macro();
  }
}

void NesterovPlace::wrapNesNetList()
{
  for (auto* net : _nes_database->_placer_db->get_design()->get_net_list()) {
    NesNet* n_net = new NesNet(net->get_name());

    wrapNesNet(net, n_net);

    _nes_database->_nNet_list.push_back(n_net);
    _nes_database->_nNet_map.emplace(net, n_net);
    _nes_database->_net_map.emplace(n_net, net);
    n_net->set_net_id(_nes_database->_nNets_range);
    _nes_database->_nNets_range += 1;
  }
}

void NesterovPlace::wrapNesNet(Net* net, NesNet* nesNet)
{
  nesNet->set_weight(net->get_net_weight());

  if (net->isDontCareNet()) {
    nesNet->set_dont_care();
  }
}

void NesterovPlace::wrapNesPinList()
{
  for (auto* pin : _nes_database->_placer_db->get_design()->get_pin_list()) {
    NesPin* n_pin = new NesPin(pin->get_name());

    wrapNesPin(pin, n_pin);

    _nes_database->_nPin_list.push_back(n_pin);
    _nes_database->_nPin_map.emplace(pin, n_pin);
    _nes_database->_pin_map.emplace(n_pin, pin);
    n_pin->set_pin_id(_nes_database->_nPins_range);
    _nes_database->_nPins_range += 1;
  }
}

void NesterovPlace::wrapNesPin(Pin* pin, NesPin* nesPin)
{
  int32_t origin_offset_x = pin->get_offset_coordi().get_x();
  int32_t origin_offset_y = pin->get_offset_coordi().get_y();

  auto* inst = pin->get_instance();
  if (!inst) {
    nesPin->set_offset_coordi(Point<int32_t>(origin_offset_x, origin_offset_y));
  } else {
    int32_t modify_offset_x = 0;
    int32_t modify_offset_y = 0;

    Orient inst_orient = inst->get_orient();
    if (inst_orient == Orient::kN_R0) {
      modify_offset_x = origin_offset_x;
      modify_offset_y = origin_offset_y;
    } else if (inst_orient == Orient::kW_R90) {
      modify_offset_x = (-1) * origin_offset_y;
      modify_offset_y = origin_offset_x;
    } else if (inst_orient == Orient::kS_R180) {
      modify_offset_x = (-1) * origin_offset_x;
      modify_offset_y = (-1) * origin_offset_y;
    } else if (inst_orient == Orient::kFW_MX90) {
      modify_offset_x = origin_offset_y;
      modify_offset_y = origin_offset_x;
    } else if (inst_orient == Orient::kFN_MY) {
      modify_offset_x = (-1) * origin_offset_x;
      modify_offset_y = origin_offset_y;
    } else if (inst_orient == Orient::kFE_MY90) {
      modify_offset_x = (-1) * origin_offset_y;
      modify_offset_y = (-1) * origin_offset_x;
    } else if (inst_orient == Orient::kFS_MX) {
      modify_offset_x = origin_offset_x;
      modify_offset_y = (-1) * origin_offset_y;
    } else if (inst_orient == Orient::kE_R270) {
      modify_offset_x = origin_offset_y;
      modify_offset_y = (-1) * origin_offset_x;
    } else {
      LOG_WARNING << inst->get_name() + " has not the orient!";
    }
    nesPin->set_offset_coordi(Point<int32_t>(modify_offset_x, modify_offset_y));
  }
  nesPin->set_center_coordi(std::move(pin->get_center_coordi()));
}

void NesterovPlace::completeConnection()
{
  for (auto pair : _nes_database->_nPin_map) {
    auto* pin = pair.first;
    auto* n_pin = pair.second;

    auto* inst = pin->get_instance();

    // skip the outside inst.
    if (inst && !inst->isOutsideInstance()) {
      auto* n_inst = _nes_database->_nInstance_map[inst];

      n_pin->set_nInstance(n_inst);
      n_inst->add_nPin(n_pin);
    }

    auto* net = pin->get_net();
    auto* n_net = _nes_database->_nNet_map[net];

    n_pin->set_nNet(n_net);

    auto* driver_pin = net->get_driver_pin();
    if (driver_pin) {
      if (driver_pin->get_name() == pin->get_name()) {
        n_net->set_driver(n_pin);
      } else {
        n_net->add_loader(n_pin);
      }
    } else {
      n_net->add_loader(n_pin);
    }
  }
}

void NesterovPlace::initGridManager()
{
  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  int32_t grid_cnt_x = _nes_config.get_bin_cnt_x();
  int32_t grid_cnt_y = _nes_config.get_bin_cnt_y();
  float target_density = _nes_config.get_target_density();

  GridManager* grid_manager = new GridManager(core_shape, grid_cnt_x, grid_cnt_y, target_density, _nes_config.get_thread_num());
  _nes_database->_grid_manager = grid_manager;

  // BinGrid specially need to parallel
  _nes_database->_bin_grid = new BinGrid(grid_manager);
  _nes_database->_bin_grid->set_thread_nums(_nes_config.get_thread_num());
  int route_cap_h = _nes_database->_placer_db->get_layout()->get_route_cap_h();
  int route_cap_v = _nes_database->_placer_db->get_layout()->get_route_cap_v();
  int partial_route_cap_h = _nes_database->_placer_db->get_layout()->get_partial_route_cap_h();
  int partial_route_cap_v = _nes_database->_placer_db->get_layout()->get_partial_route_cap_v();
  _nes_database->_bin_grid->set_route_cap_h(route_cap_h);
  _nes_database->_bin_grid->set_route_cap_v(route_cap_v);
  _nes_database->_bin_grid->set_partial_route_cap_h(partial_route_cap_h);
  _nes_database->_bin_grid->set_partial_route_cap_v(partial_route_cap_v);

  _nes_database->_density = new Density(grid_manager);
  _nes_database->_density_gradient = new ElectricFieldGradient(grid_manager);  // TODO : be optional.
}

void NesterovPlace::initTopologyManager()
{
  TopologyManager* topo_manager = new TopologyManager();
  _nes_database->_topology_manager = topo_manager;

  this->initNodes();
  this->initNetWorks();
  this->initGroups();
  this->initArcs();

  if (_nes_config.isOptTiming()) {
    topo_manager->updateALLNodeTopoId();
  }
}

void NesterovPlace::initNodes()
{
  auto* topo_manager = _nes_database->_topology_manager;
  for (auto pair : _nes_database->_pin_map) {
    auto* n_pin = pair.first;
    Node* node = new Node(n_pin->get_name());
    node->set_location(std::move(n_pin->get_center_coordi()));
    auto* pl_pin = pair.second;

    // set node type.
    if (pl_pin->isInstanceInput() || pl_pin->isIOInput()) {
      node->set_node_type(NODE_TYPE::kInput);
    } else if (pl_pin->isInstanceOutput() || pl_pin->isIOOutput()) {
      node->set_node_type(NODE_TYPE::kOutput);
    } else if (pl_pin->isInstanceInputOutput() || pl_pin->isIOInputOutput()) {
      node->set_node_type(NODE_TYPE::kInputOutput);
    } else {
      node->set_node_type(NODE_TYPE::kNone);
    }

    // set is io node
    if (pl_pin->isIOPort()) {
      node->set_is_io();
    }
    topo_manager->add_node(node);
    node->set_node_id(n_pin->get_pin_id());  // not match in origin order
  }
  topo_manager->sortNodeList();
}

void NesterovPlace::initNetWorks()
{
  auto* topo_manager = _nes_database->_topology_manager;
  for (auto pair : _nes_database->_net_map) {
    auto* n_net = pair.first;
    NetWork* network = new NetWork(n_net->get_name());

    network->set_net_weight(n_net->get_weight());

    auto* pl_net = pair.second;

    // set network type.
    if (pl_net->isClockNet()) {
      network->set_network_type(NETWORK_TYPE::kClock);
    } else if (pl_net->isFakeNet()) {
      network->set_network_type(NETWORK_TYPE::kFakeNet);
    } else if (pl_net->isSignalNet()) {
      network->set_network_type(NETWORK_TYPE::kSignal);
    } else {
      network->set_network_type(NETWORK_TYPE::kNone);
    }

    NesPin* driver = n_net->get_driver();
    if (driver) {
      Node* transmitter = topo_manager->findNodeById(driver->get_pin_id());
      transmitter->set_network(network);
      network->set_transmitter(transmitter);
    }

    for (auto* loader : n_net->get_loader_list()) {
      Node* receiver = topo_manager->findNodeById(loader->get_pin_id());
      receiver->set_network(network);
      network->add_receiver(receiver);
    }

    topo_manager->add_network(network);
    network->set_network_id(n_net->get_net_id());  // not match in origin order.
  }
  topo_manager->sortNetworkList();
}

void NesterovPlace::initGroups()
{
  auto* topo_manager = _nes_database->_topology_manager;
  for (auto pair : _nes_database->_instance_map) {
    auto* n_inst = pair.first;
    Group* group = new Group(n_inst->get_name());

    auto* pl_inst = pair.second;
    // set group type.
    auto* cell_master = pl_inst->get_cell_master();
    if (cell_master) {
      if (cell_master->isFlipflop()) {
        group->set_group_type(GROUP_TYPE::kFlipflop);
      } else if (cell_master->isClockBuffer()) {
        group->set_group_type(GROUP_TYPE::kClockBuffer);
      } else if (cell_master->isLogicBuffer()) {
        group->set_group_type(GROUP_TYPE::kLogicBuffer);
      } else if (cell_master->isMacro()) {
        group->set_group_type(GROUP_TYPE::kMacro);
      } else if (cell_master->isIOCell()) {
        group->set_group_type(GROUP_TYPE::kIOCell);
      } else if (cell_master->isLogic()) {
        group->set_group_type(GROUP_TYPE::kLogic);
      } else {
        group->set_group_type(GROUP_TYPE::kNone);
      }
    } else {
      group->set_group_type(GROUP_TYPE::kNone);
    }

    for (auto* n_pin : n_inst->get_nPin_list()) {
      Node* node = topo_manager->findNodeById(n_pin->get_pin_id());
      node->set_group(group);
      group->add_node(node);
    }
    topo_manager->add_group(group);
    group->set_group_id(pl_inst->get_inst_id());  // not match in origin order.
  }
  topo_manager->sortGroupList();
}

void NesterovPlace::initArcs()
{
  auto* topo_manager = _nes_database->_topology_manager;
  for (auto* node : topo_manager->get_node_list()) {
    // TODO: Consider the INPUT & OUTPUT Case.
    if (node->is_io_node()) {
      if (node->get_node_type() == NODE_TYPE::kOutput) {
        this->generateNetArc(node);
      }

    } else {
      if (node->get_node_type() == NODE_TYPE::kInput) {
        this->generateNetArc(node);
      } else if (node->get_node_type() == NODE_TYPE::kOutput) {
        this->generateGroupArc(node);
      }
    }
  }
  topo_manager->sortArcList();
}

void NesterovPlace::generatePortOutNetArc(Node* node)
{
  auto* topo_manager = _nes_database->_topology_manager;
  auto* network = node->get_network();
  if (network) {
    for (auto* sink_node : network->get_receiver_list()) {
      Arc* net_arc = new Arc(node, sink_node);
      net_arc->set_arc_type(ARC_TYPE::kNetArc);
      node->add_output_arc(net_arc);
      sink_node->add_input_arc(net_arc);
      topo_manager->add_arc(net_arc);
    }
  }
}

void NesterovPlace::generateNetArc(Node* node)
{
  auto* topo_manager = _nes_database->_topology_manager;
  auto* network = node->get_network();
  if (network) {
    auto* driver_node = network->get_transmitter();
    if (driver_node) {
      Arc* net_arc = new Arc(driver_node, node);
      net_arc->set_arc_type(ARC_TYPE::kNetArc);
      driver_node->add_output_arc(net_arc);
      node->add_input_arc(net_arc);
      topo_manager->add_arc(net_arc);
    }
  }
}

void NesterovPlace::generateGroupArc(Node* node)
{
  auto* topo_manager = _nes_database->_topology_manager;
  auto* group = node->get_group();
  if (group) {
    auto input_list = group->obtainInputNodes();
    for (auto* input_node : input_list) {
      NetWork* input_net = input_node->get_network();
      if (input_net->get_network_type() != NETWORK_TYPE::kClock && group->get_group_type() == GROUP_TYPE::kFlipflop) {
        continue;
      }

      Arc* group_arc = new Arc(input_node, node);
      group_arc->set_arc_type(ARC_TYPE::kGroupArc);
      input_node->add_output_arc(group_arc);
      node->add_input_arc(group_arc);
      topo_manager->add_arc(group_arc);
    }
  }
}

void NesterovPlace::initHPWLEvaluator()
{
  _nes_database->_wirelength = new HPWirelength(_nes_database->_topology_manager);
}

void NesterovPlace::initWAWLGradientEvaluator()
{
  _nes_database->_wirelength_gradient = new WAWirelengthGradient(_nes_database->_topology_manager);
}

void NesterovPlace::initTimingAnnotation()
{
  _nes_database->_timing_annotation = new TimingAnnotation(_nes_database->_topology_manager);
}

void NesterovPlace::initFillerNesInstance()
{
  std::vector<int32_t> edge_x_assemble;
  std::vector<int32_t> edge_y_assemble;

  // record area.
  int64_t nonplace_area = 0;
  int64_t occupied_area = 0;

  Rectangle<int32_t> core_rect = _nes_database->_placer_db->get_layout()->get_core_shape();
  PolygonSet ps;
  for (auto* n_inst : _nes_database->_nInstance_list) {
    Rectangle<int32_t> n_inst_shape = n_inst->get_origin_shape();
    int64_t shape_area_x = static_cast<int64_t>(n_inst_shape.get_width());
    int64_t shape_area_y = static_cast<int64_t>(n_inst_shape.get_height());

    // skip fixed nInsts.
    if (n_inst->isFixed()) {
      ps.insert(
          gtl::rectangle_data<int>(n_inst_shape.get_ll_x(), n_inst_shape.get_ll_y(), n_inst_shape.get_ur_x(), n_inst_shape.get_ur_y()));
      continue;
    }
    if (n_inst->isMacro()) {
      occupied_area += shape_area_x * shape_area_y * _nes_config.get_target_density();
    } else {
      occupied_area += shape_area_x * shape_area_y;
    }

    edge_x_assemble.push_back(shape_area_x);
    edge_y_assemble.push_back(shape_area_y);
  }
  for (auto* blockage : _nes_database->_placer_db->get_design()->get_region_list()) {
    for (auto boundary : blockage->get_boundaries()) {
      // non_place_instance_area += boundary_width * boundary_height;
      auto rect = gtl::rectangle_data<int>(boundary.get_ll_x(), boundary.get_ll_y(), boundary.get_ur_x(), boundary.get_ur_y());
      ps.insert(rect);
    }
  }
  ps &= gtl::rectangle_data<int>(core_rect.get_ll_x(), core_rect.get_ll_y(), core_rect.get_ur_x(), core_rect.get_ur_y());

  std::vector<gtl::rectangle_data<int>> rects;
  ps.get_rectangles(rects);

  for (const auto& rect : rects) {
    nonplace_area += 1LL * boost::polygon::area(rect);
  }

  // sort
  std::sort(edge_x_assemble.begin(), edge_x_assemble.end());
  std::sort(edge_y_assemble.begin(), edge_y_assemble.end());

  int64_t edge_x_sum = 0, edge_y_sum = 0;

  int min_idx = edge_x_assemble.size() * 0.05;
  int max_idx = edge_y_assemble.size() * 0.95;
  for (int i = min_idx; i < max_idx; i++) {
    edge_x_sum += edge_x_assemble[i];
    edge_y_sum += edge_y_assemble[i];
  }

  int avg_edge_x = static_cast<int>(edge_x_sum / (max_idx - min_idx));
  int avg_edge_y = static_cast<int>(edge_y_sum / (max_idx - min_idx));

  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  int64_t core_area = static_cast<int64_t>(core_shape.get_width()) * static_cast<int64_t>(core_shape.get_height());
  int64_t white_space_area = core_area - nonplace_area;
  int64_t movable_area = white_space_area * _nes_config.get_target_density();
  int64_t total_filler_area = movable_area - occupied_area;

  LOG_ERROR_IF(total_filler_area < 0) << "Detect: Negative filler area \n"
                                      << "       The reason may be target_density setting: try to improve target_density \n";

  std::mt19937 rand_val(0);

  // int32_t filler_cnt = total_filler_area / (avg_edge_x * avg_edge_y);

  // test
  int32_t filler_cnt = std::ceil(static_cast<int32_t>(static_cast<float>(total_filler_area / (avg_edge_x * avg_edge_y))));

  for (int i = 0; i < filler_cnt; i++) {
    auto rand_x = rand_val();
    auto rand_y = rand_val();

    NesInstance* filler = new NesInstance("filler_" + std::to_string(i));

    int32_t center_x = rand_x % core_shape.get_width() + core_shape.get_ll_x();
    int32_t center_y = rand_y % core_shape.get_height() + core_shape.get_ll_y();
    int32_t lower_x = center_x - avg_edge_x / 2;
    int32_t lower_y = center_y - avg_edge_y / 2;

    filler->set_filler();
    filler->set_origin_shape(Rectangle<int32_t>(lower_x, lower_y, lower_x + avg_edge_x, lower_y + avg_edge_y));

    _nes_database->_nInstance_list.push_back(filler);
    filler->set_inst_id(_nes_database->_nInstances_range);
    _nes_database->_nInstances_range += 1;
  }
}

void NesterovPlace::initNesInstanceDensitySize()
{
  for (auto* n_inst : _nes_database->_nInstance_list) {
    if (n_inst->isFixed()) {
      n_inst->set_density_shape(n_inst->get_origin_shape());
      n_inst->set_density_scale(1.0F);
      continue;
    }

    float scale_x = 0, scale_y = 0;
    int32_t density_size_x = 0, density_size_y = 0;

    Rectangle<int32_t> first_grid_shape = this->obtainFirstGridShape();

    int32_t grid_size_x = first_grid_shape.get_width();
    int32_t grid_size_y = first_grid_shape.get_height();

    Rectangle<int32_t> n_inst_shape = n_inst->get_origin_shape();
    int32_t n_inst_width = n_inst_shape.get_width();
    int32_t n_inst_height = n_inst_shape.get_height();

    if (n_inst_width < static_cast<int32_t>(SQRT2 * grid_size_x)) {
      scale_x = static_cast<float>(n_inst_width) / (SQRT2 * grid_size_x);
      density_size_x = static_cast<int32_t>(SQRT2 * grid_size_x);
    } else {
      scale_x = 1.0F;
      density_size_x = n_inst_width;
    }

    if (n_inst_height < static_cast<int32_t>(SQRT2 * grid_size_y)) {
      scale_y = static_cast<float>(n_inst_height) / (SQRT2 * grid_size_y);
      density_size_y = static_cast<int32_t>(SQRT2 * grid_size_y);
    } else {
      scale_y = 1.0F;
      density_size_y = n_inst_height;
    }

    n_inst->set_density_shape(Rectangle<int32_t>(n_inst_shape.get_ll_x(), n_inst_shape.get_ll_y(), n_inst_shape.get_ll_x() + density_size_x,
                                                 n_inst_shape.get_ll_y() + density_size_y));
    n_inst->set_density_scale(scale_x * scale_y);
  }
}

void NesterovPlace::runNesterovPlace()
{
  std::cout << std::endl;
  LOG_INFO << "-----------------Start Global Placement-----------------";
  ieda::Stats gp_status;

  std::vector<NesInstance*> placable_inst_list = std::move(this->obtianPlacableNesInstanceList());
  initNesterovPlace(placable_inst_list);

  // main
  NesterovSolve(placable_inst_list);
  PlacerDBInst.updateTopoManager();
  PlacerDBInst.updateGridManager();

  double time_delta = gp_status.elapsedRunTime();
  LOG_INFO << "Global Placement Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Global Placement-----------------";
}

void NesterovPlace::initNesterovPlace(std::vector<NesInstance*>& inst_list)
{
  size_t inst_size = inst_list.size();

  std::vector<Point<int32_t>> prev_coordi_list(inst_size, Point<int32_t>());
  std::vector<Point<float>> prev_wirelength_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> prev_density_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> prev_sum_grad_list(inst_size, Point<float>());
  std::vector<Point<int32_t>> current_coordi_list(inst_size, Point<int32_t>());
  std::vector<Point<float>> current_wirelength_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> current_density_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> current_sum_grad_list(inst_size, Point<float>());

  // initial coordi vector.
  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();

#pragma omp parallel for num_threads(_nes_config.get_thread_num())
  for (size_t i = 0; i < inst_size; i++) {
    auto* n_inst = inst_list[i];
    this->updateDensityCoordiLayoutInside(n_inst, core_shape);
    current_coordi_list[i] = Point<int32_t>(n_inst->get_density_center_coordi().get_x(), n_inst->get_density_center_coordi().get_y());
  }

  initGridFixedArea();
  // update current density gradient force.
  _nes_database->_bin_grid->updateBinGrid(inst_list, _nes_config.get_thread_num());
  _nes_database->_density_gradient->updateDensityForce(_nes_config.get_thread_num(), false);

  _total_inst_area = this->obtainTotalArea(inst_list);
  float sum_overflow = static_cast<float>(_nes_database->_bin_grid->get_overflow_area_without_filler()) / _total_inst_area;

  // update current wirelength gradient force.
  initBaseWirelengthCoef();
  updateWirelengthCoef(sum_overflow);
  float wirelength_coef = _nes_database->_wirelength_coef;
  updateTopologyManager();
  _nes_database->_wirelength_gradient->updateWirelengthForce(wirelength_coef, wirelength_coef, _nes_config.get_min_wirelength_force_bar(),
                                                             _nes_config.get_thread_num());

  // update current target penalty object.
  updatePenaltyGradient(inst_list, current_sum_grad_list, current_wirelength_grad_list, current_density_grad_list, false);

  if (_nes_database->_is_diverged) {
    return;
  }

  // update initial prev coordinates.
#pragma omp parallel for num_threads(_nes_config.get_thread_num())
  for (size_t i = 0; i < inst_size; i++) {
    auto* n_inst = inst_list[i];

    int32_t prev_coordi_x
        = current_coordi_list[i].get_x() + _nes_config.get_initial_prev_coordi_update_coef() * current_sum_grad_list[i].get_x();
    int32_t prev_coordi_y
        = current_coordi_list[i].get_y() + _nes_config.get_initial_prev_coordi_update_coef() * current_sum_grad_list[i].get_y();
    Point<int32_t> prev_coordi(prev_coordi_x, prev_coordi_y);
    n_inst->updateDensityCenterLocation(prev_coordi);
    this->updateDensityCoordiLayoutInside(n_inst, core_shape);
    prev_coordi_list[i] = Point<int32_t>(n_inst->get_density_center_coordi().get_x(), n_inst->get_density_center_coordi().get_y());
  }

  // update prev density gradient force.
  _nes_database->_bin_grid->updateBinGrid(inst_list, _nes_config.get_thread_num());
  _nes_database->_density_gradient->updateDensityForce(_nes_config.get_thread_num(), false);

  // update prev wirelength gradient force.
  updateTopologyManager();
  _nes_database->_wirelength_gradient->updateWirelengthForce(wirelength_coef, wirelength_coef, _nes_config.get_min_wirelength_force_bar(),
                                                             _nes_config.get_thread_num());

  // update prev target penalty object.
  updatePenaltyGradient(inst_list, prev_sum_grad_list, prev_wirelength_grad_list, prev_density_grad_list, false);

  if (_nes_database->_is_diverged) {
    return;
  }

  // init quad penalty
  initQuadPenaltyCoeff();

  // init density penalty.
  _nes_database->_density_penalty
      = (_nes_database->_wirelength_grad_sum / _nes_database->_density_grad_sum) * _nes_config.get_init_density_penalty();

  // init nesterov solver.
  _nes_database->_nesterov_solver->initNesterov(prev_coordi_list, prev_sum_grad_list, current_coordi_list, current_sum_grad_list);

  // Test Preconditon.
  // initDiagonalIdentityMatrix(inst_size);
  // initDiagonalHkMatrix(inst_list);
  // initDiagonalSkMatrix(inst_list);
}

std::vector<NesInstance*> NesterovPlace::obtianPlacableNesInstanceList()
{
  std::vector<NesInstance*> placable_list;
  for (auto* n_inst : _nes_database->_nInstance_list) {
    if (n_inst->isFixed()) {
      continue;
    }
    placable_list.push_back(n_inst);
  }
  return placable_list;
}

void NesterovPlace::updateDensityCoordiLayoutInside(NesInstance* n_inst, Rectangle<int32_t> core_shape)
{
  int32_t target_lower_x = n_inst->get_density_coordi().get_x();
  int32_t target_lower_y = n_inst->get_density_coordi().get_y();
  int32_t target_edge_x = n_inst->get_density_shape().get_width();
  int32_t target_edge_y = n_inst->get_density_shape().get_height();

  if (target_lower_x < core_shape.get_ll_x()) {
    target_lower_x = core_shape.get_ll_x();
  }

  if (target_lower_y < core_shape.get_ll_y()) {
    target_lower_y = core_shape.get_ll_y();
  }

  if (target_lower_x + target_edge_x > core_shape.get_ur_x()) {
    target_lower_x = core_shape.get_ur_x() - target_edge_x;
  }

  if (target_lower_y + target_edge_y > core_shape.get_ur_y()) {
    target_lower_y = core_shape.get_ur_y() - target_edge_y;
  }

  n_inst->updateDensityLocation(Point<int32_t>(target_lower_x, target_lower_y));
}

void NesterovPlace::updateDensityCenterCoordiLayoutInside(NesInstance* n_inst, Point<int32_t>& center_coordi, Rectangle<int32_t> core_shape)
{
  int32_t target_edge_x = n_inst->get_density_shape().get_width();
  int32_t target_edge_y = n_inst->get_density_shape().get_height();

  int32_t target_lower_x = center_coordi.get_x() - 0.5 * target_edge_x;
  int32_t target_lower_y = center_coordi.get_y() - 0.5 * target_edge_y;

  if (target_lower_x < core_shape.get_ll_x()) {
    target_lower_x = core_shape.get_ll_x();
  }

  if (target_lower_y < core_shape.get_ll_y()) {
    target_lower_y = core_shape.get_ll_y();
  }

  if (target_lower_x + target_edge_x > core_shape.get_ur_x()) {
    target_lower_x = core_shape.get_ur_x() - target_edge_x;
  }

  if (target_lower_y + target_edge_y > core_shape.get_ur_y()) {
    target_lower_y = core_shape.get_ur_y() - target_edge_y;
  }

  center_coordi = Point<int32_t>(target_lower_x + std::ceil(0.5 * target_edge_x), target_lower_y + std::ceil(0.5 * target_edge_y));
}

void NesterovPlace::initGridFixedArea()
{
  auto* grid_manager = _nes_database->_grid_manager;

  grid_manager->clearAllOccupiedArea();

  // #pragma omp parallel for num_threads(_nes_config.get_thread_num())
  for (auto* n_inst : _nes_database->_nInstance_list) {
    if (!n_inst->isFixed()) {
      continue;
    }

    std::vector<Grid*> overlap_grid_list;
    auto origin_shape = std::move(n_inst->get_origin_shape());
    grid_manager->obtainOverlapGridList(overlap_grid_list, origin_shape);
    for (auto* grid : overlap_grid_list) {
      int64_t overlap_area = grid_manager->obtainOverlapArea(grid, n_inst->get_origin_shape());

      // #pragma omp atomic
      grid->fixed_area += overlap_area * grid->available_ratio;
      // grid->add_fixed_area(overlap_area * grid->get_available_ratio());
    }
  }

  // add blockage.
  auto region_list = _nes_database->_placer_db->get_design()->get_region_list();
  for (auto* region : region_list) {
    if (region->isFence()) {
      std::vector<Grid*> overlap_grid_list;
      for (auto boundary : region->get_boundaries()) {
        grid_manager->obtainOverlapGridList(overlap_grid_list, boundary);
        for (auto* grid : overlap_grid_list) {
          // tmp fix overlap area between fixed inst and blockage.
          if (grid->fixed_area != 0) {
            continue;
          }

          // if (grid->get_fixed_area() != 0) {
          //   continue;
          // }
          int64_t overlap_area = grid_manager->obtainOverlapArea(grid, boundary);

          grid->fixed_area += overlap_area * grid->available_ratio;
          // grid->add_fixed_area(overlap_area * grid->get_available_ratio());
        }
      }
    }
  }
}

void NesterovPlace::updateTopologyManager()
{
  auto* topo_manager = _nes_database->_topology_manager;
  int32_t thread_num = _nes_config.get_thread_num();

  int32_t net_chunk_size = std::max(int(_nes_database->_nNet_list.size() / thread_num / 16), 1);
#pragma omp parallel for num_threads(thread_num) schedule(dynamic, net_chunk_size)
  for (auto* n_net : _nes_database->_nNet_list) {
    if (n_net->isDontCare()) {
      continue;
    }
    auto* network = topo_manager->findNetworkById(n_net->get_net_id());
    network->set_net_weight(n_net->get_weight());
  }

  int32_t pin_chunk_size = std::max(int(_nes_database->_nPin_list.size() / thread_num / 16), 1);
#pragma omp parallel for num_threads(thread_num) schedule(dynamic, pin_chunk_size)
  for (auto* n_pin : _nes_database->_nPin_list) {
    auto* node = topo_manager->findNodeById(n_pin->get_pin_id());
    node->set_location(n_pin->get_center_coordi());
  }
}

Rectangle<int32_t> NesterovPlace::obtainFirstGridShape()
{
  auto& first_grid = _nes_database->_grid_manager->get_grid_2d_list()[0][0];

  return first_grid.shape;

  // auto* first_grid = _nes_database->_grid_manager->get_row_list()[0]->get_grid_list()[0];

  // return first_grid->get_shape();
}

int64_t NesterovPlace::obtainTotalArea(std::vector<NesInstance*>& inst_list)
{
  int64_t total_area = 0;
  for (auto* n_inst : inst_list) {
    if (n_inst->isFiller()) {
      continue;
    }

    int64_t n_inst_width = static_cast<int64_t>(n_inst->get_origin_shape().get_width());
    int64_t n_inst_height = static_cast<int64_t>(n_inst->get_origin_shape().get_height());

    if (n_inst->isMacro()) {
      total_area += static_cast<int64_t>(n_inst_width * n_inst_height * _nes_config.get_target_density());
    } else {
      total_area += n_inst_width * n_inst_height;
    }
  }

  return total_area;
}

int64_t NesterovPlace::obtainTotalFillerArea(std::vector<NesInstance*>& inst_list)
{
  int64_t total_filler_area = 0;

  for (auto* n_inst : inst_list) {
    if (n_inst->isFiller()) {
      int64_t n_inst_width = static_cast<int64_t>(n_inst->get_origin_shape().get_width());
      int64_t n_inst_height = static_cast<int64_t>(n_inst->get_origin_shape().get_height());
      total_filler_area += n_inst_width * n_inst_height;
    }
  }

  return total_filler_area;
}

void NesterovPlace::initBaseWirelengthCoef()
{
  Rectangle<int32_t> first_grid_shape = this->obtainFirstGridShape();
  _nes_database->_base_wirelength_coef
      = _nes_config.get_init_wirelength_coef() / (static_cast<float>(first_grid_shape.get_half_perimeter()) * 0.5);
}

void NesterovPlace::updateWirelengthCoef(float overflow)
{
  if (overflow > 1.0) {
    _nes_database->_wirelength_coef = 0.1;
  } else if (overflow < 0.1) {
    _nes_database->_wirelength_coef = 10.0;
  } else {
    _nes_database->_wirelength_coef = 1.0 / pow(10.0, (overflow - 0.1) * 20 / 9.0 - 1.0);
  }

  _nes_database->_wirelength_coef *= _nes_database->_base_wirelength_coef;
}

void NesterovPlace::initDiagonalIdentityMatrix(int32_t inst_size)
{
  _global_diagonal_list.resize(2 * inst_size, 1.0);
}

void NesterovPlace::initDiagonalHkMatrix(std::vector<NesInstance*>& inst_list)
{
  size_t inst_size = inst_list.size();
  _global_diagonal_list.resize(2 * inst_size, 1.0);

  for (size_t i = 0; i < inst_size; i++) {
    auto* cur_n_inst = inst_list[i];
    Point<float> wirelength_precondition = std::move(obtainWirelengthPrecondition(cur_n_inst));
    Point<float> density_precondition = std::move(obtainDensityPrecondition(cur_n_inst));
    float coeff = wirelength_precondition.get_x() + _nes_database->_density_penalty * density_precondition.get_x();
    _global_diagonal_list[i] = static_cast<double>(coeff);
    _global_diagonal_list[inst_size + i] = static_cast<double>(coeff);
  }
}

void NesterovPlace::initDiagonalSkMatrix(std::vector<NesInstance*>& inst_list)
{
  size_t inst_size = inst_list.size();
  _global_diagonal_list.resize(2 * inst_size, 1.0);

  for (size_t i = 0; i < inst_size; i++) {
    auto* cur_n_inst = inst_list[i];
    Point<float> wirelength_precondition = std::move(obtainWirelengthPrecondition(cur_n_inst));
    Point<float> density_precondition = std::move(obtainDensityPrecondition(cur_n_inst));
    float coeff = wirelength_precondition.get_x() + _nes_database->_density_penalty * density_precondition.get_x();
    _global_diagonal_list[i] = static_cast<double>(1 / coeff);
    _global_diagonal_list[inst_size + i] = static_cast<double>(1 / coeff);
  }
}

void NesterovPlace::updatePenaltyGradientPre1(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                              std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads)
{
  // if((_nes_database->_nesterov_solver->get_current_iter()) % 100 == 0){
  //   LOG_INFO << "Iteration: "<< _nes_database->_nesterov_solver->get_current_iter() <<" Reset precondition.";
  //   _global_diagonal_list.clear();
  //   initDiagonalHkMatrix(nInst_list);
  // }
  _nes_database->_density_grad_sum = 0.0F;

  // const auto& front_coordi_list = _nes_database->_nesterov_solver->get_current_coordis();
  // const auto& current_coordi_list = _nes_database->_nesterov_solver->get_next_coordis();
  // const auto& front_grad_list = _nes_database->_nesterov_solver->get_current_grads();
  // const auto& current_grad_list = _nes_database->_nesterov_solver->get_next_grads();

  // // prcondition method 1
  // size_t inst_size = nInst_list.size();
  // std::vector<double> front_ski_x(inst_size);
  // std::vector<double> front_ski_y(inst_size);
  // std::vector<double> front_yki_x(inst_size);
  // std::vector<double> front_yki_y(inst_size);
  // double sk2hk_variable = 0.0;
  // double yksk_variable = 0.0;
  // for(size_t i=0; i < inst_size; i++){
  //   front_ski_x[i] = current_coordi_list[i].get_x() - front_coordi_list[i].get_x();
  //   front_ski_y[i] = current_coordi_list[i].get_y() - front_coordi_list[i].get_y();
  //   sk2hk_variable += (front_ski_x[i] * front_ski_x[i] * _global_diagonal_list[i]);
  //   sk2hk_variable += (front_ski_y[i] * front_ski_y[i] * _global_diagonal_list[i+inst_size]);

  //   front_yki_x[i] = current_grad_list[i].get_x() - front_grad_list[i].get_x();
  //   front_yki_y[i] = current_grad_list[i].get_y() - front_grad_list[i].get_y();
  //   yksk_variable += (front_yki_x[i] * front_ski_x[i] + front_yki_y[i] * front_ski_y[i]);
  // }

  for (size_t i = 0; i < nInst_list.size(); i++) {
    auto& cur_n_inst = nInst_list[i];

    wirelength_grads[i] = std::move(_nes_database->_wirelength_gradient->obtainWirelengthGradient(
        cur_n_inst->get_inst_id(), _nes_database->_wirelength_coef, _nes_database->_wirelength_coef));
    density_grads[i] = std::move(
        _nes_database->_density_gradient->obtainDensityGradient(cur_n_inst->get_density_shape(), cur_n_inst->get_density_scale(), 0, 0.0f));

    _nes_database->_wirelength_grad_sum += fabs(wirelength_grads[i].get_x());
    _nes_database->_wirelength_grad_sum += fabs(wirelength_grads[i].get_y());

    _nes_database->_density_grad_sum += fabs(density_grads[i].get_x());
    _nes_database->_density_grad_sum += fabs(density_grads[i].get_y());

    sum_grads[i].set_x(wirelength_grads[i].get_x() + _nes_database->_density_penalty * density_grads[i].get_x());
    sum_grads[i].set_y(wirelength_grads[i].get_y() + _nes_database->_density_penalty * density_grads[i].get_y());

    // // prcondition method 1
    // double hksk_x = _global_diagonal_list[i] * front_ski_x[i];
    // double hksk_y = _global_diagonal_list[inst_size + i] * front_ski_y[i];
    // float hki_x = static_cast<float>(_global_diagonal_list[i] - (hksk_x * hksk_x) / sk2hk_variable + (front_yki_x[i] * front_yki_x[i]) /
    // yksk_variable); float hki_y = static_cast<float>(_global_diagonal_list[inst_size + i] - (hksk_y * hksk_y) / sk2hk_variable +
    // (front_yki_y[i] * front_yki_y[i]) / yksk_variable); Point<float> sum_precondition(hki_x,hki_y);

    // if (sum_precondition.get_x() <= _nes_config.get_min_precondition()) {
    //   sum_precondition.set_x(_nes_config.get_min_precondition());
    // }
    // if (sum_precondition.get_y() <= _nes_config.get_min_precondition()) {
    //   sum_precondition.set_y(_nes_config.get_min_precondition());
    // }

    Point<float> sum_precondition(1.0, 1.0);

    sum_grads[i].set_x(sum_grads[i].get_x() / sum_precondition.get_x());
    sum_grads[i].set_y(sum_grads[i].get_y() / sum_precondition.get_y());

    // _global_diagonal_list[i] = hki_x;
    // _global_diagonal_list[inst_size + i] = hki_y;
  }

  if (std::isnan(_nes_database->_wirelength_grad_sum) || std::isinf(_nes_database->_wirelength_grad_sum)
      || std::isnan(_nes_database->_density_grad_sum) || std::isinf(_nes_database->_density_grad_sum)) {
    _nes_database->_is_diverged = true;
  }
}

void NesterovPlace::updatePenaltyGradientPre2(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                              std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads)
{
  _nes_database->_wirelength_grad_sum = 0.0F;
  _nes_database->_density_grad_sum = 0.0F;

  const auto& front_coordi_list = _nes_database->_nesterov_solver->get_current_coordis();
  const auto& current_coordi_list = _nes_database->_nesterov_solver->get_next_coordis();
  const auto& front_grad_list = _nes_database->_nesterov_solver->get_current_grads();
  const auto& current_grad_list = _nes_database->_nesterov_solver->get_next_grads();

  // prcondition method 2
  size_t inst_size = nInst_list.size();
  std::vector<double> front_ski_x(inst_size);
  std::vector<double> front_ski_y(inst_size);
  std::vector<double> front_yki_x(inst_size);
  std::vector<double> front_yki_y(inst_size);
  double yksk_variable = 0.0;
  for (size_t i = 0; i < inst_size; i++) {
    front_ski_x[i] = current_coordi_list[i].get_x() - front_coordi_list[i].get_x();
    front_ski_y[i] = current_coordi_list[i].get_y() - front_coordi_list[i].get_y();
    front_yki_x[i] = current_grad_list[i].get_x() - front_grad_list[i].get_x();
    front_yki_y[i] = current_grad_list[i].get_y() - front_grad_list[i].get_y();
    yksk_variable += (front_yki_x[i] * front_ski_x[i] + front_yki_y[i] * front_ski_y[i]);
  }

  for (size_t i = 0; i < nInst_list.size(); i++) {
    auto& cur_n_inst = nInst_list[i];

    wirelength_grads[i] = std::move(_nes_database->_wirelength_gradient->obtainWirelengthGradient(
        cur_n_inst->get_inst_id(), _nes_database->_wirelength_coef, _nes_database->_wirelength_coef));
    density_grads[i] = std::move(
        _nes_database->_density_gradient->obtainDensityGradient(cur_n_inst->get_density_shape(), cur_n_inst->get_density_scale(), 0, 0.0f));

    _nes_database->_wirelength_grad_sum += fabs(wirelength_grads[i].get_x());
    _nes_database->_wirelength_grad_sum += fabs(wirelength_grads[i].get_y());

    _nes_database->_density_grad_sum += fabs(density_grads[i].get_x());
    _nes_database->_density_grad_sum += fabs(density_grads[i].get_y());

    sum_grads[i].set_x(wirelength_grads[i].get_x() + _nes_database->_density_penalty * density_grads[i].get_x());
    sum_grads[i].set_y(wirelength_grads[i].get_y() + _nes_database->_density_penalty * density_grads[i].get_y());

    // prcondition method 2
    double bk_coeff_x = (1.0 - front_ski_x[i] * front_yki_x[i] / yksk_variable);
    double bk_coeff_y = (1.0 - front_ski_y[i] * front_yki_y[i] / yksk_variable);
    float bki_x
        = static_cast<float>((bk_coeff_x * bk_coeff_x * _global_diagonal_list[i]) + (front_ski_x[i] * front_ski_x[i]) / yksk_variable);
    float bki_y = static_cast<float>((bk_coeff_y * bk_coeff_y * _global_diagonal_list[inst_size + i])
                                     + (front_ski_y[i] * front_ski_y[i]) / yksk_variable);
    Point<float> sum_precondition(bki_x, bki_y);

    if (sum_precondition.get_x() >= _nes_config.get_min_precondition()) {
      sum_precondition.set_x(_nes_config.get_min_precondition());
    }
    if (sum_precondition.get_y() >= _nes_config.get_min_precondition()) {
      sum_precondition.set_y(_nes_config.get_min_precondition());
    }

    sum_grads[i].set_x(sum_grads[i].get_x() * sum_precondition.get_x());
    sum_grads[i].set_y(sum_grads[i].get_y() * sum_precondition.get_y());

    _global_diagonal_list[i] = bki_x;
    _global_diagonal_list[inst_size + i] = bki_y;
  }

  if (std::isnan(_nes_database->_wirelength_grad_sum) || std::isinf(_nes_database->_wirelength_grad_sum)
      || std::isnan(_nes_database->_density_grad_sum) || std::isinf(_nes_database->_density_grad_sum)) {
    _nes_database->_is_diverged = true;
  }
}

void NesterovPlace::updatePenaltyGradient(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                          std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads,
                                          bool is_add_quad_penalty)
{
  _nes_database->_wirelength_grad_sum = 0.0F;
  _nes_database->_density_grad_sum = 0.0F;

#pragma omp parallel for num_threads(_nes_config.get_thread_num())
  for (size_t i = 0; i < nInst_list.size(); i++) {
    auto& cur_n_inst = nInst_list[i];

    wirelength_grads[i] = std::move(_nes_database->_wirelength_gradient->obtainWirelengthGradient(
        cur_n_inst->get_inst_id(), _nes_database->_wirelength_coef, _nes_database->_wirelength_coef));
    density_grads[i] = std::move(_nes_database->_density_gradient->obtainDensityGradient(
        cur_n_inst->get_density_shape(), cur_n_inst->get_density_scale(), is_add_quad_penalty, _quad_penalty_coeff));
  }

  float density_penalty = _nes_database->_density_penalty;
  if (is_add_quad_penalty) {
    density_penalty *= (1.0 + 2 * _quad_penalty_coeff * _nes_database->_density_gradient->get_sum_phi());
    density_penalty *= 2;
  }

  for (size_t i = 0; i < nInst_list.size(); i++) {
    auto& cur_n_inst = nInst_list[i];
    _nes_database->_wirelength_grad_sum += fabs(wirelength_grads[i].get_x());
    _nes_database->_wirelength_grad_sum += fabs(wirelength_grads[i].get_y());

    _nes_database->_density_grad_sum += fabs(density_grads[i].get_x());
    _nes_database->_density_grad_sum += fabs(density_grads[i].get_y());

    sum_grads[i].set_x(wirelength_grads[i].get_x() + density_penalty * density_grads[i].get_x());
    sum_grads[i].set_y(wirelength_grads[i].get_y() + density_penalty * density_grads[i].get_y());

    Point<float> wirelength_precondition = std::move(obtainWirelengthPrecondition(cur_n_inst));
    Point<float> density_precondition = std::move(obtainDensityPrecondition(cur_n_inst));
    Point<float> sum_precondition(wirelength_precondition.get_x() + _nes_database->_density_penalty * density_precondition.get_x(),
                                  wirelength_precondition.get_y() + _nes_database->_density_penalty * density_precondition.get_y());

    if (sum_precondition.get_x() <= _nes_config.get_min_precondition()) {
      sum_precondition.set_x(_nes_config.get_min_precondition());
    }
    if (sum_precondition.get_y() <= _nes_config.get_min_precondition()) {
      sum_precondition.set_y(_nes_config.get_min_precondition());
    }

    sum_grads[i].set_x(sum_grads[i].get_x() / sum_precondition.get_x());
    sum_grads[i].set_y(sum_grads[i].get_y() / sum_precondition.get_y());
  }

  if (std::isnan(_nes_database->_wirelength_grad_sum) || std::isinf(_nes_database->_wirelength_grad_sum)
      || std::isnan(_nes_database->_density_grad_sum) || std::isinf(_nes_database->_density_grad_sum)) {
    _nes_database->_is_diverged = true;
  }
}

Point<float> NesterovPlace::obtainWirelengthPrecondition(NesInstance* n_inst)
{
  float wl_factor = 0.0;
  for (auto* n_pin : n_inst->get_nPin_list()) {
    wl_factor += n_pin->get_nNet()->get_weight();
  }
  return Point<float>(wl_factor, wl_factor);
}

Point<float> NesterovPlace::obtainDensityPrecondition(NesInstance* n_inst)
{
  float n_inst_width = static_cast<float>(n_inst->get_origin_shape().get_width());
  float n_inst_height = static_cast<float>(n_inst->get_origin_shape().get_height());

  float area_val = n_inst_width * n_inst_height;
  return Point<float>(area_val, area_val);
}

float NesterovPlace::obtainPhiCoef(float scaled_diff_hpwl, int32_t iteration_num)
{
  float ret_coef = (scaled_diff_hpwl < 0) ? _nes_config.get_max_phi_coef() * std::max(std::pow(0.9999, float(iteration_num)), 0.98)
                                          : _nes_config.get_max_phi_coef() * pow(_nes_config.get_max_phi_coef(), scaled_diff_hpwl * -1.0);
  ret_coef = std::min(_nes_config.get_max_phi_coef(), ret_coef);
  ret_coef = std::max(_nes_config.get_min_phi_coef(), ret_coef);
  return ret_coef;
}

void NesterovPlace::NesterovSolve(std::vector<NesInstance*>& inst_list)
{
  // diverged control.
  if (_nes_database->_is_diverged) {
    LOG_ERROR << "Detect diverged, The reason may be parameters setting.";
    return;
  }

  auto* solver = _nes_database->_nesterov_solver;
  size_t inst_size = inst_list.size();

  float sum_overflow;
  int64_t prev_hpwl, hpwl;
  prev_hpwl = _nes_database->_wirelength->obtainTotalWirelength();

  std::vector<Point<float>> next_slp_wirelength_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> next_slp_density_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> next_slp_sum_grad_list(inst_size, Point<float>());

  float sum_overflow_threshold = 1e25;
  float hpwl_attach_sum_overflow = 1e25;
  bool max_phi_coef_record = false;

  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();

  // opt setting
  const std::vector<float>& opt_overflow_list = _nes_config.get_opt_overflow_list();
  int32_t cur_opt_overflow_step = opt_overflow_list.size() - 1;

  // prepare for long net opt.
  int32_t long_width, long_height;
  std::ofstream long_net_stream;
  if (PRINT_LONG_NET) {
    float layout_ratio = 0.5;
    long_width = core_shape.get_width() * layout_ratio;
    long_height = core_shape.get_height() * layout_ratio;
    long_net_stream.open(iPLAPIInst.obtainTargetDir() + "/pl/AcrossLongNet_process.txt");
    if (!long_net_stream.good()) {
      LOG_WARNING << "Cannot open file for recording across long net !";
    }
  }

  // prepare for iter info record
  std::ofstream info_stream;
  if (RECORD_ITER_INFO) {
    info_stream.open(iPLAPIInst.obtainTargetDir() + "/pl/plIterInfo.csv");
    if (!info_stream.good()) {
      LOG_WARNING << "Cannot open file for iter info record !";
    }
  }

  // prepare for convergence acceleration and non-convergence treatment
  int32_t min_perturb_interval = 50;
  int32_t last_perturb_iter = -min_perturb_interval;
  bool is_add_quad_penalty = false;
  bool is_cal_phi = false;
  bool stop_placement = false;
  std::vector<Point<int32_t>> best_position_list;
  std::vector<Point<int32_t>> cur_position_list;
  best_position_list.resize(inst_size);
  cur_position_list.resize(inst_size);

  if (_nes_config.isOptCongestion()) {
    _nes_database->_bin_grid->evalRouteCap(_nes_config.get_thread_num());
    // _nes_database->_bin_grid->plotRouteCap();
  }

  // algorithm core loop.
  for (int32_t iter_num = 1; iter_num <= _nes_config.get_max_iter(); iter_num++) {
    solver->runNextIter(iter_num, _nes_config.get_thread_num());
    int32_t num_backtrack = 0;
    for (; num_backtrack < _nes_config.get_max_back_track(); num_backtrack++) {
      auto& next_coordi_list = solver->get_next_coordis();
      auto& next_slp_coordi_list = solver->get_next_slp_coordis();

#pragma omp parallel for num_threads(_nes_config.get_thread_num())
      for (size_t i = 0; i < inst_size; i++) {
        Point<int32_t> next_coordi(next_coordi_list[i].get_x(), next_coordi_list[i].get_y());
        Point<int32_t> next_slp_coordi(next_slp_coordi_list[i].get_x(), next_slp_coordi_list[i].get_y());

        updateDensityCenterCoordiLayoutInside(inst_list[i], next_coordi, core_shape);
        solver->correctNextCoordi(i, next_coordi);
        cur_position_list[i] = next_coordi;

        updateDensityCenterCoordiLayoutInside(inst_list[i], next_slp_coordi, core_shape);
        solver->correctNextSLPCoordi(i, next_slp_coordi);
        inst_list[i]->updateDensityCenterLocation(next_slp_coordi);
      }

      _nes_database->_bin_grid->updateBinGrid(inst_list, _nes_config.get_thread_num());

      // print density map for debug
      if (iter_num == 60 && PRINT_DENSITY_MAP) {
        printDensityMapToCsv("density_map_" + std::to_string(iter_num));
      }

      _nes_database->_density_gradient->updateDensityForce(_nes_config.get_thread_num(), is_cal_phi);

      updateTopologyManager();

      sum_overflow = static_cast<float>(_nes_database->_bin_grid->get_overflow_area_without_filler()) / _total_inst_area;
      if (_nes_config.isOptCongestion() && iter_num >= 340 && iter_num % 10 == 0) {
        _nes_database->_bin_grid->evalRouteDem(_nes_database->_topology_manager->get_network_list(), _nes_config.get_thread_num());
        _nes_database->_bin_grid->fastGaussianBlur();
        _nes_database->_bin_grid->evalRouteUtil();
        // _nes_database->_bin_grid->plotRouteUtil(iter_num);
      }

      if (!_nes_config.isOptCongestion()) {
        _nes_database->_wirelength_gradient->updateWirelengthForce(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                   _nes_config.get_min_wirelength_force_bar(),
                                                                   _nes_config.get_thread_num());
      } else {
        if (sum_overflow > 0.5) {
          _nes_database->_wirelength_gradient->updateWirelengthForce(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                     _nes_config.get_min_wirelength_force_bar(),
                                                                     _nes_config.get_thread_num());
        } else {
          _nes_database->_bin_grid->evalRouteDem(_nes_database->_topology_manager->get_network_list(), _nes_config.get_thread_num());
          _nes_database->_bin_grid->fastGaussianBlur();
          _nes_database->_bin_grid->evalRouteUtil();
          // _nes_database->_bin_grid->plotOverflowUtil(sum_overflow, iter_num);

          // writeBackPlacerDB();
          // PlacerDBInst.writeBackSourceDataBase();
          // eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
          // std::vector<float> gr_congestion = eval_api.evalGRCong();

          auto grid_manager = _nes_database->_bin_grid->get_grid_manager();
          _nes_database->_wirelength_gradient->updateWirelengthForceDirect(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                           _nes_config.get_min_wirelength_force_bar(),
                                                                           _nes_config.get_thread_num(), grid_manager);
        }
      }

      // update next target penalty object.
      updatePenaltyGradient(inst_list, next_slp_sum_grad_list, next_slp_wirelength_grad_list, next_slp_density_grad_list,
                            is_add_quad_penalty);

      if (_nes_database->_is_diverged) {
        break;
      }

      float current_steplength = solver->get_next_steplength();
      solver->calculateNextSteplength(next_slp_sum_grad_list);
      float next_steplength = solver->get_next_steplength();

      if (next_steplength > current_steplength * 0.95) {
        break;
      } else {
        solver->runBackTrackIter(_nes_config.get_thread_num());
      }
    }

    if (num_backtrack == _nes_config.get_max_back_track()) {
      LOG_ERROR << "Detect divergence,"
                << " The reason may be high init_density_penalty value";
      _nes_database->_is_diverged = true;
    }

    if (_nes_database->_is_diverged) {
      break;
    }

    if (RECORD_ITER_INFO) {
      if (iter_num == 1) {
        info_stream << "WireLength Grad Sum,Density Grad Sum,Density Weight,StepLength" << std::endl;
      }
      printIterInfoToCsv(info_stream, iter_num);
    }

    if (_nes_config.isOptMaxWirelength()) {
      if (cur_opt_overflow_step >= 0 && sum_overflow < opt_overflow_list[cur_opt_overflow_step]) {
        // update net weight.
        updateMaxLengthNetWeight();
        --cur_opt_overflow_step;
        LOG_INFO << "[NesterovSolve] Begin update netweight for max wirelength constraint.";
      }
    }

    if (_nes_config.isOptTiming()) {
      if (cur_opt_overflow_step >= 0 && sum_overflow < opt_overflow_list[cur_opt_overflow_step]) {
        // update net weight.
        updateTimingNetWeight();
        --cur_opt_overflow_step;
        LOG_INFO << "[NesterovSolve] Update netweight for timing improvement.";
      }
    }

    updateWirelengthCoef(sum_overflow);
    if (!max_phi_coef_record && sum_overflow < 0.35f) {
      max_phi_coef_record = true;
      _nes_config.set_max_phi_coef(0.985 * _nes_config.get_max_phi_coef());
    }

    hpwl = _nes_database->_wirelength->obtainTotalWirelength();

    float phi_coef = obtainPhiCoef(static_cast<float>(hpwl - prev_hpwl) / _nes_config.get_reference_hpwl(), iter_num);
    prev_hpwl = hpwl;
    _nes_database->_density_penalty *= phi_coef;

    // print info.
    if (iter_num == 1 || iter_num % _nes_config.get_info_iter_num() == 0) {
      LOG_INFO << "[NesterovSolve] Iter: " << iter_num << " overflow: " << sum_overflow << " HPWL: " << prev_hpwl;

      if (PRINT_LONG_NET) {
        long_net_stream << "CURRENT ITERATION: " << iter_num << std::endl;
        long_net_stream << std::endl;
        printAcrossLongNet(long_net_stream, long_width, long_height);
      }

      if (PLOT_IMAGE) {
        plotInstImage("inst_" + std::to_string(iter_num));
        plotBinForceLine("bin_" + std::to_string(iter_num));
      }

      if (isJsonOutputEnabled()) {
        plotInstJson("inst_" + std::to_string(iter_num), iter_num, sum_overflow);

        printDensityMapToCsv("density/density_map_" + std::to_string(iter_num));
      }
    }

    if (iter_num == 1 || iter_num % 5 == 0) {
      if (PRINT_COORDI) {
        saveNesterovPlaceData(iter_num);
      }
    }

    if (sum_overflow_threshold > sum_overflow) {
      sum_overflow_threshold = sum_overflow;
      hpwl_attach_sum_overflow = prev_hpwl;
    }

    if (sum_overflow < 0.32f && sum_overflow - sum_overflow_threshold >= 0.05f && hpwl_attach_sum_overflow * 1.25f < prev_hpwl) {
      LOG_ERROR << "Detect divergence. \n"
                << "    The reason may be max_phi_cof value: try to decrease max_phi_cof";
      _nes_database->_is_diverged = true;
      break;
    }

    _overflow_record_list.push_back(sum_overflow);
    _hpwl_record_list.push_back(hpwl);

    if (sum_overflow < _best_overflow) {
      _best_hpwl = hpwl;
      _best_overflow = sum_overflow;
      best_position_list.swap(cur_position_list);
    }

    if (sum_overflow < _nes_config.get_target_overflow() * 4 && sum_overflow > _nes_config.get_target_overflow() * 1.1) {
      if (checkDivergence(3, 0.03 * sum_overflow) || checkLongTimeOverflowUnchanged(100, 0.03 * sum_overflow)) {
        // rollback to best pos.
        for (size_t i = 0; i < inst_size; i++) {
          updateDensityCenterCoordiLayoutInside(inst_list[i], best_position_list[i], core_shape);
        }
        sum_overflow = _best_overflow;
        prev_hpwl = _best_hpwl;

        stop_placement = true;
      }
    }

    if (iter_num - last_perturb_iter > min_perturb_interval && checkPlateau(50, 0.01)) {
      if (sum_overflow > 0.9) {
        // quad mode
        is_add_quad_penalty = true;
        is_cal_phi = true;
        LOG_INFO << "Try to enable quadratic penalty for density to accelerate convergence";
        if (sum_overflow > 0.95) {
          float noise_intensity = std::min(std::max(40 + (120 - 40) * (sum_overflow - 0.95) * 10, 40.0), 90.0)
                                  * _nes_database->_placer_db->get_layout()->get_site_width();
          entropyInjection(0.996, noise_intensity);
          LOG_INFO << "Try to entropy injection with noise intensity = " << noise_intensity << " to help convergence";
        }
        last_perturb_iter = iter_num;
      }
    }

    // minimun iteration is 30
    if ((iter_num > 30 && sum_overflow <= _nes_config.get_target_overflow()) || stop_placement) {
      if (PRINT_LONG_NET) {
        long_net_stream << "CURRENT ITERATION: " << iter_num << std::endl;
        long_net_stream << std::endl;
        printAcrossLongNet(long_net_stream, long_width, long_height);
        long_net_stream.close();
      }

      if (RECORD_ITER_INFO) {
        info_stream.close();
      }

      if (PRINT_COORDI) {
        saveNesterovPlaceData(iter_num);
      }

      LOG_INFO << "[NesterovSolve] Finished with Overflow:" << sum_overflow << " HPWL : " << prev_hpwl;
      break;
    }
  }

  if (_nes_database->_is_diverged) {
    LOG_ERROR << "Detect divergence, The reason may be parameters setting.";
    exit(1);
  }

  notifyPLOverflowInfo(sum_overflow);
  notifyPLPlaceDensity();

  // update PlacerDB.
  writeBackPlacerDB();
}

void NesterovPlace::notifyPLOverflowInfo(float final_overflow)
{
  PlacerDBInst.gp_overflow = final_overflow;

  std::vector<Grid*> grid_list;
  _nes_database->_grid_manager->obtainOverflowIllegalGridList(grid_list);
  PlacerDBInst.gp_overflow_number = grid_list.size();
}

void NesterovPlace::notifyPLPlaceDensity()
{
  auto* grid_manager = _nes_database->_grid_manager;
  PlacerDBInst.place_density[0] = grid_manager->obtainAvgGridDensity();
}

void NesterovPlace::plotInstImage(std::string file_name)
{
#ifdef BUILD_QT
  auto core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  // auto core_shape = _nes_database->_core_shape;
  std::vector<NesInstance*>& inst_list = _nes_database->_nInstance_list;

  Image image_ploter(core_shape.get_width(), core_shape.get_height(), _nes_database->_nInstance_list.size());

  int32_t core_shift_x = core_shape.get_ll_x();
  int32_t core_shift_y = core_shape.get_ll_y();

  // plot bin
  auto bin_grid_shape = _nes_database->_grid_manager->get_shape();
  int32_t bin_cnt_x = _nes_database->_grid_manager->get_grid_cnt_x();
  int32_t bin_cnt_y = _nes_database->_grid_manager->get_grid_cnt_y();
  int32_t bin_width = _nes_database->_grid_manager->get_grid_size_x();
  int32_t bin_height = _nes_database->_grid_manager->get_grid_size_y();
  int32_t bin_grid_x = 0;
  int32_t bin_grid_y = 0;

  for (int32_t i = 0; i < bin_cnt_x; i++) {
    image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x, bin_grid_y + bin_grid_shape.get_height(), IMAGE_COLOR::klightGray);
    bin_grid_x += bin_width;
  }
  image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x, bin_grid_y + bin_grid_shape.get_height(), IMAGE_COLOR::klightGray);

  bin_grid_x = 0;
  for (int32_t i = 0; i < bin_cnt_y; i++) {
    image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x + bin_grid_shape.get_width(), bin_grid_y, IMAGE_COLOR::klightGray);
    bin_grid_y += bin_height;
  }
  image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x + bin_grid_shape.get_width(), bin_grid_y, IMAGE_COLOR::klightGray);

  for (auto* inst : inst_list = _nes_database->_nInstance_list) {
    int32_t inst_real_width = inst->get_origin_shape().get_width();
    int32_t inst_real_height = inst->get_origin_shape().get_height();
    auto inst_center = inst->get_density_center_coordi();
    inst_center.set_x(inst_center.get_x() - core_shift_x);
    inst_center.set_y(inst_center.get_y() - core_shift_y);

    if (inst->isFiller()) {
      image_ploter.drawRect(inst_center.get_x(), inst_center.get_y(), inst_real_width, inst_real_height, 0.0, IMAGE_COLOR::kBule);
    } else if (inst->isMacro()) {
      image_ploter.drawRect(inst_center.get_x(), inst_center.get_y(), inst_real_width, inst_real_height, 0.0, IMAGE_COLOR::kRed);
    } else {
      image_ploter.drawRect(inst_center.get_x(), inst_center.get_y(), inst_real_width, inst_real_height, 0.0);
    }
  }

  image_ploter.save(iPLAPIInst.obtainTargetDir() + "/pl/plot/" + file_name + ".jpg");
#endif
}

void NesterovPlace::plotInstJson(std::string file_name, int32_t cur_iter, float overflow)
{
  auto core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  std::vector<NesInstance*>& inst_list = _nes_database->_nInstance_list;
  nlohmann::json plot = nlohmann::json::object();

  // Initialize the plot object
  plot["instances"] = nlohmann::json::array();
  plot["real_width"] = core_shape.get_width();
  plot["real_height"] = core_shape.get_height();
  plot["num_obj"] = _nes_database->_nInstance_list.size();
  plot["ll_x"] = core_shape.get_ll_x();
  plot["ll_y"] = core_shape.get_ll_y();
  plot["iter"] = cur_iter;

  int32_t core_shift_x = core_shape.get_ll_x();
  int32_t core_shift_y = core_shape.get_ll_y();

  for (auto* inst : inst_list) {
    int32_t inst_real_width = inst->get_origin_shape().get_width();
    int32_t inst_real_height = inst->get_origin_shape().get_height();
    auto inst_center = inst->get_density_center_coordi();
    inst_center.set_x(inst_center.get_x() - core_shift_x);
    inst_center.set_y(inst_center.get_y() - core_shift_y);

    plot["instances"].push_back({{"id", inst->get_inst_id()},
                                 {"name", inst->get_name()},
                                 {"x", inst_center.get_x()},
                                 {"y", inst_center.get_y()},
                                 {"width", inst_real_width},
                                 {"height", inst_real_height},
                                 {"is_macro", inst->isMacro()},
                                 {"is_filler", inst->isFiller()}});
  }

  std::ofstream out_file(iPLAPIInst.obtainTargetDir() + "/pl/plot/" + file_name + ".json");
  out_file << plot.dump(2);
  out_file.close();
}

void NesterovPlace::plotBinForceLine(std::string file_name)
{
#ifdef BUILD_QT
  auto core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  // auto core_shape = _nes_database->_core_shape;
  auto& force_2d_x_list = _nes_database->_density_gradient->get_force_2d_x_list();
  auto& force_2d_y_list = _nes_database->_density_gradient->get_force_2d_y_list();

  Image image_ploter(core_shape.get_width(), core_shape.get_height(), _nes_database->_nInstance_list.size());

  // plot bin
  auto bin_grid_shape = _nes_database->_grid_manager->get_shape();
  int32_t bin_cnt_x = _nes_database->_grid_manager->get_grid_cnt_x();
  int32_t bin_cnt_y = _nes_database->_grid_manager->get_grid_cnt_y();
  int32_t bin_width = _nes_database->_grid_manager->get_grid_size_x();
  int32_t bin_height = _nes_database->_grid_manager->get_grid_size_y();
  int32_t bin_grid_x = 0;
  int32_t bin_grid_y = 0;

  for (int32_t i = 0; i < bin_cnt_x; i++) {
    image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x, bin_grid_y + bin_grid_shape.get_height(), IMAGE_COLOR::klightGray);
    bin_grid_x += bin_width;
  }
  image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x, bin_grid_y + bin_grid_shape.get_height(), IMAGE_COLOR::klightGray);

  bin_grid_x = 0;
  for (int32_t i = 0; i < bin_cnt_y; i++) {
    image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x + bin_grid_shape.get_width(), bin_grid_y, IMAGE_COLOR::klightGray);
    bin_grid_y += bin_height;
  }
  image_ploter.drawLine(bin_grid_x, bin_grid_y, bin_grid_x + bin_grid_shape.get_width(), bin_grid_y, IMAGE_COLOR::klightGray);

  float electro_force_max = 0;
  int max_len = std::numeric_limits<int>::max();
  for (int i = 0; i < bin_cnt_y; i++) {
    for (int j = 0; j < bin_cnt_x; j++) {
      electro_force_max = std::max(electro_force_max, std::hypot(force_2d_x_list[i][j], force_2d_y_list[i][j]));
      max_len = std::min({max_len, bin_width, bin_height});
    }
  }

  for (int i = 0; i < bin_cnt_y; i++) {
    for (int j = 0; j < bin_cnt_x; j++) {
      float fx = force_2d_x_list[i][j];
      float fy = force_2d_y_list[i][j];
      float f = std::hypot(fx, fy);
      float ratio = f / electro_force_max;
      float dx = fx / f * max_len * ratio;
      float dy = fy / f * max_len * ratio;

      int cx = j * bin_width + bin_width / 2;
      int cy = i * bin_height + bin_height / 2;

      image_ploter.drawArc(cx, cy, cx + dx, cy + dy);
    }
  }
  image_ploter.save(iPLAPIInst.obtainTargetDir() + "/pl/plot/" + file_name + ".jpg");
#endif
}

void NesterovPlace::printIterInfoToCsv(std::ofstream& file_stream, int32_t iter_num)
{
  file_stream << _nes_database->_wirelength_grad_sum << "," << _nes_database->_density_grad_sum * _nes_database->_density_penalty << ","
              << _nes_database->_density_penalty << "," << _nes_database->_nesterov_solver->get_next_steplength() << std::endl;
  ;
}

void NesterovPlace::printDensityMapToCsv(std::string file_name)
{
  std::ofstream file_stream;
  file_stream.open(iPLAPIInst.obtainTargetDir() + "/pl/" + file_name + ".csv");
  if (!file_stream.good()) {
    LOG_WARNING << "Cannot open file for density map calculation!";
  }

  int32_t grid_cnt_y = _nes_database->_grid_manager->get_grid_cnt_y();
  int32_t grid_cnt_x = _nes_database->_grid_manager->get_grid_cnt_x();
  float available_ratio = _nes_database->_grid_manager->get_available_ratio();
  auto& grid_2d_list = _nes_database->_grid_manager->get_grid_2d_list();

  for (int32_t i = grid_cnt_y - 1; i >= 0; i--) {
    for (int32_t j = 0; j < grid_cnt_x; j++) {
      file_stream << grid_2d_list[i][j].obtainGridDensity() / available_ratio << ",";
    }
    file_stream << std::endl;
  }

  file_stream.close();
}

void NesterovPlace::writeBackPlacerDB()
{
  // #pragma omp parallel for num_threads(_nes_config.get_thread_num())
  for (auto pair : _nes_database->_instance_map) {
    auto* n_inst = pair.first;
    auto* inst = pair.second;

    inst->update_center_coordi(n_inst->get_density_center_coordi());
  }
  PlacerDBInst.updateGridManager();
}

void NesterovPlace::updateMaxLengthNetWeight()
{
  int32_t max_wirelength_constraint = _nes_config.get_max_net_wirelength();

  for (auto* n_net : _nes_database->_nNet_list) {
    if (n_net->isDontCare()) {
      continue;
    }

    int32_t n_net_wirelength = _nes_database->_wirelength->obtainNetWirelength(n_net->get_net_id());
    int32_t delta = n_net_wirelength - max_wirelength_constraint;
    if (delta < 0) {
      continue;
    }

    float wl_overflow = static_cast<float>(delta) / max_wirelength_constraint;
    float pre_delta_weight = n_net->get_delta_weight();
    float cur_delta_weight = static_cast<float>(1 + exp(1)) / (1 + exp(-wl_overflow)) - 1;
    float delta_weight = 0.5 * pre_delta_weight + 0.5 * cur_delta_weight;

    n_net->set_weight(n_net->get_weight() + delta_weight);
    n_net->set_delta_weight(delta_weight);
  }
}

void NesterovPlace::updateTimingNetWeight()
{
  float cita = 0.2;

  auto* topo_manager = _nes_database->_topology_manager;
  auto* timing_annotation = _nes_database->_timing_annotation;
  auto& nNet_list = _nes_database->_nNet_list;
  std::vector<float> prev_miu_list;
  prev_miu_list.resize(nNet_list.size());
  float prev_max_centrality = timing_annotation->get_max_centrality();
  for (size_t i = 0; i < nNet_list.size(); i++) {
    if (Utility().isFloatApproximatelyZero(prev_max_centrality)) {
      prev_miu_list[i] = 0.0f;
      continue;
    }

    auto* n_net = nNet_list[i];
    auto* network = topo_manager->findNetworkById(n_net->get_net_id());
    if (n_net->isDontCare()) {
      prev_miu_list[i] = 0.0f;
    } else {
      prev_miu_list[i] = timing_annotation->get_network_centrality(network) / prev_max_centrality;
    }
  }

  timing_annotation->updateSTATimingFull();
  timing_annotation->updateCriticalityAndCentralityFull();

  float cur_max_centrality = timing_annotation->get_max_centrality();
  for (size_t i = 0; i < nNet_list.size(); i++) {
    if (Utility().isFloatApproximatelyZero(cur_max_centrality)) {
      break;
    }

    auto* n_net = nNet_list[i];
    auto* network = topo_manager->findNetworkById(n_net->get_net_id());
    if (n_net->isDontCare()) {
      //
    } else {
      float cur_miu = timing_annotation->get_network_centrality(network) / cur_max_centrality;
      float delta_weight = cita * prev_miu_list[i] + (1 - cita) * cur_miu;
      float cur_netweight = n_net->get_weight() + delta_weight;
      n_net->set_weight(cur_netweight);
    }
  }
}

void NesterovPlace::printNesterovDatabase()
{
  int32_t nes_inst_cnt = _nes_database->_nInstance_list.size();
  int32_t fixed_nes_inst_cnt = 0;
  int32_t macro_nes_inst_cnt = 0;
  int32_t stdcell_nes_inst_cnt = 0;
  int32_t filler_nes_inst_cnt = 0;

  for (auto n_inst : _nes_database->_nInstance_list) {
    if (n_inst->isFixed()) {
      fixed_nes_inst_cnt++;
    }

    if (n_inst->isMacro()) {
      macro_nes_inst_cnt++;
    } else {
      stdcell_nes_inst_cnt++;
      if (n_inst->isFiller()) {
        filler_nes_inst_cnt++;
      }
    }
  }

  LOG_INFO << "NesInstances Num : " << nes_inst_cnt;
  LOG_INFO << "1. Macro Num : " << macro_nes_inst_cnt;
  LOG_INFO << "2. Stdcell Num : " << stdcell_nes_inst_cnt;
  LOG_INFO << "2.1 Filler Num : " << filler_nes_inst_cnt;
  LOG_INFO << "Fixed NesInstances Num : " << fixed_nes_inst_cnt;

  int32_t nes_net_cnt = _nes_database->_nNet_list.size();
  int32_t dont_care_net_cnt = 0;
  int32_t set_weight_net_cnt = 0;

  for (auto n_net : _nes_database->_nNet_list) {
    if (n_net->isDontCare()) {
      dont_care_net_cnt++;
    }

    if (fabs(n_net->get_weight() - 1.0F) >= 1e-5) {
      set_weight_net_cnt++;
    }
  }

  LOG_INFO << "NesNets Num : " << nes_net_cnt;
  LOG_INFO << "Dont Care Num : " << dont_care_net_cnt;
  LOG_INFO << "Set NetWeight Num : " << set_weight_net_cnt;

  LOG_INFO << "NesPins Num : " << _nes_database->_nPin_list.size();

  int32_t bin_size_x = _nes_database->_grid_manager->get_grid_size_x();
  int32_t bin_size_y = _nes_database->_grid_manager->get_grid_size_y();
  LOG_INFO << "BinGrid Info";
  LOG_INFO << "BinCnt(x * y) : " << _nes_config.get_bin_cnt_x() << " * " << _nes_config.get_bin_cnt_y();
  LOG_INFO << "BinSize(width , height) : " << bin_size_x << " , " << bin_size_y;
  LOG_INFO << "Target Density : " << _nes_config.get_target_density();
}

void NesterovPlace::printAcrossLongNet(std::ofstream& long_net_stream, int32_t max_width, int32_t max_height)
{
  auto* topo_manager = _nes_database->_topology_manager;
  auto core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  int32_t core_width = core_shape.get_width();
  int32_t core_height = core_shape.get_height();

  int net_count = 0;

  for (auto* network : topo_manager->get_network_list()) {
    auto shape = network->obtainNetWorkShape();
    int32_t network_width = shape.get_width();
    int32_t network_height = shape.get_height();

    if (network_width > max_width && network_height > max_height) {
      long_net_stream << "Net : " << network->get_name() << " Width/CoreWidth " << network_width << "/" << core_width
                      << " Height/CoreHeight " << network_height << "/" << core_height << std::endl;
      ++net_count;
    } else if (network_width > max_width) {
      long_net_stream << "Net : " << network->get_name() << " Width/CoreWidth " << network_width << "/" << core_width << std::endl;
      ++net_count;
    } else if (network_height > max_height) {
      long_net_stream << "Net : " << network->get_name() << " Height/CoreHeight " << network_height << "/" << core_height << std::endl;
      ++net_count;
    }
  }

  long_net_stream << std::endl;
  long_net_stream << "SUMMARY : "
                  << "AcrossLongNets / Total Nets = " << net_count << " / " << topo_manager->get_network_list().size() << std::endl;
  long_net_stream << std::endl << std::endl;
}

void NesterovPlace::saveNesterovPlaceData(int32_t cur_iter)
{
  iplf::plInst->clearFileInstanceList();

  for (auto pair : _nes_database->_instance_map) {
    auto* nes_inst = pair.first;

    if (nes_inst->isFiller()) {
      continue;
    }

    auto* inst = pair.second;
    int32_t coordi_x = nes_inst->get_density_center_coordi().get_x() - inst->get_shape_width() / 2;
    int32_t coordi_y = nes_inst->get_density_center_coordi().get_y() - inst->get_shape_height() / 2;

    std::string orient;
    if (inst->get_orient() == Orient::kN_R0) {
      orient = "N_R0";
    } else if (inst->get_orient() == Orient::kS_R180) {
      orient = "S_R180";
    } else if (inst->get_orient() == Orient::kFN_MY) {
      orient = "FN_MY";
    } else if (inst->get_orient() == Orient::kFS_MX) {
      orient = "FS_MX";
    }

    iplf::plInst->addFileInstance(nes_inst->get_name(), coordi_x, coordi_y, (int8_t) inst->get_orient());
  }

  iplf::plInst->saveInstanceDataToDirectory(iPLAPIInst.obtainTargetDir() + "/pl/gui/");
}

void NesterovPlace::printIterationCoordi(std::ofstream& long_net_stream, int32_t cur_iter)
{
  for (auto pair : _nes_database->_instance_map) {
    auto* nes_inst = pair.first;

    if (nes_inst->isFiller()) {
      continue;
    }

    auto* inst = pair.second;
    int32_t coordi_x = nes_inst->get_density_center_coordi().get_x() - inst->get_shape_width() / 2;
    int32_t coordi_y = nes_inst->get_density_center_coordi().get_y() - inst->get_shape_height() / 2;

    std::string orient;
    if (inst->get_orient() == Orient::kN_R0) {
      orient = "N_R0";
    } else if (inst->get_orient() == Orient::kS_R180) {
      orient = "S_R180";
    } else if (inst->get_orient() == Orient::kFN_MY) {
      orient = "FN_MY";
    } else if (inst->get_orient() == Orient::kFS_MX) {
      orient = "FS_MX";
    }

    long_net_stream << nes_inst->get_name() << " " << coordi_x << "," << coordi_y << " " << orient << std::endl;
  }
}

void NesterovPlace::resetOverflowRecordList()
{
  _overflow_record_list.clear();
}

void NesterovPlace::resetHPWLRecordList()
{
  _hpwl_record_list.clear();
}

void NesterovPlace::initQuadPenaltyCoeff()
{
  int64_t cur_overflow = std::max((int64_t) 1, _nes_database->_bin_grid->get_overflow_area_without_filler());

  // density weight subgradient preconditioner
  float density_weight_grad_precond = 1.0 / cur_overflow;
  _quad_penalty_coeff = _quad_penalty_coeff / 2 * density_weight_grad_precond;
}

bool NesterovPlace::checkPlateau(int32_t window, float threshold)
{
  int32_t cur_size = static_cast<int32_t>(_overflow_record_list.size());
  if (cur_size < window) {
    return false;
  }

  float max = FLT_MIN;
  float min = FLT_MAX;
  float avg = 0.0;
  auto iter_end = _overflow_record_list.rbegin() + window;
  for (auto it = _overflow_record_list.rbegin(); it != iter_end; it++) {
    *it > max ? max = *it : max;
    *it < min ? min = *it : min;
    avg += *it;
  }

  return (max - min) / (avg / window) < threshold;
}

void NesterovPlace::entropyInjection(float shrink_factor, float noise_intensity)
{
  int64_t center_x = 0;
  int64_t center_y = 0;
  int32_t movable_inst_cnt = 0;
  // cal all movable instance mean center
  for (auto* inst : _nes_database->_nInstance_list) {
    if (inst->isFixed()) {
      continue;
    }
    auto inst_center = std::move(inst->get_density_center_coordi());
    center_x += inst_center.get_x();
    center_y += inst_center.get_y();
    movable_inst_cnt++;
  }

  center_x /= movable_inst_cnt;
  center_y /= movable_inst_cnt;

  int32_t seed = 1000;
  std::default_random_engine gen(seed);
  std::normal_distribution<float> dis(0, 1);
  for (auto* inst : _nes_database->_nInstance_list) {
    if (inst->isFixed()) {
      continue;
    }
    auto inst_center = std::move(inst->get_density_center_coordi());

    // shrink all movable insts
    int32_t new_x = (inst_center.get_x() - center_x) * shrink_factor + center_x;
    int32_t new_y = (inst_center.get_y() - center_y) * shrink_factor + center_y;

    // add some noise
    new_x += noise_intensity * dis(gen);
    new_y += noise_intensity * dis(gen);

    inst->updateDensityCenterLocation(new_x, new_y);
  }
}

bool NesterovPlace::checkDivergence(int32_t window, float threshold, bool is_routability)
{
  if (static_cast<int32_t>(_overflow_record_list.size()) < window) {
    return false;
  }

  int32_t begin_idx = static_cast<int32_t>(_overflow_record_list.size() - window);
  int32_t end_idx = static_cast<int32_t>(_overflow_record_list.size());

  float overflow_mean = 0.0f;
  float overflow_diff = 0.0f;
  float overflow_max = FLT_MIN;
  float overflow_min = FLT_MAX;
  int64_t wl_mean = 0;
  float wl_ratio, overflow_ratio;

  for (int32_t i = begin_idx; i < end_idx; i++) {
    float overflow = _overflow_record_list[i];
    overflow_mean += overflow;
    if (i + 1 < end_idx) {
      overflow_diff += std::fabs(_overflow_record_list[i + 1] - overflow);
    }
    overflow > overflow_max ? overflow_max = overflow : overflow;
    overflow < overflow_min ? overflow_min = overflow : overflow;

    wl_mean += _hpwl_record_list[i];
  }
  overflow_mean /= window;
  overflow_diff /= window;
  wl_mean /= window;
  overflow_ratio = (overflow_mean - std::max(_nes_config.get_target_overflow(), _best_overflow)) / _best_overflow;
  wl_ratio = static_cast<float>(wl_mean - _best_hpwl) / _best_hpwl;

  if (wl_ratio > threshold * 1.2) {
    if (overflow_ratio > threshold && is_routability == false) {
      LOG_WARNING << "Detect divergence: overflow increases too much than best overflow (" << overflow_ratio << " > " << threshold << ")";
      return true;
    } else if ((overflow_max - overflow_min) / overflow_mean < threshold && is_routability == false) {
      LOG_WARNING << "Detect divergence: overflow plateau ( " << (overflow_max - overflow_min) / overflow_mean << " < " << threshold << ")";
      return true;
    } else if (overflow_diff > 0.6) {
      LOG_WARNING << "Detect divergence: overflow fluctuate too frequently (" << overflow_diff << "> 0.6)";
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool NesterovPlace::checkLongTimeOverflowUnchanged(int32_t window, float threshold)
{
  if (static_cast<int32_t>(_overflow_record_list.size()) < window) {
    return false;
  }

  int32_t begin_idx = static_cast<int32_t>(_overflow_record_list.size() - window);
  int32_t end_idx = static_cast<int32_t>(_overflow_record_list.size());

  float overflow_mean = 0.0f;
  float overflow_max = FLT_MIN;
  float overflow_min = FLT_MAX;

  for (int32_t i = begin_idx; i < end_idx; i++) {
    float overflow = _overflow_record_list[i];
    overflow_mean += overflow;
    overflow > overflow_max ? overflow_max = overflow : overflow;
    overflow < overflow_min ? overflow_min = overflow : overflow;
  }

  overflow_mean /= window;

  float overflow_ratio = (overflow_max - overflow_min) / overflow_mean;
  if (overflow_ratio < 0.8 * threshold) {
    LOG_WARNING << "Detect divergence: overflow plateau ( " << overflow_ratio << " < " << (0.8 * threshold) << ")";
    return true;
  } else {
    return false;
  }
}

}  // namespace ipl
