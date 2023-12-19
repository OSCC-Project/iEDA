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
#include <cfloat>
#include <cmath>
#include <random>

#include "EvalAPI.hpp"
#include "ipl_io.h"
#include "omp.h"
#include "tool_manager.h"
#include "usage/usage.hh"

#ifdef BUILD_QT
#include "utility/Image.hh"
#endif

namespace ipl {

#define PRINT_LONG_NET 0
#define PRINT_COORDI 0
#define PLOT_IMAGE 0
#define RECORD_ITER_INFO 0
#define PRINT_DENSITY_MAP 0

#define SQRT2 1.414213562373095048801L

void NesterovPlace::initNesConfig(Config* config)
{
  _nes_config = config->get_nes_config();

  if (_nes_config.isOptMaxWirelength()) {
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

  initGridManager();
  initTopologyManager();
}

void NesterovPlace::wrapNesInstanceList()
{
  auto inst_list = _nes_database->_placer_db->get_design()->get_instance_list();

  for (auto* inst : inst_list) {
    // outside inst is none of nesterov place business.
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

    // outside inst is none of nesterov place business.
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

  _nes_database->_density = new Density(grid_manager);
  _nes_database->_density_gradient = new ElectricFieldGradient(grid_manager);  // TODO : be optional.
}

void NesterovPlace::initTopologyManager()
{
  TopologyManager* topo_manager = new TopologyManager();

  for (auto* n_pin : _nes_database->_nPin_list) {
    Node* node = new Node(n_pin->get_name());
    node->set_location(std::move(n_pin->get_center_coordi()));
    topo_manager->add_node(node);
  }

  for (auto* n_net : _nes_database->_nNet_list) {
    NetWork* network = new NetWork(n_net->get_name());

    network->set_net_weight(n_net->get_weight());

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
  }

  for (auto* n_inst : _nes_database->_nInstance_list) {
    Group* group = new Group(n_inst->get_name());

    for (auto* n_pin : n_inst->get_nPin_list()) {
      Node* node = topo_manager->findNodeById(n_pin->get_pin_id());
      node->set_group(group);
      group->add_node(node);
    }

    topo_manager->add_group(group);
  }

  _nes_database->_topology_manager = topo_manager;
  _nes_database->_wirelength = new HPWirelength(topo_manager);
  _nes_database->_wirelength_gradient = new WAWirelengthGradient(topo_manager);
}

void NesterovPlace::initFillerNesInstance()
{
  // extract average edge_x / edge_y in range(10%, 90%)
  std::vector<int32_t> edge_x_assemble;
  std::vector<int32_t> edge_y_assemble;

  // record area.
  int64_t nonplace_area = 0;
  int64_t occupied_area = 0;

  for (auto* n_inst : _nes_database->_nInstance_list) {
    Rectangle<int32_t> n_inst_shape = n_inst->get_origin_shape();
    int64_t shape_area_x = static_cast<int64_t>(n_inst_shape.get_width());
    int64_t shape_area_y = static_cast<int64_t>(n_inst_shape.get_height());

    // skip fixed nInsts.
    if (n_inst->isFixed()) {
      nonplace_area += shape_area_x * shape_area_y;
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
      int64_t boundary_width = static_cast<int64_t>(boundary.get_width());
      int64_t boundary_height = static_cast<int64_t>(boundary.get_height());
      nonplace_area += boundary_width * boundary_height;
    }
  }

  // sort
  std::sort(edge_x_assemble.begin(), edge_x_assemble.end());
  std::sort(edge_y_assemble.begin(), edge_y_assemble.end());

  // average from (10% - 90%)
  int64_t edge_x_sum = 0, edge_y_sum = 0;

  int min_idx = edge_x_assemble.size() * 0.05;
  int max_idx = edge_y_assemble.size() * 0.95;
  for (int i = min_idx; i < max_idx; i++) {
    edge_x_sum += edge_x_assemble[i];
    edge_y_sum += edge_y_assemble[i];
  }

  // the avg_edge_x and avg_edge_y will be used as filler cells' width and
  // height
  int avg_edge_x = static_cast<int>(edge_x_sum / (max_idx - min_idx));
  int avg_edge_y = static_cast<int>(edge_y_sum / (max_idx - min_idx));

  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();
  int64_t core_area = static_cast<int64_t>(core_shape.get_width()) * static_cast<int64_t>(core_shape.get_height());
  // nonPlaceInstsArea should not have targetDensity downscaling!!!
  int64_t white_space_area = core_area - nonplace_area;
  int64_t movable_area = white_space_area * _nes_config.get_target_density();
  int64_t total_filler_area = movable_area - occupied_area;

  LOG_ERROR_IF(total_filler_area < 0) << "Filler area is negative!!\n"
                                      << "       Please put higher target density or \n"
                                      << "       Re-floorplan to have enough coreArea\n";

  // mt19937 supports huge range of random values.
  // rand()'s RAND_MAX is only 32767
  std::mt19937 rand_val(0);

  // int32_t filler_cnt = total_filler_area / (avg_edge_x * avg_edge_y);

  // test
  int32_t filler_cnt = std::ceil(static_cast<int32_t>(static_cast<float>(total_filler_area / (avg_edge_x * avg_edge_y))));

  for (int i = 0; i < filler_cnt; i++) {
    // instability problem between g++ and clang++!
    auto rand_x = rand_val();
    auto rand_y = rand_val();

    // place filler cells on random coordi and set size as avgDx and avgDy
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

/*****************************Congestion-driven Placement: START*****************************/

void NesterovPlace::runNesterovRoutablityPlace()
{
  std::vector<NesInstance*> placable_inst_list = this->obtianPlacableNesInstanceList();

  initNesterovPlace(placable_inst_list);

  ieda::Stats gp_status;
  // main
  NesterovRoutablitySolve(placable_inst_list);

  double memory_delta = gp_status.memoryDelta();
  LOG_INFO << "GP memory usage " << memory_delta << "MB";
  double time_delta = gp_status.elapsedRunTime();
  LOG_INFO << "GP time elapsed " << time_delta << "s";
}

/*****************************Congestion-driven Placement: END*****************************/

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
  // if placer diverged in init() function, global placement must be skipped.
  if (_nes_database->_is_diverged) {
    LOG_ERROR << "diverged occured. please tune the parameters again";
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

  // divergence detection
  float min_sum_overflow = 1e30;
  float hpwl_with_min_sum_overflow = 1e30;

  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();

  // dynamic adjustment of max_phi_coef
  bool is_max_phi_coef_changed = false;

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
    long_net_stream.open("./result/pl/AcrossLongNet_process.txt");
    if (!long_net_stream.good()) {
      LOG_WARNING << "Cannot open file for recording across long net !";
    }
  }

  // prepare for iter info record
  std::ofstream info_stream;
  if (RECORD_ITER_INFO) {
    info_stream.open("./result/pl/plIterInfo.csv");
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

  // core Nesterov loop.
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
      _nes_database->_wirelength_gradient->updateWirelengthForce(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                 _nes_config.get_min_wirelength_force_bar(), _nes_config.get_thread_num());

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
        //
        break;
      } else {
        solver->runBackTrackIter(_nes_config.get_thread_num());
      }
    }

    // usually, max back track should be 1~3
    // 10 is the case when all of cells are not moved at all.
    if (num_backtrack == _nes_config.get_max_back_track()) {
      LOG_ERROR << "divergence detected. \n"
                << " please decrease init_density_penalty value";
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

    // _nes_database->_bin_grid->updataOverflowArea(inst_list, _nes_config.get_thread_num());
    sum_overflow = static_cast<float>(_nes_database->_bin_grid->get_overflow_area_without_filler()) / _total_inst_area;

    if (_nes_config.isOptMaxWirelength()) {
      if (cur_opt_overflow_step >= 0 && sum_overflow < opt_overflow_list[cur_opt_overflow_step]) {
        // update net weight.
        updateNetWeight();
        --cur_opt_overflow_step;
        LOG_INFO << "[NesterovSolve] Begin update netweight for max wirelength constraint.";
      }
    }

    updateWirelengthCoef(sum_overflow);
    // dynamic adjustment for better convergence with large designs
    if (!is_max_phi_coef_changed && sum_overflow < 0.35f) {
      is_max_phi_coef_changed = true;
      _nes_config.set_max_phi_coef(0.99 * _nes_config.get_max_phi_coef());
    }

    hpwl = _nes_database->_wirelength->obtainTotalWirelength();

    float phi_coef = obtainPhiCoef(static_cast<float>(hpwl - prev_hpwl) / _nes_config.get_reference_hpwl(), iter_num);
    prev_hpwl = hpwl;
    _nes_database->_density_penalty *= phi_coef;

    // print info.
    if (iter_num == 1 || iter_num % 10 == 0) {
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
    }

    if (iter_num == 1 || iter_num % 5 == 0) {
      if (PRINT_COORDI) {
        saveNesterovPlaceData(iter_num);
      }
    }

    if (min_sum_overflow > sum_overflow) {
      min_sum_overflow = sum_overflow;
      hpwl_with_min_sum_overflow = prev_hpwl;
    }

    if (sum_overflow < 0.3f && sum_overflow - min_sum_overflow >= 0.02f && hpwl_with_min_sum_overflow * 1.2f < prev_hpwl) {
      LOG_ERROR << " divergence detected. \n"
                << "    please decrease max_phi_cof value";
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
      if (checkDivergence(3, 0.03 * sum_overflow)) {
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
        LOG_INFO << "Stuck at early stage. Turn on quadratic penalty with double density factor to accelerate convergence";
        if (sum_overflow > 0.95) {
          float noise_intensity = std::min(std::max(40 + (120 - 40) * (sum_overflow - 0.95) * 10, 40.0), 90.0)
                                  * _nes_database->_placer_db->get_layout()->get_site_width();
          entropyInjection(0.996, noise_intensity);
          LOG_INFO << "Stuck at very early stage. Turn on entropy injection with noise intensity = " << noise_intensity
                   << " to help convergence";
        }
        last_perturb_iter = iter_num;
      }
    }

    // minimun iteration is 50
    if ((iter_num > 50 && sum_overflow <= _nes_config.get_target_overflow()) || stop_placement) {
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
    exit(1);
  }

  // update PlacerDB.
  writeBackPlacerDB();
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

  image_ploter.save("./result/pl/plot/" + file_name + ".jpg");
#endif
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
  image_ploter.save("./result/pl/plot/" + file_name + ".jpg");
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
  file_stream.open("./result/pl/" + file_name + ".csv");
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

/*****************************Congestion-driven Placement: START*****************************/
void NesterovPlace::NesterovRoutablitySolve(std::vector<NesInstance*>& inst_list)
{
  // if placer diverged in init() function, global placement must be skipped.
  if (_nes_database->_is_diverged) {
    LOG_ERROR << "diverged occured. please tune the parameters again";
    return;
  }

  auto* solver = _nes_database->_nesterov_solver;
  size_t inst_size = inst_list.size();
  int64_t total_area = this->obtainTotalArea(inst_list);
  int64_t total_filler_area = this->obtainTotalFillerArea(inst_list);
  LOG_INFO << "Routability-driven placement: total area before cell inflation : " << total_area;

  float sum_overflow;
  int64_t prev_hpwl, hpwl;
  prev_hpwl = _nes_database->_wirelength->obtainTotalWirelength();

  std::vector<Point<float>> next_slp_wirelength_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> next_slp_density_grad_list(inst_size, Point<float>());
  std::vector<Point<float>> next_slp_sum_grad_list(inst_size, Point<float>());

  // divergence detection
  float min_sum_overflow = 1e30;
  float hpwl_with_min_sum_overflow = 1e30;

  Rectangle<int32_t> core_shape = _nes_database->_placer_db->get_layout()->get_core_shape();

  // dynamic adjustment of max_phi_coef
  bool is_max_phi_coef_changed = false;

  // Routability-driven placement(RDP): snapshot for tracking back previous solution
  bool is_snapshot_saved = false;
  std::vector<Point<int32_t>> snapshot_next_coord_list;
  std::vector<Point<int32_t>> snapshot_next_slp_coord_list;
  std::vector<Point<float>> snapshot_next_grad_list;
  float snapshot_next_parameter = 0.f;
  float snapshot_step_length = 0.f;
  float snapshot_density_penalty = 0.f;
  float snapshot_wl_coef = 0.f;

  // Routability-driven placement(RDP): variables
  bool is_diverge_tried_revert = false;
  bool is_routability_mode = true;
  bool is_routability_need = true;
  bool is_revert_init_need = true;

  int num_call_routability = 0;
  int num_max_call_routability = 10;

  int min_rc_violated_cnt = 0;
  float min_rc = 1e30;
  float min_rc_target_density = 0;
  std::vector<std::pair<int, int>> min_rc_cell_size_list;
  min_rc_cell_size_list.resize(inst_size, std::make_pair(0, 0));
  float max_target_density = 0.9;
  float routability_check_overflow = 0.2;
  float target_rc = 1.1;
  float min_inflation_ratio = 1.01;
  float max_inflation_ratio = 2.5;
  float inflation_ratio_coef = 2.5;

  int64_t inflated_area_delta = 0;

  // for record long net.
  float layout_ratio = 0.5;
  int32_t long_width = static_cast<float>(core_shape.get_width()) * layout_ratio;
  int32_t long_height = static_cast<float>(core_shape.get_height()) * layout_ratio;
  std::ofstream long_net_stream;
  if (_nes_config.isOptMaxWirelength()) {
    layout_ratio = 0.3;
    long_width = static_cast<float>(core_shape.get_width()) * layout_ratio;
    long_height = static_cast<float>(core_shape.get_height()) * layout_ratio;
    long_net_stream.open("./result/pl/AcrossLongNet_process.txt");
    if (!long_net_stream.good()) {
      LOG_WARNING << "Cannot open file for recording across long net !";
    }
  }
  const std::vector<float>& opt_overflow_list = _nes_config.get_opt_overflow_list();
  int32_t cur_opt_overflow_step = opt_overflow_list.size() - 1;

  // core Nesterov loop.
  for (int32_t iter_num = 1; iter_num <= _nes_config.get_max_iter(); iter_num++) {
    solver->runNextIter(iter_num, 1);
    int32_t num_backtrack = 0;
    for (; num_backtrack < _nes_config.get_max_back_track(); num_backtrack++) {
      auto next_coordi_list = solver->get_next_coordis();
      auto next_slp_coordi_list = solver->get_next_slp_coordis();

      for (size_t i = 0; i < inst_size; i++) {
        updateDensityCenterCoordiLayoutInside(inst_list[i], next_coordi_list[i], core_shape);
        solver->correctNextCoordi(i, next_coordi_list[i]);
        updateDensityCenterCoordiLayoutInside(inst_list[i], next_slp_coordi_list[i], core_shape);
        solver->correctNextSLPCoordi(i, next_slp_coordi_list[i]);
        inst_list[i]->updateDensityCenterLocation(next_slp_coordi_list[i]);
      }

      // update next density gradient force.
      _nes_database->_bin_grid->updateBinGrid(inst_list, _nes_config.get_thread_num());
      _nes_database->_density_gradient->updateDensityForce(_nes_config.get_thread_num(), false);

      // update next wirelength gradient force.
      updateTopologyManager();
      _nes_database->_wirelength_gradient->updateWirelengthForce(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                 _nes_config.get_min_wirelength_force_bar(), _nes_config.get_thread_num());

      // update next target penalty object.
      updatePenaltyGradient(inst_list, next_slp_sum_grad_list, next_slp_wirelength_grad_list, next_slp_density_grad_list, false);

      if (_nes_database->_is_diverged) {
        break;
      }

      float current_steplength = solver->get_next_steplength();
      solver->calculateNextSteplength(next_slp_sum_grad_list);
      float next_steplength = solver->get_next_steplength();

      if (next_steplength > current_steplength * 0.95) {
        //
        break;
      } else {
        solver->runBackTrackIter(1);
      }
    }

    // usually, max back track should be 1~3
    // 10 is the case when all of cells are not moved at all.
    if (num_backtrack == _nes_config.get_max_back_track()) {
      LOG_ERROR << "divergence detected. \n"
                << " please decrease init_density_penalty value";
      _nes_database->_is_diverged = true;
    }

    if (_nes_database->_is_diverged) {
      break;
    }

    sum_overflow = static_cast<float>(_nes_database->_bin_grid->obtainOverflowAreaWithoutFiller()) / total_area;

    if (_nes_config.isOptMaxWirelength()) {
      if (cur_opt_overflow_step >= 0 && sum_overflow < opt_overflow_list[cur_opt_overflow_step]) {
        // update net weight.
        updateNetWeight();
        --cur_opt_overflow_step;
        LOG_INFO << "[NesterovSolve] Begin update netweight for max wirelength constraint.";
      }
    }

    updateWirelengthCoef(sum_overflow);
    // dynamic adjustment for better convergence with large designs
    if (!is_max_phi_coef_changed && sum_overflow < 0.35f) {
      is_max_phi_coef_changed = true;
      _nes_config.set_max_phi_coef(0.99 * _nes_config.get_max_phi_coef());
    }

    hpwl = _nes_database->_wirelength->obtainTotalWirelength();
    float phi_coef = obtainPhiCoef(static_cast<float>(hpwl - prev_hpwl) / _nes_config.get_reference_hpwl(), iter_num);
    prev_hpwl = hpwl;
    _nes_database->_density_penalty *= phi_coef;

    // print info.
    if (iter_num == 1 || iter_num % 10 == 0) {
      LOG_INFO << "[NesterovSolve] Iter: " << iter_num << " overflow: " << sum_overflow << " HPWL: " << prev_hpwl;

      if (PRINT_LONG_NET) {
        long_net_stream << "CURRENT ITERATION : " << iter_num << std::endl;
        long_net_stream << std::endl;
        printAcrossLongNet(long_net_stream, long_width, long_height);
      }
    }

    if (min_sum_overflow > sum_overflow) {
      min_sum_overflow = sum_overflow;
      hpwl_with_min_sum_overflow = prev_hpwl;
    }

    if (sum_overflow < 0.3f && sum_overflow - min_sum_overflow >= 0.02f && hpwl_with_min_sum_overflow * 1.2f < prev_hpwl) {
      LOG_ERROR << " divergence detected. \n"
                << "    please decrease max_phi_cof value";
      _nes_database->_is_diverged = true;
      // Routability-driven placement(RDP): snapshot for tracking back previous solution
      // revert back to the original routability solution
      if (!is_diverge_tried_revert && num_call_routability >= 1) {
        // revert back to the working rc size.
        for (size_t i = 0; i < inst_size; ++i) {
          if (inst_list[i]->isFiller() || inst_list[i]->isFixed() || inst_list[i]->isMacro()) {
            continue;
          }
          inst_list[i]->changeSize(min_rc_cell_size_list[i].first, min_rc_cell_size_list[i].second);
        }
        // revert back the current density penality
        solver->set_next_coordis(snapshot_next_coord_list);
        solver->set_next_slp_coordis(snapshot_next_slp_coord_list);
        solver->set_next_gradients(snapshot_next_grad_list);
        solver->set_next_parameter(snapshot_next_parameter);
        solver->set_next_steplength(snapshot_step_length);
        _nes_database->_density_penalty = snapshot_density_penalty;
        _nes_database->_wirelength_coef = snapshot_wl_coef;
        // update current cell location
        auto next_coodis = solver->get_next_coordis();
        for (size_t i = 0; i < inst_size; i++) {
          inst_list[i]->updateDensityCenterLocation(next_coodis[i]);
        }
        // update next density gradient force.
        _nes_database->_bin_grid->updateBinGrid(inst_list, _nes_config.get_thread_num());
        _nes_database->_density_gradient->updateDensityForce(_nes_config.get_thread_num(), false);
        // update next wirelength gradient force.
        updateTopologyManager();
        _nes_database->_wirelength_gradient->updateWirelengthForce(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                   _nes_config.get_min_wirelength_force_bar(),
                                                                   _nes_config.get_thread_num());
        // update flag, turn off the RDP forcely
        _nes_database->_is_diverged = false;
        is_diverge_tried_revert = true;
        is_routability_need = false;
      } else {
        break;
      }
    }

    // Routability-driven placement(RDP): saving snapshots for routability
    if (!is_snapshot_saved && is_routability_mode && sum_overflow <= 0.6) {
      snapshot_next_coord_list = solver->get_next_coordis();
      snapshot_next_slp_coord_list = solver->get_next_slp_coordis();
      snapshot_next_grad_list = solver->get_next_gradients();
      snapshot_next_parameter = solver->get_next_parameter();
      snapshot_step_length = solver->get_next_steplength();
      snapshot_density_penalty = _nes_database->_density_penalty;
      snapshot_wl_coef = _nes_database->_wirelength_coef;
      is_snapshot_saved = true;
      LOG_INFO << "Routability-driven placement: Save snapshot at iter = " << iter_num;
    }

    // Routability-driven placement(RDP): Core code
    if (is_routability_mode && is_routability_need && num_call_routability < num_max_call_routability
        && sum_overflow <= routability_check_overflow) {
      // increase count;
      num_call_routability++;
      LOG_INFO << "Routability-driven placement: num_call: " << num_call_routability;

      // update placeDB && dmInst, get route congestion
      writeBackPlacerDB();
      PlacerDBInst.writeBackSourceDataBase();
      eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
      std::vector<float> gr_congestion = eval_api.evalGRCong();
      LOG_INFO << "Routability-driven placement: ACE: " << gr_congestion[0] << " TOF: " << gr_congestion[1] << " MOF: " << gr_congestion[2];

      // no need if ACE is lower than target_rc
      if (gr_congestion[0] < target_rc) {
        inflated_area_delta = 0;
        is_routability_need = false;
        is_revert_init_need = false;
      } else {
        // save solution when ACE becomes lower
        if (min_rc > gr_congestion[0]) {
          min_rc = gr_congestion[0];
          min_rc_target_density = _nes_config.get_target_density();
          min_rc_violated_cnt = 0;
          for (size_t i = 0; i < inst_size; i++) {
            if (inst_list[i]->isFiller() || inst_list[i]->isFixed() || inst_list[i]->isMacro()) {
              continue;
            }
            min_rc_cell_size_list[i]
                = std::make_pair(inst_list[i]->get_origin_shape().get_width(), inst_list[i]->get_origin_shape().get_height());
          }
        } else {
          min_rc_violated_cnt++;
        }

        // get route utilizatio list, and then get inflation ratio
        std::vector<float> use_cap_ratio_list = eval_api.getUseCapRatioList();
        for (size_t i = 0; i < use_cap_ratio_list.size(); ++i) {
          float old_ratio = use_cap_ratio_list[i];
          if (old_ratio >= min_inflation_ratio) {
            float new_ratio = std::pow(old_ratio, inflation_ratio_coef);
            new_ratio = std::fmin(new_ratio, max_inflation_ratio);
            use_cap_ratio_list[i] = new_ratio;
          }
        }

        // bloat cells and get inflated_area_delta
        inflated_area_delta = 0;                                                   //  0 |  1 |  2  |  3  |  4 |  5
        const std::vector<int>& grid_info = eval_api.getTileGridCoordSizeCntXY();  // lx | ly |sizex|sizey|cntx|cnty
        for (size_t i = 0; i < inst_size; ++i) {
          if (inst_list[i]->isFiller() || inst_list[i]->isFixed() || inst_list[i]->isMacro()) {
            continue;
          }
          // match cell location and route utilization map
          int idx_x = (inst_list[i]->get_density_center_coordi().get_x() - grid_info[0]) / grid_info[2];
          int idx_y = (inst_list[i]->get_density_center_coordi().get_y() - grid_info[1]) / grid_info[3];
          float cur_ratio = use_cap_ratio_list[idx_x + idx_y * grid_info[4]];
          if (cur_ratio <= 1.0f) {
            continue;
          }
          // bloat
          int64_t prev_cell_area = static_cast<int64_t>(inst_list[i]->get_origin_shape().get_width())
                                   * static_cast<int64_t>(inst_list[i]->get_origin_shape().get_height());
          inst_list[i]->changeSize(static_cast<int32_t>(round(inst_list[i]->get_origin_shape().get_width() * sqrt(cur_ratio))),
                                   static_cast<int32_t>(round(inst_list[i]->get_origin_shape().get_height() * sqrt(cur_ratio))));
          int64_t new_cell_area = static_cast<int64_t>(inst_list[i]->get_origin_shape().get_width())
                                  * static_cast<int64_t>(inst_list[i]->get_origin_shape().get_height());
          inflated_area_delta += new_cell_area - prev_cell_area;
        }
        LOG_INFO << "Routability-driven placement: inflated_area_delta: " << inflated_area_delta;

        // compute whitespace to get new target_density
        int64_t nonplace_area = 0;
        for (auto* n_inst : _nes_database->_nInstance_list) {
          Rectangle<int32_t> n_inst_shape = n_inst->get_origin_shape();
          int64_t shape_area_x = static_cast<int64_t>(n_inst_shape.get_width());
          int64_t shape_area_y = static_cast<int64_t>(n_inst_shape.get_height());
          if (n_inst->isFixed()) {
            nonplace_area += shape_area_x * shape_area_y;
            continue;
          }
        }
        for (auto* blockage : _nes_database->_placer_db->get_design()->get_region_list()) {
          for (auto boundary : blockage->get_boundaries()) {
            int64_t boundary_width = static_cast<int64_t>(boundary.get_width());
            int64_t boundary_height = static_cast<int64_t>(boundary.get_height());
            nonplace_area += boundary_width * boundary_height;
          }
        }
        int64_t core_area = static_cast<int64_t>(core_shape.get_width()) * static_cast<int64_t>(core_shape.get_height());
        int64_t white_space_area = core_area - nonplace_area;
        int64_t total_cell_area = inflated_area_delta + total_area + total_filler_area;
        _nes_config.set_target_density(static_cast<float>(total_cell_area) / static_cast<float>(white_space_area));
        LOG_INFO << "Routability-driven placement: target_density: " << _nes_config.get_target_density();

        // max density detection or rc not improvement detection
        if (_nes_config.get_target_density() > max_target_density || min_rc_violated_cnt >= 3) {
          LOG_INFO << "Routability-driven placement: Revert procedure. current min_rc: " << min_rc
                   << " target_density: " << min_rc_target_density;
          _nes_config.set_target_density(min_rc_target_density);
          for (size_t i = 0; i < inst_size; ++i) {
            if (inst_list[i]->isFiller() || inst_list[i]->isFixed() || inst_list[i]->isMacro()) {
              continue;
            }
            inst_list[i]->changeSize(min_rc_cell_size_list[i].first, min_rc_cell_size_list[i].second);
          }
          initNesInstanceDensitySize();
          inflated_area_delta = 0;
          is_routability_need = false;
          is_revert_init_need = true;
        } else {
          // update area
          total_area = this->obtainTotalArea(inst_list);
          LOG_INFO << "Routability-driven placement: total area after cell inflation : " << total_area;
          // update density_size for all cell
          initNesInstanceDensitySize();
          // reset
          inflated_area_delta = 0;
          is_routability_need = true;
          is_revert_init_need = true;
        }
      }

      // if routability is needed
      if (is_routability_need || is_revert_init_need) {
        LOG_INFO << "Routability-driven placement: enable RDP and revert back to the snapshot";
        solver->set_next_coordis(snapshot_next_coord_list);
        solver->set_next_slp_coordis(snapshot_next_slp_coord_list);
        solver->set_next_gradients(snapshot_next_grad_list);
        solver->set_next_parameter(snapshot_next_parameter);
        solver->set_next_steplength(snapshot_step_length);
        _nes_database->_density_penalty = snapshot_density_penalty;
        _nes_database->_wirelength_coef = snapshot_wl_coef;
        // update current cell location
        auto next_coodis = solver->get_next_coordis();
        for (size_t i = 0; i < inst_size; i++) {
          inst_list[i]->updateDensityCenterLocation(next_coodis[i]);
        }
        // update next density gradient force.
        _nes_database->_bin_grid->updateBinGrid(inst_list, _nes_config.get_thread_num());
        _nes_database->_density_gradient->updateDensityForce(_nes_config.get_thread_num(), false);
        // update next wirelength gradient force.
        updateTopologyManager();
        _nes_database->_wirelength_gradient->updateWirelengthForce(_nes_database->_wirelength_coef, _nes_database->_wirelength_coef,
                                                                   _nes_config.get_min_wirelength_force_bar(),
                                                                   _nes_config.get_thread_num());
        // reset the divergence detect conditions
        min_sum_overflow = 1e30;
        hpwl_with_min_sum_overflow = 1e30;
      }
    }

    // minimun iteration is 50
    if (iter_num > 50 && sum_overflow <= _nes_config.get_target_overflow()) {
      LOG_INFO << "[NesterovSolve] Finished with Overflow:" << sum_overflow << " HPWL : " << prev_hpwl;
      LOG_INFO << "[TEST] Peak Bin Density: " << _nes_database->_density->obtainPeakBinDensity();
      LOG_INFO << "[TEST] Final Iteration: " << iter_num;
      break;
    }
  }

  if (_nes_database->_is_diverged) {
    exit(1);
  }

  if (_nes_config.isOptMaxWirelength()) {
    long_net_stream.close();
  }

  // update PlacerDB.
  writeBackPlacerDB();
}

/*****************************Congestion-driven Placement: END*****************************/

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

void NesterovPlace::updateNetWeight()
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

  iplf::plInst->saveInstanceDataToDirectory("./result/pl/gui/");
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

bool NesterovPlace::checkDivergence(int32_t window, float threshold)
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
    if (overflow_ratio > threshold) {
      LOG_WARNING << "Divergence detected: overflow increases too much than best overflow (" << overflow_ratio << " > " << threshold << ")";
      return true;
    } else if ((overflow_max - overflow_min) / overflow_mean < threshold) {
      LOG_WARNING << "Divergence detected: overflow plateau ( " << (overflow_max - overflow_min) / overflow_mean << " < " << threshold
                  << ")";
      return true;
    } else if (overflow_diff > 0.6) {
      LOG_WARNING << "Divergence detected: overflow fluctuate too frequently (" << overflow_diff << "> 0.6)";
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

}  // namespace ipl
