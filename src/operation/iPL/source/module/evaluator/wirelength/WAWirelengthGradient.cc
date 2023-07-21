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
 * @Date: 2022-03-09 11:17:53
 * @LastEditTime: 2022-10-31 11:52:16
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/WAWirelengthGradient.cc
 * Contact : https://github.com/sjchanson
 */

#include "WAWirelengthGradient.hh"

#include "omp.h"
#include "usage/usage.hh"

// debug
#include <fstream>
#include <iostream>
#include <sstream>

namespace ipl {

static float fastExp(float a);

WAWirelengthGradient::WAWirelengthGradient(TopologyManager* topology_manager) : WirelengthGradient(topology_manager)
{
  size_t pin_size = _topology_manager->get_node_list().size();
  _pin_grad_x_list.resize(pin_size);
  _pin_grad_y_list.resize(pin_size);

  // initWAInfo();
}

void WAWirelengthGradient::initWAInfo()
{
  _wa_net_info_list.resize(_topology_manager->get_network_list().size());
  _wa_pin_info_list.resize(_topology_manager->get_node_list().size());
}

void WAWirelengthGradient::updateWirelengthForce_OLD(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num)
{
  // reset all WA variables.
  for (auto& wa_pin_info : _wa_pin_info_list) {
    wa_pin_info.reset();
  }
  for (auto& wa_net_info : _wa_net_info_list) {
    wa_net_info.reset();
  }

  // NOLINTNEXTLINE
  int32_t net_chunk_size = std::max(int(_topology_manager->get_network_list().size() / thread_num / 16), 1);
#pragma omp parallel for num_threads(thread_num) schedule(dynamic, net_chunk_size)
  for (auto* network : _topology_manager->get_network_list()) {
    if (network->isIgnoreNetwork()) {
      continue;
    }

    WANetInfo wa_net_info;

    Rectangle<int32_t> network_shape = std::move(network->obtainNetWorkShape());
    for (auto* node : network->get_node_list()) {
      WAPinInfo wa_pin_info;

      Point<int32_t> node_loc = std::move(node->get_location());
      float exp_min_x = (network_shape.get_ll_x() - node_loc.get_x()) * coeff_x;
      float exp_max_x = (node_loc.get_x() - network_shape.get_ur_x()) * coeff_x;
      float exp_min_y = (network_shape.get_ll_y() - node_loc.get_y()) * coeff_y;
      float exp_max_y = (node_loc.get_y() - network_shape.get_ur_y()) * coeff_y;

      // min x.
      if (exp_min_x > min_force_bar) {
        wa_pin_info.min_ExpSum_x = fastExp(exp_min_x);
        wa_pin_info.has_MinExpSum_x = 1;
        wa_net_info.wa_ExpMinSum_x += wa_pin_info.min_ExpSum_x;
        wa_net_info.wa_X_ExpMinSum_x += node_loc.get_x() * wa_pin_info.min_ExpSum_x;
      }

      // max x.
      if (exp_max_x > min_force_bar) {
        wa_pin_info.max_ExpSum_x = fastExp(exp_max_x);
        wa_pin_info.has_MaxExpSum_x = 1;
        wa_net_info.wa_ExpMaxSum_x += wa_pin_info.max_ExpSum_x;
        wa_net_info.wa_X_ExpMaxSum_x += node_loc.get_x() * wa_pin_info.max_ExpSum_x;
      }

      // min y.
      if (exp_min_y > min_force_bar) {
        wa_pin_info.min_ExpSum_y = fastExp(exp_min_y);
        wa_pin_info.has_MinExpSum_y = 1;
        wa_net_info.wa_ExpMinSum_y += wa_pin_info.min_ExpSum_y;
        wa_net_info.wa_Y_ExpMinSum_y += node_loc.get_y() * wa_pin_info.min_ExpSum_y;
      }

      // max y.
      if (exp_max_y > min_force_bar) {
        wa_pin_info.max_ExpSum_y = fastExp(exp_max_y);
        wa_pin_info.has_MaxExpSum_y = 1;
        wa_net_info.wa_ExpMaxSum_y += wa_pin_info.max_ExpSum_y;
        wa_net_info.wa_Y_ExpMaxSum_y += node_loc.get_y() * wa_pin_info.max_ExpSum_y;
      }

      auto& cur_node_info = _wa_pin_info_list[node->get_node_id()];
      cur_node_info = std::move(wa_pin_info);
    }

    auto& cur_network_info = _wa_net_info_list[network->get_network_id()];
    cur_network_info = std::move(wa_net_info);
  }
}

void WAWirelengthGradient::updateWirelengthForce(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num)
{
  // NOLINTNEXTLINE
  int32_t net_chunk_size = std::max(int(_topology_manager->get_network_list().size() / thread_num / 16), 1);
#pragma omp parallel for num_threads(thread_num) schedule(dynamic, net_chunk_size)
  for (auto* network : _topology_manager->get_network_list()) {
    if (network->isIgnoreNetwork()) {
      continue;
    }

    float net_ExpMinSum_x, net_ExpMaxSum_x, net_ExpMinSum_y, net_ExpMaxSum_y;
    float net_X_ExpMinSum_x, net_X_ExpMaxSum_x, net_Y_ExpMinSum_y, net_Y_ExpMaxSum_y;

    net_ExpMinSum_x = net_ExpMaxSum_x = net_ExpMinSum_y = net_ExpMaxSum_y = 0.0f;
    net_X_ExpMinSum_x = net_X_ExpMaxSum_x = net_Y_ExpMinSum_y = net_Y_ExpMaxSum_y = 0.0f;

    Rectangle<int32_t> network_shape = std::move(network->obtainNetWorkShape());
    for (auto* node : network->get_node_list()) {
      float pin_ExpMin_x, pin_ExpMax_x, pin_ExpMin_y, pin_ExpMax_y;
      pin_ExpMin_x = pin_ExpMax_x = pin_ExpMin_y = pin_ExpMax_y = 0.0f;

      Point<int32_t> node_loc = std::move(node->get_location());
      float exp_min_x = (network_shape.get_ll_x() - node_loc.get_x()) * coeff_x;
      float exp_max_x = (node_loc.get_x() - network_shape.get_ur_x()) * coeff_x;
      float exp_min_y = (network_shape.get_ll_y() - node_loc.get_y()) * coeff_y;
      float exp_max_y = (node_loc.get_y() - network_shape.get_ur_y()) * coeff_y;

      // min x.
      if (exp_min_x > min_force_bar) {
        pin_ExpMin_x = fastExp(exp_min_x);
        net_ExpMinSum_x += pin_ExpMin_x;
        net_X_ExpMinSum_x += node_loc.get_x() * pin_ExpMin_x;
      }

      // max x.
      if (exp_max_x > min_force_bar) {
        pin_ExpMax_x = fastExp(exp_max_x);
        net_ExpMaxSum_x += pin_ExpMax_x;
        net_X_ExpMaxSum_x += node_loc.get_x() * pin_ExpMax_x;
      }

      // min y.
      if (exp_min_y > min_force_bar) {
        pin_ExpMin_y = fastExp(exp_min_y);
        net_ExpMinSum_y += pin_ExpMin_y;
        net_Y_ExpMinSum_y += node_loc.get_y() * pin_ExpMin_y;
      }

      // max y.
      if (exp_max_y > min_force_bar) {
        pin_ExpMax_y = fastExp(exp_max_y);
        net_ExpMaxSum_y += pin_ExpMax_y;
        net_Y_ExpMaxSum_y += node_loc.get_y() * pin_ExpMax_y;
      }
    }

    for (auto* node : network->get_node_list()) {
      float pin_ExpMin_x, pin_ExpMax_x, pin_ExpMin_y, pin_ExpMax_y;
      pin_ExpMin_x = pin_ExpMax_x = pin_ExpMin_y = pin_ExpMax_y = 0.0f;

      Point<int32_t> node_loc = std::move(node->get_location());
      float exp_min_x = (network_shape.get_ll_x() - node_loc.get_x()) * coeff_x;
      float exp_max_x = (node_loc.get_x() - network_shape.get_ur_x()) * coeff_x;
      float exp_min_y = (network_shape.get_ll_y() - node_loc.get_y()) * coeff_y;
      float exp_max_y = (node_loc.get_y() - network_shape.get_ur_y()) * coeff_y;

      float pin_grad_min_x, pin_grad_max_x, pin_grad_min_y, pin_grad_max_y;
      pin_grad_min_x = pin_grad_max_x = pin_grad_min_y = pin_grad_max_y = 0.0f;

      // min x.
      if (exp_min_x > min_force_bar) {
        pin_ExpMin_x = fastExp(exp_min_x);
        pin_grad_min_x
            = (net_ExpMinSum_x * (pin_ExpMin_x * (1.0 - coeff_x * node_loc.get_x())) + coeff_x * pin_ExpMin_x * net_X_ExpMinSum_x)
              / (net_ExpMinSum_x * net_ExpMinSum_x);
      }

      // max x.
      if (exp_max_x > min_force_bar) {
        pin_ExpMax_x = fastExp(exp_max_x);
        pin_grad_max_x
            = (net_ExpMaxSum_x * (pin_ExpMax_x * (1.0 + coeff_x * node_loc.get_x())) - coeff_x * pin_ExpMax_x * net_X_ExpMaxSum_x)
              / (net_ExpMaxSum_x * net_ExpMaxSum_x);
      }

      // min y.
      if (exp_min_y > min_force_bar) {
        pin_ExpMin_y = fastExp(exp_min_y);
        pin_grad_min_y
            = (net_ExpMinSum_y * (pin_ExpMin_y * (1.0 - coeff_y * node_loc.get_y())) + coeff_y * pin_ExpMin_y * net_Y_ExpMinSum_y)
              / (net_ExpMinSum_y * net_ExpMinSum_y);
      }

      // max y.
      if (exp_max_y > min_force_bar) {
        pin_ExpMax_y = fastExp(exp_max_y);
        pin_grad_max_y
            = (net_ExpMaxSum_y * (pin_ExpMax_y * (1.0 + coeff_y * node_loc.get_y())) - coeff_y * pin_ExpMax_y * net_Y_ExpMaxSum_y)
              / (net_ExpMaxSum_y * net_ExpMaxSum_y);
      }

      _pin_grad_x_list[node->get_node_id()] = (pin_grad_min_x - pin_grad_max_x);
      _pin_grad_y_list[node->get_node_id()] = (pin_grad_min_y - pin_grad_max_y);
    }
  }
}

void WAWirelengthGradient::waWLAnalyzeForDebug(float coeff_x, float coeff_y)
{
  std::ofstream file_stream;
  file_stream.open("./result/pl/gradientAnalyze_4pins.txt", std::ios::app);
  if (!file_stream.good()) {
    std::cout << "Cannot open file for gradientAnalyze.txt" << std::endl;
    exit(1);
  }
  std::stringstream feed;

  for (auto* network : _topology_manager->get_network_list()) {
    if (network->isIgnoreNetwork()) {
      continue;
    }

    if (network->get_node_list().size() < 4) {
      continue;
    }

    auto& wa_net_info = _wa_net_info_list[network->get_network_id()];
    float func_x_value
        = wa_net_info.wa_X_ExpMaxSum_x / wa_net_info.wa_ExpMaxSum_x - wa_net_info.wa_X_ExpMinSum_x / wa_net_info.wa_ExpMinSum_x;
    float func_y_value
        = wa_net_info.wa_Y_ExpMaxSum_y / wa_net_info.wa_ExpMaxSum_y - wa_net_info.wa_Y_ExpMinSum_y / wa_net_info.wa_ExpMinSum_y;

    feed << func_x_value << "," << func_y_value << std::endl;
    feed << network->get_node_list().size() << std::endl;

    for (auto* node : network->get_node_list()) {
      Point<int32_t> pin_coordi = node->get_location();
      Point<float> pin_gradient_pair = obtainPinWirelengthGradient(node, coeff_x, coeff_y);

      feed << pin_coordi.get_x() << "," << pin_coordi.get_y() << std::endl;
      feed << pin_gradient_pair.get_x() << "," << pin_gradient_pair.get_y() << std::endl;
    }
    feed << std::endl;
  }
  file_stream << feed.str();
  feed.clear();
  file_stream.close();

  std::cout << "append gradientAnalyze_4pins.txt to ./result/pl/" << std::endl;
}

Point<float> WAWirelengthGradient::obtainWirelengthGradient_OLD(int32_t inst_id, float coeff_x, float coeff_y)
{
  float gradient_x = 0.0F;
  float gradient_y = 0.0F;

  // debug
  // double wl_collect_inner_runtime = 0.0;

  // ieda::Stats outer_collect_status;
  auto* group = _topology_manager->findGroupById(inst_id);
  if (group) {
    for (auto* node : group->get_node_list()) {
      // ieda::Stats inner_collect_status;
      Point<float> pin_gradient_pair = std::move(obtainPinWirelengthGradient(node, coeff_x, coeff_y));
      // wl_collect_inner_runtime += inner_collect_status.elapsedRunTime();

      // add net weight.
      float net_weight = node->get_network()->get_net_weight();
      float pin_gradient_x = pin_gradient_pair.get_x() * net_weight;
      float pin_gradient_y = pin_gradient_pair.get_y() * net_weight;

      gradient_x += pin_gradient_x;
      gradient_y += pin_gradient_y;
    }
  }

  // std::cout << "DEBUG: wl grad collecting outer runtime: " << outer_collect_status.elapsedRunTime() << " s" << std::endl;
  // std::cout << "DEBUG: wl grad collecting inner runtime: " << wl_collect_inner_runtime << " s" << std::endl;

  return Point<float>(gradient_x, gradient_y);
}

Point<float> WAWirelengthGradient::obtainWirelengthGradient(int32_t inst_id, float coeff_x, float coeff_y)
{
  float gradient_x = 0.0F;

  float gradient_y = 0.0F;

  auto* group = _topology_manager->findGroupById(inst_id);
  if (group) {
    for (auto* node : group->get_node_list()) {
      // add net weight.
      float net_weight = node->get_network()->get_net_weight();
      float pin_gradient_x = _pin_grad_x_list[node->get_node_id()] * net_weight;
      float pin_gradient_y = _pin_grad_y_list[node->get_node_id()] * net_weight;

      gradient_x += pin_gradient_x;
      gradient_y += pin_gradient_y;
    }
  }

  return Point<float>(gradient_x, gradient_y);
}

Point<float> WAWirelengthGradient::obtainPinWirelengthGradient(Node* pin, float coeff_x, float coeff_y)
{
  float gradient_min_x = 0, gradient_min_y = 0;
  float gradient_max_x = 0, gradient_max_y = 0;

  auto& pin_info = _wa_pin_info_list[pin->get_node_id()];
  auto& net_info = _wa_net_info_list[pin->get_network()->get_network_id()];

  // min x.
  if (pin_info.has_MinExpSum_x == 1) {
    float wa_exp_min_sum_x = net_info.wa_ExpMinSum_x;
    float wa_x_exp_min_sum_x = net_info.wa_X_ExpMinSum_x;

    gradient_min_x = (wa_exp_min_sum_x * (pin_info.min_ExpSum_x * (1.0 - coeff_x * pin->get_location().get_x()))
                      + coeff_x * pin_info.min_ExpSum_x * wa_x_exp_min_sum_x)
                     / (wa_exp_min_sum_x * wa_exp_min_sum_x);
  }

  // max x.
  if (pin_info.has_MaxExpSum_x == 1) {
    float wa_exp_max_sum_x = net_info.wa_ExpMaxSum_x;
    float wa_x_exp_max_sum_x = net_info.wa_X_ExpMaxSum_x;

    gradient_max_x = (wa_exp_max_sum_x * (pin_info.max_ExpSum_x * (1.0 + coeff_x * pin->get_location().get_x()))
                      - coeff_x * pin_info.max_ExpSum_x * wa_x_exp_max_sum_x)
                     / (wa_exp_max_sum_x * wa_exp_max_sum_x);
  }

  // min y.
  if (pin_info.has_MinExpSum_y == 1) {
    float wa_exp_min_sum_y = net_info.wa_ExpMinSum_y;
    float wa_y_exp_min_sum_y = net_info.wa_Y_ExpMinSum_y;

    gradient_min_y = (wa_exp_min_sum_y * (pin_info.min_ExpSum_y * (1.0 - coeff_y * pin->get_location().get_y()))
                      + coeff_y * pin_info.min_ExpSum_y * wa_y_exp_min_sum_y)
                     / (wa_exp_min_sum_y * wa_exp_min_sum_y);
  }

  // max y.
  if (pin_info.has_MaxExpSum_y == 1) {
    float wa_exp_max_sum_y = net_info.wa_ExpMaxSum_y;
    float wa_y_exp_max_sum_y = net_info.wa_Y_ExpMaxSum_y;

    gradient_max_y = (wa_exp_max_sum_y * (pin_info.max_ExpSum_y * (1.0 + coeff_y * pin->get_location().get_y()))
                      - coeff_y * pin_info.max_ExpSum_y * wa_y_exp_max_sum_y)
                     / (wa_exp_max_sum_y * wa_exp_max_sum_y);
  }

  return Point<float>(gradient_min_x - gradient_max_x, gradient_min_y - gradient_max_y);
}

//
// https://codingforspeed.com/using-faster-exponential-approximation/
static float fastExp(float a)
{
  a = 1.0 + a / 1024.0;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  return a;
}

}  // namespace ipl