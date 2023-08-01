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
 * @Date: 2022-01-10 20:11:14
 * @LastEditTime: 2022-10-26 09:27:56
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/solver/nesterov/Nesterov.cc
 * Contact : https://github.com/sjchanson
 */

#include "Nesterov.hh"

#include <float.h>
#include <math.h>

#include "Log.hh"
#include "omp.h"
#include "utility/Utility.hh"

namespace ipl {

Nesterov::Nesterov()
    : _current_iter(0),
      _current_parameter(1.0),
      _next_parameter(1.0),
      _current_steplength(FLT_MIN),
      _next_steplength(FLT_MIN),
      _current_gradients(FLT_MIN),
      _next_gradients(FLT_MIN)
{
}

void Nesterov::initNesterov(std::vector<Point<int32_t>> previous_coordis, std::vector<Point<float>> previous_grads,
                            std::vector<Point<int32_t>> current_coordis, std::vector<Point<float>> current_grads)
{
  _current_coordis = _current_slp_coordis = std::move(previous_coordis);
  _current_gradients = std::move(previous_grads);
  _next_coordis = _next_slp_coordis = std::move(current_coordis);
  _next_gradients = std::move(current_grads);

  _next_steplength = calculateSteplength(_current_slp_coordis, _current_gradients, _next_slp_coordis, _next_gradients);
}

void Nesterov::calculateNextSteplength(std::vector<Point<float>> next_grads)
{
  if (!_next_gradients.empty()) {
    // print the error infomations. level 0.
  }
  _next_gradients = std::move(next_grads);
  _next_steplength = calculateSteplength(_current_slp_coordis, _current_gradients, _next_slp_coordis, _next_gradients);
}

void Nesterov::runNextIter(int next_iter, int32_t thread_num)
{
  checkIterOrder(next_iter);

  swapCoordis();
  swapSLPCoordis();
  swapSteplength();
  swapGradients();
  swapParameter();

  calculateNextParameter();
  calculateNextCoordis(thread_num);
}

void Nesterov::runBackTrackIter(int32_t thread_num)
{
  cleanNextCoordis();
  cleanNextSLPCoordis();
  swapSteplength();
  cleanNextGradients();

  calculateNextCoordis(thread_num);
}

void Nesterov::correctNextCoordi(int index, Point<int32_t> new_coordi)
{
  if (index >= static_cast<int>(_next_coordis.size())) {
    LOG_ERROR << "Error in correctNextCoordi : index out of ranget!";
  }

  _next_coordis[index] = std::move(new_coordi);
}

void Nesterov::correctNextSLPCoordi(int index, Point<int32_t> new_slp_coordi)
{
  if (index >= static_cast<int>(_next_slp_coordis.size())) {
    LOG_ERROR << "Error in correctNextSLPCoordi : index out of ranget!";
  }

  _next_slp_coordis[index] = std::move(new_slp_coordi);
}

void Nesterov::calculateNextParameter()
{
  _next_parameter = (1.0 + sqrt(4.0 * _current_parameter * _current_parameter + 1.0)) * 0.5;
}

void Nesterov::calculateNextCoordis(int32_t thread_num)
{
  // if (!_next_coordis.empty() || !_next_slp_coordis.empty()) {
  //   // LOG_ERROR << "Error in calculateNextCoordis : _next_coordis/_next_slp_coordis is not empty!";
  // }

  float coeff = (_current_parameter - 1.0) / _next_parameter;

#pragma omp parallel for num_threads(thread_num)
  for (size_t i = 0; i < _current_coordis.size(); i++) {
    Point<int32_t> next_coordi(_current_slp_coordis[i].get_x() + _current_steplength * _current_gradients[i].get_x(),
                               _current_slp_coordis[i].get_y() + _current_steplength * _current_gradients[i].get_y());
    Point<int32_t> next_slp_coordi(next_coordi.get_x() + coeff * (next_coordi.get_x() - _current_coordis[i].get_x()),
                                   next_coordi.get_y() + coeff * (next_coordi.get_y() - _current_coordis[i].get_y()));

    _next_coordis[i] = std::move(next_coordi);
    _next_slp_coordis[i] = std::move(next_slp_coordi);

    // _next_coordis.push_back(std::move(next_coordi));
    // _next_slp_coordis.push_back(std::move(next_slp_coordi));
  }
}

float Nesterov::calculateSteplength(const std::vector<Point<int32_t>>& prev_slp_coordis,
                                    const std::vector<Point<float>>& prev_slp_sum_grads, const std::vector<Point<int32_t>>& cur_slp_coordis,
                                    const std::vector<Point<float>>& cur_slp_sum_grads)
{
  Utility utility;
  float coordi_distance = utility.getDistance(prev_slp_coordis, cur_slp_coordis);
  float grad_distance = utility.getDistance(prev_slp_sum_grads, cur_slp_sum_grads);

  return coordi_distance / grad_distance;
}

void Nesterov::resetAll()
{
  _current_iter = 0;
  _current_parameter = 1.0;
  _next_parameter = FLT_MIN;
  _current_steplength = FLT_MIN;
  _next_parameter = FLT_MIN;
  _current_steplength = FLT_MIN;
  _next_steplength = FLT_MIN;

  _current_coordis.clear();
  _next_coordis.clear();
  _current_slp_coordis.clear();
  _next_slp_coordis.clear();
  _current_gradients.clear();
  _next_gradients.clear();
}

void Nesterov::checkIterOrder(int next_iter)
{
  int differential_step = next_iter - _current_iter;
  if (differential_step == 1) {
    _current_iter = next_iter;
    return;
  } else {
    LOG_ERROR << "Error in checkIterOrder : next_iter is not continuous!";
  }
}

void Nesterov::swapCoordis()
{
  _current_coordis.swap(_next_coordis);
  cleanNextCoordis();
}

void Nesterov::swapSLPCoordis()
{
  _current_slp_coordis.swap(_next_slp_coordis);
  cleanNextSLPCoordis();
}

void Nesterov::swapSteplength()
{
  _current_steplength = _next_steplength;
  cleanNextSteplength();
}

void Nesterov::swapGradients()
{
  _current_gradients.swap(_next_gradients);
  cleanNextGradients();
}

void Nesterov::swapParameter()
{
  _current_parameter = _next_parameter;
  cleanNextParameter();
}

void Nesterov::cleanNextCoordis()
{
  // _next_coordis.clear();
}

void Nesterov::cleanNextSLPCoordis()
{
  // _next_slp_coordis.clear();
}

void Nesterov::cleanNextSteplength()
{
  _next_steplength = FLT_MIN;
}

void Nesterov::cleanNextGradients()
{
  _next_gradients.clear();
}

void Nesterov::cleanNextParameter()
{
  _next_parameter = FLT_MIN;
}

}  // namespace ipl