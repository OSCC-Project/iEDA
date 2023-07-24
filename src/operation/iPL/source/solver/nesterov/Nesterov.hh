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
 * @Date: 2022-01-10 20:11:04
 * @LastEditTime: 2023-03-06 07:37:40
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/solver/nesterov/Nesterov.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_SOLVER_NESTEROV_H
#define IPL_SOLVER_NESTEROV_H

#include <math.h>

#include <vector>

#include "data/Point.hh"

namespace ipl {

class Nesterov
{
 public:
  Nesterov();
  Nesterov(const Nesterov& other) = delete;
  Nesterov(Nesterov&& other) = delete;
  ~Nesterov() = default;

  // getter.
  int get_current_iter() const { return _current_iter; }
  const std::vector<Point<int32_t>>& get_current_coordis() const { return _current_coordis; }
  const std::vector<Point<float>>& get_current_grads() const { return _current_gradients; }
  const std::vector<Point<float>>& get_next_grads() const { return _next_gradients; }
  const std::vector<Point<int32_t>>& get_next_coordis() const { return _next_coordis; }
  const std::vector<Point<int32_t>>& get_next_slp_coordis() const { return _next_slp_coordis; }
  float get_next_steplength() const { return _next_steplength; }

  // for RDP
  const std::vector<Point<float>>& get_next_gradients() const { return _next_gradients; }
  float get_next_parameter() const { return _next_parameter; }
  void set_next_coordis(const std::vector<Point<int32_t>>& next_coordis) { _next_coordis = next_coordis; }
  void set_next_slp_coordis(const std::vector<Point<int32_t>>& next_slp_coordis) { _next_slp_coordis = next_slp_coordis; }
  void set_next_gradients(const std::vector<Point<float>>& next_gradients) { _next_gradients = next_gradients; }
  void set_next_parameter(float next_parameter) { _next_parameter = next_parameter; }
  void set_next_steplength(float next_steplength) { _next_steplength = next_steplength; }

  // function.
  void initNesterov(std::vector<Point<int32_t>> previous_coordis, std::vector<Point<float>> previous_grads,
                    std::vector<Point<int32_t>> current_coordis, std::vector<Point<float>> current_grads);
  void calculateNextSteplength(std::vector<Point<float>> next_grads);

  void runNextIter(int next_iter, int32_t thread_num);
  void runBackTrackIter(int32_t thread_num);

  void correctNextCoordi(int index, Point<int32_t> new_coordi);
  void correctNextSLPCoordi(int index, Point<int32_t> new_slp_coordi);

  void resetAll();

 private:
  int _current_iter;

  float _current_parameter;
  float _next_parameter;

  float _current_steplength;
  float _next_steplength;

  std::vector<Point<int32_t>> _current_coordis;
  std::vector<Point<int32_t>> _next_coordis;

  // slp is step length prediction.
  std::vector<Point<int32_t>> _current_slp_coordis;
  std::vector<Point<int32_t>> _next_slp_coordis;

  std::vector<Point<float>> _current_gradients;
  std::vector<Point<float>> _next_gradients;

  void checkIterOrder(int next_iter);

  void swapCoordis();
  void swapSLPCoordis();
  void swapSteplength();
  void swapGradients();
  void swapParameter();

  void cleanNextCoordis();
  void cleanNextSLPCoordis();
  void cleanNextSteplength();
  void cleanNextGradients();
  void cleanNextParameter();

  void calculateNextCoordis(int32_t thread_num);
  float calculateSteplength(const std::vector<Point<int32_t>>& prev_slp_coordis, const std::vector<Point<float>>& prev_slp_sum_grads,
                            const std::vector<Point<int32_t>>& cur_slp_coordis, const std::vector<Point<float>>& cur_slp_sum_grads);

  void calculateNextParameter();
};

}  // namespace ipl

#endif  // SRC_OPERATION_IPL_SOURCE_SOLVER_NESTEROV_NESTEROV_HH_
