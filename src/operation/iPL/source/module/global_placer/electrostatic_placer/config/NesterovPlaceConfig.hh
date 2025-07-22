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
 * @Date: 2022-03-06 15:26:12
 * @LastEditTime: 2022-10-17 14:49:16
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/nesterov_place/config/NesterovPlaceConfig.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_GP_NESTEROV_PLACE_CONFIG_H
#define IPL_OPERATOR_GP_NESTEROV_PLACE_CONFIG_H

#include <string>
#include <vector>

namespace ipl {

class NesterovPlaceConfig
{
 public:
  NesterovPlaceConfig()                                 = default;
  NesterovPlaceConfig(const NesterovPlaceConfig& other) = default;
  NesterovPlaceConfig(NesterovPlaceConfig&& other)      = default;
  ~NesterovPlaceConfig()                                = default;

  NesterovPlaceConfig& operator=(const NesterovPlaceConfig& other) = default;
  NesterovPlaceConfig& operator=(NesterovPlaceConfig&& other) = default;

  // getter.
  int32_t get_thread_num() const { return _thread_num;}
  int32_t get_info_iter_num() const { return _info_iter_num; }
  float   get_init_wirelength_coef() const { return _init_wirelength_coef; }
  float   get_reference_hpwl() const { return _reference_hpwl; }
  float   get_min_wirelength_force_bar() const { return _min_wirelength_force_bar; }
  bool    isAdaptiveBin() const { return _is_adaptive_bin; }
  float   get_target_density() const { return _target_density; }
  int32_t get_bin_cnt_x() const { return _bin_cnt_x; }
  int32_t get_bin_cnt_y() const { return _bin_cnt_y; }
  float   get_min_phi_coef() const { return _min_phi_coef; }
  float   get_max_phi_coef() const { return _max_phi_coef; }
  int32_t get_max_iter() const { return _max_iter; }
  int32_t get_max_back_track() const { return _max_back_track; }
  float   get_target_overflow() const { return _target_overflow; }
  float   get_initial_prev_coordi_update_coef() const { return _initial_prev_coordi_update_coef; }
  float   get_min_precondition() const { return _min_precondition; }
  float   get_init_density_penalty() const { return _init_density_penalty; }
  bool isOptMaxWirelength() const { return _is_opt_max_wirelength;}
  bool isOptTiming() const { return _is_opt_timing;}
  bool isOptCongestion() const { return _is_opt_congestion;}
  int32_t get_max_net_wirelength() const { return _max_net_wirelength;}
  const std::vector<float>& get_opt_overflow_list() {return _opt_overflow_list;} 

  // setter.
  void set_thread_num(int32_t num_thread) { _thread_num = num_thread; }
  void set_info_iter_num(int32_t info_iter_num) { _info_iter_num = info_iter_num; }
  void set_init_wirelength_coef(float coef) { _init_wirelength_coef = coef; }
  void set_reference_hpwl(float hpwl) { _reference_hpwl = hpwl; }
  void set_min_wirelength_force_bar(float bar) { _min_wirelength_force_bar = bar; }
  void set_target_density(float target_density) { _target_density = target_density; }
  void set_adaptive_bin(bool is_adaptive) { _is_adaptive_bin = is_adaptive; }
  void set_bin_cnt_x(float bin_cnt_x) { _bin_cnt_x = bin_cnt_x; }
  void set_bin_cnt_y(float bin_cnt_y) { _bin_cnt_y = bin_cnt_y; }
  void set_min_phi_coef(float min_phi_coef) { _min_phi_coef = min_phi_coef; }
  void set_max_phi_coef(float max_phi_coef) { _max_phi_coef = max_phi_coef; }
  void set_max_iter(int32_t max_iter) { _max_iter = max_iter; }
  void set_max_back_track(int32_t max_back_track) { _max_back_track = max_back_track; }
  void set_init_density_penalty(float init_density_penalty) { _init_density_penalty = init_density_penalty; }
  void set_target_overflow(float target_overflow) { _target_overflow = target_overflow; }
  void set_initial_prev_coordi_update_coef(float coef) { _initial_prev_coordi_update_coef = coef; }
  void set_min_precondition(float precondition) { _min_precondition = precondition; }
  void set_is_opt_max_wirelength(bool flag) { _is_opt_max_wirelength = flag;}
  void set_is_opt_timing(bool flag) { _is_opt_timing = flag; }
  void set_is_opt_congestion(bool flag) { _is_opt_congestion = flag;}
  void set_max_net_wirelength(int32_t max_wirelength) { _max_net_wirelength = max_wirelength;}
  void add_opt_target_overflow(float overflow) { _opt_overflow_list.push_back(overflow);}

 private:
  int32_t _thread_num;
  int32_t _info_iter_num;
  // about wirelength.
  float _init_wirelength_coef;
  float _reference_hpwl;
  float _min_wirelength_force_bar;

  // about density.
  float   _target_density;
  bool _is_adaptive_bin;
  int32_t _bin_cnt_x;
  int32_t _bin_cnt_y;

  // about nesterov.
  int32_t _max_iter;
  int32_t _max_back_track;
  float   _init_density_penalty;
  float   _target_overflow;
  float   _initial_prev_coordi_update_coef;
  float   _min_precondition;
  float   _min_phi_coef;
  float   _max_phi_coef;

  // about maxlength constraint
  bool _is_opt_max_wirelength;
  int32_t _max_net_wirelength;

  // about timing.
  bool _is_opt_timing;

  // about congestion.
  bool _is_opt_congestion;

  // about opt target overflow list
  std::vector<float> _opt_overflow_list;
};

}  // namespace ipl

#endif