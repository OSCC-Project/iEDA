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
 * @Date: 2022-01-21 15:22:44
 * @LastEditTime: 2023-02-09 09:57:41
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/config/Configurator.cc
 * Contact : https://github.com/sjchanson
 */

#include <fstream>
#include <vector>

#include "Config.hh"
#include "module/logger/Log.hh"

namespace ipl {

void Config::setConfigFromJson(const std::string& json_file)
{
  initConfig(json_file);
  checkConfig();
  // printConfig();
}

void Config::initConfig(const std::string& json_file)
{
  std::ifstream in_stream(json_file);
  if (!in_stream.good()) {
    LOG_FATAL << "Cannot open json file : " << json_file << " for reading";
  }

  nlohmann::json json;
  in_stream >> json;
  initConfigByJson(json);

  in_stream.close();
}

void Config::initConfigByJson(nlohmann::json json)
{
  int32_t is_max_length_opt = getDataByJson(json, {"PL", "is_max_length_opt"});
  int32_t max_length_constraint = getDataByJson(json, {"PL", "max_length_constraint"});
  int32_t is_timing_effort = getDataByJson(json, {"PL", "is_timing_effort"});
  int32_t is_congestion_effort = getDataByJson(json, {"PL", "is_congestion_effort"});
  int32_t ignore_net_degree = getDataByJson(json, {"PL", "ignore_net_degree"});
  int32_t num_threads = getDataByJson(json, {"PL", "num_threads"});
  int32_t info_iter_num = getDataByJson(json, {"PL", "info_iter_num"});

  // Global Placer
  float init_wirelength_coef = getDataByJson(json, {"PL", "GP", "Wirelength", "init_wirelength_coef"});
  int32_t reference_hpwl = getDataByJson(json, {"PL", "GP", "Wirelength", "reference_hpwl"});
  float min_wirelength_force_bar = getDataByJson(json, {"PL", "GP", "Wirelength", "min_wirelength_force_bar"});

  float target_density = getDataByJson(json, {"PL", "GP", "Density", "target_density"});
  int32_t is_adaptive_bin = getDataByJson(json, {"PL", "GP", "Density", "is_adaptive_bin"});
  int32_t bin_cnt_x = getDataByJson(json, {"PL", "GP", "Density", "bin_cnt_x"});
  int32_t bin_cnt_y = getDataByJson(json, {"PL", "GP", "Density", "bin_cnt_y"});

  int32_t max_iter = getDataByJson(json, {"PL", "GP", "Nesterov", "max_iter"});
  int32_t max_backtrack = getDataByJson(json, {"PL", "GP", "Nesterov", "max_backtrack"});
  float init_density_penalty = getDataByJson(json, {"PL", "GP", "Nesterov", "init_density_penalty"});
  float target_overflow = getDataByJson(json, {"PL", "GP", "Nesterov", "target_overflow"});
  float initial_prev_coordi_update_coef = getDataByJson(json, {"PL", "GP", "Nesterov", "initial_prev_coordi_update_coef"});
  float min_precondition = getDataByJson(json, {"PL", "GP", "Nesterov", "min_precondition"});
  float min_phi_coef = getDataByJson(json, {"PL", "GP", "Nesterov", "min_phi_coef"});
  float max_phi_coef = getDataByJson(json, {"PL", "GP", "Nesterov", "max_phi_coef"});

  // Buffer
  int32_t max_buffer_num = getDataByJson(json, {"PL", "BUFFER", "max_buffer_num"});
  std::vector<std::string> buffer_master_list;
  for (std::string buffer_name : getDataByJson(json, {"PL", "BUFFER", "buffer_type"})) {
    buffer_master_list.push_back(buffer_name);
  }

  // Legalizer
  int32_t lg_max_displacement = getDataByJson(json, {"PL", "LG", "max_displacement"});
  int32_t lg_global_padding = getDataByJson(json, {"PL", "LG", "global_right_padding"});

  // Detail Placer
  int32_t dp_max_displacement = getDataByJson(json, {"PL", "DP", "max_displacement"});
  int32_t dp_global_padding = getDataByJson(json, {"PL", "DP", "global_right_padding"});
  int32_t dp_enable_networkflow = getDataByJson(json, {"PL", "DP", "enable_networkflow"});

  // Filler
  std::vector<std::vector<std::string>> filler_group_list;
  std::vector<std::string> filler_name_list;
  for (std::string filler_name : getDataByJson(json, {"PL", "Filler", "first_iter"})) {
    filler_name_list.push_back(filler_name);
  }
  filler_group_list.push_back(filler_name_list);
  filler_name_list.clear();
  for (std::string filler_name : getDataByJson(json, {"PL", "Filler", "second_iter"})) {
    filler_name_list.push_back(filler_name);
  }
  filler_group_list.push_back(filler_name_list);
  int32_t min_filler_width = getDataByJson(json, {"PL", "Filler", "min_filler_width"});

  // // macro placer
  // std::vector<std::string> fixed_macro_list;
  // std::vector<int32_t> fixed_macro_coordinate;
  // std::vector<int32_t> blockage;
  // std::vector<std::string> guidance_macro_list;
  // std::vector<int32_t> guidance;
  // for (std::string macro_name : getDataByJson(json, {"PL", "MP", "fixed_macro"})) {
  //   fixed_macro_list.emplace_back(macro_name);
  // }
  // for (int32_t coord : getDataByJson(json, {"PL", "MP", "fixed_macro_coordinate"})) {
  //   fixed_macro_coordinate.emplace_back(coord);
  // }
  // for (int32_t coord : getDataByJson(json, {"PL", "MP", "blockage"})) {
  //   blockage.emplace_back(coord);
  // }
  // for (std::string macro_name : getDataByJson(json, {"PL", "MP", "guidance_macro"})) {
  //   guidance_macro_list.emplace_back(macro_name);
  // }
  // for (int32_t coord : getDataByJson(json, {"PL", "MP", "guidance"})) {
  //   guidance.emplace_back(coord);
  // }
  // std::string solution_type = getDataByJson(json, {"PL", "MP", "solution_type"});
  // int32_t perturb_per_step = getDataByJson(json, {"PL", "MP", "SimulateAnneal", "perturb_per_step"});
  // float cool_rate = getDataByJson(json, {"PL", "MP", "SimulateAnneal", "cool_rate"});
  // int32_t parts = getDataByJson(json, {"PL", "MP", "Partition", "parts"});
  // int32_t ufactor = getDataByJson(json, {"PL", "MP", "Partition", "ufactor"});
  // float new_macro_density = getDataByJson(json, {"PL", "MP", "Partition", "new_macro_density"});
  // int32_t halo_x = getDataByJson(json, {"PL", "MP", "halo_x"});
  // int32_t halo_y = getDataByJson(json, {"PL", "MP", "halo_y"});

  /***********************************************************************/

  // Global Placer
  _nes_config.set_thread_num(num_threads);
  _nes_config.set_init_wirelength_coef(init_wirelength_coef);
  _nes_config.set_reference_hpwl(reference_hpwl);
  _nes_config.set_min_wirelength_force_bar(min_wirelength_force_bar);
  _nes_config.set_target_density(target_density);
  if(is_adaptive_bin){
    _nes_config.set_adaptive_bin(true);
  }else{
    _nes_config.set_adaptive_bin(false);
  }
  _nes_config.set_bin_cnt_x(bin_cnt_x);
  _nes_config.set_bin_cnt_y(bin_cnt_y);
  _nes_config.set_max_iter(max_iter);
  _nes_config.set_max_back_track(max_backtrack);
  _nes_config.set_init_density_penalty(init_density_penalty);
  _nes_config.set_target_overflow(target_overflow);
  _nes_config.set_initial_prev_coordi_update_coef(initial_prev_coordi_update_coef);
  _nes_config.set_min_precondition(min_precondition);
  _nes_config.set_min_phi_coef(min_phi_coef);
  _nes_config.set_max_phi_coef(max_phi_coef);
  if (is_max_length_opt) {
    _nes_config.set_is_opt_max_wirelength(true);
    _nes_config.set_max_net_wirelength(max_length_constraint);
  } else {
    _nes_config.set_is_opt_max_wirelength(false);
    _nes_config.set_max_net_wirelength(-1);
  }
  if (is_timing_effort) {
    _nes_config.set_is_opt_timing(true);
  } else {
    _nes_config.set_is_opt_timing(false);
  }
  if (is_congestion_effort) {
    _nes_config.set_is_opt_congestion(true);
  } else {
    _nes_config.set_is_opt_congestion(false);
  }

  if (info_iter_num < 0) {
    _nes_config.set_info_iter_num(10);
  } else {
    _nes_config.set_info_iter_num(info_iter_num);
  }

  // Buffer
  _buffer_config.set_thread_num(num_threads);
  _buffer_config.set_is_max_length_opt(is_max_length_opt);
  _buffer_config.set_max_wirelength_constraint(max_length_constraint);
  _buffer_config.set_max_buffer_num(max_buffer_num);
  for (std::string buffer_name : buffer_master_list) {
    _buffer_config.add_buffer_master(buffer_name);
  }

  // Legalizer
  _lg_config.set_thread_num(num_threads);
  _lg_config.set_max_displacement(lg_max_displacement);
  _lg_config.set_global_padding(lg_global_padding);

  // DetailPlacer
  _dp_config.set_thread_num(num_threads);
  _dp_config.set_max_displacement(dp_max_displacement);
  _dp_config.set_global_padding(dp_global_padding);
  _dp_config.set_enable_networkflow(dp_enable_networkflow);

  // Filler
  _filler_config.set_thread_num(num_threads);
  _filler_config.set_filler_group_list(filler_group_list);
  _filler_config.set_min_filler_width(min_filler_width);

  // // MacroPlacer
  // _mp_config.set_fixed_macro(fixed_macro_list);
  // _mp_config.set_fixed_macro_coord(fixed_macro_coordinate);
  // _mp_config.set_blockage(blockage);
  // _mp_config.set_guidance_macro(guidance_macro_list);
  // _mp_config.set_guidance(guidance);
  // _mp_config.set_solution_type(solution_type);
  // _mp_config.set_perturb_per_step(perturb_per_step);
  // _mp_config.set_cool_rate(cool_rate);
  // _mp_config.set_parts(parts);
  // _mp_config.set_ufactor(ufactor);
  // _mp_config.set_new_macro_density(new_macro_density);
  // _mp_config.set_halo_x(halo_x);
  // _mp_config.set_halo_y(halo_y);

  _ignore_net_degree = ignore_net_degree;
  _is_timing_effort = (is_timing_effort == 1);
  _is_congestion_effort = (is_congestion_effort == 1);
}

nlohmann::json Config::getDataByJson(nlohmann::json value, std::vector<std::string> flag_list)
{
  int flag_size = flag_list.size();
  if (flag_size == 0) {
    LOG_FATAL << "Config The number of json flag is zero!";
  }

  for (int i = 0; i < flag_size; i++) {
    value = value[flag_list[i]];
  }

  if (!value.is_null()) {
    return value;
  }

  std::string key;

  for (int i = 0; i < flag_size; i++) {
    key += flag_list[i];
    if (i < flag_size - 1) {
      key += ".";
    }
  }
  LOG_FATAL << "Config "
            << "The configuration file key=[ " << key << " ] is null! exit...";
}

void Config::checkConfig()
{
}

void Config::printConfig()
{
  //
}

}  // namespace ipl