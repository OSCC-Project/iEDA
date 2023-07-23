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
#pragma once

#include <string>
#include <vector>

namespace ipl::imp {

class MacroPlacerConfig
{
 public:
  MacroPlacerConfig() = default;
  MacroPlacerConfig(const MacroPlacerConfig& other) = default;
  MacroPlacerConfig(MacroPlacerConfig&& other) = default;
  ~MacroPlacerConfig() = default;

  MacroPlacerConfig& operator=(const MacroPlacerConfig& other) = default;
  MacroPlacerConfig& operator=(MacroPlacerConfig&& other) = default;

  // getter.
  const std::vector<std::string>& get_fixed_macro() const { return _fixed_macro; }
  const std::vector<int32_t>& get_fixed_macro_coordinate() const { return _fixed_macro_coordinate; }
  const std::vector<int32_t>& get_blockage() const { return _blockage; }
  const std::vector<std::string>& get_guidance_macro() const { return _guidance_macro; }
  const std::vector<int32_t>& get_guidance() const { return _guidance; }
  std::string get_solution_tpye() const { return _solution_type; }
  int32_t get_perturb_per_step() const { return _perturb_per_step; }
  float get_cool_rate() const { return _cool_rate; }
  int32_t get_parts() const { return _parts; }
  int32_t get_ufactor() const { return _ufactor; }
  float get_new_macro_density() const { return _new_macro_density; }
  int32_t get_halo_x() const { return _halo_x; }
  int32_t get_halo_y() const { return _halo_y; }
  std::string get_output_path() const { return _output_path; }

  // setter.
  void set_fixed_macro(std::vector<std::string> fixed_macro) { _fixed_macro = fixed_macro; }
  void set_fixed_macro_coord(std::vector<int32_t> coord) { _fixed_macro_coordinate = coord; }
  void set_blockage(std::vector<int32_t> coord) { _blockage = coord; }
  void set_guidance_macro(std::vector<std::string> guidance_macro) { _guidance_macro = guidance_macro; }
  void set_guidance(std::vector<int32_t> coord) { _guidance = coord; }
  void set_solution_type(std::string type) { _solution_type = type; }
  void set_perturb_per_step(int32_t itera) { _perturb_per_step = itera; }
  void set_cool_rate(float rate) { _cool_rate = rate; }
  void set_parts(int32_t parts) { _parts = parts; }
  void set_ufactor(int32_t ufactor) { _ufactor = ufactor; }
  void set_new_macro_density(float density) { _new_macro_density = density; }
  void set_halo_x(int32_t x) { _halo_x = x; }
  void set_halo_y(int32_t y) { _halo_y = y; }
  void set_output_path(std::string path) { _output_path = path; }

 private:
  // about fixed macro.
  std::vector<std::string> _fixed_macro;
  std::vector<int32_t> _fixed_macro_coordinate;

  // about blockage.
  std::vector<int32_t> _blockage;

  // about guidance macro.
  std::vector<std::string> _guidance_macro;
  std::vector<int32_t> _guidance;

  // about solution type
  std::string _solution_type;

  // about simulate anneal.
  int32_t _perturb_per_step;
  float _cool_rate;

  // about partition
  int32_t _parts;
  int32_t _ufactor;
  float _new_macro_density;

  // about halo
  int32_t _halo_x;
  int32_t _halo_y;

  // about output path
  std::string _output_path;
};

}  // namespace ipl::imp
