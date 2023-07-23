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

#include "SAParam.hh"
#include "database/Enum.hh"

namespace ipl::imp {

class Setting : public SAParam
{
 public:
  Setting();

  // set
  void set_new_macro_density(float density) { _new_macro_density = density; }
  void set_macro_halo_x(int halo) { _macro_halo_x = halo; }
  void set_macro_halo_y(int halo) { _macro_halo_y = halo; }
  void set_partition_type(PartitionType type) { _partition_type = type; }
  void set_solver_type(SolverType type) { _solver_type = type; }
  void set_solution_type(SolutionType type) { _solution_type = type; }
  void set_output_path(std::string path) { _output_path = path; }
  void set_parts(int parts) { _parts = parts; }
  void set_ncon(int ncon) { _ncon = ncon; }
  void set_ufactor(int ufactor) { _ufactor = ufactor; }
  void set_weight_area(float weight) { _weight_area = weight; }
  void set_weight_e_area(float weight) { _weight_e_area = weight; }
  void set_weight_wl(float weight) { _weight_wl = weight; }
  void set_weight_boundary(float weight) { _weight_boundary = weight; }
  void set_weight_notch(float weight) { _weight_notch = weight; }
  void set_weight_guidance(float weight) { _weight_guidance = weight; }
  void set_swap_pro(float pro) { _swap_pro = pro; }
  void set_move_pro(float pro) { _move_pro = pro; }
  void set_swap_pos_pro(float pro) { _swap_pos_pro = pro; }
  void set_swap_neg_pro(float pro) { _swap_neg_pro = pro; }
  void set_rotate_pro(float pro) { _rotate_pro = pro; }

  // get
  float get_new_macro_density() const { return _new_macro_density; }
  uint32_t get_macro_halo_x() const { return _macro_halo_x; }
  uint32_t get_macro_halo_y() const { return _macro_halo_y; }
  PartitionType get_partition_type() const { return _partition_type; }
  SolverType get_solver_type() const { return _solver_type; }
  SolutionType get_solution_type() const { return _solution_type; }
  std::string get_output_path() const { return _output_path; }
  int get_parts() const { return _parts; }
  int get_ncon() const { return _ncon; }
  int get_ufactor() const { return _ufactor; }
  float get_weight_area() const { return _weight_area; }
  float get_weight_e_area() const { return _weight_e_area; }
  float get_weight_wl() const { return _weight_wl; }
  float get_weight_boundary() const { return _weight_boundary; }
  float get_weight_notch() const { return _weight_notch; }
  float get_weight_guidance() const { return _weight_guidance; }
  float get_swap_pro() const { return _swap_pro; }
  float get_move_pro() const { return _move_pro; }
  float get_swap_pos_pro() const { return _swap_pos_pro; }
  float get_swap_neg_pro() const { return _swap_neg_pro; }
  float get_rotate_pro() const { return _rotate_pro; }

 private:
  // macroplacer
  float _new_macro_density;
  uint32_t _macro_halo_x;
  uint32_t _macro_halo_y;

  // type
  PartitionType _partition_type;
  SolverType _solver_type;
  SolutionType _solution_type;
  std::string _output_path;

  // partition
  int _parts;  // the number of cluster
  int _ncon;   // The number of balancing constraints
  int _ufactor;

  // simulate anneal
  float _weight_area;
  float _weight_e_area;
  float _weight_wl;  // wire length
  float _weight_boundary;
  float _weight_notch;
  float _weight_guidance;  // guidance

  // B* tree
  float _swap_pro;  // the probability of swap
  float _move_pro;  // the probability of move

  // Sequence pair
  float _swap_pos_pro;
  float _swap_neg_pro;
  float _rotate_pro;
};

}  // namespace ipl::imp