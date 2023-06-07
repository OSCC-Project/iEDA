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

#include "Setting.hh"

namespace ipl::imp {
Setting::Setting()
{
  _new_macro_density = 1;
  _macro_halo_x = 0;
  _macro_halo_y = 0;
  _partition_type = PartitionType::kHmetis;
  _solver_type = SolverType::kSimulate_anneal;
  _solution_type = SolutionType::kBStar_tree;
  _output_path = ".";
  _parts = 16;
  _ncon = 5;
  _ufactor = 400;
  _weight_area = 1;       // 1
  _weight_e_area = 96;    // 96
  _weight_wl = 12;        // 12
  _weight_boundary = 22;  // 22
  _weight_notch = 1;
  _weight_guidance = 3;
  _swap_pro = 0.5;
  _move_pro = 0.5;
  _swap_pos_pro = 0.3;
  _swap_neg_pro = 0.3;
  _rotate_pro = 0;
}
}  // namespace ipl::imp