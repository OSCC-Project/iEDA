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
#include "MP.hh"

#include <functional>
#include <memory>

#include "BlkClustering.hh"
#include "HierPlacer.hh"
#include "Logger.hpp"
#include "MacroAligner.hh"

namespace imp {

void MP::runMP()
{
  float macro_halo_micron = 2.0;
  float dead_space_ratio = 0.8;
  float weight_wl = 1.0;
  float weight_ol = 0.05;
  float weight_ob = 0.02;
  float weight_periphery = 0.05;
  float weight_blockage = 0.0;
  float weight_io = 0.0;
  float max_iters = 1000;
  float cool_rate = 0.96;
  float init_temperature = 2000.0;

  BlkClustering2 clustering{.l1_nparts = 200, .level_num = 1, .parser = _parser};  // one level place
  root().parallel_preorder_op(clustering);
  auto placer = SAHierPlacer<int32_t>(root(), macro_halo_micron, dead_space_ratio, weight_wl, weight_ol, weight_ob, weight_periphery,
                                      weight_blockage, weight_io, max_iters, cool_rate, init_temperature);

  placer(root());
  std::string file_name = "placement_level" + std::to_string(clustering.level_num) + "_" + std::to_string(clustering.l1_nparts) + "_"
                          + std::to_string(clustering.l2_nparts);
  // writePlacement(root(), file_name + ".txt");
  auto macro_aligner = MacroAligner<int32_t>();
  macro_aligner(root());
  // writePlacement(root(), file_name + "_aligned.txt");
  writePlacementTcl(root(), file_name + ".tcl", root().netlist().property()->get_database_unit());
  _parser->write();  // write back to idb
}

}  // namespace imp