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

#include "RandomPlace.hh"

#include <random>

namespace ipl {

void RandomPlace::runRandomPlace()
{
  auto core_shape = std::move(_placer_db->get_layout()->get_core_shape());

  int32_t seed = 1000;
  std::default_random_engine gen(seed);
  std::normal_distribution<float> dis_x(core_shape.get_center().get_x(), core_shape.get_width() * 0.001);
  std::normal_distribution<float> dis_y(core_shape.get_center().get_y(), core_shape.get_height() * 0.001);

  for (auto* inst : _placer_db->get_design()->get_instance_list()) {
    if (inst->isFixed()) {
      continue;
    }

    inst->update_center_coordi(dis_x(gen), dis_y(gen));
  }

  LOG_INFO << "Finish Random Initialize Insts Coordinates!";
}

}  // namespace ipl