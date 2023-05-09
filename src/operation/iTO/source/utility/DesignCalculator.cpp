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
#include "DesignCalculator.h"

#include "Layout.h"

namespace ito {
double DesignCalculator::calculateDesignArea(Layout *layout, int dbu) {
  double design_area = 0.0;
  for (auto inst : layout->get_insts()) {
    Master *master = inst->get_master();
    design_area += calcMasterArea(master, dbu);
  }
  return design_area;
}

double DesignCalculator::calculateCoreArea(Rectangle core, int dbu) {
  double core_x = dbuToMeters(core.get_dx(), dbu);
  double core_y = dbuToMeters(core.get_dy(), dbu);
  return core_x * core_y;
}

double DesignCalculator::calcMasterArea(Master *master, int dbu) {
  //   if (!master->isAutoPlaceable()) {
  //     return 0;
  //   }
  double width = dbuToMeters(master->get_width(), dbu);
  double height = dbuToMeters(master->get_height(), dbu);
  return width * height;
}
} // namespace ito
