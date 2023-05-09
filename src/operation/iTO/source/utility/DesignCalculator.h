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

#include "Inst.h"
#include "Master.h"
#include "Rect.h"
#include "Utility.h"

namespace ito {
class Layout;

class DesignCalculator {
 public:
  DesignCalculator() = default;
  ~DesignCalculator() = default;
  // open functions
  void calculateDesign();

  static double calculateDesignArea(Layout *layout, int dbu);

  static double calculateCoreArea(ito::Rectangle core, int dbu);

  static double calcMasterArea(Master *master, int dbu);

 private:
  // init
  void initCore();
};

} // namespace ito