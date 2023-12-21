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
#include "condition_manager.h"

namespace idrc {

/**
 * rule conditions are concepts built from tech lef drc rules, it contains a condition matrix to guide condition check orders, the rule
 * matrix index indicates the checking order,
 *
 */
class DrcBasicPoint;

class DrcConditionBuilder
{
 public:
  DrcConditionBuilder(DrcConditionManager* condition_manager) : _condition_manager(condition_manager) {}
  ~DrcConditionBuilder() {}

  bool buildCondition();

 private:
  DrcConditionManager* _condition_manager;

  /// condition builder
  void buildConditionRoutingLayer();
  void filterSpacing();
  void filterSpacingForPolygon(DrcBasicPoint* start_point, idb::IdbLayer* layer, int min_spacing, int max_spacing);
  void checkSpacingUp(DrcBasicPoint* point, idb::IdbLayer* layer, int min_spacing, int max_spacing);
  void checkSpacingRight(DrcBasicPoint* point, idb::IdbLayer* layer, int min_spacing, int max_spacing);
  void checkSpacing(DrcBasicPoint* point, idb::IdbLayer* layer, int min_spacing, int max_spacing, DrcDirection direction);
  void buildWidth();

  /// violation process
  void saveViolationSpacing(DrcBasicPoint* start_point_1, DrcBasicPoint* start_point_2, idb::IdbLayer* layer, bool b_vertical = false,
                            int min_spacing = -1);

  /// utility
  std::vector<ieda_solver::GtlPoint> get_boost_point(DrcBasicPoint* point);
};

}  // namespace idrc