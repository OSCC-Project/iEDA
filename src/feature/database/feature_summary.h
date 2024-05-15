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
/**
 * @file		summary.h
 * @date		13/05/2024
 * @version		0.1
 * @description


        summary data
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "feature_db.h"
#include "feature_ipl.h"
#include "feature_irt.h"

namespace ieda_feature {

class FeatureSummary
{
 public:
  FeatureSummary();
  ~FeatureSummary();

  // getter
  DBSummary& get_db() { return _db; }
  PlaceSummary& get_summary_ipl() { return _summary_ipl; }
  RTSummary& get_summary_irt() { return _summary_irt; }

 private:
  PlaceSummary _summary_ipl;
  RTSummary _summary_irt;
  DBSummary _db;
};

}  // namespace ieda_feature
