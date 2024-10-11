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
#include "feature_icts.h"
#include "feature_ieval.h"
#include "feature_ino.h"
#include "feature_ipl.h"
#include "feature_irt.h"
#include "feature_ito.h"

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
  CTSSummary& get_summary_icts() { return _summary_icts; }
  NetOptSummary& get_summary_ino() { return _summary_ino; }
  TimingOptSummary& get_summary_ito_optdrv() { return _summary_ito_optdrv; }
  TimingOptSummary& get_summary_ito_opthold() { return _summary_ito_opthold; }
  TimingOptSummary& get_summary_ito_optsetup() { return _summary_ito_optsetup; }

  TimingEvalSummary& get_summary_timing_eval() { return _summary_timing_eval; }
  TotalWLSummary& get_summary_wirelength_eval() { return _summary_wirelength_eval; }
  DensityMapSummary& get_summary_density_eval() { return _summary_density_eval; }
  CongestionSummary& get_summary_congestion_eval() { return _summary_congestion_eval; }

  void set_db(DBSummary db) { _db = db; }
  void set_ipl(PlaceSummary db) { _summary_ipl = db; }
  void set_irt(RTSummary db) { _summary_irt = db; }
  void set_icts(CTSSummary db) { _summary_icts = db; }
  void set_ino(NetOptSummary db) { _summary_ino = db; }
  void set_ito_optdrv(TimingOptSummary db) { _summary_ito_optdrv = db; }
  void set_ito_opthold(TimingOptSummary db) { _summary_ito_opthold = db; }
  void set_ito_optsetup(TimingOptSummary db) { _summary_ito_optsetup = db; }

  void set_timing_eval(TimingEvalSummary db) { _summary_timing_eval = db; }
  void set_wirelength_eval(TotalWLSummary db) { _summary_wirelength_eval = db; }
  void set_density_eval(DensityMapSummary db) { _summary_density_eval = db; }
  void set_congestion_eval(CongestionSummary db) { _summary_congestion_eval = db; }

 private:
  DBSummary _db;
  PlaceSummary _summary_ipl;
  RTSummary _summary_irt;
  CTSSummary _summary_icts;
  NetOptSummary _summary_ino;
  TimingOptSummary _summary_ito_optdrv;
  TimingOptSummary _summary_ito_opthold;
  TimingOptSummary _summary_ito_optsetup;

  TimingEvalSummary _summary_timing_eval;
  TotalWLSummary _summary_wirelength_eval;
  DensityMapSummary _summary_density_eval;
  CongestionSummary _summary_congestion_eval;
};

}  // namespace ieda_feature
