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
#include <fstream>

#include "Evaluation.hh"
#include "MPDB.hh"
#include "MPSolution.hh"
#include "Setting.hh"
#include "module/logger/Log.hh"

namespace ipl::imp {
class MPEvaluation : public Evaluation
{
 public:
  MPEvaluation(MPDB* mdb, Setting* set, MPSolution* solution)
  {
    _solution = solution;
    FPRect* core_rect = mdb->get_layout()->get_core_shape();
    _core_width = core_rect->get_width();
    _core_height = core_rect->get_height();
    _net_list = mdb->get_new_net_list();

    _weight_area = set->get_weight_area();
    _weight_e_area = set->get_weight_e_area();
    _weight_wl = set->get_weight_wl();
    _weight_boundary = set->get_weight_boundary();
    _weight_notch = set->get_weight_notch();
    _weight_guidance = set->get_weight_guidance();

    _blockage_list = mdb->get_blockage_list();
    _guidance_to_macro_map = mdb->get_guidance_to_macro_map();
    _macro_list = mdb->get_place_macro_list();

    init_norm(set);  // NOLINT
  }
  float get_weight_e_area() { return _weight_e_area; }
  void set_weight_e_area(float weight) { _weight_e_area = weight; }
  float evaluate() override;
  void init_norm(SAParam* param) override;
  Solution* get_solution() override { return _solution; }
  void showMassage() override;
  void alignMacro();
  void summaryTime() { LOG_INFO << "evl_wl_count: " << _evl_wl_count << " time: " << _evl_wl_time; }

 private:
  float evalHPWL();
  float evalEArea();
  float evalBlockagePenalty();
  float evalBoundaryPenalty();
  float evalLocationPenalty();
  float evalNotchPenalty();
  float eval();
  bool isOverlap();

  float _weight_area = 0;
  float _weight_e_area = 0;
  float _weight_wl = 0;
  float _weight_boundary = 0;
  float _weight_notch = 0;
  float _weight_guidance = 0;

  float _init_prob = 0.95;

  float _norm_area = 0;
  float _norm_wl = 0;
  float _norm_e_area = 0;
  float _norm_boundary = 0;
  float _norm_notch = 0;
  float _norm_guidance = 0;

  MPSolution* _solution;
  uint32_t _core_width;
  uint32_t _core_height;
  vector<FPNet*> _net_list;
  vector<FPRect*> _blockage_list;
  map<FPRect*, FPInst*> _guidance_to_macro_map;
  vector<FPInst*> _macro_list;
  int _evl_wl_count = 0;
  double _evl_wl_time = 0;

  std::ofstream _result;
};
}  // namespace ipl::imp