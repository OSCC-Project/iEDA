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

#include "RTHeader.hpp"

namespace irt {

class PAComParam
{
 public:
  PAComParam() = default;
  PAComParam(int32_t max_candidate_point_num, int32_t extra_via_master_num, int32_t ap_per_via_master)
  {
    _max_candidate_point_num = max_candidate_point_num;
    _extra_via_master_num = extra_via_master_num;
    _ap_per_via_master = ap_per_via_master;
  }
  ~PAComParam() = default;
  // getter
  int32_t get_max_candidate_point_num() const { return _max_candidate_point_num; }
  int32_t get_extra_via_master_num() const { return _extra_via_master_num; }
  int32_t get_ap_per_via_master() const { return _ap_per_via_master; }
  // setter
  void set_max_candidate_point_num(const int32_t max_candidate_point_num) { _max_candidate_point_num = max_candidate_point_num; }
  void set_extra_via_master_num(const int32_t extra_via_master_num) { _extra_via_master_num = extra_via_master_num; }
  void set_ap_per_via_master(const int32_t ap_per_via_master) { _ap_per_via_master = ap_per_via_master; }

 private:
  int32_t _max_candidate_point_num = 0;
  int32_t _extra_via_master_num = 0;
  int32_t _ap_per_via_master = 0;
};

}  // namespace irt
