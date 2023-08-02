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

#include <stdint.h>

namespace ipl {
class SAParam
{
 public:
  void set_max_num_step(uint32_t step) { _max_num_step = step; }
  void set_perturb_per_step(uint32_t step) { _perturb_per_step = step; }
  void set_cool_rate(float rate) { _cool_rate = rate; }
  void set_init_pro(float pro) { _init_pro = pro; }
  void set_init_temperature(float temperatrure) { _init_temperature = temperatrure; }

  uint32_t get_max_num_step() const { return _max_num_step; }
  uint32_t get_perturb_per_step() const { return _perturb_per_step; }
  float get_cool_rate() const { return _cool_rate; }
  float get_init_pro() const { return _init_pro; }
  float get_init_temperature() const { return _init_temperature; }

 protected:
  uint32_t _max_num_step = 500;
  uint32_t _perturb_per_step = 60;
  float _cool_rate = 0.92;
  float _init_pro = 0.95;
  float _init_temperature = 1000;  // default 30000
};
}  // namespace ipl
