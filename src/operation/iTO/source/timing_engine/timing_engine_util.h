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

#include <map>

#include "define.h"

namespace ito {

#define COE 1.1
class TOLibCellLoadMap
{
 public:
  TOLibCellLoadMap() {}
  ~TOLibCellLoadMap() {}
  float get_load(LibCell* cell) { return _cell_load_map[cell]; }
  void clear() { _cell_load_map.clear(); }
  void insert(LibCell* cell, float load) { _cell_load_map[cell] = load; }

 private:
  std::map<LibCell*, float> _cell_load_map;
};

class TOLibRepowerInstance
{
 public:
  TOLibRepowerInstance() = default;
  TOLibRepowerInstance(Pin* driver_pin);
  ~TOLibRepowerInstance() {}

  bool repowerInstance();
  bool repowerInstance(ista::LibCell* repower_size, ista::Instance* repowered_inst);

 private:
  Pin* _driver_pin = nullptr;
  ista::Instance* _inst = nullptr;
  ista::LibCell* _repower_size_best = nullptr;
  double _load = 0.0;
  float _cell_target_load_best = 0.0;
  float _load_margin_best = 0.0;
  float _cell_delay_best = 0.0;
  bool _b_buffer = false;

  bool is_repower();
  bool find_best_cell();
  bool is_best(ista::LibCell* cell);
  void set_best(ista::LibCell* cell);
  bool check_buf(float dist, float delay)
  {
    return (((dist < coe_value(_load_margin_best)) && (delay < _cell_delay_best)))
           || ((dist < _load_margin_best) && (delay < coe_value(_cell_delay_best)));
  }
  bool check_load(float dist, float load) { return dist < _load_margin_best && load > _cell_target_load_best; }
  float coe_value(float value) { return value * COE; };  ///  coefficient

  float calLoad(ista::LibCell* cell);
  float calDelay(ista::LibCell* cell);
  float calDist(float target_cell_load);
};

}  // namespace ito
