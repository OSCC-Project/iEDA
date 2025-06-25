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
/**
 * @file CtsCellLib.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ModelFactory.hh"

namespace icts {

class CtsCellLib
{
 public:
  CtsCellLib(const std::string& cell_master, const std::vector<std::vector<double>>& index_list, const std::vector<double>& delay_values,
             const std::vector<double>& slew_values)
      : _cell_master(cell_master), _index_list(index_list), _delay_values(delay_values), _slew_values(slew_values)
  {
    _slew_in_span = std::span<const double>(_index_list[0].data(), _index_list[0].size());
    _cap_out_span = std::span<const double>(_index_list[1].data(), _index_list[1].size());
  }
  ~CtsCellLib() = default;
  // get
  std::string get_cell_master() const { return _cell_master; }
  std::vector<std::vector<double>> get_index_list() const { return _index_list; }
  std::vector<double> get_delay_values() const { return _delay_values; }
  std::vector<double> get_slew_values() const { return _slew_values; }
  std::vector<double> get_delay_coef() const { return _delay_coef; }
  std::vector<double> get_slew_coef() const { return _slew_coef; }
  double get_init_cap() const { return _init_cap; }
  double getDelayIntercept() const
  {
    //  return _delay_coef[0];
    return calcInsertDelay(_slew_in_span[0], _cap_out_span[0]);
  }

  // setk
  void set_delay_coef(const std::vector<double>& coef) { _delay_coef = coef; }
  void set_slew_coef(const std::vector<double>& coef) { _slew_coef = coef; }
  void set_init_cap(const double& init_cap) { _init_cap = init_cap; }

  // calc
  double calcSlew(const double& cap_out) const
  {
    return calcLinearSlew(cap_out);
  }

  double calcInsertSlew(const double& slew_in, const double& cap_out) const
  {
    // slew_in index
    auto it1 = std::upper_bound(_slew_in_span.begin(), _slew_in_span.end(), slew_in);
    size_t i1 = std::clamp(static_cast<size_t>(std::distance(_slew_in_span.begin(), it1)), size_t{1}, _slew_in_span.size() - 1);
    double slew_low = _slew_in_span[i1 - 1];
    double slew_high = _slew_in_span[i1];

    // cap_out index
    auto it2 = std::upper_bound(_cap_out_span.begin(), _cap_out_span.end(), cap_out);
    size_t i2 = std::clamp(static_cast<size_t>(std::distance(_cap_out_span.begin(), it2)), size_t{1}, _cap_out_span.size() - 1);
    double cap_low = _cap_out_span[i2 - 1];
    double cap_high = _cap_out_span[i2];

    double y11 = _slew_values[(i1 - 1) * _cap_out_span.size() + i2 - 1];
    double y12 = _slew_values[(i1 - 1) * _cap_out_span.size() + i2];
    double y21 = _slew_values[i1 * _cap_out_span.size() + i2 - 1];
    double y22 = _slew_values[i1 * _cap_out_span.size() + i2];

    // insert value
    double res = std::lerp(std::lerp(y11, y12, std::lerp(0.0, 1.0, (cap_out - cap_low) / (cap_high - cap_low))),
                           std::lerp(y21, y22, std::lerp(0.0, 1.0, (cap_out - cap_low) / (cap_high - cap_low))),
                           std::lerp(0.0, 1.0, (slew_in - slew_low) / (slew_high - slew_low)));

    return res;
  }
  double calcDelay(const double& slew_in, const double& cap_out) const
  {
    return calcInsertDelay(slew_in, cap_out);
  }
  double calcLinearSlew(const double& cap_out) const { return _slew_coef[0] + _slew_coef[1] * cap_out; }
  double calcLinearDelay(const double& slew_in, const double& cap_out) const
  {
    return _delay_coef[0] + _delay_coef[1] * slew_in + _delay_coef[2] * cap_out;
  }
  double calcInsertDelay(const double& slew_in, const double& cap_out) const
  {
    // slew_in index
    auto it1 = std::upper_bound(_slew_in_span.begin(), _slew_in_span.end(), slew_in);
    size_t i1 = std::clamp(static_cast<size_t>(std::distance(_slew_in_span.begin(), it1)), size_t{1}, _slew_in_span.size() - 1);
    double slew_low = _slew_in_span[i1 - 1];
    double slew_high = _slew_in_span[i1];

    // cap_out index
    auto it2 = std::upper_bound(_cap_out_span.begin(), _cap_out_span.end(), cap_out);
    size_t i2 = std::clamp(static_cast<size_t>(std::distance(_cap_out_span.begin(), it2)), size_t{1}, _cap_out_span.size() - 1);
    double cap_low = _cap_out_span[i2 - 1];
    double cap_high = _cap_out_span[i2];

    double y11 = _delay_values[(i1 - 1) * _cap_out_span.size() + i2 - 1];
    double y12 = _delay_values[(i1 - 1) * _cap_out_span.size() + i2];
    double y21 = _delay_values[i1 * _cap_out_span.size() + i2 - 1];
    double y22 = _delay_values[i1 * _cap_out_span.size() + i2];

    // insert value
    double res = std::lerp(std::lerp(y11, y12, std::lerp(0.0, 1.0, (cap_out - cap_low) / (cap_high - cap_low))),
                           std::lerp(y21, y22, std::lerp(0.0, 1.0, (cap_out - cap_low) / (cap_high - cap_low))),
                           std::lerp(0.0, 1.0, (slew_in - slew_low) / (slew_high - slew_low)));

    return res;
  }

 private:
  std::string _cell_master;
  std::vector<std::vector<double>> _index_list;
  std::span<const double> _slew_in_span;
  std::span<const double> _cap_out_span;
  std::vector<double> _delay_values;
  std::vector<double> _slew_values;
  std::vector<double> _delay_coef;
  std::vector<double> _slew_coef;
  double _init_cap = 0.0;
};

class CtsLibs
{
 public:
  CtsLibs() = default;
  ~CtsLibs() = default;
  void insertLib(const std::string& cell_master, CtsCellLib* lib) { _lib_maps[cell_master] = lib; }

  CtsCellLib* findLib(const std::string& cell_master)
  {
    if (_lib_maps.find(cell_master) == _lib_maps.end()) {
      return nullptr;
    }
    return _lib_maps[cell_master];
  }
 private:
  std::unordered_map<std::string, CtsCellLib*> _lib_maps;
};
}  // namespace icts