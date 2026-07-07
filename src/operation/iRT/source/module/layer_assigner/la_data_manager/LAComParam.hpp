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

namespace irt {

class LAComParam
{
 public:
  LAComParam() = default;
  LAComParam(int32_t topo_spilt_length, double via_unit, double overflow_unit)
  {
    _topo_spilt_length = topo_spilt_length;
    _via_unit = via_unit;
    _overflow_unit = overflow_unit;
  }
  LAComParam(int32_t topo_spilt_length, int32_t mid_topo_spilt_length, int32_t long_topo_spilt_length, int32_t short_segment_length,
             int32_t mid_segment_length, int32_t long_segment_length, double via_unit, double overflow_unit, double layer_bias_unit,
             double layer_switch_unit)
  {
    _topo_spilt_length = topo_spilt_length;
    _mid_topo_spilt_length = mid_topo_spilt_length;
    _long_topo_spilt_length = long_topo_spilt_length;
    _short_segment_length = short_segment_length;
    _mid_segment_length = mid_segment_length;
    _long_segment_length = long_segment_length;
    _via_unit = via_unit;
    _overflow_unit = overflow_unit;
    _layer_bias_unit = layer_bias_unit;
    _layer_switch_unit = layer_switch_unit;
  }
  ~LAComParam() = default;
  // getter
  int32_t get_topo_spilt_length() const { return _topo_spilt_length; }
  int32_t get_mid_topo_spilt_length() const { return _mid_topo_spilt_length; }
  int32_t get_long_topo_spilt_length() const { return _long_topo_spilt_length; }
  int32_t get_short_segment_length() const { return _short_segment_length; }
  int32_t get_mid_segment_length() const { return _mid_segment_length; }
  int32_t get_long_segment_length() const { return _long_segment_length; }
  double get_via_unit() const { return _via_unit; }
  double get_overflow_unit() const { return _overflow_unit; }
  double get_layer_bias_unit() const { return _layer_bias_unit; }
  double get_layer_switch_unit() const { return _layer_switch_unit; }
  // setter
  void set_topo_spilt_length(const int32_t topo_spilt_length) { _topo_spilt_length = topo_spilt_length; }
  void set_mid_topo_spilt_length(const int32_t mid_topo_spilt_length) { _mid_topo_spilt_length = mid_topo_spilt_length; }
  void set_long_topo_spilt_length(const int32_t long_topo_spilt_length) { _long_topo_spilt_length = long_topo_spilt_length; }
  void set_short_segment_length(const int32_t short_segment_length) { _short_segment_length = short_segment_length; }
  void set_mid_segment_length(const int32_t mid_segment_length) { _mid_segment_length = mid_segment_length; }
  void set_long_segment_length(const int32_t long_segment_length) { _long_segment_length = long_segment_length; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_overflow_unit(const double overflow_unit) { _overflow_unit = overflow_unit; }
  void set_layer_bias_unit(const double layer_bias_unit) { _layer_bias_unit = layer_bias_unit; }
  void set_layer_switch_unit(const double layer_switch_unit) { _layer_switch_unit = layer_switch_unit; }

 private:
  int32_t _topo_spilt_length = 0;
  int32_t _mid_topo_spilt_length = 0;
  int32_t _long_topo_spilt_length = 0;
  int32_t _short_segment_length = 0;
  int32_t _mid_segment_length = 0;
  int32_t _long_segment_length = 0;
  double _via_unit = 0;
  double _overflow_unit = 0;
  double _layer_bias_unit = 0;
  double _layer_switch_unit = 0;
};

}  // namespace irt
