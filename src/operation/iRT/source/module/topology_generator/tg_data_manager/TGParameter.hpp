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

class TGParameter
{
 public:
  TGParameter()
  {
    _topo_spilt_length = 10;
    _congestion_unit = 1;
  }
  ~TGParameter() = default;
  // getter
  double get_topo_spilt_length() const { return _topo_spilt_length; }
  double get_congestion_unit() const { return _congestion_unit; }
  // setter
  void set_topo_spilt_length(const double topo_spilt_length) { _topo_spilt_length = topo_spilt_length; }
  void set_congestion_unit(const double congestion_unit) { _congestion_unit = congestion_unit; }

 private:
  double _topo_spilt_length = 0;
  double _congestion_unit = 0;
};

}  // namespace irt
