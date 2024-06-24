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

#include "PAPin.hpp"

namespace irt {

class ConflictAccessPoint : public PlanarCoord
{
 public:
  ConflictAccessPoint() = default;
  ~ConflictAccessPoint() = default;
  // getter
  PAPin* get_pa_pin() { return _pa_pin; }
  AccessPoint* get_access_point() { return _access_point; }
  // setter
  void set_pa_pin(PAPin* pa_pin) { _pa_pin = pa_pin; }
  void set_access_point(AccessPoint* access_point) { _access_point = access_point; }
  // function
 private:
  PAPin* _pa_pin = nullptr;
  AccessPoint* _access_point = nullptr;
};

}  // namespace irt
