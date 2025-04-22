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

class SAComParam
{
 public:
  SAComParam() = default;
  SAComParam(int32_t supply_reduction) { _supply_reduction = supply_reduction; }
  ~SAComParam() = default;
  // getter
  int32_t get_supply_reduction() const { return _supply_reduction; }
  // setter
  void set_supply_reduction(const int32_t supply_reduction) { _supply_reduction = supply_reduction; }

 private:
  int32_t _supply_reduction = 0;
};

}  // namespace irt
