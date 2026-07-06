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

#include "CapTable.hpp"
#include "Types.hh"

namespace ircx {

class CapTableQuery
{
 public:
  CapTableQuery(const parser::CapTable& cap_table, const Str& layer_name)
      : cap_table_(cap_table), layer_name_(layer_name)
  {
  }

  auto nearCap(const Str& below_layer,
               const Str& above_layer,
               Micron spacing) const -> parser::CapacitanceResult;
  auto farthestCap(const Str& below_layer,
                   const Str& above_layer) const -> parser::CapacitanceResult;

 private:
  const parser::CapTable& cap_table_;
  const Str& layer_name_;
};

}  // namespace ircx
