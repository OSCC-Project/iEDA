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
#include <string>

namespace ipl::imp {

enum class InstType : uint8_t
{
  STD,
  MACRO,
  IO,
  NEWMACRO,
  FLIPFLOP
};

enum class Orient
{
  kNone,
  N,
  E,
  S,
  W,
  FN,
  FE,
  FS,
  FW
};

class Coordinate
{
 public:
  int32_t _x = 0;
  int32_t _y = 0;
};

}  // namespace ipl::imp