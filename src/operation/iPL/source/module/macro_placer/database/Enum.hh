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

#include <limits>
#include <string>

namespace ipl::imp {

enum class InstType : uint8_t
{
  kStd_cell,
  kMacro,
  kIo_cell,
  kNew_macro,
  kFlip_flop
};

enum class Orient : uint8_t
{
  kNone,
  kN,
  kE,
  kS,
  kW,
  kFN,
  kFE,
  kFS,
  kFW
};

enum class PartitionType : uint8_t
{
  kHmetis,
  kMetis
};

enum class SolutionTYPE : uint8_t
{
  kBStar_tree,
  kSequence_pair
};

enum class MoveType : uint8_t
{
  kSwap,
  kRotate,
  kMove
};

#define UNDEFINED -1
#define INFITY UINT32_MAX
#define DEFAULT 0

}  // namespace ipl::imp