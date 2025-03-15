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

namespace idrc {

// !!! 请不要改变枚举值
// !!! Please don’t change the enumeration value
enum class GPTextPresentation
{
  kNone = -1,
  kLeftTop = 0,
  kCenterTop = 1,
  kRightTop = 2,
  kLeftMiddle = 4,
  kCenterMiddle = 5,
  kRightMiddle = 6,
  kLeftBottom = 8,
  kCenterBottom = 9,
  kRightBottom = 10
};

}  // namespace idrc
