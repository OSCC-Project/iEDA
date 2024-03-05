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

namespace idrc {

enum class ViolationEnumType
{
  kNone,
  kArea,
  kAreaEnclosed,
  kShort,
  kDefaultSpacing,
  kPRLSpacing,
  kJogToJog,
  kEOL,
  kWidth,
  kMinStep,
  kNotch,
  kConnectivity,
  kCornerFill,
  kMax
};

struct GetViolationTypeName
{
  std::string operator()(const ViolationEnumType& type) const
  {
    switch (type) {
      case ViolationEnumType::kArea:
        return "Area";
      case ViolationEnumType::kAreaEnclosed:
        return "Encolsed Area";
      case ViolationEnumType::kShort:
        return "Metal Short";
      case ViolationEnumType::kDefaultSpacing:
        return "Default Spacing";
      case ViolationEnumType::kPRLSpacing:
        return "PRL Spacing";
      case ViolationEnumType::kJogToJog:
        return "JogToJog Spacing";
      case ViolationEnumType::kEOL:
        return "EndOfLine Spacing";
      case ViolationEnumType::kWidth:
        return "Wire Width";
      case ViolationEnumType::kMinStep:
        return "Min Step";
      case ViolationEnumType::kNotch:
        return "Notch Spacing";
      case ViolationEnumType::kCornerFill:
        return "Corner Fill";
      default:
        return "None";
    }
  }
};

}  // namespace idrc