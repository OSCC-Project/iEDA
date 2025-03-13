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

#define NET_ID_ENVIRONMENT -1
#define NET_ID_OBS -2
#define NET_ID_PDN -3
#define NET_ID_VDD -4
#define NET_ID_VSS -5

struct GetViolationTypeName
{
  std::string operator()(const ViolationEnumType& type) const
  {
    switch (type) {
      case ViolationEnumType::kArea:
        return "Minimum Area";
      case ViolationEnumType::kAreaEnclosed:
        return "MinHole";
      case ViolationEnumType::kShort:
        return "Metal Short";
      case ViolationEnumType::kDefaultSpacing:
        return "Default Spacing";
      case ViolationEnumType::kPRLSpacing:
        return "ParallelRunLength Spacing";
      case ViolationEnumType::kJogToJog:
        return "JogToJog Spacing";
      case ViolationEnumType::kEOL:
        return "EndOfLine Spacing";
      case ViolationEnumType::kWidth:
        return "Wire Width";
      case ViolationEnumType::kMinStep:
        return "MinStep";
      case ViolationEnumType::kNotch:
        return "Notch Spacing";
      case ViolationEnumType::kCornerFill:
        return "Corner Fill Spacing";
      default:
        return "None";
    }
  }
};

struct GetViolationType
{
  ViolationEnumType operator()(const std::string& type_name) const
  {
    if (type_name == "Minimum Area") {
      return ViolationEnumType::kArea;
    } else if (type_name == "MinHole") {
      return ViolationEnumType::kAreaEnclosed;
    } else if (type_name == "Metal Short") {
      return ViolationEnumType::kShort;
    } else if (type_name == "Default Spacing") {
      return ViolationEnumType::kDefaultSpacing;
    } else if (type_name == "ParallelRunLength Spacing") {
      return ViolationEnumType::kPRLSpacing;
    } else if (type_name == "JogToJog Spacing") {
      return ViolationEnumType::kJogToJog;
    } else if (type_name == "EndOfLine Spacing") {
      return ViolationEnumType::kEOL;
    } else if (type_name == "Wire Width") {
      return ViolationEnumType::kWidth;
    } else if (type_name == "MinStep") {
      return ViolationEnumType::kMinStep;
    } else if (type_name == "Notch Spacing") {
      return ViolationEnumType::kNotch;
    } else if (type_name == "Corner Fill Spacing") {
      return ViolationEnumType::kCornerFill;
    } else {
      return ViolationEnumType::kNone;
    }
  }
};

}  // namespace idrc