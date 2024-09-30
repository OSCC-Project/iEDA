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

#include "Logger.hpp"

namespace irt {

enum class DERule
{
  kNone,
  kCornerFillSpacing,
  kCutEolSpacing,
  kCutShort,
  kDifferentLayerCutSpacing,
  kEndOfLineSpacing,
  kEnclosure,
  kEnclosureEdge,
  kEnclosureParallel,
  kFloatingPatch,
  kMaxViaStack,
  kMetalShort,
  kMinHole,
  kMinimumArea,
  kMinimumCut,
  kMinimumWidth,
  kMinStep,
  kNonsufficientMetalOverlap,
  kNotchSpacing,
  kOffGridorWrongWay,
  kOutOfDie,
  kParallelRunLengthSpacing,
  kSameLayerCutSpacing
};

struct GetDERuleName
{
  std::string operator()(const DERule& de_rule) const
  {
    std::string de_rule_name;
    switch (de_rule) {
      case DERule::kNone:
        de_rule_name = "none";
        break;
      case DERule::kCornerFillSpacing:
        de_rule_name = "Corner Fill Spacing";
        break;
      case DERule::kCutEolSpacing:
        de_rule_name = "Cut EolSpacing";
        break;
      case DERule::kCutShort:
        de_rule_name = "Cut Short";
        break;
      case DERule::kDifferentLayerCutSpacing:
        de_rule_name = "Different Layer Cut Spacing";
        break;
      case DERule::kEndOfLineSpacing:
        de_rule_name = "EndOfLine Spacing";
        break;
      case DERule::kEnclosure:
        de_rule_name = "Enclosure";
        break;
      case DERule::kEnclosureEdge:
        de_rule_name = "EnclosureEdge";
        break;
      case DERule::kEnclosureParallel:
        de_rule_name = "Enclosure Parallel";
        break;
      case DERule::kFloatingPatch:
        de_rule_name = "Floating Patch";
        break;
      case DERule::kMaxViaStack:
        de_rule_name = "MaxViaStack";
        break;
      case DERule::kMetalShort:
        de_rule_name = "Metal Short";
        break;
      case DERule::kMinHole:
        de_rule_name = "MinHole";
        break;
      case DERule::kMinimumArea:
        de_rule_name = "Minimum Area";
        break;
      case DERule::kMinimumCut:
        de_rule_name = "Minimum Cut";
        break;
      case DERule::kMinimumWidth:
        de_rule_name = "Minimum Width";
        break;
      case DERule::kMinStep:
        de_rule_name = "MinStep";
        break;
      case DERule::kNonsufficientMetalOverlap:
        de_rule_name = "Non-sufficient Metal Overlap";
        break;
      case DERule::kNotchSpacing:
        de_rule_name = "Notch Spacing";
        break;
      case DERule::kOffGridorWrongWay:
        de_rule_name = "Off Grid or Wrong Way";
        break;
      case DERule::kOutOfDie:
        de_rule_name = "Out Of Die";
        break;
      case DERule::kParallelRunLengthSpacing:
        de_rule_name = "ParallelRunLength Spacing";
        break;
      case DERule::kSameLayerCutSpacing:
        de_rule_name = "Same Layer Cut Spacing";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return de_rule_name;
  }
};

struct GetDERule
{
  DERule operator()(const std::string& input_name) const
  {
    DERule de_rule;
    if (input_name == "Corner Fill Spacing") {
      de_rule = DERule::kCornerFillSpacing;
    } else if (input_name == "Cut EolSpacing") {
      de_rule = DERule::kCutEolSpacing;
    } else if (input_name == "Cut Short") {
      de_rule = DERule::kCutShort;
    } else if (input_name == "Different Layer Cut Spacing") {
      de_rule = DERule::kDifferentLayerCutSpacing;
    } else if (input_name == "EndOfLine Spacing") {
      de_rule = DERule::kEndOfLineSpacing;
    } else if (input_name == "Enclosure") {
      de_rule = DERule::kEnclosure;
    } else if (input_name == "EnclosureEdge") {
      de_rule = DERule::kEnclosureEdge;
    } else if (input_name == "Enclosure Parallel") {
      de_rule = DERule::kEnclosureParallel;
    } else if (input_name == "Floating Patch") {
      de_rule = DERule::kFloatingPatch;
    } else if (input_name == "MaxViaStack") {
      de_rule = DERule::kMaxViaStack;
    } else if (input_name == "Metal Short") {
      de_rule = DERule::kMetalShort;
    } else if (input_name == "MinHole") {
      de_rule = DERule::kMinHole;
    } else if (input_name == "Minimum Area") {
      de_rule = DERule::kMinimumArea;
    } else if (input_name == "Minimum Cut") {
      de_rule = DERule::kMinimumCut;
    } else if (input_name == "Minimum Width") {
      de_rule = DERule::kMinimumWidth;
    } else if (input_name == "MinStep") {
      de_rule = DERule::kMinStep;
    } else if (input_name == "Non-sufficient Metal Overlap") {
      de_rule = DERule::kNonsufficientMetalOverlap;
    } else if (input_name == "Notch Spacing") {
      de_rule = DERule::kNotchSpacing;
    } else if (input_name == "Off Grid or Wrong Way") {
      de_rule = DERule::kOffGridorWrongWay;
    } else if (input_name == "Out Of Die") {
      de_rule = DERule::kOutOfDie;
    } else if (input_name == "ParallelRunLength Spacing") {
      de_rule = DERule::kParallelRunLengthSpacing;
    } else if (input_name == "Same Layer Cut Spacing") {
      de_rule = DERule::kSameLayerCutSpacing;
    } else {
      de_rule = DERule::kNone;
    }
    return de_rule;
  }
};

}  // namespace irt
