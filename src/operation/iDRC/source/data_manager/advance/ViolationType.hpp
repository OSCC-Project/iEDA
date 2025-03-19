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

namespace idrc {

enum class ViolationType
{
  kNone,
  kAdjacentCutSpacing,
  kCornerFillSpacing,
  kCutEOLSpacing,
  kCutShort,
  kDifferentLayerCutSpacing,
  kEndOfLineSpacing,
  kEnclosure,
  kEnclosureEdge,
  kEnclosureParallel,
  kFloatingPatch,
  kJogToJogSpacing,
  kMaximumWidth,
  kMaxViaStack,
  kMetalShort,
  kMinHole,
  kMinimumArea,
  kMinimumCut,
  kMinimumWidth,
  kMinStep,
  kNonsufficientMetalOverlap,
  kNotchSpacing,
  kOffGridOrWrongWay,
  kOutOfDie,
  kParallelRunLengthSpacing,
  kSameLayerCutSpacing
};

struct GetViolationTypeName
{
  std::string operator()(const ViolationType& violation_rule) const
  {
    std::string violation_rule_name;
    switch (violation_rule) {
      case ViolationType::kNone:
        violation_rule_name = "none";
        break;
      case ViolationType::kAdjacentCutSpacing:
        violation_rule_name = "adjacent_cut_spacing";
        break;
      case ViolationType::kCornerFillSpacing:
        violation_rule_name = "corner_fill_spacing";
        break;
      case ViolationType::kCutEOLSpacing:
        violation_rule_name = "cut_eol_spacing";
        break;
      case ViolationType::kCutShort:
        violation_rule_name = "cut_short";
        break;
      case ViolationType::kDifferentLayerCutSpacing:
        violation_rule_name = "different_layer_cut_spacing";
        break;
      case ViolationType::kEndOfLineSpacing:
        violation_rule_name = "end_of_line_spacing";
        break;
      case ViolationType::kEnclosure:
        violation_rule_name = "enclosure";
        break;
      case ViolationType::kEnclosureEdge:
        violation_rule_name = "enclosure_edge";
        break;
      case ViolationType::kEnclosureParallel:
        violation_rule_name = "enclosure_parallel";
        break;
      case ViolationType::kFloatingPatch:
        violation_rule_name = "floating_patch";
        break;
      case ViolationType::kJogToJogSpacing:
        violation_rule_name = "jog_to_jog_spacing";
        break;
      case ViolationType::kMaximumWidth:
        violation_rule_name = "maximum_width";
        break;
      case ViolationType::kMaxViaStack:
        violation_rule_name = "max_via_stack";
        break;
      case ViolationType::kMetalShort:
        violation_rule_name = "metal_short";
        break;
      case ViolationType::kMinHole:
        violation_rule_name = "min_hole";
        break;
      case ViolationType::kMinimumArea:
        violation_rule_name = "minimum_area";
        break;
      case ViolationType::kMinimumCut:
        violation_rule_name = "minimum_cut";
        break;
      case ViolationType::kMinimumWidth:
        violation_rule_name = "minimum_width";
        break;
      case ViolationType::kMinStep:
        violation_rule_name = "min_step";
        break;
      case ViolationType::kNonsufficientMetalOverlap:
        violation_rule_name = "nonsufficient_metal_overlap";
        break;
      case ViolationType::kNotchSpacing:
        violation_rule_name = "notch_spacing";
        break;
      case ViolationType::kOffGridOrWrongWay:
        violation_rule_name = "off_grid_or_wrong_way";
        break;
      case ViolationType::kOutOfDie:
        violation_rule_name = "out_of_die";
        break;
      case ViolationType::kParallelRunLengthSpacing:
        violation_rule_name = "parallel_run_length_spacing";
        break;
      case ViolationType::kSameLayerCutSpacing:
        violation_rule_name = "same_layer_cut_spacing";
        break;
      default:
        DRCLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return violation_rule_name;
  }
};

struct GetViolationTypeByName
{
  ViolationType operator()(const std::string& violation_rule_name) const
  {
    ViolationType violation_rule;
    if (violation_rule_name == "adjacent_cut_spacing") {
      violation_rule = ViolationType::kAdjacentCutSpacing;
    } else if (violation_rule_name == "corner_fill_spacing") {
      violation_rule = ViolationType::kCornerFillSpacing;
    } else if (violation_rule_name == "corner_fill_spacing") {
      violation_rule = ViolationType::kCornerFillSpacing;
    } else if (violation_rule_name == "cut_eol_spacing") {
      violation_rule = ViolationType::kCutEOLSpacing;
    } else if (violation_rule_name == "cut_short") {
      violation_rule = ViolationType::kCutShort;
    } else if (violation_rule_name == "different_layer_cut_spacing") {
      violation_rule = ViolationType::kDifferentLayerCutSpacing;
    } else if (violation_rule_name == "end_of_line_spacing") {
      violation_rule = ViolationType::kEndOfLineSpacing;
    } else if (violation_rule_name == "enclosure") {
      violation_rule = ViolationType::kEnclosure;
    } else if (violation_rule_name == "enclosure_edge") {
      violation_rule = ViolationType::kEnclosureEdge;
    } else if (violation_rule_name == "enclosure_parallel") {
      violation_rule = ViolationType::kEnclosureParallel;
    } else if (violation_rule_name == "floating_patch") {
      violation_rule = ViolationType::kFloatingPatch;
    } else if (violation_rule_name == "jog_to_jog_spacing") {
      violation_rule = ViolationType::kJogToJogSpacing;
    } else if (violation_rule_name == "maximum_width") {
      violation_rule = ViolationType::kMaximumWidth;
    } else if (violation_rule_name == "max_via_stack") {
      violation_rule = ViolationType::kMaxViaStack;
    } else if (violation_rule_name == "metal_short") {
      violation_rule = ViolationType::kMetalShort;
    } else if (violation_rule_name == "min_hole") {
      violation_rule = ViolationType::kMinHole;
    } else if (violation_rule_name == "minimum_area") {
      violation_rule = ViolationType::kMinimumArea;
    } else if (violation_rule_name == "minimum_cut") {
      violation_rule = ViolationType::kMinimumCut;
    } else if (violation_rule_name == "minimum_width") {
      violation_rule = ViolationType::kMinimumWidth;
    } else if (violation_rule_name == "min_step") {
      violation_rule = ViolationType::kMinStep;
    } else if (violation_rule_name == "nonsufficient_metal_overlap") {
      violation_rule = ViolationType::kNonsufficientMetalOverlap;
    } else if (violation_rule_name == "notch_spacing") {
      violation_rule = ViolationType::kNotchSpacing;
    } else if (violation_rule_name == "off_grid_or_wrong_way") {
      violation_rule = ViolationType::kOffGridOrWrongWay;
    } else if (violation_rule_name == "out_of_die") {
      violation_rule = ViolationType::kOutOfDie;
    } else if (violation_rule_name == "parallel_run_length_spacing") {
      violation_rule = ViolationType::kParallelRunLengthSpacing;
    } else if (violation_rule_name == "same_layer_cut_spacing") {
      violation_rule = ViolationType::kSameLayerCutSpacing;
    } else {
      violation_rule = ViolationType::kNone;
    }
    return violation_rule;
  }
};

}  // namespace idrc
