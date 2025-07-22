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
  kCornerSpacing,
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
  std::string operator()(const ViolationType& violation_type) const
  {
    std::string violation_type_name;
    switch (violation_type) {
      case ViolationType::kNone:
        violation_type_name = "none";
        break;
      case ViolationType::kAdjacentCutSpacing:
        violation_type_name = "adjacent_cut_spacing";
        break;
      case ViolationType::kCornerFillSpacing:
        violation_type_name = "corner_fill_spacing";
        break;
      case ViolationType::kCornerSpacing:
        violation_type_name = "corner_spacing";
        break;
      case ViolationType::kCutEOLSpacing:
        violation_type_name = "cut_eol_spacing";
        break;
      case ViolationType::kCutShort:
        violation_type_name = "cut_short";
        break;
      case ViolationType::kDifferentLayerCutSpacing:
        violation_type_name = "different_layer_cut_spacing";
        break;
      case ViolationType::kEndOfLineSpacing:
        violation_type_name = "end_of_line_spacing";
        break;
      case ViolationType::kEnclosure:
        violation_type_name = "enclosure";
        break;
      case ViolationType::kEnclosureEdge:
        violation_type_name = "enclosure_edge";
        break;
      case ViolationType::kEnclosureParallel:
        violation_type_name = "enclosure_parallel";
        break;
      case ViolationType::kFloatingPatch:
        violation_type_name = "floating_patch";
        break;
      case ViolationType::kJogToJogSpacing:
        violation_type_name = "jog_to_jog_spacing";
        break;
      case ViolationType::kMaximumWidth:
        violation_type_name = "maximum_width";
        break;
      case ViolationType::kMaxViaStack:
        violation_type_name = "max_via_stack";
        break;
      case ViolationType::kMetalShort:
        violation_type_name = "metal_short";
        break;
      case ViolationType::kMinHole:
        violation_type_name = "min_hole";
        break;
      case ViolationType::kMinimumArea:
        violation_type_name = "minimum_area";
        break;
      case ViolationType::kMinimumCut:
        violation_type_name = "minimum_cut";
        break;
      case ViolationType::kMinimumWidth:
        violation_type_name = "minimum_width";
        break;
      case ViolationType::kMinStep:
        violation_type_name = "min_step";
        break;
      case ViolationType::kNonsufficientMetalOverlap:
        violation_type_name = "nonsufficient_metal_overlap";
        break;
      case ViolationType::kNotchSpacing:
        violation_type_name = "notch_spacing";
        break;
      case ViolationType::kOffGridOrWrongWay:
        violation_type_name = "off_grid_or_wrong_way";
        break;
      case ViolationType::kOutOfDie:
        violation_type_name = "out_of_die";
        break;
      case ViolationType::kParallelRunLengthSpacing:
        violation_type_name = "parallel_run_length_spacing";
        break;
      case ViolationType::kSameLayerCutSpacing:
        violation_type_name = "same_layer_cut_spacing";
        break;
      default:
        DRCLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return violation_type_name;
  }
};

struct GetViolationTypeByName
{
  ViolationType operator()(const std::string& violation_type_name) const
  {
    ViolationType violation_type;
    if (violation_type_name == "adjacent_cut_spacing") {
      violation_type = ViolationType::kAdjacentCutSpacing;
    } else if (violation_type_name == "corner_fill_spacing") {
      violation_type = ViolationType::kCornerFillSpacing;
    } else if (violation_type_name == "corner_spacing") {
      violation_type = ViolationType::kCornerSpacing;
    } else if (violation_type_name == "cut_eol_spacing") {
      violation_type = ViolationType::kCutEOLSpacing;
    } else if (violation_type_name == "cut_short") {
      violation_type = ViolationType::kCutShort;
    } else if (violation_type_name == "different_layer_cut_spacing") {
      violation_type = ViolationType::kDifferentLayerCutSpacing;
    } else if (violation_type_name == "end_of_line_spacing") {
      violation_type = ViolationType::kEndOfLineSpacing;
    } else if (violation_type_name == "enclosure") {
      violation_type = ViolationType::kEnclosure;
    } else if (violation_type_name == "enclosure_edge") {
      violation_type = ViolationType::kEnclosureEdge;
    } else if (violation_type_name == "enclosure_parallel") {
      violation_type = ViolationType::kEnclosureParallel;
    } else if (violation_type_name == "floating_patch") {
      violation_type = ViolationType::kFloatingPatch;
    } else if (violation_type_name == "jog_to_jog_spacing") {
      violation_type = ViolationType::kJogToJogSpacing;
    } else if (violation_type_name == "maximum_width") {
      violation_type = ViolationType::kMaximumWidth;
    } else if (violation_type_name == "max_via_stack") {
      violation_type = ViolationType::kMaxViaStack;
    } else if (violation_type_name == "metal_short") {
      violation_type = ViolationType::kMetalShort;
    } else if (violation_type_name == "min_hole") {
      violation_type = ViolationType::kMinHole;
    } else if (violation_type_name == "minimum_area") {
      violation_type = ViolationType::kMinimumArea;
    } else if (violation_type_name == "minimum_cut") {
      violation_type = ViolationType::kMinimumCut;
    } else if (violation_type_name == "minimum_width") {
      violation_type = ViolationType::kMinimumWidth;
    } else if (violation_type_name == "min_step") {
      violation_type = ViolationType::kMinStep;
    } else if (violation_type_name == "nonsufficient_metal_overlap") {
      violation_type = ViolationType::kNonsufficientMetalOverlap;
    } else if (violation_type_name == "notch_spacing") {
      violation_type = ViolationType::kNotchSpacing;
    } else if (violation_type_name == "off_grid_or_wrong_way") {
      violation_type = ViolationType::kOffGridOrWrongWay;
    } else if (violation_type_name == "out_of_die") {
      violation_type = ViolationType::kOutOfDie;
    } else if (violation_type_name == "parallel_run_length_spacing") {
      violation_type = ViolationType::kParallelRunLengthSpacing;
    } else if (violation_type_name == "same_layer_cut_spacing") {
      violation_type = ViolationType::kSameLayerCutSpacing;
    } else {
      violation_type = ViolationType::kNone;
    }
    return violation_type;
  }
};

}  // namespace idrc
