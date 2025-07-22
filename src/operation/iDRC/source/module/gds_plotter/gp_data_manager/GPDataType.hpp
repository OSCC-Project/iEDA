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

enum class GPDataType
{
  kNone,
  kEnvShape,
  kResultShape,
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

struct GetGPDataTypeName
{
  std::string operator()(const GPDataType& data_type) const
  {
    std::string data_type_name;
    switch (data_type) {
      case GPDataType::kNone:
        data_type_name = "none";
        break;
      case GPDataType::kEnvShape:
        data_type_name = "env_shape";
        break;
      case GPDataType::kResultShape:
        data_type_name = "result_shape";
        break;
      case GPDataType::kAdjacentCutSpacing:
        data_type_name = "adjacent_cut_spacing";
        break;
      case GPDataType::kCornerFillSpacing:
        data_type_name = "corner_fill_spacing";
        break;
      case GPDataType::kCornerSpacing:
        data_type_name = "corner_spacing";
        break;
      case GPDataType::kCutEOLSpacing:
        data_type_name = "cut_eol_spacing";
        break;
      case GPDataType::kCutShort:
        data_type_name = "cut_short";
        break;
      case GPDataType::kDifferentLayerCutSpacing:
        data_type_name = "different_layer_cut_spacing";
        break;
      case GPDataType::kEndOfLineSpacing:
        data_type_name = "end_of_line_spacing";
        break;
      case GPDataType::kEnclosure:
        data_type_name = "enclosure";
        break;
      case GPDataType::kEnclosureEdge:
        data_type_name = "enclosure_edge";
        break;
      case GPDataType::kEnclosureParallel:
        data_type_name = "enclosure_parallel";
        break;
      case GPDataType::kFloatingPatch:
        data_type_name = "floating_patch";
        break;
      case GPDataType::kJogToJogSpacing:
        data_type_name = "jog_to_jog_spacing";
        break;
      case GPDataType::kMaximumWidth:
        data_type_name = "maximum_width";
        break;
      case GPDataType::kMaxViaStack:
        data_type_name = "max_via_stack";
        break;
      case GPDataType::kMetalShort:
        data_type_name = "metal_short";
        break;
      case GPDataType::kMinHole:
        data_type_name = "min_hole";
        break;
      case GPDataType::kMinimumArea:
        data_type_name = "minimum_area";
        break;
      case GPDataType::kMinimumCut:
        data_type_name = "minimum_cut";
        break;
      case GPDataType::kMinimumWidth:
        data_type_name = "minimum_width";
        break;
      case GPDataType::kMinStep:
        data_type_name = "min_step";
        break;
      case GPDataType::kNonsufficientMetalOverlap:
        data_type_name = "nonsufficient_metal_overlap";
        break;
      case GPDataType::kNotchSpacing:
        data_type_name = "notch_spacing";
        break;
      case GPDataType::kOffGridOrWrongWay:
        data_type_name = "off_grid_or_wrong_way";
        break;
      case GPDataType::kOutOfDie:
        data_type_name = "out_of_die";
        break;
      case GPDataType::kParallelRunLengthSpacing:
        data_type_name = "parallel_run_length_spacing";
        break;
      case GPDataType::kSameLayerCutSpacing:
        data_type_name = "same_layer_cut_spacing";
        break;
      default:
        DRCLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return data_type_name;
  }
};

}  // namespace idrc
