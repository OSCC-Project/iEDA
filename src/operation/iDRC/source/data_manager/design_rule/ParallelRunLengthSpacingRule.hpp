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

#include "DRCHeader.hpp"
#include "GridMap.hpp"

namespace idrc {

enum class LayerSpacingType : uint8_t
{
  kNone,
  kSpacingDefault,
  kSpacingRange,
  //!-----tbd-------------
  kSpacingRangeLenThreshold,
  kSpacingEOL,
  kMax
};
struct LayerSpacing
{
  LayerSpacingType spacing_type;
  int32_t min_spacing;
  int32_t min_width;
  int32_t max_width;
};

class ParallelRunLengthSpacingRule
{
 public:
  ParallelRunLengthSpacingRule() = default;
  ~ParallelRunLengthSpacingRule() = default;
  int32_t getMaxSpacing() { return width_parallel_length_map.back().back(); }
  int32_t getSpacing(int32_t width, int32_t parallel_length)
  {
    int32_t width_idx = static_cast<int32_t>(width_list.size()) - 1;
    for (int32_t i = 1; i <= width_idx; i++) {
      if (width <= width_list[i]) {
        width_idx = i - 1;
        break;
      }
    }
    int32_t parallel_length_idx = static_cast<int32_t>(parallel_length_list.size()) - 1;
    for (int32_t i = 1; i <= parallel_length_idx; i++) {
      if (parallel_length <= parallel_length_list[i]) {
        parallel_length_idx = i - 1;
        break;
      }
    }
    return width_parallel_length_map[width_idx][parallel_length_idx];
  }
  std::vector<int32_t> width_list;
  std::vector<int32_t> parallel_length_list;
  GridMap<int32_t> width_parallel_length_map;
  bool has_spacing_table = false;

  std::vector<LayerSpacing> spacing_list;
  bool has_spacing_list = false;

  int32_t getSpacingWithWidth(int32_t width)
  {
    int32_t spacing = -1;
    int32_t default_spacing = -1;
    for (auto& layerSpacing : spacing_list) {
      if (layerSpacing.spacing_type == LayerSpacingType::kSpacingRange) {
        if (layerSpacing.min_width <= width && width <= layerSpacing.max_width) {
          spacing = layerSpacing.min_spacing;
        }
      } else {
        default_spacing = layerSpacing.min_spacing;
      }
    }
    return spacing == -1 ? default_spacing : spacing;
  }

  int32_t getSpacingMaxWidth()
  {
    int32_t spacing = -1;
    for (auto& layerSpacing : spacing_list) {
      spacing = std::max(spacing, layerSpacing.min_spacing);
    }
    return spacing;
  }
};

}  // namespace idrc
