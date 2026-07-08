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
/**
 * @file TestDataset.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared records for iCTS test datasets.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace icts {
class Pin;
}

namespace icts_test {

struct InfoReport
{
  std::string title;
  std::string content;
};

struct GeneratedPins
{
  GeneratedPins() = default;
  ~GeneratedPins();
  GeneratedPins(GeneratedPins&&) noexcept;
  auto operator=(GeneratedPins&&) noexcept -> GeneratedPins&;
  GeneratedPins(const GeneratedPins&) = delete;
  auto operator=(const GeneratedPins&) -> GeneratedPins& = delete;

  std::vector<std::unique_ptr<icts::Pin>> storage;
  std::vector<icts::Pin*> loads;
  int width = 0;
  int height = 0;
};

struct CanvasSize
{
  int width = 0;
  int height = 0;
};

struct TopologyStats
{
  std::size_t tree_size = 0;
  std::size_t leaf_count = 0;
  std::size_t empty_leaf_count = 0;
  std::size_t min_leaf_load = 0;
  std::size_t max_leaf_load = 0;
  double avg_leaf_load = 0.0;
};

}  // namespace icts_test
