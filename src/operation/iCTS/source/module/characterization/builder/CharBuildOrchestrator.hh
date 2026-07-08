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
 * @file CharBuildOrchestrator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder sweep-build orchestrator component. Drives the
 *        wirelength sweep loop: per wirelength delegates topology enumeration
 *        to CharPatternEnumerator, aggregates running statistics, and emits the
 *        characterization-progress and characterization-results report tables.
 */

#pragma once

namespace icts::char_builder::detail {

class CharBuilderImpl;

class CharBuildOrchestrator
{
 public:
  explicit CharBuildOrchestrator(CharBuilderImpl& impl) : _impl(impl) {}
  ~CharBuildOrchestrator() = default;
  CharBuildOrchestrator(const CharBuildOrchestrator&) = delete;
  auto operator=(const CharBuildOrchestrator&) -> CharBuildOrchestrator& = delete;

  auto build() -> void;

 private:
  CharBuilderImpl& _impl;
};

}  // namespace icts::char_builder::detail
