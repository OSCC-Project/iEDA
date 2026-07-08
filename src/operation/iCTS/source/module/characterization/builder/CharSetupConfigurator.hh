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
 * @file CharSetupConfigurator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder sweep-grid configuration component. Implements the
 *        public `init(input, config)` entry: validates dependencies, discovers
 *        usable buffer cells, resolves slew / cap / wirelength-unit /
 *        routing-layer limits from explicit config or liberty fall-backs, and
 *        populates the sweep grid stored on CharBuilderImpl.
 */

#pragma once

#include "characterization/builder/CharBuilder.hh"

namespace icts::char_builder::detail {

class CharBuilderImpl;

class CharSetupConfigurator
{
 public:
  explicit CharSetupConfigurator(CharBuilderImpl& impl) : _impl(impl) {}
  ~CharSetupConfigurator() = default;
  CharSetupConfigurator(const CharSetupConfigurator&) = delete;
  auto operator=(const CharSetupConfigurator&) -> CharSetupConfigurator& = delete;

  auto init(const ::icts::CharBuilder::Input& input, const ::icts::CharBuilder::Config& config) -> void;

 private:
  CharBuilderImpl& _impl;
};

}  // namespace icts::char_builder::detail
