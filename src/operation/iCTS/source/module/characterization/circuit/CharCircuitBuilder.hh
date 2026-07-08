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
 * @file CharCircuitBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder per-pattern temporary CTS circuit lifetime component.
 *        Owns the create / parasitic-update / destroy lifecycle of the
 *        fast-STA characterization context that backs one buffering pattern's
 *        STA sampling pass.
 */

#pragma once

#include <string>
#include <vector>

namespace icts::char_builder::detail {

class CharBuilderImpl;
struct TopologyDesc;

class CharCircuitBuilder
{
 public:
  explicit CharCircuitBuilder(CharBuilderImpl& impl) : _impl(impl) {}
  ~CharCircuitBuilder() = default;
  CharCircuitBuilder(const CharCircuitBuilder&) = delete;
  auto operator=(const CharCircuitBuilder&) -> CharCircuitBuilder& = delete;

  auto createCharCircuit(const TopologyDesc& topo, const std::vector<std::string>& buf_masters) -> void;
  auto setCharParasitics(const TopologyDesc& topo, double load_pf) const -> void;
  auto destroyCharCircuit() -> void;

 private:
  CharBuilderImpl& _impl;
};

}  // namespace icts::char_builder::detail
