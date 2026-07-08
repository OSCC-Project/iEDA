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
 * @file CharPatternStorage.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief CharBuilder pattern-storage component. Materializes BufferingPattern
 *        records from a (TopologyDesc, buffer masters, total wirelength)
 *        tuple by computing the normalized buffer positions along the wire
 *        and assigning a fresh PatternId.
 */

#pragma once

#include <string>
#include <vector>

namespace icts {
struct PatternId;
}  // namespace icts

namespace icts::char_builder::detail {

class CharBuilderImpl;
struct TopologyDesc;

class CharPatternStorage
{
 public:
  explicit CharPatternStorage(CharBuilderImpl& impl) : _impl(impl) {}
  ~CharPatternStorage() = default;
  CharPatternStorage(const CharPatternStorage&) = delete;
  auto operator=(const CharPatternStorage&) -> CharPatternStorage& = delete;

  auto storeBufferingPattern(unsigned length_idx, const TopologyDesc& topo, const std::vector<std::string>& buf_masters,
                             double total_length_um) -> ::icts::PatternId;

 private:
  CharBuilderImpl& _impl;
};

}  // namespace icts::char_builder::detail
