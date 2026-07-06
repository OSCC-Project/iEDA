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

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "data/CompareSpefData.hh"

namespace ircx {
namespace compare_spef {

class ResistanceSolver
{
 public:
  auto equivalentResistance(const Net& net, const std::string& from_node, const std::string& to_node) const -> std::optional<double>;
  auto equivalentResistances(const Net& net, const std::vector<NodePair>& pairs) const -> std::vector<std::optional<double>>;
  auto equivalentResistances(const Net& net, const std::vector<NodePair>& pairs, const std::vector<std::size_t>& pair_indices) const
      -> std::vector<std::optional<double>>;
};

}  // namespace compare_spef
}  // namespace ircx
