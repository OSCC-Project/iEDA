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

#include <vector>

#include "compare/NetSelector.hh"
#include "config/CompareSpefConfig.hh"
#include "data/CompareSpefData.hh"

namespace ircx {
namespace compare_spef {

class PathPairGenerator
{
 public:
  explicit PathPairGenerator(const Config& config);

  auto generate(const Net& net) const -> std::vector<NodePair>;

 private:
  const Config& _config;
  NetSelector _net_selector;
};

}  // namespace compare_spef
}  // namespace ircx
