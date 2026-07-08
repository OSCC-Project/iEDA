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
 * @file ClockLayoutSynthesisTopology.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-29
 * @brief Synthesis topology metadata consumed by clock layout projection construction.
 */

#pragma once

#include <vector>

namespace icts {

class Inst;
class Net;

struct ClockLayoutInstTopology
{
  const Inst* inst = nullptr;
  int topology_level = -1;
};

struct ClockLayoutNetTopology
{
  const Net* net = nullptr;
  int topology_level = -1;
};

struct SinkDomainSynthesisTopology
{
  int selected_depth = -1;
  int topology_level_count = 0;
  std::vector<ClockLayoutInstTopology> inserted_insts;
  std::vector<ClockLayoutNetTopology> inserted_nets;
};

struct SourceToRootSynthesisTopology
{
  int selected_depth = -1;
  int topology_level_count = 0;
  std::vector<ClockLayoutInstTopology> inserted_insts;
  std::vector<ClockLayoutNetTopology> inserted_nets;
};

}  // namespace icts
