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
 * @file SourceTrunkStage.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-29
 * @brief Source-to-root synthesis stage formatting.
 */

#include "synthesis/topology/SourceTrunkStage.hh"

namespace icts {

auto ToString(SourceTrunkStage stage) -> const char*
{
  switch (stage) {
    case SourceTrunkStage::kSegment:
      return "top_segment";
    case SourceTrunkStage::kHTree:
      return "top_htree";
    case SourceTrunkStage::kUnknown:
      return "unknown";
  }
  return "unknown";
}

}  // namespace icts
