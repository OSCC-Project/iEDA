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
 * @file Embedding.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree selected-pattern embedding contracts.
 */

#pragma once

#include <string>

#include "Point.hh"
#include "synthesis/htree/HTree.hh"

namespace icts {
class Design;
class Wrapper;
}  // namespace icts

namespace icts::htree {

struct BufferPatternLibrary;
struct DiagnosticBuild;

auto InterpolateManhattanPoint(const Point<int>& source, const Point<int>& sink, double normalized_position) -> Point<int>;
auto ValidateRootDriverSizing(icts::Design& design, Wrapper& wrapper, const HTree::Build& result, const std::string& cell_master) -> bool;
auto ApplyRootDriverSizing(icts::Design& design, Wrapper& wrapper, htree::DiagnosticBuild& result, const std::string& cell_master) -> bool;
auto BuildEmbedding(icts::Design& design, Wrapper& wrapper, htree::DiagnosticBuild& result,
                    const BufferPatternLibrary& segment_pattern_library) -> void;

}  // namespace icts::htree
