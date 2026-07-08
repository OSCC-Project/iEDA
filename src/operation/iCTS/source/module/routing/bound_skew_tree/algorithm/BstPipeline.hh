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
 * @file BstPipeline.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Bound-skew tree A->B->C pipeline orchestrator. Iterates child
 *        merging in the bottom-up phase, then embeds the root and computes
 *        timing in the top-down phase.
 */

#pragma once

namespace icts::bst {
class Area;
}  // namespace icts::bst

namespace icts::bst::detail {

class BoundSkewTreeImpl;

class BstPipeline
{
 public:
  explicit BstPipeline(BoundSkewTreeImpl& impl) : _impl(impl) {}
  ~BstPipeline() = default;
  BstPipeline(const BstPipeline&) = delete;
  auto operator=(const BstPipeline&) -> BstPipeline& = delete;

  auto run() -> void;

  // Public so BinaryTopology can call merge(parent, left, right) via the pipeline
  auto merge(Area* parent, Area* left, Area* right) -> void;

 private:
  auto bottomUp() -> void;
  auto bottomUpAllPairBased() -> void;
  auto bottomUpTopoBased() -> void;
  auto processBottomUpTopology() -> void;
  auto topDown() -> void;
  auto embedTree() const -> void;
  auto updateEmbeddedNodeTiming(Area* current) const -> void;

  BoundSkewTreeImpl& _impl;
};

}  // namespace icts::bst::detail
