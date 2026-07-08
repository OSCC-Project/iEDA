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
 * @file SolutionReport.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief Selected H-tree solution report emission contract.
 */

#pragma once

namespace icts {
class SchemaWriter;
}  // namespace icts

namespace icts::htree {

struct CandidateBuildEvaluation;
struct DepthSummary;
struct DiagnosticBuild;

auto LogSynthesisSummary(SchemaWriter& reporter, const htree::DiagnosticBuild& result, const CandidateBuildEvaluation& selected_evaluation,
                         const DepthSummary& selected_summary) -> void;

}  // namespace icts::htree
