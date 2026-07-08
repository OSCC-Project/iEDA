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
 * @file Overview.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS report summary output implementation.
 */

#include "report/overview/Overview.hh"

#include <string>
#include <utility>

#include "logger/Schema.hh"

namespace icts {

auto Overview::emitReportMode(SchemaWriter& reporter, bool reused_evaluation_state, const ReportExportPaths& paths) -> void
{
  (void) paths;
  EmitKeyValueTable(reporter, "CTS Report Mode",
                    {
                        {"mode", reused_evaluation_state ? "reuse_evaluation_state" : "rebuild_evaluation_state"},
                    });
}

}  // namespace icts
