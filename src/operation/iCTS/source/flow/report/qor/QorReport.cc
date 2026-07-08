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
 * @file QorReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS statistics report output implementation.
 */

#include "report/qor/QorReport.hh"

#include <glog/logging.h>

#include <filesystem>
#include <ostream>

#include "Log.hh"
#include "Qor.hh"
#include "evaluation/qor/QorEvaluation.hh"
#include "report/qor/QorFiles.hh"

namespace icts {

auto QorReport::write(SchemaWriter& reporter, const EvaluationState& evaluation_state, const std::string& statistics_dir,
                      bool emit_log_tables) -> bool
{
  if (!evaluation_state.statistics.valid) {
    LOG_WARNING << "QorReport: statistics report skipped because evaluation statistics are not ready.";
    return false;
  }

  const bool success = QorFiles::writeReports(std::filesystem::path(statistics_dir), evaluation_state.statistics);
  if (emit_log_tables) {
    QorFiles::emitLogTables(reporter, evaluation_state.statistics);
  }
  return success;
}

}  // namespace icts
