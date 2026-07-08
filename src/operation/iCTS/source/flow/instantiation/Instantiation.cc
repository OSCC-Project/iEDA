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
 * @file Instantiation.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS instantiation entry facade implementation.
 */

#include "instantiation/Instantiation.hh"

#include <glog/logging.h>

#include <ostream>

#include "Log.hh"
#include "instantiation/idb_conversion/IdbConversion.hh"

namespace icts {

auto Instantiation::run(const InstantiationInput& input) -> InstantiationSummary
{
  LOG_FATAL_IF(input.design == nullptr) << "Instantiation requires a design.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Instantiation requires a wrapper.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Instantiation requires a reporter.";
  const auto idb_summary = IdbConversion::run(IdbConversionInput{
      .design = input.design,
      .wrapper = input.wrapper,
      .reporter = input.reporter,
  });
  return InstantiationSummary{
      .attempted = idb_summary.attempted,
      .design_ready = idb_summary.design_ready,
      .success = idb_summary.success,
      .design_conversion_done = idb_summary.attempted,
      .idb_conversion_done = idb_summary.success,
      .clock_count = idb_summary.clock_count,
      .failure_reason = idb_summary.failure_reason,
  };
}

}  // namespace icts
