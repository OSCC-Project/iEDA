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
 * @file Report.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS report entry facade.
 */

#pragma once

#include <string>

namespace icts {

class ClockLayout;
class Config;
class Design;
class SchemaWriter;
class Wrapper;
struct EvaluationState;

struct ReportInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  SchemaWriter* reporter = nullptr;
  std::string save_dir;
  bool evaluation_ready = false;
  const ClockLayout* clock_layout = nullptr;
  EvaluationState* evaluation_state = nullptr;
};

struct ReportSummary
{
  bool success = false;
  bool evaluation_ready = false;
  bool statistics_success = false;
  bool svg_success = false;
  bool gds_success = false;
};

class Report
{
 public:
  Report() = delete;

  static auto run(const ReportInput& input) -> ReportSummary;
};

}  // namespace icts
