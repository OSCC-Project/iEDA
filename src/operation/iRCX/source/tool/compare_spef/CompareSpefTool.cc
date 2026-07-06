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
#include "CompareSpefTool.hh"

#include "compare/Comparator.hh"
#include "log/Log.hh"
#include "reader/SpefReader.hh"
#include "report/ReportWriter.hh"

namespace ircx {

auto CompareSpefTool::run(compare_spef::Config config) -> bool
{
  const compare_spef::NetConfigReader net_config_reader;
  const compare_spef::ConfigValidator config_validator;
  if (!net_config_reader.read(config) || !config_validator.validate(config)) {
    return false;
  }

  compare_spef::Data test;
  compare_spef::Data reference;
  const compare_spef::SpefReader spef_reader;
  if (!spef_reader.read(config.test_file, test)) {
    LOG_ERROR << "compare_spef failed: read test SPEF failed: " << config.test_file;
    return false;
  }
  if (!spef_reader.read(config.reference_file, reference)) {
    LOG_ERROR << "compare_spef failed: read reference SPEF failed: " << config.reference_file;
    return false;
  }

  compare_spef::Comparator comparator(config);
  const auto result = comparator.compare(test, reference);
  const compare_spef::ReportWriter report_writer(config);
  if (!report_writer.write(result)) {
    return false;
  }

  LOG_INFO << "compare_spef wrote reports to " << config.output_dir;
  return true;
}

}  // namespace ircx
