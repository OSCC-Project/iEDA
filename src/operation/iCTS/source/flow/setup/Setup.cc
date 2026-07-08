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
 * @file Setup.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS setup entry facade implementation.
 */

#include "setup/Setup.hh"

#include <glog/logging.h>

#include <filesystem>
#include <ostream>
#include <string>
#include <system_error>
#include <utility>

#include "Log.hh"
#include "config/Config.hh"
#include "idm.h"
#include "io/Wrapper.hh"
#include "logger/Schema.hh"
#include "time/Time.hh"

namespace icts {
namespace {

auto ensureDirectory(const std::filesystem::path& dir) -> void
{
  std::error_code error_code;
  std::filesystem::create_directories(dir, error_code);
  LOG_FATAL_IF(error_code) << "failed to create CTS runtime directory " << dir.string() << ": " << error_code.message();
}

}  // namespace

auto Setup::initializeRuntime(const SetupInput& input) -> SetupSummary
{
  LOG_FATAL_IF(input.config == nullptr) << "Setup: config is null.";
  LOG_FATAL_IF(input.design == nullptr) << "Setup: design is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Setup: wrapper is null.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Setup: reporter is null.";

  auto& config = *input.config;
  auto& wrapper = *input.wrapper;
  auto& reporter = *input.reporter;
  const std::string generated_on = ieda::Time::getNowWallTime();

  const bool config_loaded = config.init(input.config_file);
  const auto dir_str = input.work_dir.empty() ? config.get_work_dir() : input.work_dir;
  auto dir = std::filesystem::path(dir_str);
  ensureDirectory(dir);
  config.set_work_dir(dir.string());

  const auto log_path = (dir / "cts.log").string();
  const auto visualization_dir = dir / "visualization";
  const auto statistics_dir = dir / "statistics";
  ensureDirectory(visualization_dir);
  ensureDirectory(statistics_dir);
  config.set_log_file(log_path);
  config.set_visualization_dir(visualization_dir.string());
  config.set_statistics_dir(statistics_dir.string());

  reporter.open(config.get_log_file(), "iCTS Run",
                {
                    {"config_file", input.config_file},
                    {"work_dir", dir.string()},
                });
  LOG_INFO << "Generate the report at " << generated_on;
  if (!config_loaded) {
    reporter.emitSection("## Runtime Setup");
    reporter.emitKeyValueTable("CTS Setup Overview", {
                                                         {"status", "failed"},
                                                         {"config_file", input.config_file},
                                                         {"reason", config.get_last_error()},
                                                     });
    LOG_ERROR << "CTS setup failed for config file " << input.config_file << ": " << config.get_last_error();
    return SetupSummary{.success = false, .reason = config.get_last_error()};
  }

  auto* idb_builder = dmInst->get_idb_builder();
  LOG_FATAL_IF(idb_builder == nullptr) << "idb builder is null";
  wrapper.init(idb_builder);
  return SetupSummary{.success = true, .reason = "n/a"};
}

auto Setup::emitRuntimeSetup(const RuntimeSetupInput& input) -> void
{
  LOG_FATAL_IF(input.config == nullptr) << "Setup: runtime config is null.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Setup: runtime wrapper is null.";
  LOG_FATAL_IF(input.reporter == nullptr) << "Setup: runtime reporter is null.";

  const auto& config = *input.config;
  const auto& wrapper = *input.wrapper;
  auto& reporter = *input.reporter;

  reporter.emitSection("## Runtime Setup");
  reporter.emitKeyValueTable("Runtime Paths",
                             {
                                 {"work_dir", config.get_work_dir()},
                                 {"cts_log", config.get_log_file()},
                                 {"cts_detail_log", reporter.getDetailPath().empty() ? "unavailable" : reporter.getDetailPath().string()},
                                 {"visualization_dir", config.get_visualization_dir()},
                                 {"statistics_dir", config.get_statistics_dir()},
                             });
  reporter.emitSection("### Runtime Configuration");
  config.emitRuntimeConfigReport(reporter, "Runtime Configuration");
  reporter.emitSection("### Runtime Routing / Wire RC");
  wrapper.emitConfiguredUnitWireRcReport(reporter, config, "Runtime Routing / Wire RC");
}

}  // namespace icts
