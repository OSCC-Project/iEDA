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
#include "RCXAPI.hh"

#include <omp.h>

#include <utility>

#include "CompareSpefTool.hh"
#include "DumpNetShapeTool.hh"
#include "Extraction.hh"
#include "PlotSpefTool.hh"
#include "RCXConfig.hh"
#include "RCXData.hh"
#include "Report.hh"
#include "Setup.hh"
#include "StageLog.hh"
#include "log/Log.hh"

namespace ircx {

RCXAPI::RCXAPI()
{
  char config[] = "iRCX";
  char* argv[] = {config, nullptr};
  ieda::Log::init(argv);
}

auto RCXAPI::init(const std::string& config_file) -> bool
{
  return run_stage("init_rcx", [&]() {
    RCX_DATA_INST.reset();
    return Setup::initialize(config_file);
  });
}

auto RCXAPI::run() -> bool
{
  return run_stage("run_rcx", []() {
    if (!RCX_CONFIG_INST.get_initialized()) {
      LOG_ERROR << "run_rcx failed: RCX config is not initialized.";
      return false;
    }

    if (!Setup::adaptDB()) {
      return false;
    }

    omp_set_num_threads(RCX_CONFIG_INST.get_thread_num());

    return Extraction::run();
  });
}

auto RCXAPI::report() -> bool
{
  return run_stage("report_spef", []() {
    return Report::dumpSpef(); 
  });
}

auto RCXAPI::compare_spef(compare_spef::Config config) -> bool
{
  return run_stage("compare_spef", [&]() {
    return CompareSpefTool::run(std::move(config));
  }, {.profile = true});
}

auto RCXAPI::dump_net_shape() -> bool
{
  return run_stage("dump_net_shape", []() {
    if (!Setup::adaptDB()) {
      return false;
    }

    return DumpNetShapeTool::run();
  }, {.profile = true});
}

auto RCXAPI::plot_spef(plot_spef::Config config) -> bool
{
  return run_stage("plot_spef", [&]() {
    return PlotSpefTool::run(std::move(config));
  }, {.profile = true});
}

}  // namespace ircx
