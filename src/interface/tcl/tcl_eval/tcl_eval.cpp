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
 * @File Name: tcl_eval.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "tcl_eval.h"

#include <iostream>

#include "wirelength_io.h"
#include "density_io.h"
#include "init_egr.h"

using namespace ieval;

namespace tcl {

CmdEvalInit::CmdEvalInit(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* bin_cnt_x = new TclIntOption("-bin_cnt_x", 0);
  auto* bin_cnt_y = new TclIntOption("-bin_cnt_y", 0);
  addOption(bin_cnt_x);
  addOption(bin_cnt_y);
}

unsigned CmdEvalInit::check()
{
  TclOption* bin_cnt_x = getOptionOrArg("-bin_cnt_x");
  TclOption* bin_cnt_y = getOptionOrArg("-bin_cnt_y");
  LOG_FATAL_IF(!bin_cnt_x);
  LOG_FATAL_IF(!bin_cnt_y);
  return 1;
}

unsigned CmdEvalInit::exec()
{
  if (!check()) {
    return 0;
  }

  auto* opt_bin_cnt_x = getOptionOrArg("-bin_cnt_x");
  auto* opt_bin_cnt_y = getOptionOrArg("-bin_cnt_y");
  auto bin_cnt_x = opt_bin_cnt_x->getIntVal();
  auto bin_cnt_y = opt_bin_cnt_y->getIntVal();

  std::cout << "bin_cnt_x=" << bin_cnt_x << " bin_cnt_y=" << bin_cnt_y << std::endl;

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdEvalTimingRun::CmdEvalTimingRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  auto* output_path_option = new TclStringOption("-eval_output_path", 1, nullptr);
  auto* route_type_option = new TclStringOption("-routing_type", 1, "HPWL");
  addOption(output_path_option);
  addOption(path_option);
  addOption(route_type_option);
}

unsigned CmdEvalTimingRun::check()
{
  const TclOption* path_option = getOptionOrArg(TCL_OUTPUT_PATH);
  const TclOption* output_path_option = getOptionOrArg("-eval_output_path");
  const TclOption* route_type_option = getOptionOrArg("-routing_type");
  LOG_FATAL_IF(!path_option);
  LOG_FATAL_IF(!output_path_option);
  LOG_INFO_IF(!route_type_option);
  return 1;
}

unsigned CmdEvalTimingRun::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* path_option = getOptionOrArg(TCL_OUTPUT_PATH);
  TclOption* output_path_option = getOptionOrArg("-eval_output_path");
  TclOption* route_type_option = getOptionOrArg("-routing_type");
  const auto path = path_option->getStringVal() != nullptr ? path_option->getStringVal() : "";
  const auto output_path = output_path_option->getStringVal() != nullptr ? output_path_option->getStringVal() : "";
  const auto route_type = route_type_option->getStringVal() != nullptr ? route_type_option->getStringVal() : "HPWL";
  std::cout << "[Evaluate Timing] path = " << path << std::endl;
  std::cout << "[Evaluate Timing] output_path = " << output_path << std::endl;
  std::cout << "[Evaluate Timing] route_type = " << route_type << std::endl;

  EvalTiming::runTimingEval(route_type);
  EvalTiming::setOutputPath(output_path);
  EvalTiming::printTimingResult();
  std::cout << path << std::endl;

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdEvalWirelengthRun::CmdEvalWirelengthRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  auto* output_path_option = new TclStringOption("-eval_output_path", 1, nullptr);
  addOption(option);
  addOption(output_path_option);
}

unsigned CmdEvalWirelengthRun::check()
{
  const TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  const TclOption* output_path_option = getOptionOrArg("-eval_output_path");
  LOG_FATAL_IF(!option);
  LOG_FATAL_IF(!output_path_option);
  return 1;
}

unsigned CmdEvalWirelengthRun::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  TclOption* output_path_option = getOptionOrArg("-eval_output_path");
  const auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";
  const auto output_path = output_path_option->getStringVal() != nullptr ? output_path_option->getStringVal() : "";
  std::cout << "[Evaluate Wirelength] path = " << path << std::endl;
  std::cout << "[Evaluate Wirelength] output_path = " << output_path << std::endl;

  EvalWirelength::setOutputPath(output_path);
  return EvalWirelength::runWirelengthEvalAndOutput();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdEvalDensityRun::CmdEvalDensityRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  auto* output_path_option = new TclStringOption("-eval_output_path", 1, nullptr);
  auto* grid_size = new TclIntOption("-grid_size", 1, 200);
  auto* stage = new TclStringOption("-stage", 1, "place");
  addOption(option);
  addOption(output_path_option);
  addOption(grid_size);
  addOption(stage);
}

unsigned CmdEvalDensityRun::check()
{
  const TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  const TclOption* output_path_option = getOptionOrArg("-eval_output_path");
  const TclOption* grid_size_option = getOptionOrArg("-grid_size");
  const TclOption* stage_option = getOptionOrArg("-stage");
  LOG_FATAL_IF(!option);
  LOG_FATAL_IF(!output_path_option);
  LOG_FATAL_IF(!grid_size_option);
  LOG_FATAL_IF(!stage_option);
  return 1;
}

unsigned CmdEvalDensityRun::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  TclOption* output_path_option = getOptionOrArg("-eval_output_path");
  TclOption* grid_size_option = getOptionOrArg("-grid_size");
  TclOption* stage_option = getOptionOrArg("-stage");

  const auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";
  const auto output_path = output_path_option->getStringVal() != nullptr ? output_path_option->getStringVal() : "";
  const auto grid_size = grid_size_option->getIntVal();
  const auto stage = stage_option->getStringVal() != nullptr ? stage_option->getStringVal() : "place";

  std::cout << "[Evaluate Density] path = " << path << std::endl;
  std::cout << "[Evaluate Density] output_path = " << output_path << std::endl;
  std::cout << "[Evaluate Density] grid_size = " << grid_size << std::endl;
  std::cout << "[Evaluate Density] stage = " << stage << std::endl;

  EvalDensity::setOutputPath(output_path);
  return EvalDensity::runDensityEvalAndOutput(grid_size, stage);
}
}  // namespace tcl


namespace tcl {

CmdEvalEgrConfig::CmdEvalEgrConfig(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption("-bottom_routing_layer", 1, nullptr));
  addOption(new TclStringOption("-top_routing_layer", 1, nullptr));
  addOption(new TclIntOption("-enable_timing", 1, 0));
  addOption(new TclStringOption("-temp_directory_path", 1, nullptr));
  addOption(new TclIntOption("-thread_number", 1, 0));
  addOption(new TclIntOption("-output_inter_result", 1, 0));
  addOption(new TclStringOption("-stage", 1, nullptr));
  addOption(new TclStringOption("-resolve_congestion", 1, nullptr));
}

unsigned CmdEvalEgrConfig::check() { return 1; }

unsigned CmdEvalEgrConfig::exec()
{
  auto* opt_bottom = getOptionOrArg("-bottom_routing_layer");
  auto* opt_top = getOptionOrArg("-top_routing_layer");
  auto* opt_timing = getOptionOrArg("-enable_timing");
  auto* opt_temp = getOptionOrArg("-temp_directory_path");
  auto* opt_threads = getOptionOrArg("-thread_number");
  auto* opt_output_inter = getOptionOrArg("-output_inter_result");
  auto* opt_stage = getOptionOrArg("-stage");
  auto* opt_resolve = getOptionOrArg("-resolve_congestion");
  auto* inst = ieval::InitEGR::getInst();
  if (opt_bottom && opt_bottom->getStringVal()) inst->setBottomRoutingLayer(opt_bottom->getStringVal());
  if (opt_top && opt_top->getStringVal()) inst->setTopRoutingLayer(opt_top->getStringVal());
  if (opt_timing) inst->setEnableTimingOverride(opt_timing->getIntVal() != 0);
  if (opt_temp && opt_temp->getStringVal()) inst->setEGRDirPath(opt_temp->getStringVal());
  if (opt_threads) inst->setThreadNumberOverride(opt_threads->getIntVal());
  if (opt_output_inter) inst->setOutputInterResultOverride(opt_output_inter->getIntVal());
  if (opt_stage && opt_stage->getStringVal()) inst->setStage(opt_stage->getStringVal());
  if (opt_resolve && opt_resolve->getStringVal()) inst->setResolveCongestion(opt_resolve->getStringVal());
  return 1;
}

}

