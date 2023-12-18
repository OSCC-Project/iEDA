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
#include "tcl_contest.h"

#include "contest_flow.h"

namespace tcl {

CmdRunContestPreprocess::CmdRunContestPreprocess(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* gcell_x = new TclDoubleOption("-gcell_x", 1, 1);
  auto* gcell_y = new TclDoubleOption("-gcell_y", 1, 1);
  auto* guide_output = new TclStringOption("-guide_output", 1, nullptr);
  addOption(gcell_x);
  addOption(gcell_y);
  addOption(guide_output);
}

unsigned CmdRunContestPreprocess::check()
{
  TclOption* gcell_x = getOptionOrArg("-gcell_x");
  TclOption* gcell_y = getOptionOrArg("-gcell_y");
  TclOption* guide_output = getOptionOrArg("-guide_output");
  LOG_FATAL_IF(!gcell_x);
  LOG_FATAL_IF(!gcell_y);
  LOG_FATAL_IF(!guide_output);
  return 1;
}

unsigned CmdRunContestPreprocess::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option_gcell_x = getOptionOrArg("-gcell_x");
  double gcell_x = option_gcell_x->getDoubleVal();

  TclOption* option_gcell_y = getOptionOrArg("-gcell_y");
  double gcell_y = option_gcell_y->getDoubleVal();

  TclOption* option_guide_output = getOptionOrArg("-guide_output");
  std::string guide_output = option_guide_output->getStringVal();

  ieda_contest::ContestFlow contest_flow;
  contest_flow.run_preprocess(gcell_x, gcell_y, guide_output);

  return 1;
}

CmdRunContest::CmdRunContest(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* guide_input = new TclStringOption("-guide_input", 1, nullptr);
  auto* guide_output = new TclStringOption("-guide_output", 1, nullptr);
  addOption(guide_input);
  addOption(guide_output);
}

unsigned CmdRunContest::check()
{
  TclOption* guide_input = getOptionOrArg("-guide_input");
  TclOption* guide_output = getOptionOrArg("-guide_output");
  LOG_FATAL_IF(!guide_input);
  LOG_FATAL_IF(!guide_output);
  return 1;
}

unsigned CmdRunContest::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option_guide_input = getOptionOrArg("-guide_input");
  std::string guide_input = option_guide_input->getStringVal();

  TclOption* option_guide_output = getOptionOrArg("-guide_output");
  std::string guide_output = option_guide_output->getStringVal();

  ieda_contest::ContestFlow contest_flow;
  contest_flow.run_flow(guide_input, guide_output);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdRunContestEvaluation::CmdRunContestEvaluation(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* guide = new TclStringOption("-guide", 1, nullptr);
  addOption(guide);
}

unsigned CmdRunContestEvaluation::check()
{
  TclOption* guide = getOptionOrArg("-guide");
  LOG_FATAL_IF(!guide);
  return 1;
}

unsigned CmdRunContestEvaluation::exec()
{
  if (!check()) {
    return 0;
  }

  std::string guide_str = "";

  TclOption* option_guide = getOptionOrArg("-guide");
  if (option_guide != nullptr) {
    guide_str = option_guide->getStringVal();
  }

  ieda_contest::ContestFlow contest_flow;
  contest_flow.run_evaluation(guide_str);

  return 1;
}

}  // namespace tcl
