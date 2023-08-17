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

#include "EvalAPI.hpp"
#include "EvalType.hpp"

using namespace eval;

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
  EvalAPI& eval_api = EvalAPI::initInst();
  eval_api.initCongDataFromIDB(bin_cnt_x, bin_cnt_y);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace tcl
