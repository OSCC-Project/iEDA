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
 * @file tcl_dump_net_shape.cpp
 * @author Yipei Xu (yipeix@163.com)
 * @brief
 * @version 0.1
 * @date 2026-05-31
 */
#include "RCXAPI.hh"
#include "tcl_ircx.h"

namespace tcl {

TclDumpNetShape::TclDumpNetShape(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned TclDumpNetShape::check()
{
  return 1;
}

unsigned TclDumpNetShape::exec()
{
  if (!check()) {
    return 0;
  }

  return RCX_API_INST.dump_net_shape() ? 1U : 0U;
}

}  // namespace tcl
