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
#include "tcl_inst.h"

#include "Str.hh"
#include "idm.h"
#include "tool_manager.h"

namespace tcl {

TclFpPlaceInst::TclFpPlaceInst(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* inst_name = new TclStringOption("-inst_name", 0, nullptr);
  auto* llx = new TclIntOption("-llx", 0);
  auto* lly = new TclIntOption("-lly", 0);
  auto* orient = new TclStringOption("-orient", 0, nullptr);
  auto* cellmaster = new TclStringOption("-cellmaster", 0, nullptr);
  auto* source = new TclStringOption("-source", 0, nullptr);
  addOption(inst_name);
  addOption(llx);
  addOption(lly);
  addOption(orient);
  addOption(cellmaster);
  addOption(source);
}

unsigned TclFpPlaceInst::check()
{
  TclOption* inst_name = getOptionOrArg("-inst_name");
  TclOption* llxv = getOptionOrArg("-llx");
  TclOption* llyv = getOptionOrArg("-lly");
  TclOption* orientv = getOptionOrArg("-orient");
  TclOption* cellmasterv = getOptionOrArg("-cellmaster");
  TclOption* sourcev = getOptionOrArg("-source");
  LOG_FATAL_IF(!inst_name);
  LOG_FATAL_IF(!llxv);
  LOG_FATAL_IF(!llyv);
  LOG_FATAL_IF(!orientv);
  LOG_FATAL_IF(!cellmasterv);
  LOG_FATAL_IF(!sourcev);
  return 1;
}

unsigned TclFpPlaceInst::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* inst_name = getOptionOrArg("-inst_name");
  TclOption* llxv = getOptionOrArg("-llx");
  TclOption* llyv = getOptionOrArg("-lly");
  TclOption* orientv = getOptionOrArg("-orient");
  TclOption* cellmasterv = getOptionOrArg("-cellmaster");
  TclOption* sourcev = getOptionOrArg("-source");

  auto instance_name = inst_name->getStringVal();
  auto llx = llxv->getIntVal();
  auto lly = llyv->getIntVal();
  auto orient = orientv->getStringVal();
  auto cellmaster = cellmasterv->getStringVal();
  auto source = sourcev->getStringVal();

  string source_str = "";
  if (source != nullptr) {
    source_str = source;
  }
  string cell_master_str = "";
  if (cellmaster != nullptr) {
    cell_master_str = cellmaster;
  }
  dmInst->placeInst(instance_name, llx, lly, orient, cell_master_str, source_str);

  return 1;
}

}  // namespace tcl
