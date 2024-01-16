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
#include "py_icts.h"

#include <tool_manager.h>

#include <CTSAPI.hh>
namespace python_interface {
bool ctsAutoRun(const std::string& cts_config, const std::string& cts_work_dir)
{
  bool cts_run_ok = iplf::tmInst->autoRunCTS(cts_config, cts_work_dir);
  return cts_run_ok;
}

void ctsReport(const std::string& path) { CTSAPIInst.report(path); }

}  // namespace python_interface