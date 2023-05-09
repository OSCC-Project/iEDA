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
#include "py_ito.h"
#include <tool_manager.h>

#include "ToApi.hpp"

namespace python_interface {
bool toAutoRun(const std::string& config)
{
  bool run_ok = iplf::tmInst->autoRunTO(config);
  return run_ok;
}

bool toRunDrv(const std::string& config)
{
  bool run_ok = iplf::tmInst->RunTODrv(config);
  return run_ok;
}

bool toRunHold(const std::string& config)
{
  bool run_ok = iplf::tmInst->RunTOHold(config);
  return run_ok;
}

bool toRunSetup(const std::string& config)
{
  bool run_ok = iplf::tmInst->RunTOSetup(config);
  return run_ok;
}
}  // namespace python_interface