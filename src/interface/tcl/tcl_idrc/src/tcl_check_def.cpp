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
#include <set>

#include "DRCInterface.hpp"
#include "flow_config.h"
#include "tcl_drc.h"
#include "tcl_util.h"
#include "usage/usage.hh"

namespace tcl {

TclCheckDef::TclCheckDef(const char* cmd_name) : TclCmd(cmd_name)
{
  // std::string golden_directory_path;     // optional
  _config_list.push_back(std::make_pair("-golden_directory_path", ValueType::kString));

  TclUtil::addOption(this, _config_list);
}

unsigned TclCheckDef::exec()
{
  if (!check()) {
    return 0;
  }
  std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);
  DRCI.checkDef(config_map);
  return 1;
}

}  // namespace tcl
