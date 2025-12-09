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
#include "DRCInterface.hpp"
#include "tcl_drc.h"
#include "tcl_util.h"

namespace tcl {

TclCmpDRC::TclCmpDRC(const char* cmd_name) : TclCmd(cmd_name)
{
  _config_list.push_back(std::make_pair("-ref1_name", ValueType::kString));
  _config_list.push_back(std::make_pair("-ref1_dir", ValueType::kString));
  _config_list.push_back(std::make_pair("-ref2_name", ValueType::kString));
  _config_list.push_back(std::make_pair("-ref2_dir", ValueType::kString));
  _config_list.push_back(std::make_pair("-ref3_name", ValueType::kString));
  _config_list.push_back(std::make_pair("-ref3_dir", ValueType::kString));

  TclUtil::addOption(this, _config_list);
}

unsigned TclCmpDRC::exec()
{
  if (!check()) {
    return 0;
  }
  std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);
  DRCI.cmpDRC(config_map);
  return 1;
}

}  // namespace tcl
