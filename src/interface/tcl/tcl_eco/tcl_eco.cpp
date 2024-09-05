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
 * @File Name: tcl_eco.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2024-08-26
 *
 */
#include "tcl_eco.h"

#include <iostream>

#include "ieco_api.h"

namespace tcl {
/**
 * type : shape, pattern
 * default type : shape
 */
CmdEcoRepairVia::CmdEcoRepairVia(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* type_option = new TclStringOption("-type", 1, nullptr);
  addOption(type_option);
}

unsigned CmdEcoRepairVia::check()
{
  return 1;
}

unsigned CmdEcoRepairVia::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg("-type");
  if (option != nullptr) {
    auto type_option = option->getStringVal();
    std::string type = type_option == nullptr ? "shape" : type_option;

    ieco::ECOApi eco_api;
    eco_api.ecoVia(type);
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace tcl
