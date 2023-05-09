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
#include "ShellCmd.hh"

namespace ista {

CmdTESTSLL::CmdTESTSLL(const char* cmd_name) : TclCmd(cmd_name) {
  auto* strll_option = new TclStringListListOption("-puts", 0);
  addOption(strll_option);
}

unsigned CmdTESTSLL::check() {
  return 1;
}

unsigned CmdTESTSLL::exec() {
  TclOption* strll_option = getOptionOrArg("-puts");
  if (strll_option->is_set_val()) {
    auto str_list_list = strll_option->getStringListList();
    for (auto&& str_list : str_list_list) {
      std::cout << "< " ;
      for (auto&& str : str_list) {
        std::cout << "\"" << str << "\" ";
      }
      std::cout << ">\n";
    }
  }

  return 1;
}
}