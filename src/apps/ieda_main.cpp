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
 * @File Name: ieda_main.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../platform/flow/flow.h"
#include "log/Log.hh"

using namespace iplf;

int main(int argc, char** argv)
{
  if (argc == 1) {
    argv[0] = const_cast<char*>("UserShell\n");
  }

  bool printVersion = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string("-v") == argv[i]) {
      printVersion = true;
    }
    
    // support specific log directory
    if (std::string("-log") == argv[i]) {
      ieda::Log::init(argv, argv[i + 1]);
    }

    if (std::string("-script") == argv[i]) {
      // discard every args before the (first) "-script"
      // pass the rest of the args to Tcl interpreter
      argc -= i;
      argv += i;
      break;
    }
  }

  if (printVersion) {
    std::cout << "Git version: " << iEDA_GIT_VERSION << std::endl;
  }

  plfInst->runTcl(argc, argv);

  return 0;
}
