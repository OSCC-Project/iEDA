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
 * @File Name: ieda_contest.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "./../../../../src/platform/flow/flow.h"

using namespace iplf;

int main(int argc, char** argv)
{
  if (argc > 2) {
    /// read iFP script if config
    // char* tcl_file_path = nullptr;
    char* tcl_file_path = nullptr;
    for (int i = 0; i < argc; ++i) {
      if (argv[i] == std::string("-script")) {
        int len = strlen(argv[i + 1]);
        tcl_file_path = new char[len + 1];
        std::memcpy(tcl_file_path, argv[i + 1], len);
        tcl_file_path[len] = '\0';
        // tcl_file_path = argv[i + 1];
      }
    }

    /// read flow config
    std::string flow_config;
    for (int i = 0; i < argc; ++i) {
      if (argv[i] == std::string("-flow_config")) {
        flow_config = argv[i + 1];
      }
    }

    // if (!flow_config.empty()) {
    //   if (plfInst->initFlow(flow_config)) {
    //     plfInst->run(tcl_file_path)
    //   }
    // }

    if (tcl_file_path != nullptr) {
      plfInst->runTcl(tcl_file_path);

      delete tcl_file_path;
    }
  } else {
    plfInst->runTcl();
  }

  return 0;
}
