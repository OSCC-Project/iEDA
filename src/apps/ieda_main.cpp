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
