#pragma once
/**
 * @File Name: flow.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-03-17
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>

#include "config/flow_config.h"
#include "tool_manager.h"

using std::string;
using std::vector;

namespace iplf {
#define plfInst Flow::getInstance()

class Flow
{
 public:
  static Flow* getInstance()
  {
    if (!_instance) {
      _instance = new Flow;
    }
    return _instance;
  }

  bool initFlow(string flow_config = "");
  void run(char* param = nullptr);
  void runFlow();
  void runTcl(char* path = nullptr);

 private:
  static Flow* _instance;

  Flow() {}
  ~Flow() = default;
};

}  // namespace iplf