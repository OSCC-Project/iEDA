#pragma once

#include <string>

namespace ieval {

class InitEGR
{
 public:
  InitEGR();
  ~InitEGR();

  void runEGR();

  int32_t getEGRWL();
  int32_t getNetEGRWL(std::string net_name);
  int32_t getPathEGRWL(std::string net_name, std::string point_name1, std::string point_name2);
};

}  // namespace ieval
