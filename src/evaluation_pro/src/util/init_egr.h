#pragma once

#include <string>

namespace ieval {

class InitEGR
{
 public:
  InitEGR();
  ~InitEGR();

  void runEGR();

  float parseEGRWL(std::string guide_path);
  float parseNetEGRWL(std::string guide_path, std::string net_name);
  float parsePathEGRWL(std::string guide_path, std::string net_name, std::string load_name);
};

}  // namespace ieval
