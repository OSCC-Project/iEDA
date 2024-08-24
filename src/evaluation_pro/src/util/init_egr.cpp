#include "init_egr.h"

#include "RTInterface.hpp"

namespace ieval {

InitEGR::InitEGR()
{
}

InitEGR::~InitEGR()
{
}

void InitEGR::runEGR()
{
  irt::RTInterface& rtInterface = irt::RTInterface::getInst();
  std::map<std::string, std::any> config_map;
  rtInterface.initRT(config_map);
  rtInterface.runEGR();
}
}  // namespace ieval