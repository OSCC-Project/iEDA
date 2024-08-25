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

int32_t InitEGR::getEGRWL()
{
  int32_t egr_wl = 0;

  // runEGR();

  // return rtInterface.getEGRWL();
  return egr_wl;
}

int32_t InitEGR::getNetEGRWL(std::string net_name)
{
  int32_t net_egr_wl = 0;

  // runEGR();

  // return rtInterface.getNetEGRWL(net_name);
  return net_egr_wl;
}

int32_t InitEGR::getPathEGRWL(std::string net_name, std::string point_name1, std::string point_name2)
{
  int32_t path_egr_wl = 0;

  // runEGR();

  // return rtInterface.getPathEGRWL(net_name, point_name1, point_name2);
  return path_egr_wl;
}

}  // namespace ieval