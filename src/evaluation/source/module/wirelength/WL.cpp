#include "WL.hpp"

#include "EvalLog.hpp"

namespace eval {

int64_t WLMWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t WLM = 0;
  for (WLNet* net : length_net_list) {
    WLM += net->wireLoadModel();
  }
  LOG_INFO << " Total wlm = " << WLM;
  return WLM;
}

int64_t HPWLWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t HPWL = 0;
  for (WLNet* net : length_net_list) {
    HPWL += net->HPWL();
  }
  LOG_INFO << " Total HPWL =  " << HPWL;
  return HPWL;
}

int64_t HTreeWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t HTree = 0;
  for (WLNet* net : length_net_list) {
    HTree += net->HTree();
  }
  LOG_INFO << " Total HTree = " << HTree;
  return HTree;
}

int64_t VTreeWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t VTree = 0;
  for (WLNet* net : length_net_list) {
    VTree += net->VTree();
  }
  LOG_INFO << " Total VTree = " << VTree;
  return VTree;
}

int64_t CliqueWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t clique = 0;
  for (WLNet* net : length_net_list) {
    clique += net->Clique();
  }
  LOG_INFO << " Total clique = " << clique;
  return clique;
}

int64_t StarWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t star = 0;
  for (WLNet* net : length_net_list) {
    star += net->Star();
  }
  LOG_INFO << " Total star = " << star;
  return star;
}

int64_t B2BWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t B2B = 0;
  for (WLNet* net : length_net_list) {
    B2B += net->B2B();
  }
  LOG_INFO << " Total Bound2Bound = " << B2B;
  return B2B;
}

int64_t FluteWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t Flute = 0;
  for (WLNet* net : length_net_list) {
    Flute += net->FluteWL();
  }
  LOG_INFO << " Total Flute = " << Flute;
  return Flute;
}

int64_t PlaneRouteWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t planeRouteWL = 0;
  for (WLNet* net : length_net_list) {
    planeRouteWL += net->planeRouteWL();
  }
  LOG_INFO << " Total plane route WL = " << planeRouteWL;
  return planeRouteWL;
}

int64_t SpaceRouteWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t spaceRouteWL = 0;
  for (WLNet* net : length_net_list) {
    spaceRouteWL += net->spaceRouteWL();
  }
  LOG_INFO << " Total space route WL = " << spaceRouteWL;
  return spaceRouteWL;
}

int64_t DRWL::getTotalWL(const std::vector<WLNet*>& length_net_list)
{
  int64_t DRWL = 0;
  for (WLNet* net : length_net_list) {
    DRWL += net->detailRouteWL();
  }
  LOG_INFO << " Total DR WL = " << DRWL;
  return DRWL;
}
}  // namespace eval
