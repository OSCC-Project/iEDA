#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ClockTopo.h"
#include "CtsNet.h"
#include "TimingCalculator.h"

namespace icts {
// balance by master clock latency
struct BalanceTrait {
  double latency;
  double clock_at;
  TimingNode* clock_node;
  TimingNode* root_node;
  CtsNet* clk_net;
};
class Balancer {
 public:
  Balancer() = default;
  ~Balancer() = default;
  void init();
  void balance();

  double calcDumpCapOut(CtsNet* clock_net) const;
  std::map<std::string, std::string> getNetNameToClockNameMap() const;
  std::map<std::set<std::string>, std::vector<CtsNet*>>
  getClkNetsGroupByMasterClocks() const;

 private:
};

}  // namespace icts