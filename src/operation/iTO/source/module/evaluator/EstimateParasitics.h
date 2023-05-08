#pragma once

#include <set>
#include <string>
#include <vector>

#include <unordered_set>

#include "DbInterface.h"
#include "RoutingTree.h"
#include "Utility.h"

namespace ito {
using ito::dbuToMeters;
using ito::metersToDbu;

class EstimateParasitics {
 public:
  EstimateParasitics(DbInterface *dbintreface);

  EstimateParasitics(TimingEngine *timing_engine, int dbu);
  EstimateParasitics() = default;
  ~EstimateParasitics() = default;

  void excuteParasiticsEstimate();

  void estimateAllNetParasitics();

  void estimateNetParasitics(Net *net);

  void parasiticsInvalid(Net *net);

  void estimateInvalidNetParasitics(DesignObject *drvr_pin_port, Net *net);

  void excuteWireParasitic(DesignObject *drvr_pin_port, Net *curr_net,
                           TimingDBAdapter *db_adapter);

 private:
  void RctNodeConnectPin(Net *net, int index, RctNode *rcnode, RoutingTree *tree);

  DbInterface     *_db_interface = nullptr;
  TimingEngine    *_timing_engine = nullptr;
  TimingDBAdapter *_db_adapter = nullptr;
  int              _dbu;

  std::unordered_set<ista::Net *> _parasitics_invalid;

  bool _have_estimated_parasitics = false;
};

} // namespace ito