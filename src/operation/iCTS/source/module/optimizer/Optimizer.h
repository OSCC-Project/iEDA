#pragma once

#include "CtsDBWrapper.h"
#include "OptiNet.h"
#include "Router.h"

namespace icts {

class Optimizer {
 public:
  typedef std::vector<CtsNet *>::iterator NetIterator;
  typedef std::vector<IdbNet *>::iterator IdbNetIterator;

  Optimizer() = default;
  Optimizer(const Optimizer &optimizer) = delete;
  ~Optimizer() = default;

  void optimize(NetIterator begin, NetIterator end);
  void update();

 private:
  void reroute(IdbNetIterator begin, IdbNetIterator end);
  std::vector<CtsSignalWire> reroute(IdbNet *idb_net);

  CtsInstance *get_cts_inst(IdbInstance *idb_inst) const;
  IdbNet *get_idb_net(const OptiNet &opti_net) const;
  Point get_location(IdbInstance *idb_inst) const;

 private:
  std::vector<IdbNet *> _idb_nets;
  std::vector<std::vector<CtsSignalWire>> _topos;
};

}  // namespace icts