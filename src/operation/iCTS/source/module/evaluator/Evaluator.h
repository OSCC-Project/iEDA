#pragma once

#include <iostream>
#include <vector>

#include "CtsConfig.h"
#include "CtsDesign.h"
#include "EvalNet.h"
#include "GDSPloter.h"

namespace icts {
using std::vector;

class Evaluator {
 public:
  Evaluator() = default;
  Evaluator(const Evaluator &) = default;
  ~Evaluator() = default;

  void init();
  void evaluate();
  void update();

  double latency() const;
  double skew() const;
  double fanout() const;
  double slew() const;
  void statistics(const std::string &save_dir) const;
  int64_t wireLength() const;
  double dataCtsNetSlack() const;
  void plotPath(const string &inst, const string &file = "debug.gds") const;
  void plotNet(const string &net_name, const string &file = "debug.gds") const;

 private:
  void printLog();
  void transferData();

  vector<EvalNet> _eval_nets;
  const int _default_size = 100;
};

}  // namespace icts