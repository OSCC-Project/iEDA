#pragma once

#include <string>
#include <vector>

#include "CtsNet.h"

namespace icts {
using std::vector;

class CtsClock {
 public:
  CtsClock() = default;
  CtsClock(const string &clock_name) : _clock_name(clock_name) {}
  CtsClock(const CtsClock &) = default;
  ~CtsClock() = default;

  // getter
  string get_clock_name() const { return _clock_name; }
  vector<CtsNet *> &get_clock_nets() { return _clock_nets; }

  // setter
  void set_clock_name(const string &clock_name) { _clock_name = clock_name; }

  void addClockNet(CtsNet *net) { _clock_nets.push_back(net); }

 private:
  string _clock_name;
  vector<CtsNet *> _clock_nets;
};
}  // namespace icts