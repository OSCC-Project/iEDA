#pragma once

#include <string>

namespace ieval {

using namespace ::std;

class CongestionEval
{
 public:
  CongestionEval();
  ~CongestionEval();

  void runEGR();
  void runRUDY();

  void computeOverflow();

  string plotEGR();
  string plotRUDY();

  int32_t get_total_overflow() { return _total_overflow; }
  int32_t get_max_overflow() { return _max_overflow; }
  int32_t get_average_overflow() { return _average_overflow; }

 private:
  int32_t _total_overflow = -1;
  int32_t _max_overflow = -1;
  int32_t _average_overflow = -1;
};
}  // namespace ieval