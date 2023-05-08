#pragma once

#include <sys/resource.h>
#include <sys/time.h>

#include <string>

namespace irt {

class Monitor
{
 public:
  Monitor() { init(); }
  ~Monitor() = default;
  // getter

  // setter

  // function
  std::string getStatsInfo();
  double getCPUTime();
  double getElapsedTime();
  double getUsageMemory();

 private:
  double _init_cpu_time = 0;      // \s
  double _init_elapsed_time = 0;  // \s
  double _init_usage_memory = 0;  // \GB

  // function
  void init();
  void updateStats();
  double getCurrCPUTime();
  double getCurrElapsedTime();
  double getCurrUsageMemory();
};
}  // namespace irt
