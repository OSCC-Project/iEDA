// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <sys/resource.h>
#include <sys/time.h>

#include <string>

namespace idrc {

class Monitor
{
 public:
  Monitor() { init(); }
  ~Monitor() = default;
  // getter

  // setter

  // function
  std::string getStatsInfo();
  std::string getElapsedTime();
  std::string getCPUTime();
  std::string getUsageMemory();

 private:
  double _init_elapsed_time = 0;  // \s
  double _init_cpu_time = 0;      // \s
  double _init_usage_memory = 0;  // \GB

  // function
  void init();
  void updateStats();
  double getCurrElapsedTime();
  double getCurrCPUTime();
  double getCurrUsageMemory();
};

}  // namespace idrc
