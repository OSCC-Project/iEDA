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

#include <sys/time.h>
#include <time.h>

#include <cstddef>  // size_t
#include <string>

namespace ieda {

/**
 * @brief Show run time and memory statistics if the "stats" debug flag is on.
 *
 */
class Stats {
 public:
  Stats();
  ~Stats() = default;
  [[nodiscard]] std::size_t memoryUsage() const;
  [[nodiscard]] double memoryDelta() const;

  std::string getCurrentWallTime() const;

  int getTimeOfDay(struct timeval *tv) const;
  [[nodiscard]] double elapsedRunTime() const;

  void restartStats();

 private:
  std::size_t _memory_begin;
  struct timeval _elapsed_begin_time;
};

/**
 * @brief macro for profiling, start and end pos should be the same.
 * 
 */
#define CPU_PROF_START(pos) \
  ieda::Stats stats##pos

#define CPU_PROF_END(pos, msg) \
  LOG_INFO << msg << " memory usage: " << stats##pos.memoryDelta() << "MB" << std::endl; \
  LOG_INFO << msg << " time elapsed: " << stats##pos.elapsedRunTime() << "s" << std::endl

}  // namespace ieda