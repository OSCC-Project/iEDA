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
#include "Monitor.hpp"

#include "Logger.hpp"
#include "Utility.hpp"

namespace idrc {

// public

std::string Monitor::getStatsInfo()
{
  std::string stats_info;
  stats_info = DRCUTIL.getString(" (elapsed = ", getElapsedTime(), ", cpu = ", getCPUTime(), ", mem = ", getUsageMemory(), ") ");
  updateStats();
  return stats_info;
}

std::string Monitor::getElapsedTime()
{
  return DRCUTIL.formatSec(getCurrElapsedTime() - _init_elapsed_time);
}

std::string Monitor::getCPUTime()
{
  return DRCUTIL.formatSec(getCurrCPUTime() - _init_cpu_time);
}

std::string Monitor::getUsageMemory()
{
  return DRCUTIL.getString(DRCUTIL.formatByTwoDecimalPlaces(getCurrUsageMemory() - _init_usage_memory), "MB");
}

// private

void Monitor::init()
{
  updateStats();
}

void Monitor::updateStats()
{
  _init_elapsed_time = getCurrElapsedTime();
  _init_cpu_time = getCurrCPUTime();
  _init_usage_memory = getCurrUsageMemory();
}

double Monitor::getCurrElapsedTime()
{
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.00;
}

double Monitor::getCurrCPUTime()
{
  struct rusage usage;
  if (0 != getrusage(RUSAGE_SELF, &usage)) {
    DRCLOG.error(Loc::current(), "Unable to get rusage!");
  }
  return static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) / 1000000.0 + static_cast<double>(usage.ru_stime.tv_sec)
         + static_cast<double>(usage.ru_stime.tv_usec) / 1000000.0;
}

double Monitor::getCurrUsageMemory()
{
  struct rusage usage;
  if (0 != getrusage(RUSAGE_SELF, &usage)) {
    DRCLOG.error(Loc::current(), "Unable to get rusage!");
  }
  return static_cast<double>(usage.ru_maxrss) / 1000.0;
}

}  // namespace idrc
