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

 private:
  std::size_t _memory_begin;
  struct timeval _elapsed_begin_time;
};

}  // namespace ieda