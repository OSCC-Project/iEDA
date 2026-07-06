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
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <optional>
#include <sstream>
#include <string>
#include <utility>

#if defined(__has_include)
#if __has_include(<source_location>)
#include <source_location>
#define IRCX_HAS_STD_SOURCE_LOCATION 1
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#define IRCX_HAS_EXPERIMENTAL_SOURCE_LOCATION 1
#endif
#endif

#include "log/Log.hh"
#include "usage/usage.hh"

namespace ircx {

#if defined(IRCX_HAS_STD_SOURCE_LOCATION)
using SourceLocation = std::source_location;
#elif defined(IRCX_HAS_EXPERIMENTAL_SOURCE_LOCATION)
using SourceLocation = std::experimental::source_location;
#else
class SourceLocation
{
 public:
  static constexpr auto current(const char* file_name = __builtin_FILE(), int line = __builtin_LINE()) -> SourceLocation
  {
    return SourceLocation(file_name, line);
  }

  constexpr auto file_name() const -> const char* { return file_name_; }
  constexpr auto line() const -> int { return line_; }

 private:
  constexpr SourceLocation(const char* file_name, int line) : file_name_(file_name), line_(line) {}

  const char* file_name_;
  int line_;
};
#endif

template <typename... Args>
inline void log_stage(const SourceLocation& location, const Args&... args)
{
  std::ostringstream stream;
  (stream << ... << args);
  google::LogMessage(location.file_name(), static_cast<int>(location.line()), google::GLOG_INFO).stream() << stream.str();
}

class StageLog
{
 public:
  explicit StageLog(std::string stage, SourceLocation location = SourceLocation::current())
      : stage_(std::move(stage)), location_(location)
  {
    log_stage(location_, stage_, " begin.");
  }
  ~StageLog() { log_stage(location_, stage_, " end: ", (success_ ? "success" : "failed"), "."); }

  StageLog(const StageLog&) = delete;
  StageLog& operator=(const StageLog&) = delete;

  void set_success(bool success = true) { success_ = success; }

 private:
  std::string stage_;
  SourceLocation location_;
  bool success_{false};
};

struct StageLogOptions
{
  bool profile{false};
};

template <typename Func>
auto run_stage(std::string stage, Func&& func, StageLogOptions options = {},
              SourceLocation location = SourceLocation::current()) -> bool
{
  std::optional<ieda::Stats> stats;
  if (options.profile) {
    stats.emplace();
  }

  StageLog stage_log(stage, location);
  const bool success = std::forward<Func>(func)();
  stage_log.set_success(success);
  if (stats.has_value()) {
    log_stage(location, "  - memory usage: ", stats->memoryDelta(), "MB");
    log_stage(location, "  - time elapsed: ", stats->elapsedRunTime(), "s");
  }
  return success;
}

}  // namespace ircx
