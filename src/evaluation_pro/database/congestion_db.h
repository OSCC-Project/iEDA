#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ieval {

using namespace ::std;

struct OverflowSummary
{
  int32_t total_overflow;
  int32_t max_overflow;
  int32_t average_overflow;
};

struct CongestionMapPathSummary
{
  string egr_path;
  string rudy_path;
};

}  // namespace ieval