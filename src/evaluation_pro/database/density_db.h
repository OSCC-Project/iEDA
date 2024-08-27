#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ieda_eval {

using namespace ::std;

struct DensityMapPathSummary
{
  string cell_density;
  string pin_density;
  string net_density;
  string channel_density;
  string whitespace_density;
};

}  // namespace ieda_eval