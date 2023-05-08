#pragma once

#include "RTU.hpp"
#include "Stage.hpp"

namespace irt {

class GPConfig
{
 public:
  GPConfig() = default;
  ~GPConfig() = default;

  std::string temp_directory_path;
};

}  // namespace irt
