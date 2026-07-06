// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "config/PlotSpefConfig.hh"

#include "PathUtils.hh"
#include "log/Log.hh"

namespace ircx::plot_spef {

auto ConfigValidator::validate(const Config& config) const -> bool
{
  if (!path::file_exists(config.spef_file, "plot_spef SPEF file")) {
    return false;
  }

  if (config.output_dir.empty()) {
    LOG_ERROR << "plot_spef requires an output directory.";
    return false;
  }

  if (config.dbu <= 0) {
    LOG_ERROR << "plot_spef requires a positive DBU.";
    return false;
  }

  if (config.cores <= 0) {
    LOG_ERROR << "plot_spef -cores must be a positive integer.";
    return false;
  }

  if (!path::ensure_dir(config.output_dir, "plot_spef output directory")) {
    return false;
  }

  return true;
}

}  // namespace ircx::plot_spef
