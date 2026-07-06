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
#include "Report.hh"

#include "PathUtils.hh"
#include "RCXConfig.hh"
#include "RCXData.hh"
#include "SpefDumper.hh"
#include "log/Log.hh"

namespace ircx {

auto Report::dumpSpef() -> bool
{
  const Str& output_dir = RCX_CONFIG_INST.get_output_dir();
  if (!path::ensure_dir(output_dir, "output_dir")) {
    return false;
  }

  RCXData& data = RCX_DATA_INST;
  const auto& corner_data = data.corner_data();
  if (corner_data.empty()) {
    LOG_ERROR << "report spef failed: process corners not loaded.";
    return false;
  }

  SpefDumper dumper;
  dumper.set_spef_context(&data.spef_context());
  dumper.set_layout_data(&data.layout());
  dumper.set_topo_pool(&data.topo_pool());
  dumper.set_rc_table(&data.rc_table());
  dumper.set_corner_data(&corner_data);
  dumper.set_layer_table(&data.layer_table());
  if (!dumper.dump(output_dir)) {
    return false;
  }

  return true;
}

}  // namespace ircx
