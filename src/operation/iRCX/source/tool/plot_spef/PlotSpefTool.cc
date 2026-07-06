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
#include "PlotSpefTool.hh"

#include "SpefParser.hh"
#include "builder/PlotSpefModelBuilder.hh"
#include "gds/PlotSpefGdsWriter.hh"
#include "log/Log.hh"
#include "lyp/PlotSpefLypWriter.hh"

namespace ircx {

auto PlotSpefTool::run(plot_spef::Config config) -> bool
{
  const plot_spef::ConfigValidator validator;
  if (!validator.validate(config)) {
    return false;
  }

  spef::SpefReader reader;
  if (!reader.read(config.spef_file)) {
    LOG_ERROR << "plot_spef failed: read SPEF failed: " << config.spef_file;
    return false;
  }
  reader.expandName();

  const spef::Exchange* exchange = reader.getSpefFile();
  if (exchange == nullptr) {
    LOG_ERROR << "plot_spef failed: SPEF reader returned empty data.";
    return false;
  }

  const plot_spef::ModelBuilder model_builder;
  const auto model = model_builder.build(*exchange, config);

  const plot_spef::GdsWriter writer;
  if (!writer.write(model, config)) {
    return false;
  }

  const plot_spef::LypWriter lyp_writer;
  if (!lyp_writer.write(model, config)) {
    return false;
  }

  LOG_INFO << "plot_spef wrote reports to " << config.output_dir;
  return true;
}

}  // namespace ircx
