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
#include "py_lm.h"

#include "lm_api.h"
#include "timing_api.hh"
#include <filesystem>

namespace python_interface {

bool layout_patchs(const std::string& path)
{
  ilm::LargeModelApi lm_api;
  return lm_api.buildLargeModelLayoutData(path);
}

bool layout_graph(const std::string& path)
{
  ilm::LargeModelApi lm_api;
  return lm_api.buildLargeModelGraphData(path);
}

bool large_model_feature(std::string dir)
{
  if (dir == "") {
    dir = "./large_model";
  }
  ilm::LargeModelApi lm_api;
  return lm_api.buildLargeModelFeature(dir);
}

ieval::TimingWireGraph get_timing_wire_graph(std::string wire_graph_yaml_path) {
  if (std::filesystem::exists(wire_graph_yaml_path)) {
    auto timing_wire_graph = ieval::RestoreTimingGraph(wire_graph_yaml_path);
    return timing_wire_graph;
  }

  ilm::LargeModelApi lm_api;
  lm_api.runLmSTA(); 

  auto* timing_wire_graph_ptr = ieval::TimingAPI::getInst()->getTimingWireGraph();
  ieval::SaveTimingGraph(*timing_wire_graph_ptr, wire_graph_yaml_path);

  auto timing_wire_graph = std::move(*timing_wire_graph_ptr);
  delete timing_wire_graph_ptr;

  return timing_wire_graph;
}

}  // namespace python_interface