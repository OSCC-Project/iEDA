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
#include "py_vec.h"

#include <filesystem>

#include "timing_api.hh"
#include "vec_api.h"

namespace python_interface {

bool layout_patchs(const std::string& path)
{
  ivec::VectorizationApi lm_api;
  return lm_api.buildVectorizationLayoutData(path);
}

bool layout_graph(const std::string& path)
{
  ivec::VectorizationApi lm_api;
  return lm_api.buildVectorizationGraphData(path);
}

bool generate_vectors(std::string dir, int patch_row_step, int patch_col_step, bool batch_mode)
{
  if (dir == "") {
    dir = "./vectors";
  }
  ivec::VectorizationApi lm_api;
  return lm_api.buildVectorizationFeature(dir, patch_row_step, patch_col_step, batch_mode);
}

ieval::TimingWireGraph get_timing_wire_graph(std::string wire_graph_path)
{
  if (std::filesystem::exists(wire_graph_path)) {
    auto timing_wire_graph = ieval::RestoreTimingGraph(wire_graph_path);
    return timing_wire_graph;
  }

  ivec::VectorizationApi lm_api;
  lm_api.runVecSTA();

  auto* timing_wire_graph_ptr = ieval::TimingAPI::getInst()->getTimingWireGraph();
  ieval::SaveTimingGraph(*timing_wire_graph_ptr, wire_graph_path);

  auto timing_wire_graph = std::move(*timing_wire_graph_ptr);
  delete timing_wire_graph_ptr;

  return timing_wire_graph;
}

ieval::TimingInstanceGraph get_timing_instance_graph(std::string instance_graph_path)
{
  if (std::filesystem::exists(instance_graph_path)) {
    auto timing_instance_graph = ieval::RestoreTimingInstanceGraph(instance_graph_path);
    return timing_instance_graph;
  }

  ivec::VectorizationApi lm_api;
  lm_api.runVecSTA();

  auto* timing_instance_graph_ptr = ieval::TimingAPI::getInst()->getTimingInstanceGraph();
  ieval::SaveTimingInstanceGraph(*timing_instance_graph_ptr, instance_graph_path);

  auto timing_instance_graph = std::move(*timing_instance_graph_ptr);
  delete timing_instance_graph_ptr;

  return timing_instance_graph;
}

bool read_vectors_nets(std::string dir)
{
  if (dir == "") {
    return false;
  }
  ivec::VectorizationApi lm_api;
  return lm_api.readVectorsNets(dir);
}

bool read_vectors_nets_patterns(std::string path)
{
  if (path == "") {
    return false;
  }
  ivec::VectorizationApi lm_api;
  return lm_api.readVectorsNetsPatterns(path);
}

}  // namespace python_interface