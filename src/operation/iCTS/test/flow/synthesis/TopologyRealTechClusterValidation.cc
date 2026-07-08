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
/**
 * @file TopologyRealTechClusterValidation.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief GTest-free cluster-buffer validation for Topology real-tech tests.
 */

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "Clustering.hh"
#include "Inst.hh"
#include "Pin.hh"
#include "Point.hh"
#include "TopologyRealTechScenario.hh"
#include "database/design/Net.hh"
#include "synthesis/topology/Topology.hh"

namespace icts_test::synthesis_realtech_smoke {
namespace {

auto AddFailure(TopologyValidationResult& validation, const std::string& message) -> void
{
  validation.failure_messages.push_back(message);
}

auto MakeClusterBufferPrefix(std::size_t cluster_buffer_index) -> std::string
{
  return "cluster_buffer[" + std::to_string(cluster_buffer_index) + "]: ";
}

}  // namespace

auto CountNonEmptyClusters(const icts::ClusterOutput& cluster_output) -> std::size_t
{
  return static_cast<std::size_t>(std::count_if(cluster_output.clusters.begin(), cluster_output.clusters.end(),
                                                [](const std::vector<icts::Pin*>& cluster) -> bool { return !cluster.empty(); }));
}

auto CollectClusterBufferInsts(const icts::Topology::Build& result) -> std::unordered_set<icts::Inst*>
{
  std::unordered_set<icts::Inst*> cluster_buffer_insts;
  cluster_buffer_insts.reserve(result.output.cluster_buffers.size());
  for (const auto& cluster_buffer : result.output.cluster_buffers) {
    if (cluster_buffer.inst != nullptr) {
      cluster_buffer_insts.insert(cluster_buffer.inst);
    }
  }
  return cluster_buffer_insts;
}

auto ValidateClusteredSinkConnectivity(const std::vector<icts::Pin*>& sinks, const std::unordered_set<icts::Inst*>& cluster_buffer_insts)
    -> TopologyValidationResult
{
  TopologyValidationResult validation;
  if (sinks.empty()) {
    AddFailure(validation, "Expected at least one clustered sink.");
  }
  if (cluster_buffer_insts.empty()) {
    AddFailure(validation, "Expected at least one cluster buffer inst.");
  }

  for (std::size_t sink_index = 0; sink_index < sinks.size(); ++sink_index) {
    auto* sink = sinks.at(sink_index);
    const std::string prefix = "sink[" + std::to_string(sink_index) + "]: ";
    if (sink == nullptr) {
      AddFailure(validation, prefix + "pin is null");
      continue;
    }
    auto* net = sink->get_net();
    if (net == nullptr) {
      AddFailure(validation, prefix + "net is null");
      continue;
    }
    auto* driver = net->get_driver();
    if (driver == nullptr) {
      AddFailure(validation, prefix + "driver is null");
      continue;
    }
    auto* driver_inst = driver->get_inst();
    if (driver_inst == nullptr) {
      AddFailure(validation, prefix + "driver inst is null");
      continue;
    }
    if (!driver_inst->is_buffer()) {
      AddFailure(validation, prefix + "driver inst is not a buffer");
    }
    if (!cluster_buffer_insts.contains(driver_inst)) {
      AddFailure(validation, prefix + "driver inst is not one of the recorded cluster buffers");
    }
  }
  return validation;
}

auto ValidateClusterBufferMastersFollowLeafSemantics(const icts::Topology::Build& result, const std::string& min_cluster_master)
    -> TopologyValidationResult
{
  TopologyValidationResult validation;
  if (!result.output.cluster_output.has_value()) {
    AddFailure(validation, "Expected clustered synthesis result to include cluster metadata.");
    return validation;
  }

  const auto& cluster_output = *result.output.cluster_output;
  const std::size_t non_empty_cluster_count = CountNonEmptyClusters(cluster_output);
  if (result.output.cluster_buffers.size() != non_empty_cluster_count) {
    AddFailure(validation, "Cluster buffer count does not match non-empty cluster count.");
  }
  if (result.output.cluster_buffers.empty()) {
    AddFailure(validation, "Expected at least one cluster buffer.");
  }

  for (std::size_t cluster_buffer_index = 0; cluster_buffer_index < result.output.cluster_buffers.size(); ++cluster_buffer_index) {
    const auto& cluster_buffer = result.output.cluster_buffers.at(cluster_buffer_index);
    const std::string prefix = MakeClusterBufferPrefix(cluster_buffer_index);
    if (cluster_buffer.cluster_index >= cluster_output.clusters.size()) {
      AddFailure(validation, prefix + "cluster index is out of range");
      continue;
    }

    const auto& cluster_sinks = cluster_output.clusters.at(cluster_buffer.cluster_index);
    if (cluster_sinks.empty()) {
      AddFailure(validation, prefix + "cluster is empty");
    }
    if (cluster_buffer.inst == nullptr) {
      AddFailure(validation, prefix + "inst is null");
    }
    if (cluster_buffer.input_pin == nullptr) {
      AddFailure(validation, prefix + "input pin is null");
    }
    if (cluster_buffer.output_pin == nullptr) {
      AddFailure(validation, prefix + "output pin is null");
    }
    if (cluster_buffer.sink_net == nullptr) {
      AddFailure(validation, prefix + "sink net is null");
    }

    if (cluster_buffer.sink_net != nullptr) {
      if (cluster_buffer.sink_net->get_driver() != cluster_buffer.output_pin) {
        AddFailure(validation, prefix + "sink net driver does not match output pin");
      }
      if (cluster_buffer.sink_net->get_loads().size() != cluster_sinks.size()) {
        AddFailure(validation, prefix + "sink net load count does not match cluster size");
      }
    }
    if (cluster_buffer.output_pin != nullptr && cluster_buffer.output_pin->get_net() != cluster_buffer.sink_net) {
      AddFailure(validation, prefix + "output pin net does not match sink net");
    }
    if (cluster_buffer.sink_count != cluster_sinks.size()) {
      AddFailure(validation, prefix + "recorded sink count does not match cluster size");
    }
    if (cluster_buffer.inst != nullptr) {
      if (cluster_buffer.inst->get_location().get_x() != cluster_buffer.location.get_x()
          || cluster_buffer.inst->get_location().get_y() != cluster_buffer.location.get_y()) {
        AddFailure(validation, prefix + "inst location does not match recorded cluster-buffer location");
      }
      if (cluster_buffer.inst->get_cell_master() != min_cluster_master) {
        AddFailure(validation, prefix + "cell master does not match expected minimum cluster buffer master");
      }
    }

    if (cluster_buffer.input_pin != nullptr) {
      auto* leaf_net = cluster_buffer.input_pin->get_net();
      if (leaf_net == nullptr) {
        AddFailure(validation, prefix + "leaf net is null");
      } else {
        if (leaf_net == cluster_buffer.sink_net) {
          AddFailure(validation, prefix + "leaf net should differ from sink net");
        }
        auto* leaf_driver = leaf_net->get_driver();
        if (leaf_driver == nullptr) {
          AddFailure(validation, prefix + "leaf driver is null");
        } else if (leaf_driver == cluster_buffer.output_pin) {
          AddFailure(validation, prefix + "leaf driver should not be the cluster output pin");
        }
      }
    }

    for (std::size_t sink_index = 0; sink_index < cluster_sinks.size(); ++sink_index) {
      auto* sink = cluster_sinks.at(sink_index);
      if (sink == nullptr) {
        AddFailure(validation, prefix + "cluster sink[" + std::to_string(sink_index) + "] is null");
        continue;
      }
      if (sink->get_net() != cluster_buffer.sink_net) {
        AddFailure(validation, prefix + "cluster sink[" + std::to_string(sink_index) + "] is not connected to sink net");
      }
    }
  }

  return validation;
}

}  // namespace icts_test::synthesis_realtech_smoke
