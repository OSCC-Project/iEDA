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
 * @file SinkLoadClustering.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Implements clustered or direct HTree sink-load preparation for Topology.
 */

#include "synthesis/topology/sink/SinkLoadClustering.hh"

#include <glog/logging.h>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "Clustering.hh"
#include "Log.hh"
#include "Pin.hh"
#include "Point.hh"
#include "TopologyConfig.hh"
#include "io/Wrapper.hh"
#include "synthesis/topology/buffer/BufferInsertion.hh"
#include "topology/fast_clustering/FastClustering.hh"

namespace icts::topology {
namespace {

auto resolveBufferPinNames(Wrapper& wrapper, const std::string& cell_master) -> std::optional<std::pair<std::string, std::string>>
{
  if (cell_master.empty()) {
    return std::nullopt;
  }

  auto [input_pin, output_pin] = wrapper.queryBufferPorts(cell_master);
  if (input_pin.empty() || output_pin.empty()) {
    LOG_WARNING << "Topology: skip buffer master \"" << cell_master << "\" because buffer ports are unresolved.";
    return std::nullopt;
  }

  return std::make_pair(std::move(input_pin), std::move(output_pin));
}

auto resolveClusterCenter(const std::vector<Point<int>>& centers, const std::vector<Pin*>& cluster, std::size_t index) -> Point<int>
{
  if (index < centers.size()) {
    return centers.at(index);
  }
  if (cluster.empty()) {
    return Point<int>(0, 0);
  }

  int sum_x = 0;
  int sum_y = 0;
  for (auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    sum_x += pin->get_location().get_x();
    sum_y += pin->get_location().get_y();
  }

  const int count = static_cast<int>(cluster.size());
  if (count <= 0) {
    return Point<int>(0, 0);
  }
  return Point<int>(sum_x / count, sum_y / count);
}

auto resolveBufferDriveCap(Wrapper& wrapper, const std::string& cell_master) -> double
{
  double drive_cap_pf = wrapper.queryCellOutPinCapLimit(cell_master);
  if (drive_cap_pf <= 0.0) {
    drive_cap_pf = wrapper.queryCellOutPinCapTableAxisMax(cell_master);
  }
  return drive_cap_pf;
}

auto resolveMinLegalClusterBufferCell(Wrapper& wrapper, const SinkTreeLoadPreparationPolicy& policy, std::string& cell_master,
                                      std::string& input_pin_name, std::string& output_pin_name) -> bool
{
  LOG_FATAL_IF(policy.buffer_cell_masters == nullptr) << "Topology: cluster buffer master policy is not bound.";
  bool has_resolved_master = false;
  double best_drive_cap_pf = std::numeric_limits<double>::infinity();
  for (const auto& candidate_cell_master : *policy.buffer_cell_masters) {
    if (candidate_cell_master.empty()) {
      continue;
    }

    const double drive_cap_pf = resolveBufferDriveCap(wrapper, candidate_cell_master);
    if (drive_cap_pf <= 0.0) {
      LOG_WARNING << "Topology: skip clustered center buffer master \"" << candidate_cell_master
                  << "\" because output drive cap is unresolved.";
      continue;
    }

    const auto buffer_pin_names = resolveBufferPinNames(wrapper, candidate_cell_master);
    if (!buffer_pin_names.has_value()) {
      continue;
    }

    if (!has_resolved_master || drive_cap_pf < best_drive_cap_pf
        || (drive_cap_pf == best_drive_cap_pf && candidate_cell_master < cell_master)) {
      cell_master = candidate_cell_master;
      input_pin_name = buffer_pin_names->first;
      output_pin_name = buffer_pin_names->second;
      best_drive_cap_pf = drive_cap_pf;
      has_resolved_master = true;
    }
  }

  return has_resolved_master;
}

auto collectSinkPinCapPfByPin(Wrapper& wrapper, const std::vector<Pin*>& loads) -> std::unordered_map<const Pin*, double>
{
  std::unordered_map<const Pin*, double> sink_pin_cap_pf_by_pin;
  sink_pin_cap_pf_by_pin.reserve(loads.size());
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    sink_pin_cap_pf_by_pin.emplace(pin, std::max(0.0, wrapper.queryPinCapacitance(pin)));
  }
  return sink_pin_cap_pf_by_pin;
}

auto buildClusteringConfigFromPolicy(Wrapper& wrapper, const SinkTreeLoadPreparationPolicy& policy, const std::vector<Pin*>& root_loads)
    -> ClusterConfig
{
  auto clustering_config = FastClustering::buildElectricalBaseConfig(policy.max_fanout, policy.max_cap_pf);
  clustering_config.clock_route_segment_rc = policy.clock_route_segment_rc;
  clustering_config.sink_pin_cap_pf_by_pin = collectSinkPinCapPfByPin(wrapper, root_loads);
  return clustering_config;
}

auto buildClusterBufferObjects(Topology::Build& result, const ClusterOutput& cluster_output, const std::string& cluster_buffer_cell_master,
                               const std::string& input_pin_name, const std::string& output_pin_name, const std::string& object_name_prefix,
                               std::vector<Pin*>& htree_sinks) -> bool
{
  const auto& clusters = cluster_output.clusters;
  result.output.cluster_buffers.reserve(clusters.size());
  htree_sinks.reserve(clusters.size());
  for (std::size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
    const auto& cluster = clusters.at(cluster_index);
    if (cluster.empty()) {
      LOG_WARNING << "Topology: skip empty cluster at index " << cluster_index << ".";
      continue;
    }

    const auto center = resolveClusterCenter(cluster_output.centers, cluster, cluster_index);
    const auto cluster_inst_name = MakeObjectName(object_name_prefix, "cluster_buf_" + std::to_string(cluster_index));
    const auto buffer
        = CreateBufferInstance(result, cluster_inst_name, cluster_buffer_cell_master, input_pin_name, output_pin_name, center);
    const auto sink_net_name = MakeObjectName(object_name_prefix, "cluster_sink_net_" + std::to_string(cluster_index));
    auto* sink_net = CreateNet(result, sink_net_name, buffer.output_pin, cluster);
    result.output.cluster_buffers.push_back(Topology::ClusterBufferMeta{
        .cluster_index = cluster_index,
        .location = center,
        .sink_count = cluster.size(),
        .inst = buffer.inst,
        .input_pin = buffer.input_pin,
        .output_pin = buffer.output_pin,
        .sink_net = sink_net,
    });
    htree_sinks.push_back(buffer.input_pin);
  }

  if (!htree_sinks.empty()) {
    return true;
  }

  LOG_ERROR << "Topology: sink clustering generated no valid centroid buffers.";
  result.summary.failure_reason = "no valid centroid buffers after clustering";
  return false;
}

}  // namespace

auto PrepareSinkTreeLoads(const SinkTreeLoadPreparationInput& input) -> SinkTreeLoadPreparation
{
  LOG_FATAL_IF(input.build == nullptr) << "Topology sink-load preparation requires a topology build.";
  LOG_FATAL_IF(input.root_loads == nullptr) << "Topology sink-load preparation requires root loads.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "Topology sink-load preparation requires an Wrapper.";
  auto& result = *input.build;
  const auto& root_loads = *input.root_loads;
  auto& wrapper = *input.wrapper;
  result.summary.sink_clustering_enabled = input.policy.enable_sink_clustering;

  SinkTreeLoadPreparation preparation;
  if (!input.policy.enable_sink_clustering) {
    preparation.success = true;
    preparation.htree_sinks = root_loads;
    return preparation;
  }

  auto clustering_config = buildClusteringConfigFromPolicy(wrapper, input.policy, root_loads);
  auto cluster_output = Clustering::defaultFastClustering(root_loads, clustering_config);
  result.output.cluster_output = std::move(cluster_output);

  std::string cluster_buffer_cell_master;
  std::string cluster_buffer_input_pin;
  std::string cluster_buffer_output_pin;
  if (!resolveMinLegalClusterBufferCell(wrapper, input.policy, cluster_buffer_cell_master, cluster_buffer_input_pin,
                                        cluster_buffer_output_pin)) {
    LOG_ERROR << "Topology: failed to resolve a legal clustered center buffer master from configured buffer_types.";
    result.summary.failure_reason = "failed to resolve clustered center buffer master";
    return preparation;
  }
  preparation.success
      = buildClusterBufferObjects(result, *result.output.cluster_output, cluster_buffer_cell_master, cluster_buffer_input_pin,
                                  cluster_buffer_output_pin, input.object_name_prefix, preparation.htree_sinks);
  return preparation;
}

}  // namespace icts::topology
