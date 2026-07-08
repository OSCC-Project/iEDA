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
 * @file FastClustering.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Fast spatial clustering facade for topology clustering.
 */

#include "FastClustering.hh"

#include <glog/logging.h>

#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Clustering.hh"
#include "Log.hh"
#include "TopologyConfig.hh"
#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {

namespace detail = fast_clustering;

namespace {

using detail::ElapsedSeconds;
using detail::FormatRatio;
using detail::FormatSeconds;
using detail::SteadyClock;

auto FormatFanoutHistogram(const std::map<std::size_t, std::size_t>& histogram) -> std::string
{
  if (histogram.empty()) {
    return "none";
  }

  std::ostringstream stream;
  bool first = true;
  for (const auto& [fanout, count] : histogram) {
    if (!first) {
      stream << ",";
    }
    stream << fanout << "=" << count;
    first = false;
  }
  return stream.str();
}

auto BuildDraftFanoutHistogram(const std::vector<detail::ClusterDraft>& drafts) -> std::map<std::size_t, std::size_t>
{
  std::map<std::size_t, std::size_t> histogram;
  for (const auto& draft : drafts) {
    if (!draft.active || draft.entry_ids.empty()) {
      continue;
    }
    ++histogram[draft.entry_ids.size()];
  }
  return histogram;
}

auto CountDraftLoads(const std::vector<detail::ClusterDraft>& drafts) -> std::size_t
{
  std::size_t load_count = 0U;
  for (const auto& draft : drafts) {
    if (!draft.active || draft.entry_ids.empty()) {
      continue;
    }
    load_count += draft.entry_ids.size();
  }
  return load_count;
}

auto CountDraftClusters(const std::vector<detail::ClusterDraft>& drafts) -> std::size_t
{
  std::size_t cluster_count = 0U;
  for (const auto& draft : drafts) {
    if (!draft.active || draft.entry_ids.empty()) {
      continue;
    }
    ++cluster_count;
  }
  return cluster_count;
}

auto BuildResultFanoutHistogram(const ClusterOutput& result) -> std::map<std::size_t, std::size_t>
{
  std::map<std::size_t, std::size_t> histogram;
  for (const auto& cluster : result.clusters) {
    if (cluster.empty()) {
      continue;
    }
    ++histogram[cluster.size()];
  }
  return histogram;
}

auto CalcAverageFanout(std::size_t load_count, std::size_t cluster_count) -> double
{
  if (cluster_count == 0U) {
    return 0.0;
  }
  return static_cast<double>(load_count) / static_cast<double>(cluster_count);
}

}  // namespace

auto FastClustering::buildElectricalBaseConfig(std::size_t max_fanout, double max_cap) -> ClusterConfig
{
  ClusterConfig config;
  config.max_fanout = max_fanout;
  config.max_diameter = std::numeric_limits<int>::max();
  config.max_cap = max_cap;
  return config;
}

auto FastClustering::runDefault(const std::vector<Pin*>& loads, const ClusterConfig& base_config) -> ClusterOutput
{
  return run(loads, base_config);
}

auto FastClustering::run(const std::vector<Pin*>& loads, const ClusterConfig& config) -> ClusterOutput
{
  const auto run_start = SteadyClock::now();
  ClusterOutput result;
  if (loads.empty()) {
    return result;
  }

  const auto collect_start = SteadyClock::now();
  auto entries = detail::CollectEntries(loads);
  LOG_INFO << "Fast clustering collect entries: input_loads=" << loads.size() << ", valid_entries=" << entries.size()
           << ", elapsed_time=" << FormatSeconds(ElapsedSeconds(collect_start)) << " s";
  if (entries.empty()) {
    LOG_WARNING << "Fast clustering skipped: no valid load pins.";
    return result;
  }

  const auto partition_start = SteadyClock::now();
  auto drafts = detail::BuildSpatialRecursiveClusters(entries, config);
  const auto partition_cluster_count = CountDraftClusters(drafts);
  LOG_INFO << "Fast clustering recursive partition: entries=" << entries.size() << ", drafts=" << partition_cluster_count
           << ", avg_fanout=" << FormatRatio(CalcAverageFanout(CountDraftLoads(drafts), partition_cluster_count))
           << ", fanout_histogram=" << FormatFanoutHistogram(BuildDraftFanoutHistogram(drafts))
           << ", elapsed_time=" << FormatSeconds(ElapsedSeconds(partition_start)) << " s";

  const auto polish_start = SteadyClock::now();
  detail::PolishSmallClusters(drafts, entries, config);
  const auto polished_cluster_count = CountDraftClusters(drafts);
  LOG_INFO << "Fast clustering polish: drafts=" << polished_cluster_count
           << ", avg_fanout=" << FormatRatio(CalcAverageFanout(CountDraftLoads(drafts), polished_cluster_count))
           << ", fanout_histogram=" << FormatFanoutHistogram(BuildDraftFanoutHistogram(drafts))
           << ", elapsed_time=" << FormatSeconds(ElapsedSeconds(polish_start)) << " s";

  const auto finalize_start = SteadyClock::now();
  auto finalized = detail::FinalizeClusters(drafts, entries, config);
  const auto finalize_elapsed_seconds = ElapsedSeconds(finalize_start);
  if (!finalized.has_value() || detail::CountAssignedLoads(*finalized) != entries.size()) {
    LOG_WARNING << "Fast clustering failed to produce a legal complete partition.";
    return result;
  }

  const auto assigned_load_count = detail::CountAssignedLoads(*finalized);
  LOG_INFO << "Fast clustering finalize: clusters=" << finalized->clusters.size() << ", assigned_loads=" << assigned_load_count
           << ", avg_fanout=" << FormatRatio(CalcAverageFanout(assigned_load_count, finalized->clusters.size()))
           << ", fanout_histogram=" << FormatFanoutHistogram(BuildResultFanoutHistogram(*finalized))
           << ", elapsed_time=" << FormatSeconds(finalize_elapsed_seconds) << " s";
  LOG_INFO << "Fast clustering done: loads=" << entries.size() << ", clusters=" << finalized->clusters.size()
           << ", avg_fanout=" << FormatRatio(CalcAverageFanout(assigned_load_count, finalized->clusters.size()))
           << ", total_elapsed_time=" << FormatSeconds(ElapsedSeconds(run_start)) << " s, strategy=recursive_spatial_bisect";
  return *finalized;
}

}  // namespace icts
